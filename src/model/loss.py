"""
World model training losses for data4wm.

Only the RSSM loss is active; the legacy E2C/UncertaintyE2C losses were removed
when the ConvE2C model path was retired.

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn


class RSSMLoss(nn.Module):
    """
    RSSM training loss: image reconstruction + (KL-annealed) posterior-prior KL.

    Notes:
    - ``free_nats`` clamps the KL to prevent it from collapsing to zero early
      in training.
    - ``kl_divergence`` here is intentionally duplicated with the
      ``_kl_diag_gaussian`` helper in :class:`ClosedLoopInformativeTrainer`;
      one is used as a training loss term, the other as an online planning
      objective, and the two are kept decoupled on purpose.
    """
    def __init__(self, num_epochs, loss_params):
        super().__init__()
        self.num_epochs = num_epochs
        self.recon_mult = loss_params['recon_mult']
        self.beta = loss_params['beta']
        self.lam = loss_params['lambda']
        self.free_nats = loss_params.get('free_nats', 0.0)
        self.anneal_mode = loss_params['kld_anneal_mode']
        self.image_loss = loss_params.get('image_loss', 'nll')

    def kld_anneal(self, epoch):
        """Return the KL multiplier for the given epoch.

        ``const`` keeps ``beta`` fixed; ``linear`` ramps from 0 to ``beta``
        over ``num_epochs``.
        """
        if self.anneal_mode == 'const':
            return self.beta
        if self.anneal_mode == 'linear':
            return self.beta * ((epoch + 1) / self.num_epochs)
        raise NotImplementedError(f"Annealing mode {self.anneal_mode} not supported!")

    def kl_divergence(self, mu_q, logvar_q, mu_p, logvar_p):
        """Analytic KL(q || p) for independent diagonal Gaussians, summed over
        the last dim (latent dim)."""
        return 0.5 * (
            logvar_p - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
            - 1
        ).sum(dim=-1)

    def expand_uncertainty(self, logvar, target):
        """
        Reshape the decoder's log-variance output so it can be consumed by
        ``F.gaussian_nll_loss`` against ``target``.

        Args:
            logvar: Uncertainty tensor of shape ``[B]``, ``[B,1]``,
                ``[B,1,1,1]``, ``[B,K,H,W]`` or ``[B,C,H,W]`` (with ``C % K == 0``).
            target: Image tensor ``[B, T, C, H, W]``.
        """
        B, T, C, H, W = target.shape

        while logvar.dim() < 5:
            logvar = logvar.unsqueeze(-1)

        if logvar.shape[2] == 1:
            logvar = logvar.expand(B, T, C, H, W)
        elif logvar.shape[2] != C:
            if C % logvar.shape[3] != 0:
                raise ValueError(
                    f"Cannot expand uncertainty with {logvar.shape[2]} channels "
                    f"to match image with {C} channels"
                )
            repeat_factor = C // logvar.shape[2]
            logvar = logvar.repeat(1, 1, repeat_factor, 1, 1)

        return logvar

    def forward(self, tr, epoch):
        x_pred_uncertainty = self.expand_uncertainty(tr['x_pred_uncertainty'], tr['x_pred'])
        if self.image_loss == 'mse':
            recon = self.recon_mult * nn.functional.mse_loss(tr['x_next'], tr['x_pred'], reduction='mean')
            # Reconstruct only the last frame in ``past_length`` against x_recon.
            recon += self.recon_mult * nn.functional.mse_loss(tr['x'][:, -1], tr['x_recon'], reduction='mean')
        elif self.image_loss == 'nll':
            # TODO: Add reconstruction loss for tr['x_recon']
            recon = nn.functional.gaussian_nll_loss(
                tr['x_next'],
                tr['x_pred'],
                torch.exp(x_pred_uncertainty) + 1e-6,
                reduction='mean',
            )
        else:
            raise NotImplementedError(f"image_loss '{self.image_loss}' not supported!")

        # Posterior-vs-prior KL with free-nats clamp + beta annealing.
        kld = self.kl_divergence(
            tr["mu_posts"],
            tr["log_var_posts"],
            tr["mu_priors"],
            tr["log_var_priors"],
        )
        kld = torch.clamp(kld, min=self.free_nats).mean()
        kld = self.kld_anneal(epoch) * kld

        loss = recon + kld
        if torch.isnan(loss):
            breakpoint()

        loss_return = {
            r"$x$ Reconstruction Loss": recon.detach().cpu().item(),
            "KLD": kld.detach().cpu().item(),
        }
        return loss, loss_return
