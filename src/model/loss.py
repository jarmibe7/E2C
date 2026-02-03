"""
A collection of loss functions for world model training

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn

class VAELoss(nn.Module):
    """
    Basic VAE Loss, as a PyTorch module.
    """
    def __init__(self, num_epochs, loss_params):
        super().__init__()
        self.num_epochs = num_epochs
        self.recon_mult = loss_params.get('recon_mult', 1000.0)
        self.beta = loss_params['beta']
        self.anneal_mode = loss_params['kld_anneal_mode']
        self.free_nats = loss_params.get('free_nats', 1.0)

    def kld_anneal(self, epoch):
        if self.anneal_mode == 'const':
            mult = self.beta
        elif self.anneal_mode == 'linear':
            mult = self.beta*((epoch + 1)/self.num_epochs)
        else:
            raise NotImplementedError(f"Annealing mode {self.anneal_mode} not supported!")

        return mult

    def forward(self, tr, epoch):
        # Reconstruction loss
        recon = self.recon_mult*nn.functional.mse_loss(tr['x'], tr['x_recon'], reduction='mean')

        # Encoding KL Divergence
        log_var, mu = tr['log_var'], tr['mu']
        kld = self.kld_anneal(epoch)*(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean())
        kld = torch.clamp(kld, min=self.free_nats, max=1e-2)

        loss = recon + kld
        if torch.isnan(loss):
            breakpoint()

        # Make return dictionary for loss values
        loss_return = {
            r"$x$ Reconstruction Loss": recon.detach().cpu().item(),
            "KLD": kld.detach().cpu().item(),
        }
        return loss, loss_return

class E2CLoss(nn.Module):
    """
    E2C loss, made with PyTorch.
    """
    def __init__(self, num_epochs, loss_params):
        super().__init__()
        self.num_epochs = num_epochs
        self.recon_mult = loss_params.get('recon_mult', 1000.0)
        self.beta = loss_params['beta']
        self.lam = loss_params['lambda']
        self.anneal_mode = loss_params['kld_anneal_mode']

    def kld_anneal(self, epoch):
        if self.anneal_mode == 'const':
            mult = self.beta
        elif self.anneal_mode == 'linear':
            mult = self.beta*((epoch + 1)/self.num_epochs)
        else:
            raise NotImplementedError(f"Annealing mode {self.anneal_mode} not supported!")

        return mult

    def forward(self, tr, epoch):
        # Reconstruction loss
        recon = self.recon_mult*nn.functional.mse_loss(tr['x'], tr['x_recon'], reduction='mean')
        recon_next = self.recon_mult*nn.functional.mse_loss(tr['x_next'], tr['x_pred'], reduction='mean')

        # Encoding KL Divergence
        log_var, mu = tr['log_var'], tr['mu']
        kld = self.kld_anneal(epoch)*(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean())

        # Transition model KLD
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        log_var_pred, mu_pred = tr['log_var_pred'], tr['mu_pred']
        log_var_next, mu_next = tr['log_var_next'], tr['mu_next']
        var_next, var_pred = torch.exp(log_var_next), torch.exp(log_var_pred)

        # Times 0.5 because formula uses standard dev
        # kld_trans_vec = 0.5*(log_var_pred - log_var_next) \
        #                    + ((var_next + (mu_next - mu_pred)**2) / (var_pred + 1e-8)) \
        #                    - 1.0
        # kld_trans = 0.5 * torch.sum(kld_trans_vec)

        # v, r = tr['v'], tr['r']
        # dot = torch.sum(v * r, dim=1)
        # dot = torch.clamp(dot, min=-0.99)  # Ensure log(1 + dot) > 0
        # sum_term = torch.sum(log_var_pred - log_var_next, axis=1)
        # log_term = torch.log(1 + dot)
        # kld_trans_vec = 2*(sum_term - log_term)
        # kld_trans = self.lam*torch.sum(kld_trans_vec)

        # def kl_diag_gaussians(mu_p, log_var_p, mu_q, log_var_q):
        #     var_p = torch.exp(log_var_p)
        #     var_q = torch.exp(log_var_q)

        #     return 0.5 * torch.sum(
        #         log_var_q - log_var_p
        #         + (var_p + (mu_p - mu_q).pow(2)) / var_q
        #         - 1,
        #         dim=-1
        #     )
        # kld_trans = self.lam * kl_diag_gaussians(
        #     mu_pred, log_var_pred,
        #     mu_next, log_var_next
        # ).mean()

        z_pred = tr['z_pred']
        kld_trans = self.lam*(-0.5 * torch.sum(1 + log_var_next - (z_pred - mu_next).pow(2) - log_var_next.exp(), dim=-1).mean())

        loss = recon + recon_next + kld + kld_trans
        if torch.isnan(loss):
            breakpoint()

        # Make return dictionary for loss values
        loss_return = {
            r"$x$ Reconstruction Loss": recon.detach().cpu().item(),
            r"$x_{next}$ Reconstruction Loss": recon_next.detach().cpu().item(),
            "KLD": kld.detach().cpu().item(),
            "Transition KLD": kld_trans.detach().cpu().item()
        }
        return loss, loss_return
    
class UncertaintyE2CLoss(E2CLoss):
    """
    E2C loss with full image prediction uncertainty estimates, with reconstruction losses as negative loss likelihood.
    """
    def __init__(self, num_epochs, loss_params):
        super().__init__(num_epochs, loss_params)

    def expand_uncertainty(self, logvar, target):
        """
        Ensure uncertainty (log variance) output from decoder is of correct shape to be used by 
        gaussian_nll loss function in PyTorch API.

        Args:
            logvar: uncertainty tensor (any of:
                    [B], [B,1], [B,1,1,1], [B,K,H,W], [B,C,H,W])
                    where C % k == 0
            target: image tensor [B,C,H,W]
        """
        B, C, H, W = target.shape

        # Ensure 4D
        while logvar.dim() < 4:
            logvar = logvar.unsqueeze(-1)

        # [B,1,1,1] -> [B,C,H,W]
        if logvar.shape[1] == 1:
            logvar = logvar.expand(B, C, H, W)

        # [B,K,H,W] where K != C
        elif logvar.shape[1] != C:
            if C % logvar.shape[1] != 0:
                raise ValueError(
                    f"Cannot expand uncertainty with {logvar.shape[1]} channels "
                    f"to match image with {C} channels"
                )
            
            # Tile uncertainty to be shape compatible
            repeat_factor = C // logvar.shape[1]
            logvar = logvar.repeat(1, repeat_factor, 1, 1)

        return logvar

    def forward(self, tr, epoch):
        # Ensure uncertainty shape is compatible with loss function
        x_recon_uncertainty = self.expand_uncertainty(tr['x_recon_uncertainty'], tr['x_recon'])
        x_pred_recon_uncertainty = self.expand_uncertainty(tr['x_pred_recon_uncertainty'], tr['x_pred'])
        recon_nll = nn.functional.gaussian_nll_loss(
            tr['x_recon'],
            tr['x'], 
            torch.exp(x_recon_uncertainty) + 1e-6,    # Train decoder uncertainty outputs to be log variance
            reduction='mean'
        )
        pred_nll = nn.functional.gaussian_nll_loss(
            tr['x_pred'],
            tr['x_next'], 
            torch.exp(x_pred_recon_uncertainty) + 1e-6,
            reduction='mean'
        )
        # recon_nll = nn.functional.gaussian_nll_loss(
        #     tr['x_recon'],
        #     tr['x'], 
        #     torch.exp(tr['x_recon_uncertainty'].view(-1, 1, 1, 1).expand_as(tr['x_recon'])) + 1e-6,    # Train decoder uncertainty outputs to be log variance
        #     reduction='mean'
        # )
        # pred_nll = nn.functional.gaussian_nll_loss(
        #     tr['x_pred'],
        #     tr['x_next'], 
        #     torch.exp(tr['x_pred_recon_uncertainty'].view(-1, 1, 1, 1).expand_as(tr['x_pred'])) + 1e-6,
        #     reduction='mean'
        # )
        # recon_logvar = tr['x_recon_uncertainty'].expand_as(tr['x_recon']) + 1e-6
        # recon_var = torch.exp(recon_logvar ** 2) + 1e-6
        # recon_nll = -((tr['x'] - tr['x_recon']) ** 2) / (2 * recon_var) - recon_logvar - np.log(np.sqrt(2 * np.pi))
        # recon_nll = recon_nll.mean()

        # pred_logvar = tr['x_pred_recon_uncertainty'].expand_as(tr['x_pred']) + 1e-6
        # pred_var = torch.exp(pred_logvar ** 2) + 1e-6
        # pred_nll = -((tr['x_next'] - tr['x_pred']) ** 2) / (2 * pred_var) - pred_logvar - np.log(np.sqrt(2 * np.pi))
        # pred_nll = pred_nll.mean()

        recon = recon_nll + self.recon_mult*nn.functional.mse_loss(tr['x'], tr['x_recon'], reduction='mean')
        recon_next = pred_nll + self.recon_mult*nn.functional.mse_loss(tr['x_next'], tr['x_pred'], reduction='mean')

        # Encoding KL Divergence
        log_var, mu = tr['log_var'], tr['mu']
        kld = self.kld_anneal(epoch)*(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean())

        # Transition model KLD
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        log_var_pred, mu_pred = tr['log_var_pred'], tr['mu_pred']
        log_var_next, mu_next = tr['log_var_next'], tr['mu_next']
        z_pred = tr['z_pred']
        kld_trans = self.lam*(-0.5 * torch.sum(1 + log_var_next - (z_pred - mu_next).pow(2) - log_var_next.exp(), dim=-1).mean())

        loss = recon + recon_next + kld + kld_trans
        if torch.isnan(loss):
            breakpoint()

        # Make return dictionary for loss values
        loss_return = {
            r"$x$ Reconstruction Loss": recon.detach().cpu().item(),
            r"$x_{next}$ Reconstruction Loss": recon_next.detach().cpu().item(),
            r"$x$ NLL Reconstruction Loss": recon_nll.detach().cpu().item(),
            r"$x_{next}$ NLL Reconstruction Loss": pred_nll.detach().cpu().item(),
            "KLD": kld.detach().cpu().item(),
            "Transition KLD": kld_trans.detach().cpu().item()
        }
        return loss, loss_return
    

class RSSMLoss(nn.Module):
    """
    RSSM loss, made with PyTorch.
    """
    def __init__(self, num_epochs, loss_params):
        super().__init__()
        self.num_epochs = num_epochs
        self.recon_mult = loss_params['recon_mult']
        self.beta = loss_params['beta']
        self.free_nats = loss_params.get('free_nats', 0.0)
        self.anneal_mode = loss_params['kld_anneal_mode']
        self.image_loss = loss_params.get('image_loss', 'nll')

    def kld_anneal(self, epoch):
        if self.anneal_mode == 'const':
            mult = self.beta
        elif self.anneal_mode == 'linear':
            mult = min([self.beta / 10, self.beta*((epoch + 1)/self.num_epochs), self.beta*((epoch + 1)/self.num_epochs / 2)])
        elif self.anneal_mode == 'reverse':
            mult = self.beta*((self.num_epochs - 1)/self.num_epochs) + self.beta / 10
        else:
            raise NotImplementedError(f"Annealing mode {self.anneal_mode} not supported!")

        return mult
    
    def kl_divergence(self, mu_q, logvar_q, mu_p, logvar_p):
        return 0.5 * (
            logvar_p - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
            - 1
        ).sum(dim=-1)
    
    def expand_uncertainty(self, logvar, target):
        """
        Ensure uncertainty (log variance) output from decoder is of correct shape to be used by 
        gaussian_nll loss function in PyTorch API.

        Args:
            logvar: uncertainty tensor (any of:
                    [B], [B,1], [B,1,1,1], [B,K,H,W], [B,C,H,W])
                    where C % k == 0
            target: image tensor [B,C,H,W]
        """
        B, T, C, H, W = target.shape

        # Ensure 5D
        while logvar.dim() < 5:
            logvar = logvar.unsqueeze(-1)

        # [B,T,1,1,1] -> [B,T,C,H,W]
        if logvar.shape[2] == 1:
            logvar = logvar.expand(B, T, C, H, W)

        # [B,T,K,H,W] where K != C
        elif logvar.shape[2] != C:
            if C % logvar.shape[3] != 0:
                raise ValueError(
                    f"Cannot expand uncertainty with {logvar.shape[2]} channels "
                    f"to match image with {C} channels"
                )
            
            # Tile uncertainty to be shape compatible
            repeat_factor = C // logvar.shape[2]
            logvar = logvar.repeat(1, 1, repeat_factor, 1, 1)

        return logvar

    def forward(self, tr, epoch):
        # Reconstruction loss
        x_pred_uncertainty = self.expand_uncertainty(tr['x_pred_uncertainty'], tr['x_pred'])
        if self.image_loss == 'mse':
            recon = self.recon_mult*nn.functional.mse_loss(tr['x_next'], tr['x_pred'], reduction='mean')
            recon += self.recon_mult*nn.functional.mse_loss(tr['x'][:, -1], tr['x_recon'], reduction='mean') # only reconstruct last in past_length
        elif self.image_loss == 'nll':
            # TODO: Add reconstruction loss for tr['x_recon']
            recon = nn.functional.gaussian_nll_loss(
                tr['x_next'],
                tr['x_pred'], 
                torch.exp(x_pred_uncertainty) + 1e-6,
                reduction='mean'
            )

        # Encoding KL Divergence
        # KL loss (posterior vs prior)
        kld = self.kl_divergence(
            tr["mu_posts"],
            tr["log_var_posts"],
            tr["mu_priors"],
            tr["log_var_priors"]
        )
        kld = torch.clamp(kld, min=self.free_nats)#, max=0.1)
        kld = kld.mean()
        kld = self.kld_anneal(epoch)*kld

        loss = recon + kld
        if torch.isnan(loss):
            breakpoint()

        # Make return dictionary for loss values
        loss_return = {
            r"$x$ Reconstruction Loss": recon.detach().cpu().item(),
            "KLD": kld.detach().cpu().item(),
        }
        return loss, loss_return