"""
RSSM world model used by data4wm.

Architecture (see ``.cursor/prompts/network_architecture.png`` and Eq. (1) in
the paper):

- ``encoder`` -- CNN that maps each frame to a flat embedding.
- ``rnn``     -- GRU that carries the deterministic belief ``h_t``.
- ``prior``   -- dynamics head p(z_t | h_t); produces (mu, log_var).
- ``post``    -- representation head q(z_t | encoder(o_t)); produces (mu, log_var).
- ``decoder`` -- CNN that maps z_t back to pixel space, optionally with per-pixel uncertainty.

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn

from src.model.encoder import ConvEncoder
from src.model.decoder import ConvDecoder, ScalarUncertaintyConvDecoder


class RSSME2C(nn.Module):
    """
    RSSM-style E2C model with a convolutional encoder/decoder.

    Args:
        enc_latent_size: Latent dimension of encoder.
        stochastic_size: Stochastic state latent dimension (``z_t``).
        deterministic_size: Deterministic state latent dimension (``h_t``).
        control_size: Dimension of control vector.
        past_length: Length of training observation history fed into the encoder.
        pred_length: Prediction horizon length rolled out in ``forward``.
        conv_params: Dict of CNN params for encoder/decoder (see ``vae`` section
            of the yaml configs).
        device: Torch device object.
        output_uncertainty: If True, use the scalar-uncertainty decoder head.
    """
    def __init__(self, enc_latent_size, stochastic_size, deterministic_size,
                 control_size, past_length, pred_length, conv_params, device, output_uncertainty=False):
        super().__init__()
        self.device = device
        self.output_uncertainty = output_uncertainty

        self.enc_latent_size = enc_latent_size
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.control_size = control_size
        self.past_length = past_length
        self.pred_length = pred_length

        self.dummy_u = torch.zeros((1, self.control_size)).to(self.device)

        # Encoder / decoder. ``in_image_shape[0]`` stacks ``past_length`` RGB
        # frames along the channel axis; divide out the factor of 3 here.
        in_channels = conv_params['in_image_shape'][0] // 3
        self.encoder = ConvEncoder(enc_latent_size, in_channels, conv_params)
        if self.output_uncertainty:
            self.decoder = ScalarUncertaintyConvDecoder(
                stochastic_size, conv_params, self.encoder.out_dim_flat, self.encoder.out_shape
            )
        else:
            self.decoder = ConvDecoder(
                stochastic_size, conv_params, self.encoder.out_dim_flat, self.encoder.out_shape
            )
        self.out_image_shape = self.decoder.out_image_shape

        # Dreamer-style RSSM dynamics
        self.num_layers = 2
        self.rnn = nn.GRU(
            self.stochastic_size + self.control_size,
            self.deterministic_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.prior = nn.Sequential(                     # Transition model p(z_t | h_t)
            nn.Linear(self.deterministic_size, 200),
            nn.ReLU(),
            nn.Linear(200, 2 * self.stochastic_size),
        )
        self.post = nn.Sequential(                      # Representation model q(z_t | enc)
            nn.Linear(self.enc_latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.stochastic_size),
        )

    def reparameterize(self, mu, log_var):
        """Standard VAE reparameterization trick with std clamped for stability."""
        std = torch.exp(0.5 * log_var)
        std = torch.clamp(std, min=1e-5, max=1e5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def rssm_step(self, h, z, u):
        """
        Single RSSM step.

        Args:
            h: Deterministic state ``[num_layers, B, deterministic_size]``.
            z: Stochastic latent ``[B, T, stochastic_size]`` (T is typically 1).
            u: Control input ``[B, control_size]``.

        Returns:
            Tuple ``(h_next, z_next, mu, log_var)``.
        """
        u = u.unsqueeze(1).repeat(1, z.size(1), 1)
        rnn_input = torch.cat([z, u], dim=-1)
        _, h_next = self.rnn(rnn_input, h)

        stats = self.prior(h_next[-1])
        mu, log_var = stats.chunk(2, dim=-1)
        z_next = self.reparameterize(mu, log_var)
        return h_next, z_next, mu, log_var

    def encode_posterior(self, obs, actions=None):
        """Encode a sequence of observations ``[B, T, C, H, W]`` into the
        posterior ``q(z_t | o_{1:t})`` at each time step.

        Returns stacked ``(mus, log_vars, zs)`` each of shape ``[B, T, ...]``.
        ``actions`` is currently unused -- kept for future extensions where the
        posterior may condition on past actions.
        """
        T = obs.shape[1]
        mus, log_vars, zs = [], [], []

        for t in range(T):
            x = obs[:, t]
            enc = self.encoder(x)
            stats = self.post(enc)
            mu, log_var = stats.chunk(2, dim=-1)
            z = self.reparameterize(mu, log_var)
            mus.append(mu)
            log_vars.append(log_var)
            zs.append(z)
        return (
            torch.stack(mus, dim=1),
            torch.stack(log_vars, dim=1),
            torch.stack(zs, dim=1),
        )

    def forward(self, x, x_next, u):
        """Roll out posterior + prior over ``pred_length`` steps.

        Args:
            x: Past-context frames ``[B, past_length, C, H, W]``.
            x_next: Future frames ``[B, pred_length, C, H, W]`` (only shape[1]
                is used; RSSM is trained on its own imagined rollouts).
            u: Full action sequence ``[B, past_length + pred_length, A]``.

        Returns dict with reconstruction, open-loop predictions, and the
        prior/posterior means and log-vars used by :class:`RSSMLoss`.
        """
        mus, log_vars, zs = self.encode_posterior(x, u[:, :x.size(1)])
        h = torch.zeros(self.num_layers, x.size(0), self.deterministic_size, device=self.device)
        z = zs[:, -1]

        if self.output_uncertainty:
            x_recon, x_recon_uncertainty = self.decoder(zs[:, -1])
        else:
            x_recon = self.decoder(zs[:, -1])

        mu_priors, log_var_priors = [], []
        mu_posts, log_var_posts = [], []
        x_preds = []
        if self.output_uncertainty:
            x_pred_uncerts = []

        window = x
        for t in range(x_next.size(1)):
            h, z_prior, mu_p, log_var_p = self.rssm_step(h, z.unsqueeze(1), u[:, t])
            mu_priors.append(mu_p)
            log_var_priors.append(log_var_p)

            if self.output_uncertainty:
                x_pred, x_pred_uncert = self.decoder(z_prior)
                x_pred_uncerts.append(x_pred_uncert)
            else:
                x_pred = self.decoder(z_prior)
            x_preds.append(x_pred)

            if self.past_length > 1:
                window_frames = window[:, 1:]
                window = torch.cat([window_frames, x_pred.unsqueeze(1).detach()], dim=1)
            else:
                window = x_pred.detach()
            mu_q, log_var_q, zs = self.encode_posterior(window)

            mu_posts.append(mu_q[:, -1])
            log_var_posts.append(log_var_q[:, -1])

        outputs = {
            "x_recon": x_recon,
            "x_pred": torch.stack(x_preds, dim=1),
            "mu_posts": torch.stack(mu_posts, dim=1),
            "log_var_posts": torch.stack(log_var_posts, dim=1),
            "mu_priors": torch.stack(mu_priors, dim=1),
            "log_var_priors": torch.stack(log_var_priors, dim=1),
        }

        if self.output_uncertainty:
            outputs["x_recon_uncertainty"] = x_recon_uncertainty
            outputs["x_pred_uncertainty"] = torch.stack(x_pred_uncerts, dim=1)

        return outputs

    def sample_traj(self, x0, u_seq):
        """Open-loop rollout starting from a single initial observation.

        Args:
            x0: Initial image ``[C, H, W]`` or ``[B, C, H, W]``.
            u_seq: Action sequence ``[T, A]``.

        Returns:
            Decoded frames ``[T+1, H, W, C]`` (starting with the decoded
            reconstruction of ``x0``).
        """
        self.eval()
        with torch.no_grad():
            if x0.dim() == 3:
                x0 = x0.unsqueeze(0)

            batch_size = x0.size(0)
            seq_len = u_seq.size(0)

            h = torch.zeros(self.num_layers, batch_size, self.deterministic_size, device=self.device)
            enc = self.encoder(x0.to(self.device))
            stats = self.post(enc)
            mu, log_var = stats.chunk(2, dim=-1)
            z = self.reparameterize(mu, log_var)

            frames = []
            if self.output_uncertainty:
                x_dec, _ = self.decoder(z)
            else:
                x_dec = self.decoder(z)
            frames.append(x_dec)

            for t in range(seq_len):
                u_t = u_seq[t].unsqueeze(0).to(self.device)
                h, z, _, _ = self.rssm_step(h, z.unsqueeze(1), u_t)
                if self.output_uncertainty:
                    x_dec, _ = self.decoder(z)
                else:
                    x_dec = self.decoder(z)
                frames.append(x_dec)

            frames = torch.cat(frames, dim=0)
            frames = frames.permute(0, 2, 3, 1).cpu()
            return frames

    def sample(self, x, u):
        """Single-step prediction.

        Encodes the current window ``x``, steps the prior once with ``u``,
        and decodes both the reconstruction of the current frame and the
        predicted next frame.
        """
        self.eval()
        with torch.no_grad():
            sample_return = {}

            mu, log_var, z = self.encode_posterior(x)
            h = torch.zeros(self.num_layers, mu.shape[0], self.deterministic_size, device=self.device)

            h_next, z_pred, mu_pred, log_var_pred = self.rssm_step(h, z, u)

            if self.output_uncertainty:
                x_recon, x_recon_uncertainty = self.decoder(z)
                x_pred, x_pred_recon_uncertainty = self.decoder(z_pred)
                sample_return['x_recon_uncertainty'] = x_recon_uncertainty
                sample_return['x_pred_recon_uncertainty'] = x_pred_recon_uncertainty
            else:
                x_recon = self.decoder(z)
                x_pred = self.decoder(z_pred)

            sample_return['x_recon'] = x_recon
            sample_return['x_pred'] = x_pred
            sample_return['mu'] = mu
            sample_return['log_var'] = log_var
            sample_return['h_next'] = h_next
            sample_return['mu_pred'] = mu_pred
            sample_return['log_var_pred'] = log_var_pred
            sample_return['z_pred'] = z_pred

            return x_recon.squeeze(0), x_pred.squeeze(0)
