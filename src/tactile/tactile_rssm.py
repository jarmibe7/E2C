"""
tactile_rssm.py

RSSM model architecture made with PyTorch, specifically for tactile data.

Author: Jared Berry
"""
import torch
from torch import nn
import numpy as np

from src.model.encoder import ConvEncoder
from src.model.decoder import ConvDecoder, ChannelUncertaintyConvDecoder, ScalarUncertaintyConvDecoder
from src.model.rssm import RSSME2C



class TactileRSSM(RSSME2C):
    """
    An RSSM with convolutional encoder-decoder and transition model.

    Args:
        enc_latent_size: Latent dimension of tactile image encoder
        feature_latent_size: Latent dimension of encoded feature vector
        stochastic_size: Stochastic state latent dimension
        deterministic_size: Deterministic state latent dimension
        control_size: Dimension of control vector
        past_length: Length of training observation history
        pred_length: Prediction horizon length
        conv_params: Dictionary containing CNN params for encoder/decoder
        device: Torch device object
        uncertainty_output: Whether to use ChannelUncertainty decoder
    """
    def __init__(self, feature_latent_size, enc_latent_size, stochastic_size, deterministic_size,
                 control_size, past_length, pred_length, conv_params, device, output_uncertainty=False):
        super().__init__(enc_latent_size, stochastic_size, deterministic_size, control_size, past_length, 
                         pred_length, conv_params, device, output_uncertainty=output_uncertainty, img_channel_count=1)
        
        self.feature_latent_size = feature_latent_size

        # TODO: Don't hardcode the feature size
        self.feature_size = 6
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_latent_size),
            nn.ReLU(),
        )

        self.feature_decoder = nn.Sequential(
            nn.Linear(self.stochastic_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.feature_size),
        )

        # Overwrite posterior model to account for feature vector
        self.post = nn.Sequential(                      # Representation model
            nn.Linear(self.enc_latent_size + self.feature_latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.stochastic_size)
        )
    
    def encode_posterior(self, tactile, feature, actions=None):
        B, T = tactile.shape[:2]
        mus, log_vars, zs, hs = [], [], [], []

        for t in range(T):
            # Encode tactile and feature
            tac = tactile[:, t]
            feat = feature[:, t]
            tac_enc = self.encoder(tac)
            feat_enc = self.feature_encoder(feat)
            enc = torch.cat([tac_enc, feat_enc], dim=-1)

            stats = self.post(enc)
            mu, log_var = stats.chunk(2, dim=-1)
            # log_var = torch.clamp(log_var, min=1e-5, max=1e5)
            z = self.reparameterize(mu, log_var)

            mus.append(mu)
            log_vars.append(log_var)
            zs.append(z)
        return (
            torch.stack(mus, dim=1),
            torch.stack(log_vars, dim=1),
            torch.stack(zs, dim=1),
        )
    
    def transition(self, tactile, feature, tactile_next, feature_next, u, zs):
        h = torch.zeros(self.num_layers, tactile.size(0), self.deterministic_size, device=self.device)
        # Take last belief state as start
        z = zs[:, -1]

        # reconstruct current observation
        tactile_recon = self.decoder(zs[:, -1])
        feature_recon = self.feature_decoder(zs[:, -1])

        # Iterate over pred_length
        mu_priors, log_var_priors = [], []
        mu_posts, log_var_posts = [], [] # old version - i added encoded t0 mu in post? why...
        tactile_preds = []
        feature_preds = []

        tactile_window = tactile
        feature_window = feature
        for t in range(tactile_next.size(1)):
            # prior
            h, z_prior, mu_p, log_var_p = self.rssm_step(h, z.unsqueeze(1), u[:, t])
            
            # old way (takes into account past_length)
            # h, z_prior, mu_p, log_var_p = self.rssm_step(h, zs, u[:, t])
            mu_priors.append(mu_p)
            log_var_priors.append(log_var_p)

            # decode prior (open loop prediction)
            tactile_pred = self.decoder(z_prior)
            feature_pred = self.feature_decoder(z_prior)
            tactile_preds.append(tactile_pred)
            feature_preds.append(feature_pred)

            if self.past_length > 1:
                tactile_frames = tactile_window[:, 1:]   # drop first frame
                feature_frames = feature_window[:, 1:]
                tactile_window = torch.cat([tactile_frames, tactile_pred.unsqueeze(1).detach()], dim=1)
                feature_window = torch.cat([feature_frames, feature_pred.unsqueeze(1).detach()], dim=1)
            else:
                tactile_window = tactile_pred.detach()  # past_length==1, just use pred image
                feature_window = feature_pred.detach()
            mu_q, log_var_q, zs = self.encode_posterior(tactile_window, feature_window)

            # # posterior update using real next frame
            # enc = self.encoder(x_next[:, t])
            # stats = self.post(enc)
            # mu_q, log_var_q = stats.chunk(2, dim=-1)
            # z = self.reparameterize(mu_q, log_var_q)

            mu_posts.append(mu_q[:, -1])
            log_var_posts.append(log_var_q[:, -1])

        # Stack accumulated priors + posteriors
        outputs = {
            "tactile_recon": tactile_recon,
            "feature_recon": feature_recon,
            "tactile_pred": torch.stack(tactile_preds, dim=1),
            "feature_pred": torch.stack(feature_preds, dim=1),
            "mu_posts": torch.stack(mu_posts, dim=1),
            "log_var_posts": torch.stack(log_var_posts, dim=1),
            "mu_priors": torch.stack(mu_priors, dim=1),
            "log_var_priors": torch.stack(log_var_priors, dim=1),
        }

        return outputs
    
    def forward(self, tactile, feature, tactile_next, feature_next, u):
        # Infer belief over past context
        _, _, zs = self.encode_posterior(tactile, feature, u[:, :tactile.size(1)])
        outputs = self.transition(tactile, feature, tactile_next, feature_next, u, zs)
        return outputs
    
class RSSMLoss(nn.Module):
    """
    RSSM loss, made with PyTorch.
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
        if self.anneal_mode == 'const':
            mult = self.beta
        elif self.anneal_mode == 'linear':
            # mult = min(self.beta*((epoch + 1)/self.num_epochs), 0.5 * self.beta)
            mult = self.beta*((epoch + 1)/self.num_epochs)
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
        kld = torch.clamp(kld, min=self.free_nats)
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