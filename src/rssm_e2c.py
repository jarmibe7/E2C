"""
e2c.py

Embed to Control model architecture made with PyTorch.

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn
import numpy as np

from src.encoder import ConvEncoder
from src.decoder import ConvDecoder, ChannelUncertaintyConvDecoder, ScalarUncertaintyConvDecoder



class RSSME2C(nn.Module):
    """
    An E2C model with convolutional encoder-decoder and an RSSM transition model.

    Args:
        enc_latent_size: Latent dimension of encoder
        stochastic_size: Stochastic state latent dimension
        deterministic_size: Deterministic state latent dimension
        control_size: Dimension of control vector
        past_length: Length of training observation history
        pred_length: Prediction horizon length
        conv_params: Dictionary containing CNN params for encoder/decoder
        device: Torch device object
        uncertainty_output: Whether to use ChannelUncertainty decoder
    """
    def __init__(self, enc_latent_size, stochastic_size, deterministic_size,
                 control_size, past_length, pred_length, conv_params, device, output_uncertainty=False):
        super().__init__()
        self.device = device
        self.output_uncertainty = output_uncertainty

        # Set number of hidden units
        self.enc_latent_size = enc_latent_size
        self.stochastic_size = stochastic_size              # Stochastic state
        self.deterministic_size = deterministic_size       # Deterministic state
        self.control_size = control_size
        self.past_length = past_length
        self.pred_length = pred_length

        # Dummy zero control vector
        self.dummy_u = torch.zeros((1, self.control_size)).to(self.device)

        # Encoder and decoder
        in_channels = conv_params['in_image_shape'][0]
        self.encoder = ConvEncoder(enc_latent_size, in_channels, conv_params)
        if self.output_uncertainty:
            self.decoder = ChannelUncertaintyConvDecoder(stochastic_size, conv_params, self.encoder.out_dim_flat, self.encoder.out_shape)
        else:
            self.decoder = ConvDecoder(stochastic_size, conv_params, self.encoder.out_dim_flat, self.encoder.out_shape)

        # Dreamer dynamics model definition
        self.rnn = nn.GRUCell(                          # Recurrent model
            self.stochastic_size + self.control_size,
            self.deterministic_size
        )
        self.prior = nn.Sequential(                     # Transition model
            nn.Linear(self.deterministic_size, 200),
            nn.ReLU(),
            nn.Linear(200, 2 * self.stochastic_size)
        )

        self.post = nn.Sequential(                      # Representation model
            nn.Linear(self.enc_latent_size + self.deterministic_size, 200),
            nn.ReLU(),
            nn.Linear(200, 2 * self.stochastic_size)
        )

    def reparameterize(self, mu, log_var):
        # Get standard deviation from log variance
        std = torch.exp(0.5 * log_var)
        std = torch.clamp(std, min=1e-5, max=1e5) # Prevent std from being too small

        # Generate random noise epsilon of same shape std
        eps = torch.randn_like(std)

        # Return reparameterized sample
        return mu + eps * std
    
    def rssm_step(self, h, z, u):
        """
        Single RSSM step
        
        Args:
            h: deterministic state
            z: stochastic latent
            u: control input
        Returns: h_next, z_next, mu, log_var
        """
        rnn_input = torch.cat([z, u], dim=-1)
        h_next = self.rnn(rnn_input, h)

        stats = self.prior(h_next)
        mu, log_var = stats.chunk(2, dim=-1)
        z_next = self.reparameterize(mu, log_var)

        return h_next, z_next, mu, log_var

    def forward(self, x, x_next, u):
        batch_size = x.size(0)

        # Initialize current deterministic state h to zeros
        h = torch.zeros(batch_size, self.deterministic_size, device=self.device)

        # Encode current and next observations
        enc = self.encoder(x)
        enc_next = self.encoder(x_next)

        # Posterior for current observation
        post_in = torch.cat([enc, h], dim=-1)            # [batch, enc_latent + deterministic]
        post_stats = self.post(post_in)                  # [batch, 2*stochastic]
        mu, log_var = post_stats.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)        # Current stochastic latent state

        # RSSM prior rollout (predict next latent)
        h_next, z_pred, mu_pred, log_var_pred = self.rssm_step(h, z, u)

        # Posterior for next observation (used for KL computation)
        post_next_in = torch.cat([enc_next, h_next], dim=-1)
        post_next_stats = self.post(post_next_in)
        mu_next, log_var_next = post_next_stats.chunk(2, dim=-1)
        z_next = self.reparameterize(mu_next, log_var_next)     # Next stochastic latent state

        # Decode reconstructions and predictions
        if self.output_uncertainty:
            x_recon, x_recon_unc = self.decoder(z)
            x_next_recon, x_next_recon_unc = self.decoder(z_next)
            x_pred, x_pred_unc = self.decoder(z_pred)

            return {
                "x_recon": x_recon,
                "x_next_recon": x_next_recon,
                "x_pred": x_pred,
                "z": z,
                "z_next": z_next,
                "z_pred": z_pred,
                "mu": mu,
                "log_var": log_var,
                "mu_next": mu_next,
                "log_var_next": log_var_next,
                "mu_pred": mu_pred,
                "log_var_pred": log_var_pred,
                "x_recon_uncertainty": x_recon_unc,
                "x_next_recon_uncertainty": x_next_recon_unc,
                "x_pred_recon_uncertainty": x_pred_unc,
            }
        else:
            return {
                "x_recon": self.decoder(z),
                "x_next_recon": self.decoder(z_next),
                "x_pred": self.decoder(z_pred),
                "z": z,
                "z_next": z_next,
                "z_pred": z_pred,
                "mu": mu,
                "log_var": log_var,
                "mu_next": mu_next,
                "log_var_next": log_var_next,
                "mu_pred": mu_pred,
                "log_var_pred": log_var_pred,
            }
        
    def reconstruct(self, x_traj):
        """
        Reconstruct an entire trajectory to test encoder/decoder
        """
        with torch.no_grad():
            frames = []
            for x in x_traj:
                # Encode current state
                encoded = self.encoder(x.unsqueeze(0))
                flattened = encoded.view(encoded.size(0), -1)

                # Get latent variable
                mu = self.mu(flattened)
                log_var = self.log_var(flattened)
                z = self.reparameterize(mu, log_var)
                frames.append(self.decoder(z))

            return torch.concat(frames, dim=0).squeeze(0).to('cpu').permute(0, 2, 3, 1)

    def sample_traj(self, x0, seq_len):
        """
        Sample an entire trajectory, starting from an initial condition
        """
        with torch.no_grad():
            # Encode current state
            encoded = self.encoder(x0.unsqueeze(0).to(self.device))
            flattened = encoded.view(encoded.size(0), -1)

            # Get latent variable
            mu = self.mu(flattened)
            log_var = self.log_var(flattened)
            z = self.reparameterize(mu, log_var)

            # Transition model
            frames = []
            frames.append(self.decoder(z))
            for t in range(seq_len):
                mu, log_var, z, _, _ = self.transition(z, mu, log_var, self.dummy_u)
                frames.append(self.decoder(z))

            return torch.concat(frames, dim=0).squeeze(0).to('cpu').permute(0, 2, 3, 1)
        
    def sample(self, x, u, return_all=False):
        """
        Predict the next image in a sequence
        """
        with torch.no_grad():
            sample_return = {}

            # Encode current state
            encoded = self.encoder(x.to(self.device))
            flattened = encoded.view(encoded.size(0), -1)

            # Initialize deterministic state h to zeros
            h = torch.zeros(encoded.shape[0], self.deterministic_size, device=self.device)

            # Posterior for current observation
            post_in = torch.cat([encoded, h], dim=-1)            # [batch, enc_latent + deterministic]
            post_stats = self.post(post_in)                      # [batch, 2*stochastic]
            mu, log_var = post_stats.chunk(2, dim=-1)
            z = self.reparameterize(mu, log_var)

            # RSSM prior rollout (predict next latent)
            h_next, z_pred, mu_pred, log_var_pred = self.rssm_step(h, z, u)
            sample_return['mu'] = mu
            sample_return['log_var'] = log_var

            sample_return['h_next'] = h_next
            sample_return['mu_pred'] = mu_pred
            sample_return['log_var_pred'] = log_var_pred
            sample_return['z_pred'] = z_pred
            if self.output_uncertainty:
                x_recon, x_recon_uncertainty = self.decoder(z)
                x_pred, x_pred_recon_uncertainty = self.decoder(z_pred)
                sample_return['x_recon_uncertainty'] = x_recon_uncertainty
                sample_return['x_pred_recon_uncertainty'] = x_pred_recon_uncertainty
            else:
                x_recon = self.decoder(z)
                x_pred = self.decoder(z_pred)

            if return_all and self.output_uncertainty:
                return x_recon.squeeze(0), x_pred.squeeze(0), sample_return
            else:
                return x_recon.squeeze(0), x_pred.squeeze(0)