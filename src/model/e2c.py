"""
e2c.py

Embed to Control model architecture made with PyTorch.

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn
import numpy as np

from src.model.encoder import ConvEncoder
from src.model.decoder import ConvDecoder, ChannelUncertaintyConvDecoder, ScalarUncertaintyConvDecoder


class ConvE2C(nn.Module):
    """
    An E2C model with convolutional encoder-decoder.

    Args:
        enc_latent_size: Latent dimension of encoder
        latent_size: E2C latent dimension
        control_size: Dimension of control vector
        past_length: Length of training observation history
        pred_length: Prediction horizon length
        conv_params: Dictionary containing CNN params for encoder/decoder
        device: Torch device object
        uncertainty_output: Whether to use ChannelUncertainty decoder
    """
    def __init__(self, enc_latent_size, latent_size, control_size, past_length, pred_length, conv_params, device, output_uncertainty=False):
        super().__init__()
        self.device = device
        self.output_uncertainty = output_uncertainty

        # Set number of hidden units
        self.enc_latent_size = enc_latent_size
        self.latent_size = latent_size
        self.control_size = control_size
        self.past_length = past_length
        self.pred_length = pred_length
        assert self.pred_length == 1, 'pred_length > 1 not supported for regular E2C'

        # Dummy zero control vector
        self.dummy_u = torch.zeros((1, self.control_size)).to(self.device)

        # Encoder and decoder
        in_channels = conv_params['in_image_shape'][0]
        self.encoder = ConvEncoder(enc_latent_size, in_channels, conv_params)
        if self.output_uncertainty:
            self.decoder = ScalarUncertaintyConvDecoder(latent_size, conv_params, self.encoder.out_dim_flat, self.encoder.out_shape)
        else:
            self.decoder = ConvDecoder(latent_size, conv_params, self.encoder.out_dim_flat, self.encoder.out_shape)

        # VAE
        self.mu = nn.Linear(self.enc_latent_size, self.latent_size)
        self.log_var = nn.Linear(self.enc_latent_size, self.latent_size)

        # Locally linear transition model
        self.fc_trans = nn.Sequential(
            nn.Linear(self.latent_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 32)
        )
        self.v = nn.Linear(32, self.latent_size)  # A jacobian
        self.r = nn.Linear(32, self.latent_size)
        self.B = nn.Linear(32, self.latent_size*self.control_size)
        self.o = nn.Sequential(
            nn.Linear(32, self.latent_size),
            nn.Tanh()
        )
        self.H = nn.Sequential(
            nn.Linear(32, self.latent_size),
            nn.Softplus() # Ensure positive covariances
        )

    def reparameterize(self, mu, log_var):
        # Get standard deviation from log variance
        std = torch.exp(0.5 * log_var)
        std = torch.clamp(std, min=1e-5, max=1e5) # Prevent std from being too small

        # Generate random noise epsilon of same shape std
        eps = torch.randn_like(std)

        # Return reparameterized sample
        return mu + eps * std

    def transition(self, z, mu, log_var, u):
        # Pass through transition network
        trn = self.fc_trans(z)

        # Get Jacobians and linear model parameters
        v = self.v(trn).unsqueeze(-1)                                                    #   [batch, z, 1]
        r = self.r(trn).unsqueeze(-1)                                                    #   [batch, z, 1]
        A = torch.eye(self.latent_size, device=self.device).repeat(z.size(0), 1, 1) \
            + torch.bmm(v, r.transpose(1, 2))                                            #   [batch, z, z]
        B = self.B(trn).reshape((-1, self.latent_size, self.control_size))               #   [batch, z, u]
        o = self.o(trn).unsqueeze(-1)                                                    #   [batch, z, 1]
        H = torch.diag_embed(self.H(trn))                                                #   [batch, z, z]

        # Use linear model and reparam to create posterior distribution
        mu_pred = torch.bmm(A, mu.unsqueeze(-1)) + torch.bmm(B, u.unsqueeze(-1)) + o     #   [batch, z, 1]
        mu_pred = mu_pred.squeeze(-1)                                                    #   [batch, z]
        sigma = torch.diag_embed(torch.exp(log_var))                                     #   [batch, z, z]
        C = torch.bmm(A, torch.bmm(sigma, A.transpose(1, 2))) + H                        #   [batch, z, z]
        log_var_pred = torch.log(torch.diagonal(C, dim1=-2, dim2=-1) + 1e-8)             #   [batch, z]
        z_pred = self.reparameterize(mu_pred, log_var_pred)                              #   [batch, z]

        return mu_pred, log_var_pred, z_pred

    def forward(self, x, x_next, u):
        # Encode current and next state
        enc_out = self.encoder(x)
        enc_out_next = self.encoder(x_next)

        # Get latent variable
        mu = self.mu(enc_out)
        log_var = self.log_var(enc_out)
        z = self.reparameterize(mu, log_var)

        # Get next latent variable
        mu_next = self.mu(enc_out_next)
        log_var_next = self.log_var(enc_out_next)
        z_next = self.reparameterize(mu_next, log_var_next)

        # Transition model
        mu_pred, log_var_pred, z_pred = self.transition(z, mu, log_var, u)

        # Get reconstruction and prediction
        if self.output_uncertainty:
            x_recon, x_recon_uncertainty = self.decoder(z)
            x_next_recon, x_next_recon_uncertainty = self.decoder(z_next)
            x_pred, x_pred_recon_uncertainty = self.decoder(z_pred)
            train_return = {
                'x_recon': x_recon,
                'mu': mu,
                'log_var': log_var,
                'x_next_recon': x_next_recon,
                'mu_next': mu_next,
                'log_var_next': log_var_next,
                'x_pred': x_pred,
                'z_pred': z_pred,
                'mu_pred': mu_pred,
                'log_var_pred': log_var_pred,
                'x_recon_uncertainty': x_recon_uncertainty,
                'x_next_recon_uncertainty': x_next_recon_uncertainty,
                'x_pred_recon_uncertainty': x_pred_recon_uncertainty
            }
        else:
            x_recon = self.decoder(z)
            x_next_recon = self.decoder(z_next)
            x_pred = self.decoder(z_pred)
            train_return = {
                'x_recon': x_recon,
                'mu': mu,
                'log_var': log_var,
                'x_next_recon': x_next_recon,
                'mu_next': mu_next,
                'log_var_next': log_var_next,
                'x_pred': x_pred,
                'z_pred': z_pred,
                'mu_pred': mu_pred,
                'log_var_pred': log_var_pred,
            }

        return train_return

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

            # Get latent variable
            mu = self.mu(flattened)
            log_var = self.log_var(flattened)
            z = self.reparameterize(mu, log_var)
            sample_return['mu'] = mu
            sample_return['log_var'] = log_var

            # Predict transition and decode current pred and next pred
            mu_pred, log_var_pred, z_pred = self.transition(z, mu, log_var, u)
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

            if return_all:
                return x_recon.squeeze(0), x_pred.squeeze(0), sample_return
            else:
                return x_recon.squeeze(0), x_pred.squeeze(0)