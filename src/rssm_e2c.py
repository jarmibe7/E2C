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
        self.out_image_shape = self.decoder.out_image_shape

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

        # Posterior for current observations
        post_in = torch.cat([enc, h], dim=-1)            # [batch, enc_latent + deterministic]
        post_stats = self.post(post_in)                  # [batch, 2*stochastic]
        mu, log_var = post_stats.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)        # Current stochastic latent state
        if self.output_uncertainty:
            x_recon, x_recon_uncertainty = self.decoder(z)
        else:
            x_recon = self.decoder(z)

        # Iterate over pred_length
        z_preds, mu_priors, log_var_priors = [], [], []
        mu_posts, log_var_posts = [mu], [log_var]
        x_preds = []

        # Initialize training encoding window
        window = x
        
        for t in range(x_next.shape[1]):
            # RSSM prior rollout (predict next latent)
            h, z_pred, mu_p, log_var_p = self.rssm_step(h, z, u[:,t])
            z_preds.append(z_pred)
            mu_priors.append(mu_p)
            log_var_priors.append(log_var_p)
            
            # Decode predicted latent
            if self.output_uncertainty:
                x_pred, _ = self.decoder(z_pred)
            else:
                x_pred = self.decoder(z_pred)
            x_preds.append(x_pred)
            
            # # Encode next observation
            # x_next_frame = x_next[:, t]
            # x_next_enc = self.encoder(torch.cat([x_next_frame]*self.past_length, dim=1))  # Need to stack for encoding

            # Build next window for encoder: shift old window and append predicted frame
            # window: [B, past_length*C, H, W] => keep last past_length-1 frames
            if self.past_length > 1:
                window_frames = window[:, (self.out_image_shape[0] * 1):, :, :]   # drop first frame
                window = torch.cat([window_frames, x_pred.detach()], dim=1)
            else:
                window = x_pred.detach()  # past_length==1, just use pred image

            # Incorporate posterior
            x_next_enc = self.encoder(window)
            post_in = torch.cat([x_next_enc, h], dim=-1)            # [batch, enc_latent + deterministic]
            post_stats = self.post(post_in)                         # [batch, 2*stochastic]
            mu, log_var = post_stats.chunk(2, dim=-1)
            z = self.reparameterize(mu, log_var)        # Current stochastic latent state

            mu_posts.append(mu)
            log_var_posts.append(log_var)


        # Stack accumulated priors + posteriors
        if self.output_uncertainty:
            return {
                "x_recon": x_recon,
                "x_pred": torch.stack(x_preds, dim=1),
                "mu_posts": torch.stack(mu_posts[:-1], dim=1),
                "log_var_posts": torch.stack(log_var_posts[:-1], dim=1),
                "mu_priors": torch.stack(mu_priors, dim=1),
                "log_var_priors": torch.stack(log_var_priors, dim=1),
                "x_recon_uncertainty": x_recon_uncertainty
            }
        else:
            return {
                "x_recon": x_recon,
                "x_pred": torch.stack(x_preds, dim=1),
                "mu_posts": torch.stack(mu_posts[:-1], dim=1),
                "log_var_posts": torch.stack(log_var_posts[:-1], dim=1),
                "mu_priors": torch.stack(mu_priors, dim=1),
                "log_var_priors": torch.stack(log_var_priors, dim=1),
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

    def sample_traj(self, x0, u_seq):
        """
        Sample an entire trajectory, starting from an initial condition
        """
        self.eval()
        with torch.no_grad():
            # Ensure batch dimension
            if x0.dim() == 3:
                x0 = x0.unsqueeze(0)

            batch_size = x0.size(0)
            seq_len = u_seq.size(0)

            # Initialize deterministic state
            h = torch.zeros(batch_size, self.deterministic_size, device=self.device)

            # Encode initial observation
            enc = self.encoder(x0.to(self.device))

            # Initial posterior
            post_in = torch.cat([enc, h], dim=-1)
            stats = self.post(post_in)
            mu, log_var = stats.chunk(2, dim=-1)
            z = self.reparameterize(mu, log_var)

            frames = []

            # Decode initial state
            if self.output_uncertainty:
                x_dec, _ = self.decoder(z)
            else:
                x_dec = self.decoder(z)
            frames.append(x_dec)

            # Rollout using prior only
            for t in range(seq_len):
                u_t = u_seq[t].unsqueeze(0).to(self.device)

                h, z, mu_p, log_var_p = self.rssm_step(h, z, u_t)

                if self.output_uncertainty:
                    x_dec, _ = self.decoder(z)
                else:
                    x_dec = self.decoder(z)

                frames.append(x_dec)

            # Format output
            frames = torch.cat(frames, dim=0)           # [T+1, C, H, W]
            frames = frames.permute(0, 2, 3, 1).cpu()   # [T+1, H, W, C]

            return frames
        
    def sample(self, x, u, return_all=False):
        """
        Predict the next image in a sequence
        """
        self.eval()
        with torch.no_grad():
            sample_return = {}

            # Encode current state
            enc = self.encoder(x.to(self.device))

            # Initialize deterministic state h to zeros
            h = torch.zeros(enc.shape[0], self.deterministic_size, device=self.device)

            # Posterior for current observation
            post_in = torch.cat([enc, h], dim=-1)            # [batch, enc_latent + deterministic]
            post_stats = self.post(post_in)                      # [batch, 2*stochastic]
            mu, log_var = post_stats.chunk(2, dim=-1)
            z = self.reparameterize(mu, log_var)

            # RSSM prior rollout (predict next latent)
            h_next, z_pred, mu_pred, log_var_pred = self.rssm_step(h, z, u)

             # Decode next observation
            if self.output_uncertainty:
                x_recon, x_recon_uncertainty = self.decoder(z)
                x_pred, x_pred_recon_uncertainty = self.decoder(z_pred)
                sample_return['x_recon_uncertainty'] = x_recon_uncertainty
                sample_return['x_pred_recon_uncertainty'] = x_pred_recon_uncertainty
            else:
                x_recon = self.decoder(z)
                x_pred = self.decoder(z_pred)

            # Save results
            sample_return['x_recon'] = x_recon
            sample_return['x_pred'] = x_pred
            sample_return['mu'] = mu
            sample_return['log_var'] = log_var
            sample_return['h_next'] = h_next
            sample_return['mu_pred'] = mu_pred
            sample_return['log_var_pred'] = log_var_pred
            sample_return['z_pred'] = z_pred

            if return_all:
                return x_recon.squeeze(0), x_pred.squeeze(0), sample_return
            else:
                return x_recon.squeeze(0), x_pred.squeeze(0)