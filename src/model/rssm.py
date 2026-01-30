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
        in_channels = conv_params['in_image_shape'][0] // 3 # hacky, assuming RGB, and input has already stacked frames
        self.encoder = ConvEncoder(enc_latent_size, in_channels, conv_params)
        if self.output_uncertainty:
            self.decoder = ScalarUncertaintyConvDecoder(stochastic_size, conv_params, self.encoder.out_dim_flat, self.encoder.out_shape)
        else:
            self.decoder = ConvDecoder(stochastic_size, conv_params, self.encoder.out_dim_flat, self.encoder.out_shape)
        self.out_image_shape = self.decoder.out_image_shape

        # Dreamer dynamics model definition
        self.num_layers = 2
        self.rnn = nn.GRU(
            self.stochastic_size + self.control_size,
            self.deterministic_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.prior = nn.Sequential(                     # Transition model
            nn.Linear(self.deterministic_size, 200),
            nn.ReLU(),
            nn.Linear(200, 2 * self.stochastic_size)
        )

        self.post = nn.Sequential(                      # Representation model
            nn.Linear(self.enc_latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.stochastic_size)
        )
        # self.post = nn.Sequential(                      # Representation model
        #     nn.Linear(self.enc_latent_size + self.deterministic_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 2 * self.stochastic_size)
        # )

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
        # repeat u across z's time dimension
        u = u.unsqueeze(1).repeat(1, z.size(1), 1)  # u: [B, past_length, control_size]
        rnn_input = torch.cat([z, u], dim=-1)
        _, h_next = self.rnn(rnn_input, h)  # h_next: [B, 1, deterministic]

        stats = self.prior(h_next[-1])
        mu, log_var = stats.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=1e-5, max=1e5)
        z_next = self.reparameterize(mu, log_var)

        return h_next, z_next, mu, log_var
    
    def encode_posterior(self, obs, actions=None):
        B, T = obs.shape[:2]
        mus, log_vars, zs, hs = [], [], [], []

        for t in range(T):
            x = obs[:, t]
            enc = self.encoder(x)

            stats = self.post(enc)
            mu, log_var = stats.chunk(2, dim=-1)
            log_var = torch.clamp(log_var, min=1e-5, max=1e5)
            z = self.reparameterize(mu, log_var)

            mus.append(mu)
            log_vars.append(log_var)
            zs.append(z)
        return (
            # torch.stack(hs, dim=1),
            torch.stack(mus, dim=1),
            torch.stack(log_vars, dim=1),
            torch.stack(zs, dim=1),
        )
    
    def forward(self, x, x_next, u):
        # Infer belief over past context
        mus, log_vars, zs = self.encode_posterior(x, u[:, :x.size(1)])
        
        h = torch.zeros(self.num_layers, x.size(0), self.deterministic_size, device=self.device)
        # Take last belief state as start
        z = zs[:, -1]

        # reconstruct current observation
        if self.output_uncertainty:
            x_recon, x_recon_uncertainty = self.decoder(zs[:, -1])
        else:
            x_recon = self.decoder(zs[:, -1])

        # Iterate over pred_length
        mu_priors, log_var_priors = [], []
        mu_posts, log_var_posts = [], [] # old version - i added encoded t0 mu in post? why...
        x_preds = []
        if self.output_uncertainty:
            x_pred_uncerts = []

        window = x
        for t in range(x_next.size(1)):
            # prior
            h, z_prior, mu_p, log_var_p = self.rssm_step(h, z.unsqueeze(1), u[:, t])
            
            # old way (takes into account past_length)
            # h, z_prior, mu_p, log_var_p = self.rssm_step(h, zs, u[:, t])
            mu_priors.append(mu_p)
            log_var_priors.append(log_var_p)

            # decode prior (open loop prediction)
            if self.output_uncertainty:
                x_pred, x_pred_uncert = self.decoder(z_prior)
                x_pred_uncerts.append(x_pred_uncert)
            else:
                x_pred = self.decoder(z_prior)
            x_preds.append(x_pred)

            if self.past_length > 1:
                window_frames = window[:, 1:]   # drop first frame
                window = torch.cat([window_frames, x_pred.unsqueeze(1).detach()], dim=1)
            else:
                window = x_pred.detach()  # past_length==1, just use pred image
            mu_q, log_var_q, zs = self.encode_posterior(window)
            z = zs[:, -1]

            # # posterior update using real next frame
            # enc = self.encoder(x_next[:, t])
            # stats = self.post(enc)
            # mu_q, log_var_q = stats.chunk(2, dim=-1)
            # z = self.reparameterize(mu_q, log_var_q)

            mu_posts.append(mu_q[:, -1])
            log_var_posts.append(log_var_q[:, -1])

        # Stack accumulated priors + posteriors
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
            mu, log_var, z = self.encode_posterior(x)

            # Initialize deterministic state h to zeros
            h = torch.zeros(self.num_layers, mu.shape[0], self.deterministic_size, device=self.device)

            # Posterior for current observation
            # post_in = torch.cat([enc, h], dim=-1)            # [batch, enc_latent + deterministic]
            # post_stats = self.post(post_in)                      # [batch, 2*stochastic]
            # mu, log_var = post_stats.chunk(2, dim=-1)
            # z = self.reparameterize(mu, log_var)

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