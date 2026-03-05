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
        self.feature_encoder = nn.Sequential(
            nn.Linear(6, self.feature_latent_size),
            nn.ReLU(),
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
    
    def forward(self, tactile, feature, tactile_next, feature_next, u):
        # Infer belief over past context
        _, _, zs = self.encode_posterior(tactile, feature, u[:, :tactile.size(1)])
        outputs = self.transition(tactile, tactile_next, u, zs)
        return outputs