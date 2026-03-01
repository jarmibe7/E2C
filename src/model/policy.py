"""
Policies for world model active learning

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn
from torch.distributions import Normal
from src.model.encoder import ConvEncoder

class BasePolicy(nn.Module):
    """
    Base Policy class to be inherited by other policies
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class ConvStochasticPolicy(nn.Module):
    """
    A policy that encodes an image observation and outputs an action.
    """
    def __init__(self, control_size, enc_latent_size, conv_params, encoder=None):
        super().__init__()
        # CNN parameters
        k = conv_params['enc_kernel_size']
        s = conv_params['stride']
        p = conv_params['pad']
        self.control_size = control_size

        # Define convolutional encoder
        if encoder is None:
            in_channels = conv_params['in_image_shape'][0] // 3 # hacky, assuming RGB, and input has already stacked frames
            self.encoder_cnn = ConvEncoder(
                enc_latent_size,
                in_channels,
                conv_params
            )
        else:
            self.encoder_cnn = encoder


        # Actor and critic share encoder
        self.fc_actor = nn.Sequential(
            nn.Linear(enc_latent_size, 64),
            nn.ReLU(),
        )

        self.fc_critic = nn.Sequential(
            nn.Linear(enc_latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

        self.mu = nn.Linear(64, self.control_size)

        # Log variance is state independent
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py
        self.log_std = nn.Parameter(torch.zeros(self.control_size)) 

    def get_value(self, x):
        # Get value of state
        encoded = self.encoder_cnn(x)
        return self.fc_critic(encoded)

    def forward(self, x, train=True):
        encoded = self.encoder_cnn(x)
        h = self.fc_actor(encoded)

        mu = self.mu(h)
        std = torch.exp(self.log_std)

        dist = Normal(mu, std)

        policy_return = {
            'dist': dist,
        }

        return policy_return

