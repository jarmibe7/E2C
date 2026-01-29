"""
Convolutional encoder made with PyTorch.

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn

class ConvEncoder(nn.Module):
    def __init__(self, 
                 latent_size,
                 in_channels,
                 conv_params):
        super().__init__()

        self.latent_size = latent_size
        self.in_channels = in_channels

        # CNN parameters
        k = conv_params['enc_kernel_size']
        s = conv_params['stride']
        p = conv_params['pad']

        # Define encoder part of autoencoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=k+2, stride=s-1, padding=p+1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
        )

        with torch.no_grad():
            x = torch.zeros(1, in_channels, conv_params['in_image_shape'][1], conv_params['in_image_shape'][2])
            enc_out = self.encoder_cnn(x)
            self.out_dim_flat = enc_out.view(enc_out.size(0), -1).shape[1] # Keep batch dim, determine number of elements
            self.out_shape = enc_out.shape

        self.fc_encode = nn.Sequential(
            nn.Linear(self.out_dim_flat, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_size),
        )

    def forward(self, x):
        encoded = self.encoder_cnn(x)
        flattened = encoded.reshape(encoded.size(0), -1)
        out = self.fc_encode(flattened)
        return out