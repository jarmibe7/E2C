"""
Convolutional encoder used by :class:`RSSME2C`.

Maps a ``[B, in_channels, H, W]`` image to a flat ``latent_size`` embedding
through a 4-layer CNN followed by a 2-layer MLP.

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn


class ConvEncoder(nn.Module):
    """CNN image encoder.

    Args:
        latent_size: Output embedding dimension.
        in_channels: Number of input channels (RGB per frame; callers stack
            past frames along the channel dim before passing in).
        conv_params: Dict with ``enc_kernel_size``, ``stride``, ``pad``, and
            ``in_image_shape``.
    """
    def __init__(self, latent_size, in_channels, conv_params):
        super().__init__()

        self.latent_size = latent_size
        self.in_channels = in_channels

        k = conv_params['enc_kernel_size']
        s = conv_params['stride']
        p = conv_params['pad']

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

        # Run a dry-forward pass to compute the pre-FC flat dim; the decoder
        # uses the same shape to re-shape its input.
        with torch.no_grad():
            x = torch.zeros(1, in_channels, conv_params['in_image_shape'][1], conv_params['in_image_shape'][2])
            enc_out = self.encoder_cnn(x)
            self.out_dim_flat = enc_out.view(enc_out.size(0), -1).shape[1]
            self.out_shape = enc_out.shape

        self.fc_encode = nn.Sequential(
            nn.Linear(self.out_dim_flat, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_size),
        )

    def forward(self, x):
        encoded = self.encoder_cnn(x)
        flattened = encoded.reshape(encoded.size(0), -1)
        return self.fc_encode(flattened)
