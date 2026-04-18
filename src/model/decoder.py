"""
Image decoders used by :class:`RSSME2C`.

Two variants:

- :class:`ConvDecoder` -- plain CNN decoder producing pixel values in [0, 1].
- :class:`ScalarUncertaintyConvDecoder` -- same CNN plus an MLP head that
  emits a single scalar log-variance per image, consumed by
  :class:`RSSMLoss` as a Gaussian NLL.

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn


class ConvDecoder(nn.Module):
    """CNN decoder mapping a latent vector back to a [0, 1] RGB image.

    Args:
        latent_size: Dimension of the input latent ``z``.
        conv_params: Dict with ``dec_kernel_size``, ``enc_kernel_size``,
            ``stride``, ``pad``, and ``out_image_shape``.
        enc_out_dim: Flat dim of the encoder's pre-FC feature map; we project
            into it before reshaping + upsampling.
        enc_out_shape: Full encoder pre-FC shape ``[1, C, H, W]``.
    """
    def __init__(self, latent_size, conv_params, enc_out_dim, enc_out_shape):
        super().__init__()

        self.latent_size = latent_size

        k = conv_params['dec_kernel_size']
        s = conv_params['stride']
        p = conv_params['pad']
        self.out_image_shape = conv_params['out_image_shape']
        self.enc_out_shape = enc_out_shape

        self.fc_decode = nn.Sequential(
            nn.Linear(self.latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, enc_out_dim),
            nn.ReLU(),
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.out_image_shape[0], kernel_size=conv_params['enc_kernel_size'], stride=s-1, padding=p),
            nn.Sigmoid(),
        )

    def forward(self, z):
        to_decode = self.fc_decode(z)
        to_decode = to_decode.view(z.shape[0], *self.enc_out_shape[1:])
        return self.decoder_cnn(to_decode)


class ScalarUncertaintyConvDecoder(ConvDecoder):
    """Variant of :class:`ConvDecoder` with an extra scalar log-variance head.

    The uncertainty is a single number per sample (not per pixel), emitted by
    a small MLP applied to the same latent ``z`` used by the image CNN.
    """
    def __init__(self, latent_size, conv_params, enc_out_dim, enc_out_shape):
        super().__init__(latent_size, conv_params, enc_out_dim, enc_out_shape)

        k = conv_params['dec_kernel_size']
        s = conv_params['stride']
        p = conv_params['pad']
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.out_image_shape[0], kernel_size=conv_params['enc_kernel_size'], stride=s-1, padding=p),
            nn.Sigmoid(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),   # Scalar log variance output
        )

    def forward(self, z):
        to_decode = self.fc_decode(z)
        to_decode = to_decode.view(z.shape[0], *self.enc_out_shape[1:])
        decoded_image = self.decoder_cnn(to_decode)

        decoded_log_var = self.uncertainty_head(z)
        decoded_log_var = torch.clamp(decoded_log_var, min=-10.0, max=10.0)

        return decoded_image, decoded_log_var
