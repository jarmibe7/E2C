"""
Convolutional encoder and decoder made with PyTorch.

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
            x = torch.zeros(1, in_channels, conv_params['out_image_shape'][1], conv_params['out_image_shape'][2])
            enc_out = self.encoder_cnn(x)
            self.out_dim_flat = enc_out.view(enc_out.size(0), -1).shape[1] # Keep batch dim, determine number of elements
            self.out_shape = enc_out.shape

        self.fc_encode = nn.Sequential(
            nn.Linear(self.out_dim_flat, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_size),
        )

    def forward(self, x):
        encoded = self.encoder_cnn(x)
        flattened = encoded.reshape(encoded.size(0), -1)
        out = self.fc_encode(flattened)
        return out

class ConvDecoder(nn.Module):
    def __init__(self, latent_size, conv_params, enc_out_dim, enc_out_shape):
        super().__init__()

        self.latent_size = latent_size

        # CNN parameters
        k = conv_params['dec_kernel_size']
        s = conv_params['stride']
        p = conv_params['pad']
        self.out_image_shape = conv_params['out_image_shape']
        self.enc_out_shape = enc_out_shape

        self.fc_decode = nn.Linear(self.latent_size, enc_out_dim)

        self.fc_decode = nn.Sequential(
            nn.Linear(self.latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, enc_out_dim),
            nn.ReLU(),
        )

        # Define convolutional decoder
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
            nn.Sigmoid(), # Keep between 0 and 1
        )

    def forward(self, z):
        to_decode = self.fc_decode(z)
        to_decode = to_decode.view(z.shape[0], *self.enc_out_shape[1:])
        decoded = self.decoder_cnn(to_decode)
        return decoded
    
class ChannelUncertaintyConvDecoder(ConvDecoder):
    """
    Convolutional decoder that also outputs a log variance uncertainty estimate
    """
    def __init__(self, latent_size, conv_params, enc_out_dim, enc_out_shape):
        super().__init__(latent_size, conv_params, enc_out_dim, enc_out_shape)

        # self.num_var_channels = self.out_image_shape[0]
        self.num_var_channels = 1

        # Output image channels + uncerainty channel for each pixel
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
            nn.Conv2d(32, self.out_image_shape[0] + self.num_var_channels, kernel_size=conv_params['enc_kernel_size'], stride=s-1, padding=p),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        to_decode = self.fc_decode(z)
        to_decode = to_decode.view(z.shape[0], *self.enc_out_shape[1:])
        decoded = self.decoder_cnn(to_decode)
        decoded_image = self.sigmoid(decoded[:, :-self.num_var_channels])
        decoded_log_var = decoded[:, -self.num_var_channels:]
        decoded_log_var = torch.clamp(decoded_log_var, min=-10.0, max=10.0)   # Clamp log variance
        return decoded_image, decoded_log_var
    
class ScalarUncertaintyConvDecoder(ConvDecoder):
    """
    Convolutional decoder that also outputs a log variance uncertainty estimate
    """
    def __init__(self, latent_size, conv_params, enc_out_dim, enc_out_shape):
        super().__init__(latent_size, conv_params, enc_out_dim, enc_out_shape)

        # Output image channels + uncerainty channel for each pixel
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
            nn.Sigmoid()
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # Scalar log variance output
        )

    def forward(self, z):
        # Image reconstruction
        to_decode = self.fc_decode(z)
        to_decode = to_decode.view(z.shape[0], *self.enc_out_shape[1:])
        decoded_image = self.decoder_cnn(to_decode)

        # Log variance output
        decoded_log_var = self.uncertainty_head(z)
        decoded_log_var = torch.clamp(decoded_log_var, min=-10.0, max=10.0)

        return decoded_image, decoded_log_var