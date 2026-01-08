"""
A pretrained state estimation model that infers states from gym.env images

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn

class StateRepresentationModel(nn.Module):
    """
    A state representation model infers robot states from images of a Gym environment. 

    Args:
        state_size: Dimension of control vector
        conv_params: Dictionary containing CNN params for encoder/decoder
        device: Torch device object
    """
    def __init__(self, state_size, conv_params, device):
        super().__init__()
        self.device = device
        self.state_size = state_size    # Output size
        
        # CNN parameters
        in_channels = conv_params['out_image_shape'][0]
        k = conv_params['kernel_size']
        s = conv_params['stride']
        p = conv_params['pad']

        # Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=k+2, stride=s-1, padding=p+1),
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
            enc_out = self.cnn(x)
            self.out_dim_flat = enc_out.view(enc_out.size(0), -1).shape[1] # Keep batch dim, determine number of elements
            self.out_shape = enc_out.shape

        # Linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.out_dim_flat, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.state_size),
        )

    def forward(self, x):
        conv_out = self.cnn(x)
        flattened = conv_out.reshape(conv_out.size(0), -1)
        out = self.fc(flattened)
        return out
    
