"""
Policies for world model active learning

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn

class BasePolicy(nn.Module):
    """
    Base Policy class to be inherited by other policies
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class ConvPolicy(nn.Module):
    """
    A policy that encodes an image observation and outputs an action.
    """
    def __init__(self, control_size, in_channels, conv_params):
        super().__init__()
        # CNN parameters
        k = conv_params['enc_kernel_size']
        s = conv_params['stride']
        p = conv_params['pad']
        self.control_size = control_size

        # Define convolutional encoder
        self.conv = nn.Sequential(
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
            conv_out = self.conv(x)
            self.out_dim_flat = conv_out.view(conv_out.size(0), -1).shape[1] # Keep batch dim, determine number of elements

        self.fc = nn.Sequential(
            nn.Linear(self.out_dim_flat, 512),
            nn.ReLU(),
            nn.Linear(512, self.control_size),
        )

    def forward(self, x):
        conv_out = self.conv(x)
        flattened = conv_out.reshape(conv_out.size(0), -1)
        action = self.fc(flattened)
        return action

