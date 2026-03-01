"""
A collection of loss functions for policy training

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn

class OffPolicyStochasticActorLoss(nn.Module):
    """
    A2C-style actor loss for an off-policy, stochastic RL algorithm
    """
    def __init__(self):
        super().__init__()
        # TODO: Make this a param
        self.clip_range = 1.0

    def forward(self, log_probs, advantages):
        return -(log_probs * advantages.detach()).mean()