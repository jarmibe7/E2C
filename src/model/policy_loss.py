"""
A collection of loss functions for policy training

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn

class PPOActorLoss(nn.Module):
    """
    Actor loss for the A2C algorithm
    """
    def __init__(self):
        # TODO: Make this a param
        self.clip_range = 1.0

    def forward(self, actions, new_action_probs, old_action_probs, advantages):
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
        ratio = torch.exp(new_action_probs - old_action_probs)     # Given probs are log probs

        # Clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        return policy_loss