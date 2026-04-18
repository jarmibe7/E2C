"""
Replay buffers for online policy training.
"""
import random
from typing import Tuple

import torch


class ReplayBuffer:
    """Simple ring buffer with random (without-replacement) minibatch sampling.

    Args:
        img_shape: Shape of stacked image obs ``(past_length * C, H, W)``.
        control_size: Dimension of the control vector.
        capacity: Maximum number of transitions stored at once.
        device: Torch device to keep tensors on.
        config: Loaded yaml config; used to pull ``past_length`` / ``pred_length``.
    """
    def __init__(self, img_shape, control_size, capacity, device, config):
        self.ptr = 0
        self.size = 0
        self.capacity = capacity
        self.device = device

        C_tot, H, W = img_shape
        C = C_tot // config['trans']['past_length']
        self.X = torch.zeros((capacity, config['trans']['past_length'], C, H, W), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)

        pred_length = config['trans']['pred_length']
        if pred_length > 1:
            self.U = torch.zeros((capacity, pred_length, control_size), device=device)
            self.X_next = torch.zeros((capacity, pred_length, C, H, W), device=device)
        else:
            self.U = torch.zeros((capacity, control_size), device=device)
            self.X_next = torch.zeros((capacity, C, H, W), device=device)

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def add(
        self,
        img: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_img: torch.Tensor,
        done: bool,
    ) -> None:
        """Insert one transition, evicting the oldest if capacity is reached."""
        self.X[self.ptr] = img.to(self.device).contiguous()
        self.U[self.ptr] = action.to(self.device)
        self.rewards[self.ptr] = torch.tensor([reward], device=self.device, dtype=torch.float32)
        self.X_next[self.ptr] = next_img.to(self.device).contiguous()
        self.dones[self.ptr] = torch.tensor([done], device=self.device, dtype=torch.float32)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a minibatch *without* replacement.

        If ``batch_size > len(self)``, returns all available transitions.
        """
        idx = random.sample(range(self.size), min(batch_size, self.size))
        return (
            self.X[idx],
            self.U[idx],
            self.rewards[idx],
            self.X_next[idx],
            self.dones[idx],
        )
