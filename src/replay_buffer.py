"""
Replay buffers for online policy training
"""
import torch

class ReplayBuffer:
    """
    Simple ring buffer ReplayBuffer class, with random sampling.

    Args:
        obs_shape: Shape of image obs is (past_length*C, H, W)
        control_dim: Dimension of control vector
        capacity: How many samples to hold at once
        config: Config dictionary
    """
    def __init__(self, img_shape, control_size, capacity, device, config):
        # Initialize ring buffer params
        self.ptr = 0
        self.size = 0
        self.capacity = capacity
        self.device = device

        # Image and control buffers
        assert config['trans']['pred_length'] == 1, 'Replay buffer requires pred_length=1'
        C_tot, H, W = img_shape
        C = C_tot // config['trans']['past_length']
        self.X = torch.zeros((capacity, config['trans']['past_length'], C, H, W), device=device)
        self.U = torch.zeros((capacity, control_size), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.X_next = torch.zeros((capacity, C, H, W), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)

    def __len__(self):
        return self.size

    @torch.no_grad()
    def add(self, img, action, reward, next_img, done):
        self.X[self.ptr] = img.to(self.device).contiguous()
        self.U[self.ptr] = action.to(self.device)
        self.rewards[self.ptr] = torch.tensor([reward], device=self.device, dtype=torch.float32)
        self.X_next[self.ptr] = next_img.to(self.device).contiguous()
        self.dones[self.ptr] = torch.tensor([done], device=self.device, dtype=torch.float32)

        # Update ring buffer params
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Generate batch_size random samples from the replay buffer
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.X[idx],
            self.U[idx],
            self.rewards[idx],
            self.X_next[idx],
            self.dones[idx],
        )
