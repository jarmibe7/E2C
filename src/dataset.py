"""
Contains a dataset for training E2C-like world models

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from pathlib import Path

# Get paths relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"

class E2CDataset(torch.utils.data.Dataset):
    """
    An E2C Dataset consists of a current image, a future image, and a control input.
    """
    def __init__(self, config):
        # Load raw dataset
        env_name = config['train']['dataset'].split('_')[0]
        if config['train']['dataset'].endswith('.pt'):
            dataset_dir = PROJECT_ROOT / "runs" / env_name
            dataset_file = config['train']['dataset']
        else:
            dataset_dir = DATA_PATH / env_name
            dataset_file = f"{config['train']['dataset']}.pt"
        data = torch.load(dataset_dir / dataset_file)
        self.X = data["prev_images"].permute(0, 1, 4, 2, 3)  # Shape: [num_samples, past_length, C, H, W]
        self.X_next = data["next_images"].permute(0, 1, 4, 2, 3)  # Shape: [num_samples, pred_length, C, H, W]
        self.in_img_shape = [self.X[0, 0].shape[0] * config['trans']['past_length'], *self.X[0, 0].shape[1:]]
        if len(self.X.shape) == 5:
            self.past_length = self.X.shape[1]
            assert self.past_length == config['trans']['past_length'], f"past_length={config['trans']['past_length']} in config file, but dataset has past_length={self.past_length}"
        else:
            self.past_length = 1
        self.pred_length = config['trans']['pred_length']
        control = data["actions"]
        # Filter out samples where any tensor has all zeros
        # Check if each sample has any non-zero values
        prev_images_nonzero = (self.X.reshape(self.X.shape[0], -1) != 0).any(dim=1)
        next_images_nonzero = (self.X_next.reshape(self.X_next.shape[0], -1) != 0).any(dim=1)
        actions_nonzero = (control.reshape(control.shape[0], -1) != 0).any(dim=1)
        valid_indices = prev_images_nonzero & next_images_nonzero & actions_nonzero
        self.X = self.X[valid_indices]
        self.X_next = self.X_next[valid_indices]
        self.U = control[valid_indices]

        # Normalize actions
        # if control.max() - control.min() > 0:
        #     control = (control - control.min()) / (control.max() - control.min())
        self.U = control

        # Reshape for learning
        # flat = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4]).permute(0, 3, 1, 2)   # Shape: [batch*seq_len, C, H, W]
        # X = flat[:-1]   # Shape: [(batch*seq_len) - 1, C, H, W]
        # X_next = flat[1:]
        # U = control.reshape(-1, control.shape[-1])[:-1]
        # Create windowed samples for past_length and pred_length
        """H, W = self.img_shape[1:]
        num_batches = img.shape[0]
        seq_len = img.shape[1]
        x_windows = []
        x_next_windows = []
        u_list = []
        window_size = self.past_length + self.pred_length
        for b in range(num_batches):
            for t in range(seq_len - window_size + 1):
                # Get windows
                x_windows.append(img[b, t:t+self.past_length].reshape(-1, H, W))
                x_next_windows.append(img[b, t+self.past_length:t+self.past_length+self.pred_length].reshape(-1, H, W))
                u_list.append(control[b, t+self.past_length])          

        self.X = torch.stack(x_windows)
        self.X_next = torch.stack(x_next_windows)
        self.U = torch.stack(u_list)"""

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_next[idx], self.U[idx]