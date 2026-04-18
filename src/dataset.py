"""
``E2CDataset`` -- torch dataset used for offline pretraining.

Expects a ``.pt`` file with three keys:

- ``prev_images``: ``[N, past_length, H, W, C]`` uint/float tensor.
- ``next_images``: ``[N, pred_length, H, W, C]`` uint/float tensor.
- ``actions``:     ``[N, past_length + pred_length, A]`` float tensor.

Images are permuted to ``[N, T, C, H, W]`` on load. Actions are passed through
unmodified (per-env normalization is handled inside the trainer).

Authors: Jared Berry, Ayush Gaggar
"""
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"


class E2CDataset(torch.utils.data.Dataset):
    """(past_images, future_images, controls) tuples for world-model pretraining."""
    def __init__(self, config):
        env_name = config['train']['dataset'].split('_')[0]
        dataset_dir = DATA_PATH / env_name
        data = torch.load(dataset_dir / f"{config['train']['dataset']}.pt")
        self.X = data["prev_images"].permute(0, 1, 4, 2, 3)      # [N, past_length, C, H, W]
        self.X_next = data["next_images"].permute(0, 1, 4, 2, 3) # [N, pred_length, C, H, W]
        self.in_img_shape = [
            self.X[0, 0].shape[0] * config['trans']['past_length'],
            *self.X[0, 0].shape[1:],
        ]
        if len(self.X.shape) == 5:
            self.past_length = self.X.shape[1]
            assert self.past_length == config['trans']['past_length'], (
                f"past_length={config['trans']['past_length']} in config file, "
                f"but dataset has past_length={self.past_length}"
            )
        else:
            self.past_length = 1
        self.pred_length = config['trans']['pred_length']
        self.U = data["actions"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_next[idx], self.U[idx]
