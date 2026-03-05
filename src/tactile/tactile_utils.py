import torch
import random
from pathlib import Path

from src.replay_buffer import ReplayBuffer

# Get paths relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"

class TactileDataset():
    """
    An tactile sample consists of a current tactile image + feature, a future tactile image + feature, and a control input.
    """
    def __init__(self, config):
        # Load raw dataset
        env_name = config['train']['dataset'].split('_')[0]
        dataset_dir = DATA_PATH / env_name
        data = torch.load(dataset_dir / f"{config['train']['dataset']}.pt")

        # Observations
        self.tactile = data["prev_tactile"]
        self.next_tactile = data["next_tactile"]
        self.in_img_shape = [self.tactile[0, 0].shape[0] * config['trans']['past_length'], *self.tactile[0, 0].shape[1:]]

        self.feature = data["prev_feature"]
        self.next_feature = data["next_feature"]
        self.num_features = self.feature.shape[-1]

        if len(self.tactile.shape) == 5:
            self.past_length = self.tactile.shape[1]
            assert self.past_length == config['trans']['past_length'], f"past_length={config['trans']['past_length']} in config file, but dataset has past_length={self.past_length}"
        else:
            self.past_length = 1

        self.pred_length = config['trans']['pred_length']
        control = data["actions"]
        self.U = control

    def __len__(self):
        return len(self.tactile)

    def __getitem__(self, idx):
        return self.tactile[idx], self.next_tactile[idx], self.feature[idx], self.next_feature[idx], self.U[idx]

class TactileReplayBufferSample():
    def __init__(self, t, f, u, r, t_next, f_next, d):
        self.tactile = t
        self.feature = f
        self.u = u
        self.rewards = r
        self.tactile_next = t_next
        self.feature_next = f_next
        self.dones = d

class TactileReplayBuffer(ReplayBuffer):
    """
    Simple ring buffer ReplayBuffer class, with random sampling. Has tactile and feature data.

    Args:
        img_shape: Shape of tactile pixel obs is (past_length*1, H, W)
        num_features: Length of non-tactile feature vector
        control_size: Dimension of control vector
        capacity: How many samples to hold at once
        device: CPU or GPU
        config: Config dictionary
    """
    def __init__(self, img_shape, num_features, control_size, capacity, device, config):
        super().__init__(img_shape, control_size, capacity, device, config)

        # Feature buffers
        past_length = config['trans']['past_length']
        pred_length = config['trans']['pred_length']
        self.features = torch.zeros((capacity, past_length, num_features), device=device)      # TODO: Don't hardcode this
        self.features_next = torch.zeros((capacity, pred_length, num_features), device=device)

    @torch.no_grad()
    def add(self, tactile, feature, action, reward, next_tactile, next_feature, done):
        super().add(tactile, action, reward, next_tactile, done, update_ptr=False)
        self.features[self.ptr] = feature.to(self.device)
        self.features_next[self.ptr] = next_feature.to(self.device)

        # Update ring buffer params
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Generate batch_size random samples from the replay buffer
        # idx = torch.randint(0, self.size, (batch_size,), device=self.device)  # Samples with replacement
        idx = random.sample(range(self.size), min(batch_size, self.size))  # Samples without replacement

        x, u, r, x_next, d = super().sample(batch_size, idx=idx)
        f = self.features[idx]
        f_next = self.features_next[idx]

        return TactileReplayBufferSample(x, f, u, r, x_next, f_next, d)