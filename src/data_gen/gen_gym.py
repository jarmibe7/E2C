"""
Generate dataset from Gymnasium environment

Author: Jared Berry, Ayush Gaggar
"""
import numpy as np
import torch
import gymnasium as gym
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils import set_seed

# ---------------------------------
# Parameters for simulation
dt = 0.01               # Timestep
seq_len = 50            # Number of timesteps per episode (traj sequence length)

# Parameters for dataset
env_name = 'reacher'
n_samples = 100 # Number of total trajectories (number of episodes)
image_shape = (64, 64, 3)
# ---------------------------------

# Get data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

set_seed(42)
name_to_gym = {'reacher': 'Reacher-v5'}

def process_image(image):
    """
    Image processing
    """
    image = torch.from_numpy(image.copy()).permute(2, 0, 1)  # Get image tensor into (C, H, W)

    # Image processing
    normalized = image.unsqueeze(0).float() / 255.0 # Normalize to [0,1]
    image_resized = torchvision.transforms.functional.resize(normalized, image_shape[0:2], interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)   # Downscaling
      
    return image_resized.permute(0, 2, 3, 1) # Permute back to raw shape


def main():
    print('*** STARTING ***\n')
    # Create env
    env = gym.make(name_to_gym[env_name], render_mode="rgb_array")
    img = torch.zeros((n_samples, seq_len, *image_shape))
    control = torch.zeros((n_samples, seq_len, env.action_space.shape[0]))
    
    # Collect n_samples trajectories
    for episode in range(n_samples):
        if episode % 100 == 0: print(f'On sample {episode} out of {n_samples}')
        obs, _ = env.reset()
        done = False

        # Collect seq_len timesteps per traj
        for t in range(seq_len):
            # If done, just fill with final frame
            if done:
                img[episode, t] = process_image(rend)
                control[episode, t] = torch.zeros(env.action_space.shape)
                continue

            # Save image and action pair
            rend = env.render()
            img[episode, t] = process_image(rend)
            action = env.action_space.sample()
            control[episode, t] = torch.from_numpy(action)

            # Take action
            obs, rew, done, _, _ = env.step(action)

    print(f'\nDataset shape: {img.shape}')

    # Saving dataset
    dataset_dir = DATA_PATH / env_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    img_filepath = dataset_dir / 'img.pt'
    control_filepath = dataset_dir / 'control.pt'
    print(f'\nSaving dataset to {dataset_dir}')
    torch.save(img, img_filepath)
    torch.save(control, control_filepath)

    print('\n*** DONE ***')
    return

if __name__ == '__main__':
    main()