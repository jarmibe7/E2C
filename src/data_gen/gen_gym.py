"""
Generate dataset from Gymnasium environment

scp -r data/reacher jarmibe7@dingo.mech.northwestern.edu:~/E2C/
scp jarmibe7@dingo.mech.northwestern.edu:~/E2C/videos/e2c_cartpole.mp4 C:\\Users\\jarmi\\MS_Thesis\\Media\\Videos

Author: Jared Berry, Ayush Gaggar
"""
import numpy as np
import torch
import gymnasium as gym
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from pyvirtualdisplay import Display
from tqdm import tqdm

from src.utils import set_seed

# Parameters for dataset
env_name = 'reacher'
OUTPUT_NAME = env_name + '_500k'
dataset_size = int(5e5)
image_shape = (64, 64, 3)
# ---------------------------------

# Get data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

set_seed(42)
name_to_env = {'reacher': 'Reacher-v5', 'cartpole': 'CartPole-v1'}
env_to_aspace = {'reacher': 'continuous', 'cartpole': 'discrete'}

def process_image(image, dataset_name=env_name):
    """
    Image processing
    """
    if dataset_name == 'cartpole': image = image[50:350, 100:400]  # Zoom on cartpole
    if dataset_name == 'reacher': image = image[100:-50, 100:-100]  # Zoom on reacher
    image = torch.from_numpy(image.copy()).permute(2, 0, 1)  # Get image tensor into (C, H, W)

    # Image processing
    normalized = image.unsqueeze(0).float() / 255.0 # Normalize to [0,1]
    image_resized = torchvision.transforms.functional.resize(normalized, image_shape[0:2], interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)   # Downscaling
      
    return image_resized.permute(0, 2, 3, 1) # Permute back to raw shape


def main():
    # Create virtual display for running on server
    disp = Display(visible=0, size=(480, 480))
    disp.start()
    
    # buffers
    num_prev = 3
    frame_buffer = []
    act_buffer = []

    # Create env
    env = gym.make(name_to_env[env_name], render_mode="rgb_array")
    obs, _ = env.reset()
    continuous = (env_to_aspace[env_name] == 'continuous')
    prev_img = torch.zeros((dataset_size, num_prev, *image_shape))
    next_img = torch.zeros((dataset_size, *image_shape))
    if continuous: control = torch.zeros((dataset_size, env.action_space.shape[0]))
    else: control = torch.zeros((dataset_size, 1))     # Discrete action space
    done = False
    
    # Collect n_samples trajectories
    idx = 0
    pbar = tqdm(total=dataset_size)
    while idx < dataset_size:
        if len(frame_buffer) == 0:
            frame_buffer.append(process_image(env.render()))

        act = env.action_space.sample()
        act_buffer.append(act)
        next_obs, rew, done, _, _ = env.step(act)

        if done:
            obs, _ = env.reset()
            done = False
            frame_buffer = []
            act_buffer = []
            continue
        else:
            if len(frame_buffer) == num_prev + 1:
                frame_buffer.pop(0)
                act_buffer.pop(0)
            next_image = process_image(env.render())
            frame_buffer.append(next_image)
            if len(frame_buffer) == num_prev + 1:
                prev_img[idx] = torch.cat(frame_buffer[0:num_prev], dim=0)
                next_img[idx] = frame_buffer[num_prev]
                if continuous:
                    control[idx] = torch.from_numpy(act_buffer[-1])
                else:
                    control[idx] = act_buffer[-1]
                idx += 1
                pbar.update(1)

    pbar.close()

    # Saving dataset
    dataset_dir = DATA_PATH / env_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prev_images": prev_img,
        "actions": control,
        "next_images": next_img,
    }, f"{dataset_dir / OUTPUT_NAME}.pt")
    print(f'\nSaved dataset to {dataset_dir / OUTPUT_NAME}.pt')
    print('\n*** DONE ***')
    return

if __name__ == '__main__':
    main()