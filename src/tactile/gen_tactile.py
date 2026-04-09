"""
Generate dataset from Gymnasium environment

scp -r data/reacher jarmibe7@dingo.mech.northwestern.edu:~/E2C/
scp jarmibe7@dingo.mech.northwestern.edu:~/E2C/videos/e2c_cartpole.mp4 C:\\Users\\jarmi\\MS_Thesis\\Media\\Videos

Author: Jared Berry
"""
import os
import re
import numpy as np
import torch
import time
import gymnasium as gym
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from pyvirtualdisplay import Display
from tqdm import tqdm
import yaml
from datetime import datetime
from PIL import Image
import metaworld

from src.utils import set_seed, format_time
from src.data_gen.gen_fetch import process_image, debug_render, update_dataset_metadata
import gymnasium_robotics
from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env import (
    ObjectPushEnv,
)
gym.register_envs(gymnasium_robotics)

# Parameters for dataset
ENV_NAME = 'tactilepush'                                           # Gym environment name
DATASET_SIZE = int(1e3)                                     # Number of samples: (img, next_img, control) tuple
OUTPUT_NAME = ENV_NAME + f'_{DATASET_SIZE // 1000}k'        # Output name of dataset
IMAGE_SHAPE = [128, 128]                                      # Downsampled image shape
PAST_LENGTH = 3                                             # Number of previous observations to use for training
PRED_LENGTH = 3                                             # Number of timesteps to predict in the future
NEW_DT = None                                               # Desired new timestep in seconds (control timestep =0.1, sim timestep = 1/240)
# ---------------------------------

# Get data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

def process_tactile(t):
    normalized = torch.from_numpy(t).float() / 255.0 # Normalize to [0,1]
    return normalized.permute(2, 0, 1)

def process_feature(f):
    # 0:3 = EE position
    # 3:6 = EE orientation
    # 6:9 = Goal position
    # 9:12 = Goal orientation
    return torch.from_numpy(f[0:6])

def get_env_modes():
    env_modes = {
        # which dofs can have movement (environment dependent)
        # 'movement_mode':'y',          # y only, action dim = 1
        # 'movement_mode':'yRz',        # y + rotate z, action dim = 2
        # "movement_mode": "xyRz",      # x, y + rotate z, action dim = 3
        # 'movement_mode': 'TyRz',        # Move relative to TCP frame (perp + yaw), action dim = 2
        'movement_mode':'TxTyRz',     # Full TCP-parallel/perp motion + yaw, action dim = 3

        # specify arm
        "arm_type": "ur5",
        # "arm_type": "mg400",

        # specify tactile sensor
        # "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        "tactile_sensor_name": "digitac",

        # the type of control used
        'control_mode':'TCP_position_control',
        # "control_mode": "TCP_velocity_control",

        # randomisations
        "rand_init_pos": False,      # Object position
        "rand_init_orn": True,     # Object orientation
        "rand_obj_mass": False,     # Object mass

        # whether to render task goal
        "use_goal": False,

        # straight or random trajectory
        # "traj_type": "straight",
        'traj_type': 'simplex',

        # which observation type to return
        # feature includes EE position and goal position
        # 'observation_mode':'oracle',
        "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',

        # the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }
    return env_modes

seed = 42
set_seed(seed)
tactile_envs = {
    'tactilepush': ObjectPushEnv
}

def main():
    start_time = time.perf_counter()
    import os
    # os.environ["PYOPENGL_PLATFORM"] = "egl"
    # Create virtual display for running on server
    # disp = Display(visible=0, size=(480, 480))
    # disp.start()

    # Create env
    max_steps = 10000            # Max number of env steps before forced termination

    # These will slow down runtime, but are necessary for eval
    show_gui = False             # Enable PyBullet GUI
    show_tactile = False         # Display tactile sensor images
    
    env = tactile_envs[ENV_NAME](
        max_steps=max_steps,
        env_modes=get_env_modes(),
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=IMAGE_SHAPE,
    )
    
    obs, _ = env.reset()

    # Buffers
    tactile_buffer = []
    feature_buffer = []
    act_buffer = []
    prev_tactile = torch.zeros((DATASET_SIZE, PAST_LENGTH, 1, *IMAGE_SHAPE))    # tactile has one channel
    next_tactile = torch.zeros((DATASET_SIZE, PRED_LENGTH, 1, *IMAGE_SHAPE))
    prev_feature = torch.zeros((DATASET_SIZE, PAST_LENGTH, 6))
    next_feature = torch.zeros((DATASET_SIZE, PRED_LENGTH, 6))
    control = torch.zeros((DATASET_SIZE, PRED_LENGTH, env.action_space.shape[0]))
    terminated = False; truncated = False

    # Seed initial observation into buffer
    tactile_buffer.append(process_tactile(obs["tactile"]))
    feature_buffer.append(process_feature(obs["extended_feature"]))
    
    # Collect n_samples trajectories
    idx = 0
    pbar = tqdm(total=DATASET_SIZE)
    while idx < DATASET_SIZE:
        # Sample and take action
        act = env.action_space.sample()
        act_buffer.append(act)
        next_obs, rew, terminated, truncated, _ = env.step(act)

        tactile = process_tactile(next_obs['tactile'])
        feature = process_feature(next_obs['extended_feature'])

        tactile_buffer.append(tactile)
        feature_buffer.append(feature)
        act_buffer.append(act)

        # If done reset env, otherwise add sample to dataset
        if terminated or truncated:
            obs, _ = env.reset()
            tactile_buffer = []
            feature_buffer = []
            act_buffer = []
            continue

        # Maintain sliding window size
        total_len = PAST_LENGTH + PRED_LENGTH
        if len(tactile_buffer) > total_len:
            tactile_buffer.pop(0)
            feature_buffer.pop(0)
            act_buffer.pop(0)

        # Save sample when buffer full
        if len(tactile_buffer) == total_len:

            prev_tactile[idx] = torch.stack(
                tactile_buffer[:PAST_LENGTH], dim=0
            )
            next_tactile[idx] = torch.stack(
                tactile_buffer[PAST_LENGTH:], dim=0
            )

            prev_feature[idx] = torch.stack(
                feature_buffer[:PAST_LENGTH], dim=0
            )
            next_feature[idx] = torch.stack(
                feature_buffer[PAST_LENGTH:], dim=0
            )

            control[idx] = torch.stack(
                [
                    torch.from_numpy(a).float()
                    for a in act_buffer[PAST_LENGTH - 1 : PAST_LENGTH - 1 + PRED_LENGTH]
                ]
            )

            idx += 1
            pbar.update(1)

    pbar.close()

    # Saving dataset as dictionary
    dataset_dir = DATA_PATH / ENV_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prev_tactile": prev_tactile,
        "next_tactile": next_tactile,
        "prev_feature": prev_feature,
        "next_feature": next_feature,
        "actions": control,
    }, f"{dataset_dir / OUTPUT_NAME}.pt")
    print(f'\nSaved dataset to {dataset_dir / OUTPUT_NAME}.pt')

    # Update metadata file
    try: 
        dt = env.unwrapped.dt
    except:
        dt = 0.1
    update_dataset_metadata(
        dataset_dir=dataset_dir,
        dataset_name=OUTPUT_NAME,
        params={
            "ENV_NAME": ENV_NAME,
            "runtime": format_time(time.perf_counter() - start_time),
            "dataset_size": DATASET_SIZE,
            "IMAGE_SHAPE": list(IMAGE_SHAPE),
            "PAST_LENGTH": PAST_LENGTH,
            "PRED_LENGTH": PRED_LENGTH,
            "seed": seed,
            "dt": None if ENV_NAME =='cartpole' else dt
        },
    )
    print('\n*** DONE ***')
    return

if __name__ == '__main__':
    main()