"""
Generate dataset from Gymnasium environment

scp -r data/reacher jarmibe7@dingo.mech.northwestern.edu:~/E2C/
scp jarmibe7@dingo.mech.northwestern.edu:~/E2C/videos/e2c_cartpole.mp4 C:\\Users\\jarmi\\MS_Thesis\\Media\\Videos

Author: Jared Berry, Ayush Gaggar
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
ENV_NAME = 'tactile_push'                                           # Gym environment name
DATASET_SIZE = int(2e3)                                     # Number of samples: (img, next_img, control) tuple
OUTPUT_NAME = ENV_NAME + f'_isom_{DATASET_SIZE // 1000}k'        # Output name of dataset
IMAGE_SHAPE = [128, 128]                                      # Downsampled image shape
PAST_LENGTH = 3                                             # Number of previous observations to use for training
PRED_LENGTH = 3                                             # Number of timesteps to predict in the future
NEW_DT = None                                               # Desired new timestep in seconds (control timestep =0.1, sim timestep = 1/240)
# ---------------------------------

# Get data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

seed = 42
set_seed(seed)
tactile_envs = {
    'tactile_push': ObjectPushEnv
}

def main():
    start_time = time.perf_counter()
    import os
    # os.environ["PYOPENGL_PLATFORM"] = "egl"
    # Create virtual display for running on server
    # disp = Display(visible=0, size=(480, 480))
    # disp.start()
    
    # Buffers
    frame_buffer = []
    act_buffer = []

    # Create env
    max_steps = 10000            # Max number of env steps before forced termination
    show_gui = False             # Enable PyBullet GUI
    show_tactile = False         # Display tactile sensor images
    env_modes = {
        # which dofs can have movement (environment dependent)
        # 'movement_mode':'y',          # y only, action dim = 1
        # 'movement_mode':'yRz',        # y + rotate z, action dim = 2
        # "movement_mode": "xyRz",      # x, y + rotate z, action dim = 3
        'movement_mode': 'TyRz',        # Move relative to TCP frame (perp + yaw), action dim = 2
        # 'movement_mode':'TxTyRz',     # Full TCP-parallel/perp motion + yaw, action dim = 3

        # specify arm
        "arm_type": "ur5",
        # "arm_type": "mg400",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        # "tactile_sensor_name": "digitac",

        # the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        # randomisations
        "rand_init_orn": False,     # Object orientation
        "rand_obj_mass": False,     # Object mass

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
    env = tactile_envs[ENV_NAME](
        max_steps=max_steps,
        env_modes=env_modes,
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=IMAGE_SHAPE,
    )
    
    breakpoint()
    obs, _ = env.reset()
    prev_img = torch.zeros((DATASET_SIZE, PAST_LENGTH, *IMAGE_SHAPE))
    next_img = torch.zeros((DATASET_SIZE, PRED_LENGTH, *IMAGE_SHAPE))
    control = torch.zeros((DATASET_SIZE, PRED_LENGTH, env.action_space.shape[0]))
    terminated = False; truncated = False
    
    # Collect n_samples trajectories
    idx = 0
    pbar = tqdm(total=DATASET_SIZE)
    while idx < DATASET_SIZE:
        if len(frame_buffer) == 0:
            frame_buffer.append(process_image(env.render(), ENV_NAME))

        # Sample and take action
        act = env.action_space.sample()
        act_buffer.append(act)
        if ENV_NAME in meta_world_envs:
            for _ in range(meta_ts):
                next_obs, rew, terminated, truncated, _ = env.step(act)
        else:
            next_obs, rew, terminated, truncated, _ = env.step(act)

        # If done reset env, otherwise add sample to dataset
        if terminated or truncated:
            obs, _ = env.reset()
            terminated = False; truncated = False
            frame_buffer = []
            act_buffer = []
            continue
        else:
            # Slide frame obs history frame buffer to next image
            if len(frame_buffer) == PAST_LENGTH + PRED_LENGTH:
                frame_buffer.pop(0)
                act_buffer.pop(0)
            next_image = process_image(env.render(), ENV_NAME)
            # debug_render(next_image)
            frame_buffer.append(next_image)

            # Add obs history buffer to dataset
            if len(frame_buffer) == PAST_LENGTH + PRED_LENGTH:
                prev_img[idx] = torch.stack(frame_buffer[0:PAST_LENGTH], dim=0)
                next_img[idx] = torch.stack(frame_buffer[PAST_LENGTH:(PAST_LENGTH+PRED_LENGTH)], dim=0)

                # Get controls for entire PRED_LENGTH
                if continuous:
                    control[idx] = torch.stack(
                        [torch.from_numpy(a) for a in act_buffer[PAST_LENGTH-1:PAST_LENGTH-1+PRED_LENGTH]]
                    )
                else:
                    control[idx] = torch.tensor(
                        act_buffer[PAST_LENGTH-1:PAST_LENGTH-1+PRED_LENGTH]
                    ).unsqueeze(-1)
                idx += 1
                pbar.update(1)

    pbar.close()

    # Saving dataset as dictionary
    dataset_dir = DATA_PATH / ENV_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prev_images": prev_img,
        "actions": control,
        "next_images": next_img,
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