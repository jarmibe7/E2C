"""
Generate (image, state) dataset from a Gymnasium environment

Author: Jared Berry
"""

import time
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from pyvirtualdisplay import Display
import yaml
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from src.utils import set_seed, format_time
from src.data_gen.gen_gym import process_image, name_to_env

# ------------------------
# Configuration
# ------------------------
ENV_NAME = "push"
DATASET_SIZE = int(1e3)
IMAGE_SHAPE = (64, 64, 3)
SEED = 42

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# ------------------------
# Utilities
# ------------------------
def get_env_state(env, obs, env_name):    
    """
    Return true joint state when available (MuJoCo),
    otherwise fall back to observation.
    """
    unwrapped = env.unwrapped
    breakpoint()

    # Reacher: first 2 are joint angles
    # Remaining qpos entries are target-related
    if "reacher" in env_name:
        qpos = unwrapped.data.qpos.copy()
        qvel = unwrapped.data.qvel.copy()
        joint_pos = qpos[:2]
        joint_vel = qvel[:2]
        state = np.concatenate([joint_pos, joint_vel])
    elif 'push' in env_name:
        state = obs['observation'][0:6]    # [0:3] == end effector pose, [3:6] == block pose
    else:
        # Fallback to observation
        raise ValueError(f'Gym environment {env_name} is not specified in get_env_state()')
    
    if torch.is_tensor(state): return state
    else: return torch.from_numpy(state).float()


def update_metadata(dataset_dir, name, params):
    path = dataset_dir / "metadata.yaml"
    if path.exists():
        with open(path, "r") as f:
            metadata = yaml.safe_load(f) or {}
    else:
        metadata = {}

    metadata[name] = {
        **params,
        "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    with open(path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

# ------------------------
# Main
# ------------------------
def main():
    set_seed(SEED)
    start_time = time.perf_counter()

    disp = Display(visible=0, size=(480, 480))
    disp.start()

    env = gym.make(name_to_env[ENV_NAME], render_mode="rgb_array")
    obs, _ = env.reset(seed=SEED)

    # Infer state dimension
    state = get_env_state(env, obs)
    state_dim = state.numel()

    images = torch.zeros((DATASET_SIZE, *IMAGE_SHAPE), dtype=torch.float32)
    states = torch.zeros((DATASET_SIZE, state_dim), dtype=torch.float32)

    idx = 0
    pbar = tqdm(total=DATASET_SIZE)

    while idx < DATASET_SIZE:
        # Render + store
        img = process_image(env.render(), dataset_name=ENV_NAME, image_shape=IMAGE_SHAPE)
        state = get_env_state(env, obs)

        images[idx] = img
        states[idx] = state.view(-1)

        # Step environment
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

        idx += 1
        pbar.update(1)

    pbar.close()
    env.close()
    disp.stop()

    # Save
    dataset_dir = DATA_PATH / ENV_NAME / 'state_rep'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{ENV_NAME}_{DATASET_SIZE//1000}k_state_from_image"
    torch.save(
        {
            "images": images,
            "states": states,
        },
        dataset_dir / f"{output_name}.pt",
    )

    update_metadata(
        dataset_dir,
        output_name,
        {
            "env": ENV_NAME,
            "dataset_size": DATASET_SIZE,
            "image_shape": list(IMAGE_SHAPE),
            "state_dim": state_dim,
            "seed": SEED,
            "runtime": format_time(time.perf_counter() - start_time),
        },
    )

    print(f"\nSaved dataset to {dataset_dir / (output_name + '.pt')}")
    print("*** DONE ***")


if __name__ == "__main__":
    main()