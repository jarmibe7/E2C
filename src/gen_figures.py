"""
Script for generating extra figures by loading model checkpoint
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import time
import gymnasium as gym
from datetime import datetime
from pathlib import Path
from pyvirtualdisplay import Display

from src.e2c import E2CDataset, E2CLoss, E2C
from src.utils import set_seed, anim_frames
from src.eval import Plotter, Evaluator
from src.data_gen.gen_gym import process_image, name_to_env, env_to_aspace

set_seed(42)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RUN_PATH = PROJECT_ROOT / "runs" / "2025-12-04_18-40-16"

# Parameters
TIME_HORIZON = 8    # How many timesteps to predict into future

name_to_shape = {'cartpole': (64, 64, 3)}

def get_action(state, env_name):
    """
    Compute controller based on true env state to make figure
    """
    if env_name == 'cartpole':
        # Simple PID
        pole_angle = state[2]
        pole_vel = state[3]
        Kp = 100.0
        Kd = 2.0
        action_val = (Kp * pole_angle + Kd * pole_vel)
        action = 1 if action_val > 0 else 0
    return action

def main():
    print('*** STARTING ***\n')
    # Load config
    with open(RUN_PATH / 'config.yaml', "r") as f:
        config = yaml.safe_load(f)
    device = torch.device(config['train']['device'])
    if 'cuda' in config['train']['device']: 
        assert torch.cuda.is_available(), f"{config['train']['device']} selected in config, but is unavailable!"

    # Create virtual display for running on server
    disp = Display(visible=0, size=(480, 480))
    disp.start()

    # Create env and figure
    env = gym.make(name_to_env[config['train']['dataset']], render_mode="rgb_array")
    fig, axes = plt.subplots(TIME_HORIZON + 1, 3, figsize=(18, TIME_HORIZON*7), dpi=200, tight_layout=True)
    for ax in axes.flatten():
        # ax.axis('off')
        ax.set_aspect('equal')
    fontsize = 16
    axes[0, 0].set_title('Real\nt=0', fontsize=fontsize)
    axes[0, 1].set_title('Imagined\nt=0', fontsize=fontsize)
    axes[0, 1].imshow(np.ones(name_to_shape[config['train']['dataset']])) # No predictions at t=0
    axes[0, 2].set_title('One Step\nt=0', fontsize=fontsize)
    axes[0, 2].imshow(np.ones(name_to_shape[config['train']['dataset']]))

    # Create mini-dataset containers
    img = torch.zeros((TIME_HORIZON + 1, *name_to_shape[config['train']['dataset']]))
    continuous = (env_to_aspace[config['train']['dataset']] == 'continuous')
    if continuous: control_size = env.action_space.shape[0]
    else: control_size = 1     # Discrete action space
    control = torch.zeros((TIME_HORIZON + 1, control_size))

    # Load model
    config['vae']['out_image_shape'] = torch.tensor(name_to_shape[config['train']['dataset']])[torch.tensor([2, 0, 1])]
    model = E2C(
        enc_latent_size=config['vae']['enc_latent_size'],
        latent_size=config['trans']['latent_size'],
        control_size=control_size,
        past_length=config['trans']['past_length'],
        pred_length=config['trans']['pred_length'],
        conv_params=config['vae'],
        device=device
    )
    model.to(device)
    model_path = RUN_PATH / 'model.pt'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Predict a given number of steps into future from random starting position
    collected = 0
    initial_state = np.array([0.0, 0.0, -0.2, 0.0])
    raw_env = env.env.env.env
    while collected <= TIME_HORIZON:
        obs, _ = env.reset()

        # Start with initial error
        raw_env.state = initial_state
        for t in range(TIME_HORIZON + 1):
            # Save image and action pair
            rend = env.render()
            img[t] = process_image(rend, dataset_name=config['train']['dataset'])
            action = env.action_space.sample()

            action = get_action(raw_env.state, config['train']['dataset'])
            control[t] = torch.tensor([action]) if continuous else action

            # Take action
            obs, rew, done, _, _ = env.step(action)
            collected += 1

    # Create figure
    imagined = img[0].permute(2, 0, 1)
    for t, (x, u) in enumerate(zip(img, control)):
        # Plot true image
        axes[t, 0].imshow(x)

        if t > 0: 
            # Plot imagined prediction
            x, u = x.to(device), u.to(device)
            _, imagined = model.sample(imagined.unsqueeze(0), u.unsqueeze(0))
            axes[t, 1].imshow(imagined.permute(1, 2, 0).detach().cpu().numpy())

            # Plot one step prediction
            _, one_step = model.sample(x.permute(2, 0, 1).unsqueeze(0), u.unsqueeze(0))
            axes[t, 2].imshow(one_step.permute(1, 2, 0).detach().cpu().numpy())

            axes[t, 0].set_title(f't={t}', fontsize=fontsize)
            axes[t, 1].set_title(f't={t}', fontsize=fontsize)
            axes[t, 2].set_title(f't={t}', fontsize=fontsize)

    filepath = RUN_PATH / 'time_horizon_fig.png'
    print(f'\nSaved time horizon prediction figure to {filepath}')
    fig.savefig(filepath)
    plt.close(fig)
    
    print('\n*** DONE ***')


if __name__ == '__main__':
    main()
