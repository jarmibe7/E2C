"""
main.py

Main training script for E2C

Authors: Jared Berry, Ayush Gaggar
"""
import os
os.environ["DISPLAY"] = ":98"          # Ayush: If getting display errors, change this
os.environ["MUJOCO_GL"] = "glfw"
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import time
from datetime import datetime
from pathlib import Path
import copy
import traceback

from src.model.e2c import ConvE2C
from src.model.rssm import RSSME2C
from src.dataset import E2CDataset
from src.utils import set_seed, anim_frames, format_time
from src.model.policy import ConvPolicy
from src.trainer import E2CPretrainer, RSSMPretrainer, ClosedLoopPolicyTrainer, ClosedLoopRandomTrainer, ClosedLoopInformativeTrainer
import argparse

# # Configure Matplotlib to use LaTeX for text rendering
# import matplotlib as mpl
# plt.rcParams['text.usetex'] = False
# plt.rcParams['pdf.fonttype'] = 42  # Ensures TrueType fonts are used
# plt.rcParams['ps.fonttype'] = 42   # Ensures TrueType fonts for PostScript output
# mpl.rcParams['lines.linewidth'] = 5
# mpl.rcParams['lines.markersize'] = 10
# mpl.rcParams['lines.markeredgewidth'] = 1
# mpl.rcParams['axes.linewidth'] = 2
# mpl.rcParams['figure.titlesize'] = 25
# mpl.rcParams['figure.titleweight'] = 'bold'
# mpl.rcParams['axes.titlesize'] = 20
# # mpl.rcParams['axes.titleweight'] = 'bold'
# mpl.rcParams['axes.labelsize'] = 20
# mpl.rcParams['axes.titlepad'] = 5
# mpl.rcParams['xtick.major.size'] = 6
# mpl.rcParams['xtick.major.width'] = 3
# mpl.rcParams['xtick.labelsize'] = 20
# mpl.rcParams['xtick.major.pad'] = 3
# mpl.rcParams['ytick.major.size'] = 6
# mpl.rcParams['ytick.major.width'] = 3
# mpl.rcParams['ytick.labelsize'] = 20
# mpl.rcParams['ytick.major.pad'] = 3
# mpl.rcParams['figure.subplot.hspace'] = 0.85
# mpl.rcParams["axes.labelpad"] = 5

# Set random seed globally
set_seed(42)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"
RUNS_PATH = PROJECT_ROOT / "runs"

def main():
    start_time = time.perf_counter()
    print('*** STARTING ***\n')
    print("DISPLAY =", os.environ.get("DISPLAY"))
    # Load config, make run path, and choose torch device
    # ---------- CONFIG HERE ----------
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='E2C Training Script')
    parser.add_argument(
        '--config', 
        type=str, 
        default='rssm_push_active_v0',
        help='Name of the config file (without .yaml extension)'
    )
    args = parser.parse_args()
    config_name = args.config
    with open(CONFIG_PATH / f'{config_name}.yaml', "r") as f:
        config = yaml.safe_load(f)
    config['config_name'] = config_name
    config_save = copy.deepcopy(config)
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    run_path = RUNS_PATH / Path(config['train']['dataset'].split('_')[0]) / timestamp
    config['run_path'] = run_path
    if 'cuda' in config['train']['device']: 
        assert torch.cuda.is_available(), f"{config['train']['device']} selected in {config_name}, but is unavailable!"
    device = torch.device(config['train']['device'])

    # Make E2CDataset object
    print(f"Loading dataset: {config['train']['dataset']}\n")
    config['env_name'] = config['train']['dataset'].split('_')[0]
    dataset = E2CDataset(config)
    config['vae']['in_image_shape'] = dataset.in_img_shape
    num_out_channels = config['vae']['in_image_shape'][0] // config['trans']['past_length']    # Output only single frame
    config['vae']['out_image_shape'] = (num_out_channels, *config['vae']['in_image_shape'][1:])     
    config['trans']['control_size'] = dataset.U.shape[-1]

    # Create or load model
    if 'e2c' in config_name:
        model = ConvE2C(
            enc_latent_size=config['vae']['enc_latent_size'],
            latent_size=config['trans']['latent_size'],
            control_size=config['trans']['control_size'],
            past_length=config['trans']['past_length'],
            pred_length=config['trans']['pred_length'],
            conv_params=config['vae'],
            device=device,
            output_uncertainty=(config['loss']['loss_type'] == 'uncertainty' or 'rssm' in config['loss']['loss_type'])
        )
    else:
        model = RSSME2C(
            enc_latent_size=config['vae']['enc_latent_size'],
            stochastic_size=config['trans']['stochastic_size'],
            deterministic_size=config['trans']['deterministic_size'],
            control_size=config['trans']['control_size'],
            past_length=config['trans']['past_length'],
            pred_length=config['trans']['pred_length'],
            conv_params=config['vae'],
            device=device,
            output_uncertainty=(config['loss']['loss_type'] == 'uncertainty' or 'rssm' in config['loss']['loss_type'])
        )
    load_path = config['train'].get('load_path', None)
    if load_path is None:
        print(f'Training model from scratch\n')
        config['run_path'].mkdir(parents=True, exist_ok=True)
        config_save['load_path'] = config['run_path']
    else:
        # Load existing model to train from checkpoint
        print(f'Loading model from checkpoint\n')
        model_path = load_path + '/model.pt'
        model.load_state_dict(torch.load(model_path))
        config['run_path'] = PROJECT_ROOT / Path(load_path)
    
    # Make Trainer
    # If active learning, just use env specified by dataset name
    # TODO: Should policy use past_length too?
    if config.get('closed_loop', None) is not None and config['closed_loop']['closed_loop']:
        env = None
        policy_type = config['closed_loop'].get('policy', None)
        if policy_type == 'conv':
            policy = ConvPolicy(config['trans']['control_size'], 
                                config['vae']['in_image_shape'][0] // config['trans']['past_length'],
                                config['vae'])
            trainer = ClosedLoopPolicyTrainer(dataset, model, config, device, policy)
        elif policy_type == 'random':
            trainer = ClosedLoopRandomTrainer(dataset, model, config, device)
        elif policy_type == 'informative':
            trainer = ClosedLoopInformativeTrainer(dataset, model, config, device)
        elif policy_type == "direct_reward":
            # TODO: Implement shallow reward-based closed loop trainer
            pass
        else: 
            raise NotImplementedError(f'Policy type "{policy_type}" not supported!')
        config_save['env_interactions'] = trainer.num_env_inters
        config_save['train_interactions'] = trainer.num_train_inters
    else:
        if 'rssm' in config_name:
            trainer = RSSMPretrainer(dataset, model, config, device)
        else:
            trainer = E2CPretrainer(dataset, model, config, device)

    if config['train'].get('eval_only', False):
            print('*** EVAL ONLY ***')
            # trainer.evaluate(config['run_path'])
            # trainer.evaluator.eval_traj(config['run_path'], max_frames=25)
            # trainer.evaluator.visualize_planner(trainer, config['run_path'], max_steps=50, closed_loop=True)
            trainer.evaluator.eval_state_rep(trainer, config['run_path'], max_steps=50, closed_loop=True)
            print('\n*** DONE ***')
            return
    
    # Train, save, and evaluate
    try:
        trainer.learn()

        # Save and evaluate
        config_save['runtime'] = format_time(time.perf_counter() - start_time)
        if config['train']['save']: trainer.save(config_save, config['run_path'])
        if config['train']['eval']: trainer.evaluate(config['run_path'])
    except Exception:
        print('\n\n'); traceback.print_exc(); print('\n\n')
        if config['train']['save']:
            config_save['runtime'] = format_time(time.perf_counter() - start_time)
            trainer.save(config_save, config['run_path'])
            if config['train']['eval']: trainer.evaluate(config['run_path'])    
            print(f'\nException occured, saving current checkpoint')
        else: 
            print('Exception occured, ending training')
    except KeyboardInterrupt:
        if config['train']['save']:
            config_save['runtime'] = format_time(time.perf_counter() - start_time)
            trainer.save(config_save, config['run_path'])
            if config['train']['eval']: trainer.evaluate(config['run_path'])    
            print(f'\nManual interrupt occured, saving current checkpoint')
        else: 
            print('Manual interrupt occured, ending training')

    

    print('\n*** DONE ***')
    return

if __name__ == '__main__':
    main()