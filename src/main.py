"""
main.py

Main training script for E2C

Authors: Jared Berry, Ayush Gaggar
"""
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
from src.trainer import E2CPretrainer, RSSMPretrainer, ClosedLoopRandomTrainer, ClosedLoopInformativeTrainer, ClosedLoopHardwareTrainer
import argparse

def posixpath_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return Path(*seq)

# define yaml_safe load constructor to handle PosixPath
yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    posixpath_constructor,
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"
RUNS_PATH = PROJECT_ROOT / "runs"

def main():
    start_time = time.perf_counter()
    print('*** STARTING ***\n')
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
    config_file = config_name if config_name.endswith('.yaml') else f"{config_name}.yaml"
    with open(CONFIG_PATH / config_file, "r") as f:
        config = yaml.safe_load(f)
    config['config_name'] = config_name
    # Set random seed globally
    set_seed(config.get('seed', 0))
    config_save = copy.deepcopy(config)
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    policy = config_name.split('/')[-1].split('_')[1]
    if policy in ['eig', 'maxdyn', 'random']:
        if policy == 'eig':
            objective = 'pixel'
        elif policy == 'maxdyn':
            objective = 'dynamics'
        else:
            objective = 'random'
        save_name = config['train']['dataset'].split('_')[0] + '_' + objective + '_' + str(config.get('seed', 0))
    else:
        save_name = config['train']['dataset'].split('_')[0] + '_' + policy + '_' + str(config.get('seed', 0))
    run_path = RUNS_PATH / Path(config['train']['dataset'].split('_')[0]) / save_name
    config['run_path'] = run_path
    if 'cuda' in config['train']['device']: 
        assert torch.cuda.is_available(), f"{config['train']['device']} selected in {config_name}, but is unavailable!"
    device = torch.device(config['train']['device'])

    # Make E2CDataset object
    # TODO: Make temp dataset for hardware experiments
    print(f"Loading dataset: {config['train']['dataset']}\n")
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
        curr_epoch = 0
    else:
        # Load existing model to train from checkpoint
        load_path = load_path.split("model.pt")[0] if load_path.endswith('model.pt') else load_path
        model_path = load_path + '/model.pt'
        model.load_state_dict(torch.load(model_path))
        config['run_path'] = PROJECT_ROOT / Path(load_path)
        with open(config['run_path'] / 'config.yaml', "r") as f:
            loaded_config = yaml.safe_load(f)
        curr_epoch = loaded_config['train']['num_epochs']
        config_save['load_path'] = config['run_path']
        print(f'Loading model from checkpoint: {model_path} at epoch {curr_epoch}\n')
    
    # Make Trainer
    # If active learning, just use env specified by dataset name
    # TODO: Should policy use past_length too?
    if config.get('closed_loop', None) is not None and config['closed_loop']['closed_loop']:
        env = None
        policy_type = config['closed_loop'].get('policy', None)
        if policy_type == 'random':
            trainer = ClosedLoopRandomTrainer(dataset, model, config, device)
        elif policy_type == 'informative':
            trainer = ClosedLoopInformativeTrainer(dataset, model, config, device)
        elif policy_type == 'maxdyn':
            trainer = ClosedLoopInformativeTrainer(dataset, model, config, device)
        elif policy_type == "hardware":
            trainer = ClosedLoopHardwareTrainer(dataset, model, config, device)
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
    trainer.curr_epoch = curr_epoch

    if config['train'].get('eval_only', False):
            print('*** EVAL ONLY ***')
            # trainer.evaluate(config['run_path'])
            # trainer.evaluator.eval_traj(config['run_path'], max_frames=25)
            saved_state = trainer.evaluator.visualize_planner(trainer, config['run_path'], max_steps=100, closed_loop=True)
            torch.save(saved_state, config['run_path'] / 'eval_saved_state.pt')
            print('\n*** DONE ***')
            return
    
    # Train, save, and evaluate
    try:
        trainer.learn()

        # Save and evaluate
        config_save['runtime'] = format_time(time.perf_counter() - start_time)
        if config['train']['save']: trainer.save(config_save, config['run_path'])
        if config['train']['eval']: trainer.evaluate(config['run_path'])
        saved_state = trainer.evaluator.visualize_planner(trainer, config['run_path'], max_steps=150, closed_loop=True)
        torch.save(saved_state, config['run_path'] / 'eval_saved_state.pt')
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