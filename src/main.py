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

from src.e2c import E2CDataset, E2CLoss, ConvE2C
from src.utils import set_seed, anim_frames, format_time
from src.policy import ConvPolicy
from src.trainer import WorldModelPretrainer, ClosedLoopPolicyTrainer, ClosedLoopUncertaintyTrainer

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
    # Load config, make run path, and choose torch device
    # ---------- CONFIG HERE ----------
    config_name = 'e2c_reacher_v0'
    # ---------- CONFIG HERE ----------
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
    dataset = E2CDataset(config)
    config['vae']['out_image_shape'] = dataset.img_shape
    config['trans']['control_size'] = dataset.U.shape[-1]

    # Train on training dataet
    model = ConvE2C(
        enc_latent_size=config['vae']['enc_latent_size'],
        latent_size=config['trans']['latent_size'],
        control_size=config['trans']['control_size'],
        past_length=config['trans']['past_length'],
        pred_length=config['trans']['pred_length'],
        conv_params=config['vae'],
        device=device
    )
    load_path = config['train'].get('load_path', None)
    if load_path is None:
        print(f'Training model from scratch\n')
        config['run_path'].mkdir(parents=True, exist_ok=True)
        
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
                                config['vae']['out_image_shape'][0] // config['trans']['past_length'],
                                config['vae'])
            trainer = ClosedLoopPolicyTrainer(dataset, model, config, device, policy)
        else:
            raise NotImplementedError('no policy closed loop training not yet implemented')
            trainer = ClosedLoopUncertaintyTrainer(dataset, model, config, device, policy)
    else:
        trainer = WorldModelPretrainer(dataset, model, config, device)

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
            trainer.save(config_save, config['run_path'])
            print(f'\nException occured, saving current checkpoint')
        else: 
            print('Exception occured, ending training')

    

    print('\n*** DONE ***')
    return

if __name__ == '__main__':
    main()