"""
Minimal eval script for generating videos N number of times
"""
import os
import torch
import yaml
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from src.model.rssm import RSSME2C
from src.dataset import E2CDataset
from src.trainer import ClosedLoopRandomTrainer, ClosedLoopInformativeTrainer

def posixpath_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return Path(*seq)

# define yaml_safe load constructor to handle PosixPath
yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    posixpath_constructor,
)

def load_trainer(config_name):
    """
    Load trainer from config
    """
    config_file = config_name if config_name.endswith('.yaml') else f"{config_name}.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    if 'cuda' in config['train']['device']: 
        assert torch.cuda.is_available(), f"{config['train']['device']} selected in {config_name}, but is unavailable!"
    device = torch.device(config['train']['device'])
    dataset = E2CDataset(config)
    config['vae']['in_image_shape'] = dataset.in_img_shape
    num_out_channels = config['vae']['in_image_shape'][0] // config['trans']['past_length']    # Output only single frame
    config['vae']['out_image_shape'] = (num_out_channels, *config['vae']['in_image_shape'][1:])     
    config['trans']['control_size'] = dataset.U.shape[-1]
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

    config['closed_loop']['samples'] = 500
    config['closed_loop']['plan_horizon'] = 8
    config['closed_loop']['sigma_init'] = 1.5
    config['closed_loop']['sigma_min'] = 0.05
    config['closed_loop']['elite_frac'] = 0.1
    config['closed_loop']['iters'] = 6
    config['closed_loop']['alpha'] = 0.1

    # config['closed_loop']['sigma_init'] = 1.5
    # config['closed_loop']['sigma_min'] = 0.1
    # config['closed_loop']['elite_frac'] = 0.4
    # config['closed_loop']['iters'] = 4
    
    load_path = config.get('load_path', None)
    if load_path is None:
        raise ValueError("load_path must be specified in config to load trainer for evaluation")
    if type(load_path) == str:
        load_path = Path(load_path)
    # Load existing model to train from checkpoint
    load_path = load_path.split("model.pt")[0] if str(load_path).endswith('model.pt') else load_path
    model_path = load_path / 'model.pt' # can select model_200.pt if needed
    print(f'Loading model from checkpoint: {model_path}')
    model.load_state_dict(torch.load(model_path))
    policy_type = config['closed_loop'].get('policy', None)
    if policy_type == 'random':
        trainer = ClosedLoopRandomTrainer(dataset, model, config, device)
    elif policy_type == 'informative':
        trainer = ClosedLoopInformativeTrainer(dataset, model, config, device, prints=False)
    elif policy_type == 'maxdyn':
        trainer = ClosedLoopInformativeTrainer(dataset, model, config, device, prints=False)
    else:
        raise ValueError(f"Unknown control policy type: {policy_type}")
    return trainer

def render_video(trainer: ClosedLoopRandomTrainer, max_steps=100, env_seed=0):
    trainer.curr_epoch = env_seed
    trainer.config['load_path'] = trainer.config['load_path'] if type(trainer.config['load_path']) == Path else Path(trainer.config['load_path'])
    saved_state = trainer.evaluator.render_video(trainer, trainer.config['load_path'], max_steps=max_steps, closed_loop=True, env_reset_seed=env_seed)
    print(f"# of contacts: {torch.count_nonzero(saved_state[:, -1]).item()}\n")
    torch.save(saved_state, trainer.config['load_path'] / f'eval_states_{env_seed}.pt')

if __name__ == "__main__":
    for i in range(1, 2):
        # for policy, objective in tqdm(zip(['random', 'eig', 'maxdyn'], ['random', 'pixel', 'dynamics'])):
        # for policy, objective in tqdm(zip(['maxdyn', 'eig'], ['pixel', 'dynamics'])):
        # for policy, objective in tqdm(zip(['random'], ['random'])):
        for policy, objective in zip(['eig'], ['pixel']):
        # for policy, objective in zip(['maxdyn', 'random'], ['dynamics', 'random']):
            # for env in ['drawer', 'faucet', 'button', 'coffee', 'door']: # looks good...
            for env in ['faucet']: # [button, 'door', 'drawer', 'faucet']: #, 'button', 'coffee', 'door', 'drawer', 'faucet']:
                config = "runs/{env}/{env}_{objective}_{i}/config.yaml".format(env=env, objective=objective, i=i)
                # print(f"Evaluating config: {config}")
                trainer = load_trainer(config)
                for render_seed in range(12, 13):
                    render_video(trainer, max_steps=100, env_seed=render_seed)