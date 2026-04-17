"""
Minimal eval script for generating videos N number of times
"""
import os
import torch
import yaml
import time
import json
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
    device = torch.device('cuda:0')
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

    config['closed_loop']['samples'] = 300
    config['closed_loop']['plan_horizon'] = 8
    config['closed_loop']['sigma_init'] = 0.15
    config['closed_loop']['sigma_min'] = 0.05
    config['closed_loop']['elite_frac'] = 0.1
    config['closed_loop']['iters'] = 3
    config['closed_loop']['alpha'] = 0.1
    # if config['closed_loop'].get('policy', None) == 'maxdyn':
    #     config['closed_loop']['sigma_min'] = 0.3

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

def render_video(trainer: ClosedLoopRandomTrainer, max_steps=100, env_seed=0, video_save_path=None):
    trainer.curr_epoch = env_seed
    if video_save_path is None:
        video_save_path = trainer.config['load_path'] if type(trainer.config['load_path']) == Path else Path(trainer.config['load_path'])
    else:
        video_save_path = Path(video_save_path)
        video_save_path.mkdir(parents=True, exist_ok=True)
    saved_state = trainer.evaluator.render_video(trainer, video_save_path, max_steps=max_steps, closed_loop=True, env_reset_seed=env_seed)
    contact_count = torch.count_nonzero(saved_state[:, -1]).item()
    print(f"# of contacts: {contact_count}\n")
    torch.save(saved_state, video_save_path / f'{trainer.env_name}_{env_seed}.pt')
    return contact_count
    

if __name__ == "__main__":
    contacts_by_objective_env = {
        'coffee': {},
        'button': {},
        'door': {},
        'drawer': {},
        'faucet': {}
    }
    try:
        for i in range(100, 101):
            # for env in tqdm(['button', 'coffee', 'door', 'drawer', 'faucet']): #, 'button', 'door', 'drawer', 'faucet', 'coffee']:
            for env in tqdm(['coffee']): #, 'button', 'door', 'drawer', 'faucet', 'coffee']:
            # for env in tqdm(['coffee', 'door', 'drawer', 'faucet']): #, 'button', 'door', 'drawer', 'faucet', 'coffee']:
            # for env in ['button']: #, 'button', 'door', 'drawer', 'faucet', 'coffee']:
                # for policy, objective in tqdm(zip(['eig', 'maxdyn', 'random'], ['pixel', 'dynamics', 'random'])):
                for policy, objective in tqdm(zip(['maxdyn'], ['dynamics'])):
                # for policy, objective in zip(['random'], ['random']):
                # for env in ['drawer', 'faucet', 'button', 'coffee', 'door']: # looks good...
                    # if policy == 'random':
                    #     config = "runs/{env}/{env}_{objective}_{i}/config.yaml".format(env=env, objective=objective, i=20)
                    # else:
                    config = "runs/{env}/{env}_{objective}_{i}/config.yaml".format(env=env, objective=objective, i=i)
                    # print(f"Evaluating config: {config}")
                    trainer = load_trainer(config)
                    for render_seed in range(0, 2):
                        # if env == 'button':
                        #     contact_count = render_video(trainer, max_steps=100, env_seed=int(10*render_seed), video_save_path=f"src/data_gen/contact/videos/{env}_{objective}_{i}")
                        # else:
                        contact_count = render_video(trainer, max_steps=250, env_seed=render_seed, video_save_path=f"src/data_gen/contact/videos/{env}_{objective}_{i}_150")
                        contacts_by_objective_env[env][objective] = contacts_by_objective_env[env].get(objective, []) + [contact_count]
                print(f"Contacts so far: {contacts_by_objective_env}")
                # # save results by environment after all policies
                # out_path = Path("src/data_gen/contact/") / f"{env}_{i}.json"
                # with open(out_path, "w") as f:
                #     json.dump(contacts_by_objective_env, f, indent=2)
                # print(f"Saved contacts to {out_path}")


            # # save results to json
            out_path = Path("src/data_gen/contact/") / f"contacts_seed_{i}_150.json"
            with open(out_path, "w") as f:
                json.dump(contacts_by_objective_env, f, indent=2)
            print(f"Saved contacts to {out_path}")
    except Exception as e:
        pass
    # except KeyboardInterrupt:
    #     print("Interrupted by user, saving results...")
    #     out_path = Path("src/data_gen/contact/") / f"contacts_seed_{i}.json"
    #     with open(out_path, "w") as f:
    #         json.dump(contacts_by_objective_env, f, indent=2)
    #     print(f"Saved contacts to {out_path}")
    # except Exception as e:
    #     print(f"Exception occurred: {e}, saving results...")
    #     out_path = Path("src/data_gen/contact/") / f"contacts_seed_{i}.json"
    #     with open(out_path, "w") as f:
    #         json.dump(contacts_by_objective_env, f, indent=2)
    #     print(f"Saved contacts to {out_path}")