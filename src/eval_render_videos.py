"""
Minimal eval script for rendering N single-panel rollout videos from a saved
training run.

Example:

    python -m src.eval_render_videos \\
        --envs button coffee --policies eig maxdyn --seeds 0 1 --i 0 --steps 250

The script looks up each run's config at
``runs/<env>/<env>_<objective>_<i>/config.yaml``, loads the trained model, and
writes one MP4 per (env, policy, seed) combination under
``src/data_gen/contact/videos/<env>_<objective>_<i>_<steps>/``. A matching
``contacts_seed_<i>_<steps>.json`` summarising robot-object contact counts per
rollout is written to ``src/data_gen/contact/``.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.dataset import E2CDataset
from src.model.rssm import RSSME2C
from src.trainer import ClosedLoopInformativeTrainer, ClosedLoopRandomTrainer


POLICY_TO_OBJECTIVE = {'eig': 'pixel', 'maxdyn': 'dynamics', 'random': 'random'}


def posixpath_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return Path(*seq)


yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    posixpath_constructor,
)


def load_trainer(config_name):
    """Instantiate a closed-loop trainer from a saved run's ``config.yaml``."""
    config_file = config_name if config_name.endswith('.yaml') else f"{config_name}.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    if 'cuda' in config['train']['device']:
        assert torch.cuda.is_available(), (
            f"{config['train']['device']} selected in {config_name}, but is unavailable!"
        )
    device = torch.device(config['train']['device'])
    dataset = E2CDataset(config)
    config['vae']['in_image_shape'] = dataset.in_img_shape
    num_out_channels = config['vae']['in_image_shape'][0] // config['trans']['past_length']
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
        output_uncertainty=(
            config['loss']['loss_type'] == 'uncertainty'
            or 'rssm' in config['loss']['loss_type']
        ),
    )

    load_path = config.get('load_path', None)
    if load_path is None:
        raise ValueError("load_path must be specified in config to load trainer for evaluation")
    if isinstance(load_path, str):
        load_path = Path(load_path)
    load_path = load_path.split("model.pt")[0] if str(load_path).endswith('model.pt') else load_path
    model_path = load_path / 'model.pt'
    print(f'Loading model from checkpoint: {model_path}')
    model.load_state_dict(torch.load(model_path))

    policy_type = config['closed_loop'].get('policy', None)
    if policy_type == 'random':
        trainer = ClosedLoopRandomTrainer(dataset, model, config, device)
    elif policy_type in ('informative', 'maxdyn'):
        trainer = ClosedLoopInformativeTrainer(dataset, model, config, device, prints=False)
    else:
        raise ValueError(f"Unknown control policy type: {policy_type}")
    return trainer


def render_video(trainer, max_steps=100, env_seed=0, video_save_path=None):
    """Render a single rollout video for ``trainer`` and return contact count."""
    trainer.curr_epoch = env_seed
    if video_save_path is None:
        load_path = trainer.config['load_path']
        video_save_path = load_path if isinstance(load_path, Path) else Path(load_path)
    else:
        video_save_path = Path(video_save_path)
        video_save_path.mkdir(parents=True, exist_ok=True)
    saved_state = trainer.evaluator.render_video(
        trainer, video_save_path, max_steps=max_steps,
        closed_loop=True, env_reset_seed=env_seed,
    )
    contact_count = torch.count_nonzero(saved_state[:, -1]).item()
    print(f"# of contacts: {contact_count}\n")
    torch.save(saved_state, video_save_path / f'{trainer.env_name}_{env_seed}.pt')
    return contact_count


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--envs', nargs='+', default=['coffee'],
                   help='Environments to render (e.g. button coffee door drawer faucet).')
    p.add_argument('--policies', nargs='+', default=['maxdyn'],
                   help='Policy tags in the config filename (eig | maxdyn | random).')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1],
                   help='env_reset_seed values to sweep over.')
    p.add_argument('--i', type=int, default=100,
                   help='Training run index suffix used in <env>_<objective>_<i>/.')
    p.add_argument('--steps', type=int, default=250,
                   help='Number of env steps per rollout.')
    p.add_argument('--runs-root', type=str, default='runs',
                   help='Root directory containing per-env run folders.')
    p.add_argument('--contact-root', type=str, default='src/data_gen/contact',
                   help='Root directory to write contact summaries/videos.')
    return p.parse_args()


def main():
    args = parse_args()
    contacts_by_objective_env = {env: {} for env in args.envs}
    video_root = Path(args.contact_root) / 'videos'

    for env in tqdm(args.envs):
        for policy in tqdm(args.policies):
            objective = POLICY_TO_OBJECTIVE.get(policy, policy)
            config = f"{args.runs_root}/{env}/{env}_{objective}_{args.i}/config.yaml"
            trainer = load_trainer(config)
            for render_seed in args.seeds:
                save_dir = video_root / f"{env}_{objective}_{args.i}_{args.steps}"
                contact_count = render_video(
                    trainer, max_steps=args.steps,
                    env_seed=render_seed, video_save_path=save_dir,
                )
                contacts_by_objective_env[env].setdefault(objective, []).append(contact_count)
        print(f"Contacts so far: {contacts_by_objective_env}")

    out_path = Path(args.contact_root) / f"contacts_seed_{args.i}_{args.steps}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(contacts_by_objective_env, f, indent=2)
    print(f"Saved contacts to {out_path}")


if __name__ == '__main__':
    main()
