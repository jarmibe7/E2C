"""
Main training entry point for data4wm.

Usage:
    python -m src.main --config configs_final/button_eig_0

The ``--config`` argument is the stem (without ``.yaml``) of a file under
``config/``. Naming convention is load-bearing: the second underscore-separated
token (``eig`` / ``maxdyn`` / ``random`` / ``hardware``) selects the closed-loop
policy, and the seed suffix is appended to the run directory name. See
:func:`src.utils.build_run_path`.

Authors: Jared Berry, Ayush Gaggar
"""
import argparse
import copy
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.dataset import E2CDataset
from src.model.rssm import RSSME2C
from src.trainer import (
    ClosedLoopHardwareTrainer,
    ClosedLoopInformativeTrainer,
    ClosedLoopRandomTrainer,
    RSSMPretrainer,
)
from src.utils import build_run_path, format_time, set_seed


def posixpath_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return Path(*seq)


# Teach PyYAML how to rehydrate the PosixPath objects we write out in configs.
yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    posixpath_constructor,
)


PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"
RUNS_PATH = PROJECT_ROOT / "runs"


def _load_config(config_name):
    """Load the yaml at ``config/<config_name>.yaml`` and stash the stem."""
    config_file = config_name if config_name.endswith('.yaml') else f"{config_name}.yaml"
    with open(CONFIG_PATH / config_file, "r") as f:
        config = yaml.safe_load(f)
    config['config_name'] = config_name
    return config


def _build_model(config, device):
    """Instantiate the RSSM world model from config."""
    return RSSME2C(
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


def _build_trainer(dataset, model, config, device):
    """Pick the right trainer based on ``config['closed_loop']['policy']``."""
    if config.get('closed_loop', {}).get('closed_loop', False):
        policy_type = config['closed_loop'].get('policy', None)
        if policy_type == 'random':
            return ClosedLoopRandomTrainer(dataset, model, config, device)
        if policy_type in ('informative', 'maxdyn'):
            return ClosedLoopInformativeTrainer(dataset, model, config, device)
        if policy_type == 'hardware':
            return ClosedLoopHardwareTrainer(dataset, model, config, device)
        raise NotImplementedError(f'Policy type "{policy_type}" not supported!')
    return RSSMPretrainer(dataset, model, config, device)


def _save_on_exit(trainer, config_save, config, start_time, reason):
    """Shared save-and-maybe-eval path used by all exit branches of ``main``."""
    if not config['train']['save']:
        print(f'{reason}, ending training (save=False)')
        return
    config_save['runtime'] = format_time(time.perf_counter() - start_time)
    trainer.save(config_save, config['run_path'])
    if config['train']['eval']:
        trainer.evaluate(config['run_path'])
    print(f'\n{reason}, saving current checkpoint')


def main():
    start_time = time.perf_counter()
    print('*** STARTING ***\n')

    parser = argparse.ArgumentParser(description='data4wm training entry point')
    parser.add_argument(
        '--config',
        type=str,
        default='configs_final/button_eig_0',
        help='Name of the config file under config/ (with or without .yaml).',
    )
    args = parser.parse_args()

    config_name = args.config
    config = _load_config(config_name)
    set_seed(config.get('seed', 0))
    config_save = copy.deepcopy(config)

    # Run path
    run_path = build_run_path(config, config_name, RUNS_PATH)
    config['run_path'] = run_path

    # Device
    if 'cuda' in config['train']['device']:
        assert torch.cuda.is_available(), (
            f"{config['train']['device']} selected in {config_name}, but is unavailable!"
        )
    device = torch.device(config['train']['device'])

    # Dataset (also fills in derived image/control shapes)
    print(f"Loading dataset: {config['train']['dataset']}\n")
    dataset = E2CDataset(config)
    config['vae']['in_image_shape'] = dataset.in_img_shape
    num_out_channels = config['vae']['in_image_shape'][0] // config['trans']['past_length']
    config['vae']['out_image_shape'] = (num_out_channels, *config['vae']['in_image_shape'][1:])
    config['trans']['control_size'] = dataset.U.shape[-1]

    # Model (from scratch or checkpoint)
    model = _build_model(config, device)
    load_path = config['train'].get('load_path', None)
    if load_path is None:
        print('Training model from scratch\n')
        config['run_path'].mkdir(parents=True, exist_ok=True)
        config_save['load_path'] = config['run_path']
        curr_epoch = 0
    else:
        load_path = load_path.split("model.pt")[0] if load_path.endswith('model.pt') else load_path
        model_path = load_path + '/model.pt'
        model.load_state_dict(torch.load(model_path))
        config['run_path'] = PROJECT_ROOT / Path(load_path)
        with open(config['run_path'] / 'config.yaml', "r") as f:
            loaded_config = yaml.safe_load(f)
        curr_epoch = loaded_config['train']['num_epochs']
        config_save['load_path'] = config['run_path']
        print(f'Loading model from checkpoint: {model_path} at epoch {curr_epoch}\n')

    # Trainer
    trainer = _build_trainer(dataset, model, config, device)
    if config.get('closed_loop', {}).get('closed_loop', False):
        config_save['env_interactions'] = trainer.num_env_inters
        config_save['train_interactions'] = trainer.num_train_inters
    trainer.curr_epoch = curr_epoch

    if config['train'].get('eval_only', False):
        print('*** EVAL ONLY ***')
        saved_state = trainer.evaluator.visualize_planner(
            trainer, config['run_path'], max_steps=100, closed_loop=True
        )
        torch.save(saved_state, config['run_path'] / 'eval_saved_state.pt')
        print('\n*** DONE ***')
        return

    # Train, save, and evaluate
    try:
        trainer.learn()
        config_save['runtime'] = format_time(time.perf_counter() - start_time)
        if config['train']['save']:
            trainer.save(config_save, config['run_path'])
        if config['train']['eval']:
            trainer.evaluate(config['run_path'])
        saved_state = trainer.evaluator.visualize_planner(
            trainer, config['run_path'], max_steps=150, closed_loop=True
        )
        torch.save(saved_state, config['run_path'] / 'eval_saved_state.pt')
    except KeyboardInterrupt:
        _save_on_exit(trainer, config_save, config, start_time, 'Manual interrupt occured')
    except Exception:
        print('\n\n'); traceback.print_exc(); print('\n\n')
        _save_on_exit(trainer, config_save, config, start_time, 'Exception occured')

    print('\n*** DONE ***')


if __name__ == '__main__':
    main()
