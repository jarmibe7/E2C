"""
Minimal sweep harness that launches ``src.main`` as a subprocess for each
(env, policy, seed) combo. Override device / epochs / seed at the top and
un-comment the policy/env tuples you want to run.
"""
import subprocess
from pathlib import Path

import yaml

from src.eval_render_videos import posixpath_constructor


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config"
RUNS_PATH = PROJECT_ROOT / "runs"

yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    posixpath_constructor,
)

#### DEFINE WHAT CUDA TO USE HERE ####
DEVICE_TO_USE = 'cuda:1'    # e.g. 'cuda:0' / 'cuda:1' / 'cpu' / None (leave as-is)
NUM_EPOCHS = 150            # None (leave as-is)

for i in range(100, 101):
    # for policy in ['eig', 'maxdyn', 'random']:
    # for policy in ['maxdyn', 'random']:
    for policy in ['maxdyn']:
        for env in ['faucet', 'coffee', 'button', 'door', 'drawer']:
            config_name = f'configs_gripper/{env}_{policy}_{100}'
            print(f"Loading config: {config_name}")
            with open(CONFIG_PATH / f"{config_name}.yaml", "r") as f:
                config = yaml.safe_load(f)
            config_file = f"{config_name[:-3]}{i}.yaml"
            config['seed'] = i

            if DEVICE_TO_USE is not None:
                config['train']['device'] = DEVICE_TO_USE
            if NUM_EPOCHS is not None:
                config['train']['num_epochs'] = NUM_EPOCHS
            config['config_name'] = config_name

            new_config_path = CONFIG_PATH / config_file
            with open(new_config_path, "w") as f:
                yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

            subprocess.run(
                ["python3.10", "-m", "src.main", "--config", str(new_config_path)],
                check=True,
            )
