import subprocess
import yaml
from pathlib import Path, PosixPath
from src.eval_render_videos import posixpath_constructor, load_trainer

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config"
RUNS_PATH = PROJECT_ROOT / "runs"

# define yaml_safe load constructor to handle PosixPath
yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    posixpath_constructor,
)

#### DEFINE WHAT CUDA TO USE HERE ####
DEVICE_TO_USE = 'cuda:1' # None
NUM_EPOCHS = 150

for i in range(23, 24):
    # for config in [f'configs_final/{env}_{policy}_{i}' for policy in ['eig', 'maxdyn', 'random']]:
    # for policy in ['eig', 'maxdyn', 'random']:
    # for policy in ['maxdyn', 'random']:
    for policy in ['eig']:                                                      # CHANGE POLICY HERE
        for env in (['coffee', 'door', 'drawer']):                              # CHANGE ENV HERE
            config_name = f'{env}_{policy}'                                 # CHANGE CONFIG HERE
            # print(f"Loading config: {config_name}")
            # with open(CONFIG_PATH / f"{config_name}.yaml", "r") as f:
            #     config = yaml.safe_load(f)
            # config_file = f"{config_name[:-1]}{i}.yaml"
            # config['seed'] = i
            
            # if DEVICE_TO_USE is not None:
            #     config['train']['device'] = DEVICE_TO_USE 
            # if NUM_EPOCHS is not None:
            #     config['train']['num_epochs'] = NUM_EPOCHS
            # config['config_name'] = config_name
            # config['loss']['recon_mult'] = 300
            # config['trans']['alpha'] = 2e-5
            
            # config['loss']['free_nats'] = 0.0
            # config['closed_loop']['sigma_init'] = 1.5
            # config['closed_loop']['sigma_min'] = 0.25
            # config['closed_loop']['elite_frac'] = 0.2
            # config['closed_loop']['iters'] = 4
            # config['closed_loop']['alpha'] = 0.3
            # policy = config_name.split('/')[-1].split('_')[1]
            # if policy in ['eig']:                           # CHANGE POLICY HERE
            #     if policy == 'eig':
            #         objective = 'pixel'
            #     elif policy == 'maxdyn':
            #         objective = 'dynamics'
            #     else:
            #         objective = 'random'
            #     save_name = config['train']['dataset'].split('_')[0] + '_' + objective + '_' + str(config.get('seed', 0))
            # else:
            #     save_name = config['train']['dataset'].split('_')[0] + '_' + policy + '_' + str(config.get('seed', 0))
            # run_path = RUNS_PATH / Path(config['train']['dataset'].split('_')[0]) / save_name
            # model_path = run_path / 'model.pt'
            # if model_path.exists() and policy != 'random':
            #     config['train']['load_path'] = str(run_path)
            # else:
            #     print(f"I couldn't find a checkpoint, training from scratch")
            # save updated config with device and load_path
            # new_config_path = CONFIG_PATH / Path(str(config_file).split('.yaml')[0] + "_test.yaml")
            # new_config_path = CONFIG_PATH / config_file
            # with open(new_config_path, "w") as f:
            #     yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False) # Save original config so model can be loaded later
            
            subprocess.run(
                ["python3.10", "-m", "src.main", "--config", str(config_name)],
                check=True,
            )
