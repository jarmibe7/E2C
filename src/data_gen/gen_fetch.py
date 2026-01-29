"""
Generate dataset from Gymnasium environment

scp -r data/reacher jarmibe7@dingo.mech.northwestern.edu:~/E2C/
scp jarmibe7@dingo.mech.northwestern.edu:~/E2C/videos/e2c_cartpole.mp4 C:\\Users\\jarmi\\MS_Thesis\\Media\\Videos

Author: Jared Berry, Ayush Gaggar
"""
import os
import re
import numpy as np
import torch
import time
import gymnasium as gym
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from pyvirtualdisplay import Display
from tqdm import tqdm
import yaml
from datetime import datetime
from PIL import Image

from src.utils import set_seed, format_time
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

# Parameters for dataset
env_name = 'push'                                           # Gym environment name
dataset_size = int(1e3)                                     # Number of samples: (img, next_img, control) tuple
OUTPUT_NAME = env_name + f'_{dataset_size // 1000}k'        # Output name of dataset
image_shape = (64, 64, 3)                                   # Downsampled image shape
past_length = 3                                             # Number of previous observations to use for training
pred_length = 3                                             # Number of timesteps to predict in the future
new_dt = None                                               # Desired new timestep in seconds
metaworld_cam_name = 'corner3' if OUTPUT_NAME.__contains__('isom') else 'corner'                        # Camera angle for metaworld environments: corner | behindGripper         
# ---------------------------------
# Only modify XML if new_dt is set
if new_dt is not None:
    mj_path = Path(os.path.dirname(gym.__file__)) / "envs" / "mujoco" / "assets"
    xml_file = mj_path / f"{env_name}.xml"
    xml_text = xml_file.read_text()
    xml_text = re.sub(r'timestep="[^"]+"', f'timestep="{new_dt:.4f}"', xml_text)

    # Save new XML
    new_xml_filename = f"{env_name}_timestep_{int(new_dt*1000)}_ms.xml"
    new_xml_path = mj_path / new_xml_filename
    new_xml_path.write_text(xml_text)
else:
    new_xml_filename = None

# Get data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

seed = 42
set_seed(seed)
meta_world_envs = ['shelf', 'sweep', 'assembly', 'test', 'plate', 'button', 'door', 'drawer', 'window', 'lever', 'coffee', 'faucet']
name_to_env = {'reacher': 'Reacher-v5', 
                'cartpole': 'CartPole-v1', 
                'push': 'FetchPushDense-v3', 
                'pointmaze': 'PointMaze_UMaze-v3', 
                'antmaze': 'AntMaze_UMaze-v5', 
                'mountaincar': 'MountainCarContinuous-v0',
                'shelf': 'shelf-place-v3', 
                'sweep': 'sweep-into-v3', 
                'assembly': 'assembly-v3', 
                'plate': 'plate-slide-v3',
                'button': 'button-press-v3',
                'door': 'door-open-v3',
                'drawer': 'drawer-open-v3',
                'window': 'window-open-v3',
                'lever': 'lever-pull-v3',
                'coffee': 'coffee-button-v3',
                'faucet': 'faucet-open-v3'
               }
env_to_aspace = {'reacher': 'continuous', 'cartpole': 'discrete', 'push': 'continuous', 
                 'pointmaze': 'continuous', 'antmaze': 'continuous', 'mountaincar': 'continuous',
                 'shelf': 'continuous', 'sweep': 'continuous', 'assembly': 'continuous', 'test': 'continuous'}

def update_dataset_metadata(dataset_dir, dataset_name, params):
    """
    Update or create metadata YAML in dataset_dir.
    Overwrites fields for dataset_name, preserves others.
    """
    metadata_path = dataset_dir / "metadata.yaml"

    # Load existing metadata if present
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f) or {}
    else:
        metadata = {}

    if dataset_name in metadata:
        print(f"Overwriting metadata for dataset '{dataset_name}'")

    # Update this dataset's entry
    metadata[dataset_name] = {
        **params,
        "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    # Write back to file
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

def debug_render(img):
    if len(img.shape) == 4: plt.imshow(img[0])
    else: plt.imshow(img)
    plt.savefig('debug.png')

def process_image(image, dataset_name, image_shape=image_shape, downscale=True):
    """
    Image processing
    """
    dataset_name = dataset_name.split('_')[0]
    if 'cartpole' in dataset_name: image = image[50:350, 100:400]   # Zoom on cartpole
    elif 'reacher' in dataset_name: image = image[100:-50, 100:-100]  # Zoom on reacher
    elif 'push' in dataset_name: image = image[100:-50, :-100, :]     # Zoom on robot # TODO: need to zoom on push?
    elif 'pointmaze' in dataset_name: image = image[110:-70, 20:-20, :]
    elif 'antmaze' in dataset_name: image = image[110:-70, 20:-20, :]
    elif dataset_name in meta_world_envs: # and metaworld_cam_name == 'corner': 
        # always rotate Metaworld images
        image = np.rot90(image, k=2)    # Metaworld corner images are upside down
    elif dataset_name in meta_world_envs and metaworld_cam_name == 'behindGripper':
        pass 
    else: pass

    image = torch.from_numpy(image.copy()).permute(2, 0, 1)  # Get image tensor into (C, H, W)

    # Image processing
    normalized = image.float() / 255.0 # Normalize to [0,1]
    # if 'mountaincar' in dataset_name:
    if downscale:
        image_resized = torchvision.transforms.Resize(size=image_shape[0:2], antialias=True)(normalized)
        image_resized = image_resized.clamp(0.0, 1.0)
    else:
        image_resized = normalized
    # else:
        # image_resized = torchvision.transforms.functional.resize(normalized, image_shape[0:2], interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)   # Downscaling
    
    return image_resized.permute(1, 2, 0) # Permute back to raw shape

def get_obj_site_id(dataset_name, mj_model):
    if 'button' in dataset_name:
        return mj_model.site("buttonStart").id
    elif 'coffee' in dataset_name:
        return mj_model.site("objSite").id
    elif 'door' in dataset_name or 'drawer' in dataset_name or 'plate' in dataset_name:
        return mj_model.site("goal").id

def get_mujoco_geom_keys_index(dataset_name):
    """
    Return dicts with collision geometry geom_index -> geom_name for both robot and obj
    """
    robot_geom = {
            25: 'right_l6',
            26: 'right_l6',
            27: 'right_hand',
            28: 'right_hand',
            29: 'hand',
            30: 'rightclaw',
            31: 'rightpad',
            32: 'leftclaw',
            33: 'leftpad',
        }
    if 'button' in dataset_name:
        obj_geom = {
            36: 'buttonbox',
            37: 'buttonbox',
            38: 'buttonbox',
            39: 'buttonbox',
            40: 'buttonbox',
            41: 'buttonbox',
            42: 'buttonbox',
            43: 'button',
            44: 'button',
            45: 'button',
            46: 'button',
            47: 'button',
        }
    elif 'drawer' in dataset_name:
        obj_geom = {
            36: 'drawercase_link',
            37: 'drawercase_link',
            38: 'drawercase_link',
            39: 'drawercase_link',
            40: 'drawercase_link',
            41: 'drawercase_link',
            42: 'drawer_link',
            43: 'drawer_link',
            44: 'drawer_link',
            45: 'drawer_link',
            46: 'drawer_link',
            47: 'drawer_link',
            48: 'drawer_link',
            49: 'drawer_link',
            50: 'drawer_link',
            51: 'drawer_link',
        }
    elif 'coffee' in dataset_name:
        robot_geom = {
            29: 'right_l6',
            30: 'right_l6',
            31: 'right_hand',
            32: 'right_hand',
            33: 'hand',
            34: 'rightclaw',
            35: 'rightpad',
            36: 'leftclaw',
            37: 'leftpad',
        }
        obj_geom = {
            8: 'mug',
            9: 'mug',
            10: 'mug',
            11: 'mug',
            40: 'cm_link',
            41: 'cm_link',
            42: 'cm_link',
            43: 'cm_link',
            44: 'cm_link',
            45: 'cm_link',
            46: 'cm_link',
            47: 'cm_link',
            48: 'cm_link',
            49: 'cm_link',
            50: 'cm_link',
            51: 'cm_link',
            52: 'cm_link',
            53: 'cm_link',
            54: 'cmbutton',
            55: 'cmbutton',
        }
    elif 'door' in dataset_name:
        obj_geom = {
            36: 'doorlockB',
            37: 'doorlockB',
            38: 'doorlockB',
            39: 'doorlockB',
            40: 'doorlockB',
            41: 'doorlockB',
            42: 'door_link',
            43: 'door_link',
            44: 'door_link',
            45: 'door_link',
            46: 'door_link',
            47: 'door_link',
            48: 'door_link',
            49: 'door_link',
            50: 'door_link',
            51: 'door_link',
        }
    elif 'faucet' in dataset_name:
        obj_geom = {
            36: 'faucet_link',
            37: 'faucet_link',
            38: 'faucet_link',
            39: 'faucet_link2',
            40: 'faucet_link2',
            41: 'faucet_link2',
            42: 'faucet_link2',
            43: 'faucet_link2',
            44: 'faucet_link2',
        }
    else:
        robot_geom = {}
        obj_geom = {}
        print("Could not find env name in src.data_gen.gen_fetch:get_mujoco_geom_keys_index, will not check collisions\n")

    return robot_geom, obj_geom

def is_robot_contact_geometry(data, robot_geom, object_geom):
    """
    Determine whether robot contacting task object(s)
    """
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2

        robot_object = (
            (g1 in robot_geom and g2 in object_geom) or
            (g2 in robot_geom and g1 in object_geom)
        )

        if robot_object:
            # print(f"Robot collided with {object_geom.get(g1, object_geom.get(g2, 'unknown object...'))}")
            return True

    return False

def main():
    start_time = time.perf_counter()
    # import os
    # os.environ['MUJOCO_GL'] = 'glx'
    # Create virtual display for running on server
    disp = Display(visible=0, size=(480, 480))
    disp.start()
    
    # Buffers
    frame_buffer = []
    act_buffer = []

    # Create env
    if not env_name == 'cartpole' and new_xml_filename is not None:
        env = gym.make(name_to_env[env_name], render_mode="rgb_array",
                    xml_file=new_xml_filename if new_xml_filename else None)
    elif env_name in meta_world_envs:
        # Camera mode: https://metaworld.farama.org/rendering/rendering/
        meta_ts = 4
        env = gym.make('Meta-World/MT1', env_name=name_to_env[env_name], render_mode='rgb_array', camera_name=metaworld_cam_name)
    else:
        env = gym.make(name_to_env[env_name], render_mode="rgb_array")
    obs, _ = env.reset()
    continuous = (env_to_aspace.get(env_name, 'continuous') == 'continuous')
    prev_img = torch.zeros((dataset_size, past_length, *image_shape))
    next_img = torch.zeros((dataset_size, pred_length, *image_shape))
    if continuous: control = torch.zeros((dataset_size, pred_length, env.action_space.shape[0]))
    else: control = torch.zeros((dataset_size, pred_length, 1))     # Discrete action space
    terminated = False; truncated = False
    
    # Collect n_samples trajectories
    idx = 0
    pbar = tqdm(total=dataset_size)
    while idx < dataset_size:
        if len(frame_buffer) == 0:
            frame_buffer.append(process_image(env.render(), env_name))

        # Sample and take action
        act = env.action_space.sample()
        act_buffer.append(act)
        if env_name in meta_world_envs:
            for _ in range(meta_ts):
                next_obs, rew, terminated, truncated, _ = env.step(act)
        else:
            next_obs, rew, terminated, truncated, _ = env.step(act)

        # If done reset env, otherwise add sample to dataset
        if terminated or truncated:
            obs, _ = env.reset()
            terminated = False; truncated = False
            frame_buffer = []
            act_buffer = []
            continue
        else:
            # Slide frame obs history frame buffer to next image
            if len(frame_buffer) == past_length + pred_length:
                frame_buffer.pop(0)
                act_buffer.pop(0)
            next_image = process_image(env.render(), env_name)
            # debug_render(next_image)
            frame_buffer.append(next_image)

            # Add obs history buffer to dataset
            if len(frame_buffer) == past_length + pred_length:
                prev_img[idx] = torch.stack(frame_buffer[0:past_length], dim=0)
                next_img[idx] = torch.stack(frame_buffer[past_length:(past_length+pred_length)], dim=0)

                # Get controls for entire pred_length
                if continuous:
                    control[idx] = torch.stack(
                        [torch.from_numpy(a) for a in act_buffer[past_length-1:past_length-1+pred_length]]
                    )
                else:
                    control[idx] = torch.tensor(
                        act_buffer[past_length-1:past_length-1+pred_length]
                    ).unsqueeze(-1)
                idx += 1
                pbar.update(1)

    pbar.close()

    # Saving dataset as dictionary
    dataset_dir = DATA_PATH / env_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prev_images": prev_img,
        "actions": control,
        "next_images": next_img,
    }, f"{dataset_dir / OUTPUT_NAME}.pt")
    print(f'\nSaved dataset to {dataset_dir / OUTPUT_NAME}.pt')

    # Update metadata file
    try: 
        dt = env.unwrapped.dt
    except:
        dt = 0.1
    update_dataset_metadata(
        dataset_dir=dataset_dir,
        dataset_name=OUTPUT_NAME,
        params={
            "env_name": env_name,
            "runtime": format_time(time.perf_counter() - start_time),
            "dataset_size": dataset_size,
            "image_shape": list(image_shape),
            "past_length": past_length,
            "pred_length": pred_length,
            "seed": seed,
            "dt": None if env_name =='cartpole' else dt
        },
    )
    print('\n*** DONE ***')
    return

if __name__ == '__main__':
    main()