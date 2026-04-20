"""
Generate tactile-gym datasets with configurable observation modalities.

Author: Jared Berry
"""
import numpy as np
import torch
import time
import gymnasium as gym
import argparse
from pathlib import Path
from tqdm import tqdm

from src.utils import set_seed, format_time
from src.data_gen.gen_fetch import update_dataset_metadata
import gymnasium_robotics
from tactile_gym.rl_envs.nonprehensile_manipulation.object_balance.object_balance_env import (
    ObjectBalanceEnv,
)
from tactile_gym.rl_envs.nonprehensile_manipulation.object_push.object_push_env import (
    ObjectPushEnv,
)
from tactile_gym.rl_envs.nonprehensile_manipulation.object_roll.object_roll_env import (
    ObjectRollEnv,
)

gym.register_envs(gymnasium_robotics)

# Parameters for dataset
ENV_NAME = "object_push"                                    # tactile-gym env alias
DATASET_SIZE = int(1e3)                                     # Number of samples: (obs, next_obs, control)
EXPERIMENT_PRESET = "egocentric_tactile_ee_pose_block_pose"  # see EXPERIMENT_PRESETS below
OUTPUT_NAME = ENV_NAME + f"_{DATASET_SIZE // 1000}k"       # Output dataset stem
IMAGE_SHAPE = [128, 128]                                    # Downsampled image shape
PAST_LENGTH = 3                                             # Number of previous observations to use for training
PRED_LENGTH = 3                                             # Number of timesteps to predict in the future
# ---------------------------------

# Get data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

EXPERIMENT_PRESETS = {
    # Exocentric RGB image only
    "exocentric_rgb_only": {
        "image_source": "visual",
        "camera_mode": "exocentric",
        "feature_components": [],
    },
    # Exocentric RGB + tactile image, no feature vector
    "tactile_rgb": {
        "image_source": "visuotactile",
        "camera_mode": "exocentric",
        "feature_components": [],
    },
    # RGB image + end-effector pose
    "egocentric_rgb_ee_pose": {
        "image_source": "visual",
        "camera_mode": "egocentric",
        "feature_components": ["ee_pose"],
    },
    # Tactile image + end-effector pose + object pose relative to end-effector
    "egocentric_tactile_ee_pose_block_pose": {
        "image_source": "tactile",
        "camera_mode": "exocentric",
        "feature_components": ["ee_pose", "block_pose"],
    },
    # Backward-compatible behavior from previous code
    "legacy_tactile_ee_pose": {
        "image_source": "tactile",
        "camera_mode": "exocentric",
        "feature_components": ["ee_pose"],
    },
}


def get_env_modes(config=None, observation_mode=None, camera_mode=None):
    env_modes = {
        # which dofs can have movement (environment dependent)
        # 'movement_mode':'y',          # y only, action dim = 1
        # 'movement_mode':'yRz',        # y + rotate z, action dim = 2
        # "movement_mode": "xyRz",      # x, y + rotate z, action dim = 3
        # 'movement_mode': 'TyRz',        # Move relative to TCP frame (perp + yaw), action dim = 2
        'movement_mode':'TxTyRz',     # Full TCP-parallel/perp motion + yaw, action dim = 3

        # specify arm
        "arm_type": "ur5",
        # "arm_type": "mg400",

        # specify tactile sensor
        # "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        "tactile_sensor_name": "digitac",

        # the type of control used
        'control_mode':'TCP_position_control',
        # "control_mode": "TCP_velocity_control",

        # randomisations
        "rand_init_pos": True,      # Object position
        "rand_init_orn": True,     # Object orientation
        "rand_obj_mass": False,     # Object mass

        # whether to render task goal
        "use_goal": False,

        # straight or random trajectory
        # "traj_type": "straight",
        'traj_type': 'simplex',

        # which observation type to return
        # feature includes EE position and goal position
        # 'observation_mode':'oracle',
        "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',

        # the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }
    if config is not None:
        overrides = config.get("tactile", {}).get("env_modes", {})
        env_modes.update(overrides)

    if observation_mode is not None:
        env_modes["observation_mode"] = observation_mode

    if camera_mode is not None:
        env_modes["camera_mode"] = camera_mode

    return env_modes


class ObservationAdapter:
    """
    Converts tactile-gym observations into model inputs for configurable experiments.
    """

    _FEATURE_DIMS = {
        "ee_pose": 6,
        "goal_pose": 6,
        "block_pose": 6,
    }

    def __init__(self, image_source, feature_components, camera_mode="exocentric"):
        self.image_source = image_source
        self.feature_components = list(feature_components)
        self.camera_mode = camera_mode

    @property
    def observation_mode(self):
        needs_feature_obs = any(c in ("ee_pose", "goal_pose") for c in self.feature_components)
        if self.image_source == "visuotactile":
            return "visuotactile_and_feature" if needs_feature_obs else "visuotactile"
        if self.image_source == "tactile":
            return "tactile_and_feature" if needs_feature_obs else "tactile"
        if self.image_source == "visual":
            if self.camera_mode == "egocentric":
                return "egocentric_visual_and_feature" if needs_feature_obs else "egocentric_visual"
            return "visual_and_feature" if needs_feature_obs else "visual"
        raise ValueError(f"Unsupported image_source={self.image_source}")

    @property
    def feature_dim(self):
        return int(sum(self._FEATURE_DIMS[c] for c in self.feature_components))

    def _process_single_image(self, image_array):
        arr = image_array
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        tensor = torch.from_numpy(arr).float() / 255.0
        return tensor.permute(2, 0, 1)

    def process_image(self, obs):
        if self.image_source == "visuotactile":
            visual = self._process_single_image(obs["visual"])
            tactile = self._process_single_image(obs["tactile"])
            return torch.cat([visual, tactile], dim=0)

        return self._process_single_image(obs[self.image_source])

    def _block_pose(self, obs, env):
        if env is not None and hasattr(env, "get_obj_pos_workframe") and hasattr(env, "_pb"):
            obj_pos, obj_orn = env.get_obj_pos_workframe()
            obj_rpy = env._pb.getEulerFromQuaternion(obj_orn)
            return np.asarray([*obj_pos, *obj_rpy], dtype=np.float32)

        oracle = obs.get("oracle", None)
        if oracle is not None and len(oracle) >= 18:
            # Object pose in object_* envs: [12:18] = pos(3) + rpy(3)
            return np.asarray(oracle[12:18], dtype=np.float32)

        raise KeyError("Could not infer block pose from env or observation.")

    def process_feature(self, obs, env=None):
        parts = []
        for component in self.feature_components:
            if component == "ee_pose":
                parts.append(np.asarray(obs["extended_feature"][0:6], dtype=np.float32))
            elif component == "goal_pose":
                parts.append(np.asarray(obs["extended_feature"][6:12], dtype=np.float32))
            elif component == "block_pose":
                parts.append(self._block_pose(obs, env))
            else:
                raise ValueError(f"Unsupported feature component: {component}")

        if not parts:
            return torch.empty(0, dtype=torch.float32)

        return torch.from_numpy(np.concatenate(parts, axis=0).astype(np.float32))


def build_observation_adapter(config=None, preset_name=None, obs_cfg=None):
    if config is None:
        preset_name = preset_name or EXPERIMENT_PRESET
        obs_cfg = obs_cfg or {}
    else:
        preset_name = preset_name or config.get("train", {}).get("experiment", "legacy_tactile_ee_pose")
        obs_cfg = obs_cfg or config.get("tactile", {}).get("observation", {})

    preset = EXPERIMENT_PRESETS.get(preset_name)
    if preset is None:
        valid = ", ".join(sorted(EXPERIMENT_PRESETS.keys()))
        raise ValueError(f"Unknown experiment preset '{preset_name}'. Valid presets: {valid}")

    image_source = obs_cfg.get("image_source", preset["image_source"])
    camera_mode = obs_cfg.get("camera_mode", preset.get("camera_mode", "exocentric"))
    feature_components = obs_cfg.get("feature_components", preset["feature_components"])

    return ObservationAdapter(
        image_source=image_source,
        feature_components=feature_components,
        camera_mode=camera_mode,
    )


def process_tactile(t):
    """Backward-compatible tactile preprocessing helper."""
    if t.ndim == 2:
        t = t[..., np.newaxis]
    normalized = torch.from_numpy(t).float() / 255.0
    return normalized.permute(2, 0, 1)


def process_feature(f):
    """Backward-compatible feature helper (EE pose only)."""
    return torch.from_numpy(np.asarray(f[0:6], dtype=np.float32))

seed = 42
set_seed(seed)
tactile_envs = {
    "object_push": ObjectPushEnv,
    "tactilepush": ObjectPushEnv,
    "object_roll": ObjectRollEnv,
    "object_balance": ObjectBalanceEnv,
}


def resolve_env_name_from_dataset(dataset_name):
    for env_name in sorted(tactile_envs.keys(), key=len, reverse=True):
        if dataset_name == env_name or dataset_name.startswith(env_name + "_"):
            return env_name
    return dataset_name.split("_")[0]

def main():
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Generate tactile-gym dataset for a given experiment preset")
    parser.add_argument("--env", default=ENV_NAME, help="Environment alias in tactile_envs mapping")
    parser.add_argument("--preset", default=EXPERIMENT_PRESET, help="Experiment preset name")
    parser.add_argument("--size", type=int, default=DATASET_SIZE, help="Number of dataset samples")
    parser.add_argument("--past-length", type=int, default=PAST_LENGTH, help="History window length")
    parser.add_argument("--pred-length", type=int, default=PRED_LENGTH, help="Prediction horizon")
    parser.add_argument("--image-size", type=int, nargs=2, default=IMAGE_SHAPE, help="Image size as H W")
    parser.add_argument("--output-name", default=None, help="Dataset file stem (without .pt)")
    args = parser.parse_args()

    env_name = args.env
    dataset_size = args.size
    past_length = args.past_length
    pred_length = args.pred_length
    image_shape = list(args.image_size)
    output_name = args.output_name or (env_name + f"_{dataset_size // 1000}k")

    # os.environ["PYOPENGL_PLATFORM"] = "egl"
    # Create virtual display for running on server
    # disp = Display(visible=0, size=(480, 480))
    # disp.start()

    # Create env
    obs_adapter = build_observation_adapter(preset_name=args.preset)
    max_steps = 10000            # Max number of env steps before forced termination

    # These will slow down runtime, but are necessary for eval
    show_gui = False             # Enable PyBullet GUI
    show_tactile = False         # Display tactile sensor images
    
    env = tactile_envs[env_name](
        max_steps=max_steps,
        env_modes=get_env_modes(
            observation_mode=obs_adapter.observation_mode,
            camera_mode=obs_adapter.camera_mode,
        ),
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=image_shape,
    )
    
    obs, _ = env.reset()

    # Buffers
    obs_buffer = []
    feature_buffer = []
    act_buffer = []

    image_0 = obs_adapter.process_image(obs)
    feature_0 = obs_adapter.process_feature(obs, env)
    img_channels = image_0.shape[0]
    feature_dim = feature_0.shape[0]

    prev_obs = torch.zeros((dataset_size, past_length, img_channels, *image_shape))
    next_obs_tensor = torch.zeros((dataset_size, pred_length, img_channels, *image_shape))
    prev_feature = torch.zeros((dataset_size, past_length, feature_dim))
    next_feature = torch.zeros((dataset_size, pred_length, feature_dim))
    control = torch.zeros((dataset_size, pred_length, env.action_space.shape[0]))
    terminated = False; truncated = False

    # Seed initial observation into buffer
    obs_buffer.append(image_0)
    feature_buffer.append(feature_0)
    
    # Collect n_samples trajectories
    idx = 0
    pbar = tqdm(total=dataset_size)
    while idx < dataset_size:
        # Sample and take action
        act = env.action_space.sample()
        act_buffer.append(act)
        next_obs_dict, rew, terminated, truncated, _ = env.step(act)

        image = obs_adapter.process_image(next_obs_dict)
        feature = obs_adapter.process_feature(next_obs_dict, env)

        obs_buffer.append(image)
        feature_buffer.append(feature)
        act_buffer.append(act)

        # If done reset env, otherwise add sample to dataset
        if terminated or truncated:
            obs, _ = env.reset()
            obs_buffer = []
            feature_buffer = []
            act_buffer = []
            image_reset = obs_adapter.process_image(obs)
            feature_reset = obs_adapter.process_feature(obs, env)
            obs_buffer.append(image_reset)
            feature_buffer.append(feature_reset)
            continue

        # Maintain sliding window size
        total_len = past_length + pred_length
        if len(obs_buffer) > total_len:
            obs_buffer.pop(0)
            feature_buffer.pop(0)
            act_buffer.pop(0)

        # Save sample when buffer full
        if len(obs_buffer) == total_len:

            prev_obs[idx] = torch.stack(
                obs_buffer[:past_length], dim=0
            )
            next_obs_tensor[idx] = torch.stack(
                obs_buffer[past_length:], dim=0
            )

            prev_feature[idx] = torch.stack(
                feature_buffer[:past_length], dim=0
            )
            next_feature[idx] = torch.stack(
                feature_buffer[past_length:], dim=0
            )

            control[idx] = torch.stack(
                [
                    torch.from_numpy(a).float()
                    for a in act_buffer[past_length - 1 : past_length - 1 + pred_length]
                ]
            )

            idx += 1
            pbar.update(1)

    pbar.close()

    # Saving dataset as dictionary
    dataset_dir = DATA_PATH / env_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        # New generic keys
        "prev_obs": prev_obs,
        "next_obs": next_obs_tensor,
        # Backward-compatible tactile key aliases
        "prev_tactile": prev_obs,
        "next_tactile": next_obs_tensor,
        "prev_feature": prev_feature,
        "next_feature": next_feature,
        "actions": control,
        "experiment_preset": args.preset,
        "image_source": obs_adapter.image_source,
        "camera_mode": obs_adapter.camera_mode,
        "feature_components": obs_adapter.feature_components,
    }, f"{dataset_dir / output_name}.pt")
    print(f'\nSaved dataset to {dataset_dir / output_name}.pt')

    # Update metadata file
    try: 
        dt = env.unwrapped.dt
    except:
        dt = 0.1
    update_dataset_metadata(
        dataset_dir=dataset_dir,
        dataset_name=output_name,
        params={
            "ENV_NAME": env_name,
            "runtime": format_time(time.perf_counter() - start_time),
            "dataset_size": dataset_size,
            "IMAGE_SHAPE": list(image_shape),
            "PAST_LENGTH": past_length,
            "PRED_LENGTH": pred_length,
            "experiment_preset": args.preset,
            "image_source": obs_adapter.image_source,
            "camera_mode": obs_adapter.camera_mode,
            "feature_components": obs_adapter.feature_components,
            "seed": seed,
            "dt": None if env_name == 'cartpole' else dt
        },
    )
    print('\n*** DONE ***')
    return

if __name__ == '__main__':
    main()