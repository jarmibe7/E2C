"""Render and inspect a single tactile-gym frame.

This is a lightweight script version of the notebook-based inspector. It creates
one environment, captures the current observation and a rendered RGB frame, then
saves a side-by-side preview for quick inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.tactile.gen_tactile import build_observation_adapter, get_env_modes, tactile_envs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "env_inspect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a single tactile-gym frame")
    parser.add_argument("--env", default="object_push", help="Environment alias")
    parser.add_argument(
        "--experiment",
        default="egocentric_rgb_ee_pose",
        choices=[
            "exocentric_rgb_only",
            "egocentric_rgb_ee_pose",
            "egocentric_tactile_ee_pose_block_pose",
            "legacy_tactile_ee_pose",
        ],
        help="Experiment preset to inspect",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))
    parser.add_argument("--show-gui", action="store_true", help="Enable the environment GUI")
    parser.add_argument("--show-tactile", action="store_true", help="Show tactile sensor overlays in GUI mode")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where preview images will be saved",
    )
    parser.add_argument(
        "--save-prefix",
        default=None,
        help="Optional custom file prefix for saved images",
    )
    parser.add_argument(
        "--observation-mode",
        default=None,
        help="Override the preset observation mode",
    )
    parser.add_argument(
        "--camera-mode",
        default=None,
        choices=["exocentric", "egocentric"],
        help="Override the preset camera mode",
    )
    parser.add_argument(
        "--image-source",
        default=None,
        choices=["visual", "tactile"],
        help="Override the preset image source",
    )
    return parser.parse_args()


def select_image_key(obs: dict, adapter_image_source: str, override_image_key: str | None = None) -> str:
    if override_image_key is not None:
        return override_image_key
    if adapter_image_source in obs:
        return adapter_image_source
    if "visual" in obs:
        return "visual"
    if "tactile" in obs:
        return "tactile"
    return next(iter(obs.keys()))


def main() -> None:
    args = parse_args()

    adapter = build_observation_adapter(preset_name=args.experiment)
    image_source = args.image_source or adapter.image_source
    camera_mode = args.camera_mode or adapter.camera_mode
    observation_mode = args.observation_mode or adapter.observation_mode
    env_modes = get_env_modes(observation_mode=observation_mode, camera_mode=camera_mode)

    env_cls = tactile_envs[args.env]
    env = env_cls(
        max_steps=100,
        env_modes=env_modes,
        show_gui=args.show_gui,
        show_tactile=args.show_tactile,
        image_size=args.image_size,
    )

    try:
        obs, _ = env.reset()
        image_key = select_image_key(obs, image_source)
        image = np.asarray(obs[image_key])
        render_image = np.asarray(env.render(mode="rgb_array"))

        print(f"observation_mode={observation_mode}, camera_mode={camera_mode}")
        print(f"observation keys: {list(obs.keys())}")
        print(f"displaying key: {image_key}, shape={image.shape}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.save_prefix or f"{args.env}_{args.experiment}_{camera_mode}_{image_key}"

        observation_path = args.output_dir / f"{prefix}_obs.png"
        render_path = args.output_dir / f"{prefix}_render.png"
        combined_path = args.output_dir / f"{prefix}_combined.png"

        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            plt.imsave(observation_path, np.squeeze(image), cmap="gray")
        else:
            plt.imsave(observation_path, image)
        plt.imsave(render_path, render_image)

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            axes[0].imshow(np.squeeze(image), cmap="gray")
        else:
            axes[0].imshow(image)
        axes[0].set_title(f"Observation: {image_key}")
        axes[0].axis("off")
        axes[1].imshow(render_image)
        axes[1].set_title("env.render(rgb_array)")
        axes[1].axis("off")
        plt.tight_layout()
        figure.savefig(combined_path, dpi=150)
        plt.close(figure)

        print(f"Saved: {observation_path}")
        print(f"Saved: {render_path}")
        print(f"Saved: {combined_path}")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
