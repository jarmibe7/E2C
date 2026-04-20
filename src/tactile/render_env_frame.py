"""Render and inspect a single tactile-gym frame.

Simple script to render environment frames and save a combined image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os

# Set OpenGL to use EGL for headless rendering (SSH-compatible)
os.environ["PYOPENGL_PLATFORM"] = "egl"

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Create virtual display for SSH/headless environments - must be before importing tactile_gym
try:
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(480, 480))
    display.start()
except Exception as e:
    print(f"Warning: Could not start virtual display: {e}")
    display = None

from src.tactile.gen_tactile import build_observation_adapter, get_env_modes, tactile_envs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "env_inspect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and save a tactile-gym environment frame")
    parser.add_argument("--env", default="object_push", help="Environment name (default: object_push)")
    parser.add_argument(
        "--experiment",
        default="exocentric_rgb_only",
        choices=[
            "exocentric_rgb_only",
            "tactile_rgb",
            "egocentric_rgb_ee_pose",
            "egocentric_tactile_ee_pose_block_pose",
            "legacy_tactile_ee_pose",
        ],
        help="Experiment preset (default: exocentric_rgb_only)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output image path (default: figures/env_inspect/{env}.png)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[64, 64], help="Image size (default: 64 64)")
    parser.add_argument(
        "--hide-tactile-geometry",
        action="store_true",
        help="Hide tactile sensor mesh geometry (useful for egocentric camera setup)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Set up output path
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"{args.env}_{args.experiment}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build observation adapter from experiment preset
    adapter = build_observation_adapter(preset_name=args.experiment)
    env_modes = get_env_modes(
        observation_mode=adapter.observation_mode,
        camera_mode=adapter.camera_mode,
    )
    if args.hide_tactile_geometry or adapter.camera_mode == "egocentric":
        env_modes["hide_tactile_geometry"] = True

    # Create environment
    env = tactile_envs[args.env](
        max_steps=100,
        env_modes=env_modes,
        show_gui=False,
        show_tactile=False,
        image_size=args.image_size,
    )

    try:
        obs, _ = env.reset()
        image = np.asarray(obs.get("visual", next(iter(obs.values()))))
        
        # Get render image
        render_image = env.render(mode="rgb_array")
        if render_image is None or render_image.size == 0:
            render_image = image
        render_image = np.asarray(render_image)

        # Create combined side-by-side visualization
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Left: observation
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            axes[0].imshow(np.squeeze(image), cmap="gray")
        else:
            axes[0].imshow(image)
        axes[0].set_title("Observation")
        axes[0].axis("off")
        
        # Right: render
        axes[1].imshow(render_image)
        axes[1].set_title("Render")
        axes[1].axis("off")
        
        plt.tight_layout()
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)

        print(f"✓ Saved: {output_path}")
    finally:
        env.close()
        # Clean up virtual display
        if display is not None:
            try:
                display.stop()
            except Exception as e:
                print(f"Warning: Could not stop virtual display: {e}")


if __name__ == "__main__":
    main()
