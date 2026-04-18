# E2C: Embed to Control

PyTorch codebase for latent world models and control (E2C + RSSM variants), with tactile-gym dataset generation and rendering utilities.

## Quick Setup

```bash
cd /home/jarmibe7/E2C
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_linux.txt
pip install -e .
```

## Main Training

Run with a config in `config/final`:

```bash
source .venv/bin/activate
python -m src.main --config rssm_push_active_v0
```

## Tactile Dataset Generation (New)

Generate tactile-gym datasets with experiment presets:

```bash
source .venv/bin/activate
python -m src.tactile.gen_tactile \
	--env object_push \
	--preset egocentric_rgb_ee_pose \
	--size 1000 \
	--past-length 3 \
	--pred-length 3 \
	--image-size 128 128 \
	--output-name object_push_1k
```

Available presets:
- `exocentric_rgb_only`
- `egocentric_rgb_ee_pose`
- `egocentric_tactile_ee_pose_block_pose`
- `legacy_tactile_ee_pose`

## Render/Inspect One Frame

Render a single environment frame and save a combined observation+render image:

```bash
source .venv/bin/activate
python -m src.tactile.render_env_frame --experiment egocentric_rgb_ee_pose
```

Useful options:

```bash
# exocentric baseline
python -m src.tactile.render_env_frame --experiment exocentric_rgb_only

# custom output
python -m src.tactile.render_env_frame \
	--experiment egocentric_rgb_ee_pose \
	--output figures/env_inspect/custom.png

# force-hide tactile sensor geometry (egocentric already defaults to hidden)
python -m src.tactile.render_env_frame \
	--experiment egocentric_rgb_ee_pose \
	--hide-tactile-geometry
```

## Tactile Training
```
python -m src.tactile.main_subprocess_tactile --objectives random --trials 1

python -m src.tactile.main_subprocess_tactile  --objectives pixel --trials 1
```



## Citation

M. Watter, J. T. Springenberg, J. Boedecker, and M. Riedmiller, “Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images,” NIPS 2015. https://arxiv.org/abs/1506.07365


