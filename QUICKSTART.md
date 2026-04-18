# Quickstart

This repository implements the RSSM-based world model used in our paper on
unsupervised robot play. This quickstart takes a new reader from a fresh
checkout to a rendered evaluation video in five numbered steps.

All commands below assume you are at the repo root.

## 1. Install

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements_linux.txt    # requirements_win.txt on Windows
pip install -e .
```

You'll additionally need:

- **MuJoCo** with headless GL (we set `MUJOCO_GL=egl` automatically).
- **Meta-World**: before running data generation, bump
  ``max_path_length`` in ``metaworld/sawyer_xyz_env.py`` (line ~153) to
  ``5000`` so the long rollouts we collect fit inside a single episode.
- **ffmpeg** on your `$PATH` for video writing (``matplotlib`` uses
  ``FFMpegWriter``).

## 2. Generate a dataset

Pick a Meta-World environment (``button`` / ``door`` / ``drawer`` /
``coffee`` / ``faucet`` / etc. -- see ``meta_world_envs`` in
`src/data_gen/gen_fetch.py`) and edit the ``env_name`` global near the top of
that file. Then run:

```bash
python3.10 -m src.data_gen.gen_fetch
```

This collects 2k (history, future, actions) tuples of length
``past_length + pred_length = 6`` and saves them to
``data/<env>/<env>_gripper_2k.pt``. A summary is appended to
``data/<env>/metadata.yaml``. See the module docstring in
``src/data_gen/gen_fetch.py`` for the exact tensor shapes.

## 3. Understand the model

The world model is an RSSM (Recurrent State-Space Model); the reference
diagram lives at `.cursor/prompts/network_architecture.png` and the core
math is Eq. (1) of the paper:

| Part of Eq. (1)                    | Attribute in `src/model/rssm.py::RSSME2C` |
|------------------------------------|-------------------------------------------|
| image encoder `enc(o_t)`           | `self.encoder` (see `src/model/encoder.py`) |
| deterministic recurrence `h_t = f(h_{t-1}, z_{t-1}, u_{t-1})` | `self.rnn` (2-layer GRU) |
| prior / transition `p(z_t | h_t)`  | `self.prior` (MLP → mu, log_var), wrapped by `rssm_step` |
| posterior / representation `q(z_t | enc(o_t))` | `self.post` (MLP → mu, log_var), wrapped by `encode_posterior` |
| observation decoder `p(o_t | z_t)` | `self.decoder` (see `src/model/decoder.py`) |

Training loss (posterior-prior KL with free nats + image NLL) lives in
`src/model/loss.py::RSSMLoss`. The online action-selection objective (pixel
or dynamics information gain) lives in
`src/trainer.py::ClosedLoopInformativeTrainer.rollout_info_gain_decoded_batch`.

## 4. Train

Pick a config from ``config/configs_final/`` (offline pretraining on a fixed
dataset) or ``config/configs_gripper/`` (closed-loop MPC training from the
gripper-POV camera). Filenames follow the
``<env>_<policy_tag>_<seed>.yaml`` convention, where `policy_tag` is one of:

- ``eig`` -- **pixel information gain** (decode priors, KL against new posterior).
- ``maxdyn`` -- **dynamics information gain** (KL between successive priors only).
- ``random`` -- random-action baseline.

Example:

```bash
python3.10 -m src.main --config configs_final/button_eig_0
```

Checkpoints, loss curves, and an end-of-training rollout video are written to
``runs/<env>/<env>_<objective>_<seed>/`` (see
`src/utils.py::build_run_path` for the exact rule).

Note: be very vareful with your seed, data folder, etc. You do not want to accidentally overwrite data

## 5. Evaluate + render videos

After training, render evaluation rollouts and count robot-object contacts:

```bash
python3.10 -m src.eval_render_videos \
    --envs button --policies eig --seeds 0 1 --i 0 --steps 250
```

Videos are written under
``src/data_gen/contact/videos/<env>_<objective>_<i>_<steps>/`` and a
``contacts_seed_<i>_<steps>.json`` summary is written to
``src/data_gen/contact/``. Run ``--help`` for the full CLI.
