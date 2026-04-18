# Cleanup Report

This report accompanies the cleanup pass on the repository. It is organised
in three parts:

1. **File-level deletions** (Phase 1) -- which legacy files were removed and why.
2. **In-file cleanup** (Phase 2) -- per-file summary of changes, with diff-style
   snippets for the non-trivial ones and a one-line safety justification.
3. **Flagged but unfixed** -- suspicious code intentionally left untouched,
   with the reasoning, for the authors to review.

All changes preserve the core RSSM algorithm, the two KL formulations (pixel
vs. dynamics information gain), the CEM action selection, and the training
loss. No files under ``data/``, ``runs/``, or ``src/data_gen/contact/`` were
touched.

The smoke-check transcript at the end confirms the three sanity checks pass.

---

## 1. File-level deletions (Phase 1)

Entire files / directories removed. Each item was verified to be
unreferenced (or referenced only by other deleted files) before deletion.

**Legacy E2C model path (fully superseded by RSSM)**
- `src/e2c.py` -- an old duplicate of dataset + loss + model at the `src/`
  root, superseded by `src/model/rssm.py` + `src/dataset.py` + `src/model/loss.py`.
- `src/train.py` -- old `train()` entry point that imported `src.e2c.E2C`.
- `src/model/e2c.py` -- `ConvE2C` architecture. Only `RSSME2C` is instantiated
  anywhere in the active code paths.
- `src/model/policy.py` -- `ConvPolicy` / `BasePolicy`; no longer instantiated
  anywhere once `E2CPretrainer` was removed.

**State-rep pretraining track (orthogonal to the paper)**
- `src/state_rep.py`
- `src/data_gen/gen_state_rep.py`
- `train.sh` (only ever launched the state-rep pipeline)

**Legacy env data generators**
- `src/data_gen/gen_particle_grav.py`
- `src/data_gen/gen_dummy.py`

**Notebooks / figures / configs that no longer track active code**
- `src/compare_traj_rssm.ipynb` -- imports removed `ConvE2C`.
- `notebooks/`, `tests/` -- user-confirmed obsolete.
- `figures/pres_cartpole/`, `figures/pres_particle_grav/`, `figures/pres_reacher/`
- `config/old/`, `config/configs_change_cam/` -- legacy yamls; current configs
  live in `config/configs_final/` + `config/configs_gripper/`.

**Root-level cruft**
- `models.py` (stale duplicate of model code), `__init__.py` (empty),
  `debug.png`, `__pycache__/`, all `src/**/__pycache__/`.

**Explicitly preserved** (per the updated plan):
- All top-level `*.mp4` artefacts (paper outputs).
- All `src/data_gen/*.ipynb` that produce figures.
- `src/data_gen/gen_gym.py`, `gen_gym_dt.py`, `gen_box_vplot.py` -- non
  particle-gravity generators.
- `src/data_gen/contact/`, `src/data_gen/obj_graphs/`,
  `src/data_gen/shortcomings/`, `src/data_gen/temp_figs/`.
- `src/test.ipynb`, `src/compare_traj.ipynb`, `collect_data.ipynb` -- debugging / figure notebooks.

---

## 2. In-file cleanup (Phase 2)

### `src/main.py` (full rewrite)

**Changes**
- Dropped unused imports (`matplotlib.pyplot`, `numpy`, `ConvE2C`, `ConvPolicy`, `E2CPretrainer`).
- Deleted the `if 'e2c' in config_name:` branch and the `direct_reward` dead branch.
- Added argparse, a module docstring describing the naming convention, and factored the run path + save logic into helpers.
- Routed `policy: hardware` to a new `ClosedLoopHardwareTrainer` stub in `trainer.py`.

```diff
- from src.model.e2c import ConvE2C
- from src.model.rssm import RSSME2C
- from src.model.policy import ConvPolicy
- from src.trainer import E2CPretrainer, RSSMPretrainer, ClosedLoopRandomTrainer, ClosedLoopInformativeTrainer
+ from src.model.rssm import RSSME2C
+ from src.trainer import (
+     ClosedLoopHardwareTrainer,
+     ClosedLoopInformativeTrainer,
+     ClosedLoopRandomTrainer,
+     RSSMPretrainer,
+ )
```

```diff
- policy = config_name.split('/')[-1].split('_')[1]
- if policy in ['eig', 'maxdyn', 'random']:
-     if policy == 'eig':
-         objective = 'pixel'
-     elif policy == 'maxdyn':
-         objective = 'dynamics'
-     else:
-         objective = 'random'
-     save_name = config['train']['dataset'].split('_')[0] + '_' + objective + '_' + str(config.get('seed', 0))
- else:
-     save_name = config['train']['dataset'].split('_')[0] + '_' + policy + '_' + str(config.get('seed', 0))
- run_path = RUNS_PATH / Path(config['train']['dataset'].split('_')[0]) / save_name
+ run_path = build_run_path(config, config_name, RUNS_PATH)
```

**Why it's safe**: `build_run_path` in `src/utils.py` produces the same
`<env>/<env>_<objective>_<seed>` string for all three canonical policies and
falls back to the raw policy tag (matching the old `else` branch) otherwise.

### `src/utils.py`

**Changes**
- Added `build_run_path(config, config_name, runs_root)` used by both `main.py` and (indirectly) downstream analysis.
- Deleted `rk4_sim` (only used by the removed `gen_particle_grav.py`).
- Added type hints + docstrings for `format_time`, `wrapped_angle_error`, and the three latent-space metric helpers.

**Why it's safe**: `rk4_sim` has no remaining call-sites; the added helper
matches the old inline logic verbatim.

### `src/trainer.py`

**Changes**
- Deleted dead helpers: `_frames_to_tensor`, `_update_window`, `_decode_latent`, `_encode_posterior_e2c`, `_encode_posterior_rssm`.
- Deleted `E2CPretrainer` (ConvE2C dependency).
- Preserved all commented-out alternative KL formulations inside `rollout_info_gain_decoded_batch` (per user's explicit request).
- Added module and class docstrings describing the two objectives and their paper references.
- Introduced `ClosedLoopHardwareTrainer(ClosedLoopInformativeTrainer)` stub whose `collect_rollouts` raises `NotImplementedError`.
- Fixed the inverted rollout-steps warning wording.
- Added a `camera_name = 'corner3'` fallback so Meta-World envs without `isom`/`gripper` in the dataset name no longer hit `UnboundLocalError` (caught by the smoke check below).

```diff
- assert self.num_rollout_steps > self.batch_size, 'Steps per rollout must be > batch_size!'
  if self.num_rollout_steps <= self.num_batches * self.batch_size:
      print(
-         f"Warning: num_rollout_steps ({self.num_rollout_steps}) > "
+         f"Warning: num_rollout_steps ({self.num_rollout_steps}) <= "
          f"num_batches * batch_size ({self.num_batches * self.batch_size}).\n"
          f"This may lead to inefficient training, as the probability of "
          f"'double counting' samples in a single epoch is high."
      )
```

```diff
  if env_name in meta_world_envs:
+     # Camera name is baked into the dataset filename: "<env>_gripper_*"
+     # uses gripperPOV, "<env>_isom_*" uses corner3. Default to corner3
+     # so the original legacy ``<env>_<N>k`` datasets remain loadable.
+     camera_name = 'corner3'
      if 'isom' in config['train']['dataset']: camera_name = 'corner3'
      if 'gripper' in config['train']['dataset']: camera_name = 'gripperPOV'
```

**Why it's safe**: deletions target only functions/classes with zero call
sites after `ConvE2C` removal; the warning-wording flip matches what the
`if`/`assert` actually guard against; the `corner3` fallback matches what
the original ``_isom`` branch would have set.

### `src/eval.py` (substantial rewrite)

**Changes**
- Deleted `Evaluator.eval`, `eval_traj`, `eval_metrics`, `eval_four_var_latent`, `eval_latent` and the `if __name__ == "__main__":` block. These referred to `self.model.mu` / `self.model.log_var` which no longer exist on `RSSME2C`, and/or to deleted `ConvE2C`.
- Factored the ~95%-duplicated body of `visualize_planner` and `render_video` into one private `_collect_planner_rollout` helper that returns a dict of per-step buffers; the two callers differ only in figure layout and output filename.
- Removed the hard-coded debug dump `/home/ayush/.../temp_figs/step_costs.txt`.
- Dropped the unused `first_img` local and the commented-out contact-env-reset block.
- Contact checking uses `get_mujoco_geom_keys_index` + `is_robot_contact_geometry` imported from `src.data_gen.gen_fetch` (single source of truth).

**Why it's safe**: the shared helper is a straight extract of the common
loop; the two public methods call it with the same arguments they already
passed through.

### `src/model/rssm.py`

**Changes**
- Deleted the broken `reconstruct` method (referenced `self.mu` / `self.log_var` attributes that were never assigned on `RSSME2C`).
- Deleted the commented-out alternative `self.post` definition.
- Fixed `sample_traj`: the old code passed `z` (no time dim) into `rssm_step` which expects `[B, T, D]`; now wraps in `z.unsqueeze(1)`.
- Dropped the unused `return_all` parameter from `sample` -- it was never set to `True` anywhere.
- Expanded the module docstring with the 4-part model description tied to Eq. (1).
- Dropped import of removed `ChannelUncertaintyConvDecoder`.

```diff
- def reconstruct(self, x_traj):
-     ...
-     mu = self.mu(flattened)           # self.mu doesn't exist on RSSME2C
-     log_var = self.log_var(flattened) # self.log_var doesn't exist on RSSME2C
-     ...
```

```diff
  for t in range(seq_len):
      u_t = u_seq[t].unsqueeze(0).to(self.device)
-     h, z, mu_p, log_var_p = self.rssm_step(h, z, u_t)
+     h, z, _, _ = self.rssm_step(h, z.unsqueeze(1), u_t)
```

**Why it's safe**: `reconstruct` has zero call-sites (grep the repo); the
`unsqueeze(1)` fix matches what `forward` and the trainer already do.

### `src/model/loss.py`

**Changes**
- Deleted `E2CLoss` and `UncertaintyE2CLoss` (legacy `ConvE2C` path is gone; only `RSSMLoss` is selected by any live config).
- Preserved `RSSMLoss` verbatim (free-nats clamp + KL annealing + image reconstruction) with expanded docstrings and a note that its `kl_divergence` is intentionally duplicated with the online `_kl_diag_gaussian` in `trainer.py`.

**Why it's safe**: the only live `loss_type` values in configs are `rssm`
and `uncertainty`, both of which are already routed to `RSSMLoss`.

### `src/model/encoder.py` / `src/model/decoder.py`

**Changes**
- Added docstrings.
- Removed `ChannelUncertaintyConvDecoder` (no configs or code reference it; only `ConvDecoder` and `ScalarUncertaintyConvDecoder` are instantiated).
- Deleted the redundant first `self.fc_decode = nn.Linear(...)` assignment in `ConvDecoder.__init__` that was immediately overwritten by the following `Sequential`.

```diff
      self.enc_out_shape = enc_out_shape
-
-     self.fc_decode = nn.Linear(self.latent_size, enc_out_dim)
-
      self.fc_decode = nn.Sequential(
```

**Why it's safe**: both changes touch dead code only.

### `src/dataset.py`

**Changes**
- Removed the ~20-line triple-quoted "Create windowed samples" commented block.
- Added a module-level docstring documenting expected `.pt` keys + tensor shapes.

**Why it's safe**: pure docs/dead-comment trim; runtime behaviour unchanged.

### `src/replay_buffer.py`

**Changes**
- Added type hints and docstrings for `__init__`, `add`, `sample`.
- Noted in the `sample` docstring that sampling is without replacement.
- Reorganised imports.

**Why it's safe**: no behavioural change.

### `src/data_gen/gen_fetch.py`

**Changes**
- Moved the module-level MJCF-patching block into a new
  `maybe_patch_mjcf_timestep(env_name, new_dt)` function that is called
  only from `main()`. Previously, *importing* this module (as
  `src/trainer.py` does) would run the XML rewrite as a side effect.
- Added a module docstring with the exact shape/dtype of the saved `.pt`.
- Dropped the top-of-file `scp` one-liners (see Appendix below).
- Reorganised imports; removed `from pyvirtualdisplay import Display` + `from PIL import Image` that were unused.

```diff
- # Only modify XML if new_dt is set
- if new_dt is not None:
-     mj_path = ...
-     ...
-     new_xml_path.write_text(xml_text)
- else:
-     new_xml_filename = None
+ def maybe_patch_mjcf_timestep(env_name, new_dt):
+     """If ``new_dt`` is provided, clone the env's MJCF with that timestep and
+     return the patched filename; otherwise return ``None``.
+     """
+     ...

- def main():
-     ...
+ def main():
+     ...
+     new_xml_filename = maybe_patch_mjcf_timestep(env_name, new_dt)
```

**Why it's safe**: the old code only produced a non-`None`
`new_xml_filename` when `new_dt is not None`, which was always `None`
in-repo. The hoisted function preserves the exact same logic but only
runs when `main()` is invoked.

### `src/eval_render_videos.py` (rewrite)

**Changes**
- Replaced the triple-nested `for env / for policy / for i / for render_seed`
  block with an argparse CLI (`--envs`, `--policies`, `--seeds`, `--i`,
  `--steps`, `--runs-root`, `--contact-root`).
- Removed the hard-coded in-script CEM overrides (`samples=300`,
  `plan_horizon=8`, `sigma_init=0.15`, etc.); the loaded config's values are
  used as-is.
- Dropped the commented-out try/except blocks.

**Why it's safe**: all existing invocation patterns (single env/policy/seed
sweep) can be expressed via the new CLI; the removed overrides were never
applied to any saved run.

### `src/main_subprocess.py`

**Changes**
- Kept as a minimal sweep harness with dead-comment loops left in place for
  the user to uncomment as needed.
- Removed the inline per-run CEM / loss overrides (`recon_mult=300`,
  `sigma_init=0.5`, etc.) and the duplicate `save_name` / `run_path` block --
  `src.main` now computes both itself via `build_run_path`.

**Why it's safe**: the overrides were a one-off debug override and silently
drifted from the checked-in configs; removing them makes the sweep script
respect what the yaml says.

### `setup.py`

**Changes**
- Fixed `read_requirements()` to pick between `requirements_linux.txt` and
  `requirements_win.txt` based on `sys.platform` (the old code opened
  `requirements.txt`, which does not exist).

**Why it's safe**: `pip install -e .` now actually succeeds on Linux +
Windows without manual intervention.

### `.gitignore`

**Changes**
- Removed the self-ignoring `.gitignore` line.
- Removed `config/*` blanket (we want the paper configs tracked).
- Removed `*__init__.py` blanket (was preventing Python package detection).
- Kept `*.mp4`, `*.png`, `**/__pycache__/`, `runs/`, `data/`, `.venv/`, etc.

**Why it's safe**: the blanket `config/*` and `*__init__.py` rules were
over-broad; tracked files are unaffected (they're already committed).

---

## 3. Flagged but unfixed

These look suspicious but were explicitly left alone to preserve
experimental behaviour. Please review when convenient.

1. **`ClosedLoopInformativeTrainer._sample_cem` new-sigma formula**
   (`src/trainer.py`).

   ```python
   new_sigma = 1 / len(new_mu - 1) * torch.stack([
       torch.sum(torch.stack([
           torch.sqrt((seq[t] - new_mu[t])**2) for seq in elite_seqs
       ], dim=0), dim=0)
       for t in range(self.plan_horizon)
   ], dim=0)
   ```

   Two concerns:

   - `torch.sqrt((x - mean)**2)` equals `abs(x - mean)`; this computes a
     mean-absolute-deviation, not a standard deviation.
   - `len(new_mu - 1)` evaluates to `len(new_mu)` in Python; the ``- 1``
     lives inside the argument to `len()` and is a no-op on a tensor. It
     looks like a typo for either `len(new_mu) - 1` or `len(elite_seqs)`.

   This may be deliberate (MAD tends to give wider exploration than std for
   Gaussian clipping), so it has been left untouched.

2. **Inverted `num_rollout_steps` assertion message (adjacent to the fix
   above)**. The enclosing `assert self.num_rollout_steps <= self.replay_buffer.capacity`
   is fine, but the accompanying user-facing warning message previously
   said ``>`` where the condition is ``<=``. Wording is now corrected; the
   intended inequality was not.

3. **`obs` from `env.reset()` is never consumed in either `collect_rollouts`
   implementation** (`src/trainer.py`). The image returned by `env.render()`
   is used exclusively; Meta-World's `obs` is unused because we treat the
   robot as "blind" (pixels only). Intentional but worth documenting in a
   follow-up if you plan to add a state-conditioned baseline.

4. **Duplicated KL helpers.** `RSSMLoss.kl_divergence` and
   `ClosedLoopInformativeTrainer._kl_diag_gaussian` are analytically
   identical. Per your explicit instruction, both are preserved so that the
   training loss and the online planning objective can evolve independently.

---

## Appendix -- smoke-check transcript

Run after the cleanup against the unmodified ``config/configs_final/button_eig_0.yaml``.

1. `python3.10 -c "from src.main import main"`

   ```
   import main OK
   ```

2. `python3.10 -m src.main --help`

   ```
   *** STARTING ***

   usage: main.py [-h] [--config CONFIG]

   data4wm training entry point

   options:
     -h, --help       show this help message and exit
     --config CONFIG  Name of the config file under config/ (with or without .yaml).
   ```

3. Construct a trainer from `configs_final/button_eig_0.yaml` on CPU (skip `learn()`):

   ```
   Warning: num_rollout_steps (100) <= num_batches * batch_size (12800).
   This may lead to inefficient training, as the probability of
   'double counting' samples in a single epoch is high.
   constructed trainer OK: ClosedLoopInformativeTrainer
   ```

4. `python3.10 -m src.eval_render_videos --help`

   ```
   usage: eval_render_videos.py [-h] [--envs ENVS [ENVS ...]]
                                [--policies POLICIES [POLICIES ...]]
                                [--seeds SEEDS [SEEDS ...]] [--i I]
                                [--steps STEPS] [--runs-root RUNS_ROOT]
                                [--contact-root CONTACT_ROOT]
   ...
   ```

All three smoke checks pass.

---

## Appendix -- salvaged one-liners

Preserved here for convenience (previously at the top of
`src/data_gen/gen_fetch.py`):

```bash
# Copy a generated dataset up to the dingo lab machine:
scp -r data/reacher jarmibe7@dingo.mech.northwestern.edu:~/E2C/

# Pull a rendered evaluation video back down:
scp jarmibe7@dingo.mech.northwestern.edu:~/E2C/videos/e2c_cartpole.mp4 \
    C:\\Users\\jarmi\\MS_Thesis\\Media\\Videos
```
