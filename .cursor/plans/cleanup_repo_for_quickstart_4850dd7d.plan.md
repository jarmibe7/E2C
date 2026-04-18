---
name: Cleanup repo for quickstart
overview: Aggressively delete the legacy E2C / state-rep / notebook / test / figure code paths, then clean each remaining file in place (dead comments, unused imports/methods, broken snippets, duplication) without touching the RSSM core algorithm. Ship a `CLEANUP_REPORT.md` with bulleted issues + diff-style patches + explanations, plus a `QUICKSTART.md` so a new reader can go data-gen → train → eval → render in one pass.
todos:
  - id: phase1_deletes
    content: "Delete legacy files: src/e2c.py, src/train.py, src/state_rep.py, src/model/e2c.py, src/model/policy.py, gen_particle_grav.py, gen_dummy.py, gen_state_rep.py, notebooks/, tests/, debug.png, train.sh. KEEP: all *.mp4 files, keep src/data_gen/ files associated with non particle_gravity. Keep debugging jupyter notebooks. Keep notebooks that generate figures of any kind."
    status: completed
  - id: clean_main
    content: "Clean src/main.py: drop ConvE2C/E2C branch, drop matplotlib/numpy/ConvPolicy imports, remove direct_reward pass, factor _build_run_path + _save_on_exit helpers, keep hardware branch wired to new stub"
    status: completed
  - id: clean_trainer
    content: "Clean src/trainer.py: delete unused helpers (_frames_to_tensor et al), delete E2CPretrainer, strip commented KL alternatives from rollout_info_gain_decoded_batch, add docstrings tying _kl_diag_gaussian to paper eqs, add minimal ClosedLoopHardwareTrainer stub, fix inverted rollout-steps warning"
    status: completed
  - id: clean_eval
    content: "Clean src/eval.py: delete eval/eval_traj/eval_metrics/eval_four_var_latent/eval_latent and __main__ block, factor visualize_planner + render_video into one _rollout_with_planner helper, remove hardcoded /home/ayush path, wire contact helper"
    status: completed
  - id: clean_models
    content: Clean src/model/rssm.py (drop broken reconstruct, fix sample_traj unsqueeze, delete commented post variant, trim sample signature), src/model/loss.py (delete E2CLoss + UncertaintyE2CLoss + commented KLs, keep RSSMLoss), src/model/encoder.py + decoder.py (drop ChannelUncertaintyConvDecoder, delete dead fc_decode reassignment, add docstrings)
    status: completed
  - id: clean_misc
    content: Clean src/dataset.py (drop triple-quoted block, add docstring), src/replay_buffer.py (types+docstrings), src/utils.py (delete rk4_sim, add hints), src/data_gen/gen_fetch.py (move XML hack into main(), docstring), src/eval_render_videos.py (argparse CLI, drop CEM overrides), src/main_subprocess.py (trim commented code + inline overrides)
    status: completed
  - id: fix_meta
    content: Fix setup.py (broken requirements.txt reference + rename to data4wm), tighten .gitignore (remove self-ignore + blanket config/*)
    status: completed
  - id: write_quickstart
    content: Write QUICKSTART.md with 5-step install/gen/understand/train/eval runbook citing network_architecture.png and eq. (1) mapping to rssm.py modules
    status: completed
  - id: write_report
    content: "Write CLEANUP_REPORT.md: bullet list of every change with git-style diff patches + one-line safety justification per change, plus flagged-but-unfixed section (_sample_cem std formula, inverted warning wording, unused obs return)"
    status: completed
  - id: smoke_check
    content: Run 3 smoke checks (import main, construct trainer, eval_render_videos --help) and paste output into CLEANUP_REPORT.md
    status: completed
  - id: todo-1776465749570-vqpagq25d
    content: "Clean src/trainer.py: delete unused helpers (_frames_to_tensor et al), delete E2CPretrainer, add docstrings tying _kl_diag_gaussian to paper eqs, add minimal ClosedLoopHardwareTrainer stub, fix inverted rollout-steps warning"
    status: pending
isProject: false
---

## Ground rules

- **Do not touch core algorithmic logic** in `RSSME2C`, `RSSMLoss`, `ClosedLoopInformativeTrainer.rollout_info_gain_decoded_batch`, `_sample_cem`, `collect_rollouts`. The two KL formulations (pixel vs dynamics) stay as-is (both `_kl_diag_gaussian` in `trainer.py` and `kl_divergence` in `RSSMLoss` are preserved — the user explicitly called this out).
- **Preserve hardware hook:** keep a minimal `ClosedLoopHardwareTrainer` stub in `trainer.py` so `main.py`'s `elif policy_type == "hardware"` branch stays valid.
- **Never touch data:** no deletions under `data/`, `runs/`, `src/data_gen/contact/`, root `*.mp4`, `*.json`, or `*.npy`. Only source-level artifacts and obsolete notebook / figure / config folders.
- **Deliverable format:** one `CLEANUP_REPORT.md` at repo root with every change as a git-style diff snippet + explanation, plus one `QUICKSTART.md` with the new runbook. Code changes are applied directly after the report is written.

## Phase 1 — File-level deletions (aggressive)

Legacy E2C track (entirely superseded by RSSM):

- [src/e2c.py](src/e2c.py) (old duplicate of dataset+loss+model at `src/` root, not `src/model/`)
- [src/train.py](src/train.py) (old `train()` entry, references deleted `src.e2c.E2C`)
- [src/model/e2c.py](src/model/e2c.py) → rewritten to remove `ConvE2C`, `UncertaintyE2CLoss` usage. Since `src/model/rssm.py::RSSME2C` is the only model used, keep that file and delete `src/model/e2c.py` entirely. Update `main.py` imports.

State-rep pretraining track (orthogonal to the paper):

- [src/state_rep.py](src/state_rep.py)
- [src/data_gen/gen_state_rep.py](src/data_gen/gen_state_rep.py)
- [train.sh](train.sh) (only invoked state_rep pipeline)

Legacy env data generators not used by the paper (Metaworld is the only one currently exercised):

- [src/data_gen/gen_particle_grav.py](src/data_gen/gen_particle_grav.py), [src/data_gen/gen_dummy.py](src/data_gen/gen_dummy.py) 
- `src/compare_traj_rssm.ipynb`

Obsolete root / meta:

- [notebooks/](notebooks/) (user confirmed obsolete)
- [tests/](tests/) (user confirmed obsolete)
- [figures/pres_cartpole/](figures/pres_cartpole/), [figures/pres_particle_grav/](figures/pres_particle_grav/), [figures/pres_reacher/](figures/pres_reacher/)
- [config/old/](config/old/) (legacy yamls)
- [config/configs_change_cam/](config/configs_change_cam/) (camera sweep artifacts; `configs_final/` + `configs_gripper/` are the paper configs)
- Root [models.py](models.py) (stale copy of model code), root `__init__.py` (empty), root `debug.png`, root `__pycache__/`, all `src/**/__pycache__/`

Kept: `config/configs_final/`, `config/configs_gripper/`, all of `data/`, `runs/`, `src/data_gen/contact/`, root mp4/png artifacts that are paper outputs. KEEP: all *.mp4 files, keep src/data_gen/ files associated with non particle_gravity. Keep debugging jupyter notebooks. Keep notebooks that generate figures of any kind.

## Phase 2 — In-file cleanup (no algorithmic changes)

### [src/main.py](src/main.py)

- Drop unused imports: `matplotlib.pyplot`, `numpy`, `ConvE2C`, `ConvPolicy`, `E2CPretrainer`.
- Remove the `if 'e2c' in config_name:` branch (only RSSM remains).
- Remove the `direct_reward` dead branch (`pass`).
- Keep `hardware` branch → maps to new stub `ClosedLoopHardwareTrainer`.
- Factor the `save_name` computation (lines 65–76) into `_build_run_path(config, config_name)` in `utils.py`.
- Replace the `try/except Exception/KeyboardInterrupt` copy-paste with one helper `_save_on_exit(trainer, config_save, start_time, reason)`.

### [src/main_subprocess.py](src/main_subprocess.py)

- Keep as a minimal sweep harness, keeping the dead comments. Leave only: iterate seeds × policies × envs, override `device`/`num_epochs`/`seed` and launch.

### [src/trainer.py](src/trainer.py)

- Delete unused helpers: `_frames_to_tensor`, `_update_window`, `_decode_latent`, `_encode_posterior_e2c`, `_encode_posterior_rssm` (none are referenced anywhere).
- Delete `E2CPretrainer` class (ConvE2C was dropped).
- DO NOT delete commented-out alternative KL formulations inside `rollout_info_gain_decoded_batch`. Add a short docstring summarising the two objectives (`maxdyn` = KL(prior_t+1 || posterior_t0) for dynamics; `eig/informative` = KL(posterior_t+1 || prior_t+1) for pixel) with a reference to the paper equations.
- Add module + class docstrings (purpose, args table, which env classes supported).
- Add a minimal `ClosedLoopHardwareTrainer(ClosedLoopInformativeTrainer)` stub that overrides `collect_rollouts` with `raise NotImplementedError("hardware trainer lives in the private hardware repo")` and keeps the same constructor signature. This preserves `main.py`'s branching for future hardware work without leaving a dangling reference.
- Fix the `num_rollout_steps` guard wording (current one is inverted — it says `>` but triggers on `<=`; the warning is backwards).
- **Do NOT** touch the `1 / len(new_mu - 1)` line in `_sample_cem` (line 712) — flag in the report as a possible bug but leave alone per "preserve experimental logic / ask if uncertain".

### [src/eval.py](src/eval.py)

- Delete broken / ConvE2C-only methods: `eval`, `eval_traj`, `eval_metrics`, `eval_four_var_latent`, `eval_latent` (per your answer). Keep `Plotter`, `Evaluator.__init`__, `Evaluator.visualize_planner`, `Evaluator.render_video`.
- Delete the `if __name__ == "__main__":` block (uses deleted `ConvE2C`).
- Factor the ~95% duplicated body of `visualize_planner` vs `render_video` into one private `_rollout_with_planner(...)` and have both call it; the only differences are the figure grid (2×cols vs 1×1) and the output filename — pass those as params.
- Fix: remove hard-coded path `/home/ayush/Desktop/.../temp_figs/step_costs.txt` in `render_video` (was a dev debug dump).
- Drop unused `first_img` local and the commented-out contact-env-reset block.

### [src/trainer.py](src/trainer.py) + [src/eval.py](src/eval.py) helpers

- Move `get_mujoco_geom_keys_index`, `is_robot_contact_geometry` call-site wiring into one small helper `count_contacts(env, dataset_name)` in a new `src/contacts.py` — two callers (`visualize_planner`, `render_video`) currently open-code it.

### [src/dataset.py](src/dataset.py)

- Drop the giant triple-quoted commented-out "Create windowed samples" block.
- Add a docstring documenting expected `.pt` keys (`prev_images`, `next_images`, `actions`) and tensor shapes.

### [src/replay_buffer.py](src/replay_buffer.py)

- Already clean; just add types + docstrings for `add` / `sample` and note that sampling is without replacement.

### [src/model/rssm.py](src/model/rssm.py)

- Delete `reconstruct` (uses `self.mu` / `self.log_var` which don't exist on RSSM — broken).
- Fix `sample_traj`: currently passes `z` without time-dim to `rssm_step` which expects `[B, T, D]`. Wrap in `z.unsqueeze(1)`.
- Clean `forward` docstring with the 4-part model description from the paper (encoder, recurrent, prior, decoder).
- Remove the unused `sample(return_all=...)` parameter — it's dead code.

### [src/model/encoder.py](src/model/encoder.py) / [src/model/decoder.py](src/model/decoder.py)

- Add docstrings. Remove `ChannelUncertaintyConvDecoder` — only `ScalarUncertaintyConvDecoder` and `ConvDecoder` are referenced.
- Delete the redundant first `self.fc_decode = nn.Linear(...)` in `ConvDecoder.__init`__ that is immediately overwritten by the `Sequential` (line 22 is dead).

### [src/model/loss.py](src/model/loss.py)

- Delete `E2CLoss` and `UncertaintyE2CLoss` (legacy path deleted). Keep `RSSMLoss`.
- Keep `free_nats` clamping and `kld_anneal` exactly as-is.

### [src/model/policy.py](src/model/policy.py)

- Delete — `ConvPolicy` and `BasePolicy` are never instantiated anywhere after E2CPretrainer removal.

### [src/data_gen/gen_fetch.py](src/data_gen/gen_fetch.py)

- Keep; this is the live data generator. Tighten:
  - Replace the module-level XML-editing hack (`new_xml_filename = ...`) that runs on import with a function `maybe_patch_mjcf_timestep()` called only from `main()`.
  - Keep entries for envs we dropped (push/antmaze/pointmaze/mountaincar/reacher/cartpole). Default: **keep**, because `trainer.py` still dispatches on env name generically and it's cheap.
  - Move top-of-file `scp` one-liner comments into `CLEANUP_REPORT.md` appendix.
  - Add a module docstring with expected output shape of the `.pt` file.

### [src/eval_render_videos.py](src/eval_render_videos.py)

- Delete the commented-out `try/except KeyboardInterrupt` at the bottom and the sweep-of-sweeps commented blocks; collapse the `for env in ['coffee']` loop into an argparse CLI (`--envs`, `--seeds`, `--policies`, `--i`, `--steps`).
- Remove the in-script CEM overrides (`samples=300`, `plan_horizon=8`, etc.) — accept them via CLI, default to whatever is in the loaded config's yaml.

### [src/utils.py](src/utils.py)

- Keep; delete `rk4_sim` (unused since `gen_particle_grav.py` was removed).
- Add type hints + docstrings for metric helpers (`central_mass_ratio`, `excess_kurtosis`, `shoulder_mass`).

### [setup.py](setup.py)

- Fix broken `read_requirements()`: it opens `requirements.txt` which doesn't exist. Point to `requirements_linux.txt` or make it conditional on `sys.platform`.

## Phase 3 — New docs

### `QUICKSTART.md` (new)

Runbook for a new reader, in 5 numbered steps:

1. **Install**: `pip install -r requirements_linux.txt && pip install -e .` (plus MuJoCo/Metaworld system notes).
2. **Generate a dataset**: edit `env_name` in `src/data_gen/gen_fetch.py` → `python3.10 -m src.data_gen.gen_fetch` → produces `data/<env>/<env>_gripper_2k.pt`.
3. **Understand the model**: one paragraph pointing to the `network_architecture.png` + the four parts in `src/model/rssm.py` (`encoder`, `rnn`, `prior`, `post`, `decoder`) with the Eq. (1) mapping.
4. **Train** (single config): `python3.10 -m src.main --config configs_final/button_eig_0`. Explain what `eig` / `maxdyn` / `random` mean (pixel IG / dynamics IG / random baseline).
5. **Evaluate + render videos**: `python3.10 -m src.eval_render_videos --envs button --policies eig --seeds 0 1`.

### `CLEANUP_REPORT.md` (new)

For every Phase 2 change above, one section with:

- **File** and **reason**
- Git-style diff patch showing the exact lines removed/added
- One-sentence "why this is safe" note

Flag (but do not fix):

- `_sample_cem` std formula uses `torch.sqrt((x - mean)**2)` = `abs()` and divides by `len(new_mu - 1)` which evaluates to `len(new_mu)`. Possible MAD-for-std substitution, or a typo for `len(new_mu) - 1`. Needs your call.
- The `obs` variable returned from `env.reset()` in both collect_rollouts is never used — rendered image is used instead. Probably intentional for Metaworld (state is blind).
- `ClosedLoopRandomTrainer` inequality warning message is inverted.

## Phase 4 — Sanity verification (no unit tests per your "don't overfocus on testing" rule)

Run these three smoke checks and paste output into `CLEANUP_REPORT.md`:

1. `python3.10 -c "from src.main import main"` — import check after deletions.
2. `python3.10 -m src.main --config configs_final/button_eig_0 --help`-equivalent dry run (just construct the trainer, skip `.learn()`).
3. `python3.10 -m src.eval_render_videos --help` — argparse wires up.

## Explicit non-goals

- Not touching the CEM algorithm, KL objectives, RSSM forward, or loss math.
- Not renaming the two `kl_divergence` / `_kl_diag_gaussian` functions — duplication is deliberate per your note.
- Not merging `configs_final` and `configs_gripper` — naming conventions are load-bearing in `main.py::save_name`.
- Not deleting alternative KL formulations in commented code

