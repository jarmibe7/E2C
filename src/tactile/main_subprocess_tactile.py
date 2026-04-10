"""
Run batches of tactile experiments via subprocess.

Default behavior runs all object_push tactile configs. For each config,
you can run multiple trials with deterministic seed offsets.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TACTILE_CONFIG_PATH = PROJECT_ROOT / "config" / "tactile"
GENERATED_CONFIG_DIR = TACTILE_CONFIG_PATH / "generated_subprocess"


def discover_configs_for_env(env_name: str) -> list[Path]:
    """Find tactile config files for an env from <env_name>_*.yaml."""
    pattern = f"{env_name}_*.yaml"
    return sorted(p for p in TACTILE_CONFIG_PATH.glob(pattern) if p.is_file())


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def build_trial_config(base_config: dict, seed: int) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["seed"] = int(seed)
    cfg.setdefault("train", {})["seed"] = int(seed)
    return cfg


def to_config_arg(config_path: Path) -> str:
    """Convert config path to --config argument relative to config/tactile."""
    rel = config_path.relative_to(TACTILE_CONFIG_PATH)
    return str(rel.with_suffix("")).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tactile experiments in subprocesses with optional multi-trial seeds."
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["object_push"],
        help="Env prefixes to run. Configs are discovered as <env>_*.yaml.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per config.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help=(
            "Optional explicit starting seed. If omitted, each config uses its own "
            "seed as the base and increments by trial index."
        ),
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to launch subprocesses.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running them.",
    )
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")

    planned_runs: list[tuple[str, str, int, list[str]]] = []

    for env_name in args.envs:
        env_configs = discover_configs_for_env(env_name)
        if not env_configs:
            print(f"[warn] No configs found for env='{env_name}' in {TACTILE_CONFIG_PATH}")
            continue

        for config_path in env_configs:
            base_config = load_yaml(config_path)
            base_seed = int(base_config.get("seed", base_config.get("train", {}).get("seed", 0)))

            for trial_idx in range(args.trials):
                trial_seed = (args.seed_start + trial_idx) if args.seed_start is not None else (base_seed + trial_idx)

                trial_config = build_trial_config(base_config, trial_seed)
                generated_path = (
                    GENERATED_CONFIG_DIR
                    / env_name
                    / f"{config_path.stem}__trial_{trial_idx:02d}.yaml"
                )
                save_yaml(generated_path, trial_config)

                config_arg = to_config_arg(generated_path)
                cmd = [
                    args.python,
                    "-m",
                    "src.tactile.main_tactile",
                    "--config",
                    config_arg,
                ]
                planned_runs.append((env_name, config_path.name, trial_seed, cmd))

    if not planned_runs:
        raise RuntimeError("No runs were planned. Check --envs and config naming.")

    print(f"Planned {len(planned_runs)} runs")
    for idx, (env_name, config_name, trial_seed, cmd) in enumerate(planned_runs, start=1):
        print(
            f"[{idx}/{len(planned_runs)}] env={env_name} config={config_name} "
            f"seed={trial_seed}"
        )
        print("  $ " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
