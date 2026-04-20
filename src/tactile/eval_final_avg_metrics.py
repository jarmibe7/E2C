"""
Evaluate tactile run directories under runs/<env>/<obs_type>/<policy_type>/final
and write averaged metrics to avg_metric.json in each run directory and
avg_metrics.json in each final directory.

python -m src.tactile.eval_final_avg_metrics --runs-root /home/jarmibe7/E2C/runs/object_push --trajectories 4
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

try:
    from scipy.stats import mannwhitneyu, ttest_ind

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from src.tactile.gen_tactile import resolve_env_name_from_dataset
from src.tactile.tactile_rssm import TactileRSSM
from src.tactile.tactile_trainer import ClosedLoopTactileTrainer
from src.tactile.tactile_utils import TactileDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def posixpath_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return Path(*seq)


yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    posixpath_constructor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run N tactile eval trajectories for each run under "
            "runs/<env>/<obs_type>/<policy_type>/final and save avg_metric.json "
            "and avg_metrics.json."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "object_push",
        help="Root directory in format runs/<env>.",
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        default=4,
        help="Number of evaluation trajectories per run directory.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override rollout horizon; defaults to train.num_eval_timesteps or 100.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First evaluation seed.",
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1,
        help="Seed increment between trajectories.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override (e.g. cpu, cuda:0).",
    )
    parser.add_argument(
        "--obs-types",
        nargs="+",
        default=None,
        help="Optional obs type filter (e.g. egocentric exocentric tactile).",
    )
    parser.add_argument(
        "--policy-types",
        nargs="+",
        default=None,
        help="Optional policy type filter (e.g. pixel random).",
    )
    parser.add_argument(
        "--compare-policy-a",
        type=str,
        default="pixel",
        help="First policy name for per-observation significance tests.",
    )
    parser.add_argument(
        "--compare-policy-b",
        type=str,
        default="random",
        help="Second policy name for per-observation significance tests.",
    )
    parser.add_argument(
        "--significance-test",
        type=str,
        choices=["mannwhitney", "welch_t", "both", "none"],
        default="both",
        help="Trajectory-level significance test for policy comparison.",
    )
    parser.add_argument(
        "--min-samples-per-policy",
        type=int,
        default=2,
        help="Minimum trajectory samples per policy required to test a metric.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered run directories without evaluating.",
    )
    return parser.parse_args()


def _is_numeric(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def _aggregate_metrics(values: list[Any]) -> Any:
    """Recursively average numeric values and numeric leaves in nested dicts."""
    if not values:
        return None

    if all(isinstance(v, dict) for v in values):
        keys: set[str] = set()
        for value in values:
            keys.update(key for key in value.keys() if key != "_meta")

        out: dict[str, Any] = {}
        for key in sorted(keys):
            child_values = [value[key] for value in values if key in value]
            agg = _aggregate_metrics(child_values)
            if agg is not None:
                out[key] = agg
        return out

    numeric_values = [float(v) for v in values if _is_numeric(v) and np.isfinite(v)]
    if numeric_values:
        return float(np.mean(numeric_values))

    return None


def _sample_std(values: list[float]) -> float:
    """Sample standard deviation (ddof=1), returning 0 for n<=1."""
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _ci95_halfwidth(std: float, n: int) -> float:
    """95% CI half-width for the mean using normal approximation."""
    if n <= 1:
        return 0.0
    return float(1.96 * std / np.sqrt(n))


def _contact_stats_from_values(values: list[float]) -> dict[str, Any]:
    """Compute contact summary stats from raw per-trajectory values."""
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "ci95_halfwidth": None,
            "values": [],
        }

    mean = float(np.mean(values))
    std = _sample_std(values)
    ci_half = _ci95_halfwidth(std, n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - ci_half),
        "ci95_high": float(mean + ci_half),
        "ci95_halfwidth": ci_half,
        "values": [float(v) for v in values],
    }


def _flatten_numeric_metrics(payload: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested dict metrics into dot keys with finite numeric leaves."""
    if isinstance(payload, dict):
        out: dict[str, float] = {}
        for key, value in payload.items():
            if key == "_meta":
                continue
            next_prefix = f"{prefix}.{key}" if prefix else key
            out.update(_flatten_numeric_metrics(value, next_prefix))
        return out

    if _is_numeric(payload) and np.isfinite(payload):
        if not prefix:
            return {}
        return {prefix: float(payload)}

    return {}


def _sample_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    mean = float(np.mean(values))
    std = _sample_std(values)
    return {"n": len(values), "mean": mean, "std": std}


def _compute_significance_test(
    values_a: list[float],
    values_b: list[float],
    method: str,
) -> dict[str, Any]:
    if not SCIPY_AVAILABLE:
        return {
            "method": method,
            "ok": False,
            "reason": "scipy_unavailable",
            "p_value": None,
        }

    if method == "mannwhitney":
        stat, p_value = mannwhitneyu(values_a, values_b, alternative="two-sided")
        return {
            "method": method,
            "ok": True,
            "u_statistic": float(stat),
            "p_value": float(p_value),
        }

    if method == "welch_t":
        stat, p_value = ttest_ind(values_a, values_b, equal_var=False)
        return {
            "method": method,
            "ok": True,
            "t_statistic": float(stat),
            "p_value": float(p_value),
        }

    raise ValueError(f"Unsupported test method: {method}")


def _combine_contact_stats(run_stats: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Combine per-run contact stats into one pooled summary over trajectories."""
    valid_stats: list[dict[str, Any]] = []
    for stat in run_stats:
        n = stat.get("n")
        mean = stat.get("mean")
        std = stat.get("std")
        if not _is_numeric(n) or int(n) <= 0:
            continue
        if not _is_numeric(mean):
            continue
        std_val = float(std) if _is_numeric(std) else 0.0
        valid_stats.append({"n": int(n), "mean": float(mean), "std": std_val})

    if not valid_stats:
        return None

    total_n = sum(stat["n"] for stat in valid_stats)
    if total_n <= 0:
        return None

    pooled_mean = float(
        sum(stat["mean"] * stat["n"] for stat in valid_stats) / total_n
    )

    if total_n <= 1:
        pooled_std = 0.0
    else:
        ss = 0.0
        for stat in valid_stats:
            n_i = stat["n"]
            mean_i = stat["mean"]
            std_i = stat["std"]
            ss += (n_i - 1) * (std_i**2) + n_i * ((mean_i - pooled_mean) ** 2)
        pooled_std = float(np.sqrt(max(ss / (total_n - 1), 0.0)))

    ci_half = _ci95_halfwidth(pooled_std, total_n)
    return {
        "n": int(total_n),
        "mean": pooled_mean,
        "std": pooled_std,
        "ci95_low": float(pooled_mean - ci_half),
        "ci95_high": float(pooled_mean + ci_half),
        "ci95_halfwidth": ci_half,
        "ci_method": "normal_approximation",
    }


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_trajectory_metrics(
    run_dir: Path,
    per_trajectory_metrics: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "_meta": {
            **run_meta,
            "num_trajectories": len(per_trajectory_metrics),
            "run_dir": str(run_dir),
        },
        "trajectories": per_trajectory_metrics,
    }
    _write_json(run_dir / "trajectory_metrics.json", payload)


def discover_final_dirs(
    runs_root: Path,
    obs_filters: set[str] | None,
    policy_filters: set[str] | None,
) -> list[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root does not exist: {runs_root}")

    final_dirs: list[Path] = []
    for obs_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if obs_filters and obs_dir.name not in obs_filters:
            continue

        for policy_dir in sorted(path for path in obs_dir.iterdir() if path.is_dir()):
            if policy_filters and policy_dir.name not in policy_filters:
                continue

            final_dir = policy_dir / "final"
            if final_dir.is_dir():
                final_dirs.append(final_dir)

    return final_dirs


def discover_run_dirs(
    runs_root: Path,
    obs_filters: set[str] | None,
    policy_filters: set[str] | None,
) -> list[Path]:
    run_dirs: list[Path] = []
    for final_dir in discover_final_dirs(runs_root, obs_filters, policy_filters):
        for run_dir in sorted(path for path in final_dir.iterdir() if path.is_dir()):
            if (run_dir / "config.yaml").exists() and (run_dir / "model.pt").exists():
                run_dirs.append(run_dir)

    return run_dirs


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_trainer(run_dir: Path, config: dict[str, Any], device_override: str | None):
    train_cfg = config.setdefault("train", {})
    train_cfg["load_path"] = str(run_dir)
    config["run_path"] = run_dir

    if device_override:
        train_cfg["device"] = device_override

    requested_device = str(train_cfg.get("device", "cpu"))
    if "cuda" in requested_device and not torch.cuda.is_available():
        print(
            f"[warn] CUDA requested but unavailable for {run_dir.name}; "
            "falling back to cpu"
        )
        requested_device = "cpu"
        train_cfg["device"] = requested_device

    device = torch.device(requested_device)

    dataset = TactileDataset(config)
    config["vae"]["in_image_shape"] = dataset.in_img_shape
    num_out_channels = config["vae"]["in_image_shape"][0] // config["trans"]["past_length"]
    config["vae"]["out_image_shape"] = (
        num_out_channels,
        *config["vae"]["in_image_shape"][1:],
    )
    config["trans"]["control_size"] = dataset.U.shape[-1]

    model = TactileRSSM(
        enc_latent_size=config["vae"]["enc_latent_size"],
        feature_latent_size=config["tactile"]["feature_latent_size"],
        feature_size=dataset.num_features,
        stochastic_size=config["trans"]["stochastic_size"],
        deterministic_size=config["trans"]["deterministic_size"],
        control_size=config["trans"]["control_size"],
        past_length=config["trans"]["past_length"],
        pred_length=config["trans"]["pred_length"],
        conv_params=config["vae"],
        device=device,
        output_uncertainty=False,
    )

    state_dict = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    trainer = ClosedLoopTactileTrainer(dataset, model, config, device)
    return trainer


def evaluate_run_dir(
    run_dir: Path,
    trajectories: int,
    seed_start: int,
    seed_stride: int,
    max_steps: int | None,
    device_override: str | None,
) -> dict[str, Any]:
    config = load_config(run_dir / "config.yaml")
    trainer = build_trainer(run_dir, config, device_override)

    rollout_steps = max_steps
    if rollout_steps is None:
        rollout_steps = int(config.get("train", {}).get("num_eval_timesteps", 100))

    per_trajectory_metrics: list[dict[str, Any]] = []
    for traj_idx in range(trajectories):
        seed = seed_start + traj_idx * seed_stride
        eval_return = trainer.evaluator.eval(
            trainer,
            run_dir,
            max_steps=rollout_steps,
            closed_loop=True,
            env_reset_seed=seed,
            count_contacts=True,
        )

        if isinstance(eval_return, tuple) and len(eval_return) == 2:
            metrics = eval_return[1]
        elif isinstance(eval_return, dict):
            metrics = eval_return
        else:
            raise RuntimeError(
                f"Unexpected eval return type for {run_dir}: {type(eval_return)}"
            )
        per_trajectory_metrics.append(metrics)

    avg_metrics = _aggregate_metrics(per_trajectory_metrics)
    if not isinstance(avg_metrics, dict):
        raise RuntimeError(f"No aggregatable metrics found in {run_dir}")

    # Preserve per-trajectory contact dispersion for accurate final aggregation.
    contact_values = [
        float(metric["contact_timesteps"])
        for metric in per_trajectory_metrics
        if isinstance(metric, dict)
        and "contact_timesteps" in metric
        and _is_numeric(metric["contact_timesteps"])
    ]
    if contact_values:
        avg_metrics["contact_timesteps_stats"] = _contact_stats_from_values(contact_values)

    avg_metrics["_meta"] = {
        "num_trajectories": trajectories,
        "seed_start": seed_start,
        "seed_stride": seed_stride,
        "max_steps": rollout_steps,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
    }

    out_path = run_dir / "avg_metric.json"
    _write_json(out_path, avg_metrics)

    _write_trajectory_metrics(
        run_dir=run_dir,
        per_trajectory_metrics=per_trajectory_metrics,
        run_meta={
            "seed_start": seed_start,
            "seed_stride": seed_stride,
            "max_steps": rollout_steps,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    return avg_metrics


def evaluate_final_dir(final_dir: Path) -> dict[str, Any]:
    avg_metric_paths = sorted(
        run_dir / "avg_metric.json"
        for run_dir in final_dir.iterdir()
        if run_dir.is_dir() and (run_dir / "avg_metric.json").exists()
    )
    if not avg_metric_paths:
        raise RuntimeError(f"No avg_metric.json files found in {final_dir}")

    per_run_metrics = [_load_json(path) for path in avg_metric_paths]
    avg_metrics = _aggregate_metrics(per_run_metrics)
    if not isinstance(avg_metrics, dict):
        raise RuntimeError(f"No aggregatable metrics found in {final_dir}")

    # Build pooled contact stats from run-level trajectory summaries.
    run_contact_stats = [
        run_metric["contact_timesteps_stats"]
        for run_metric in per_run_metrics
        if isinstance(run_metric, dict)
        and isinstance(run_metric.get("contact_timesteps_stats"), dict)
    ]

    combined_contact_stats = _combine_contact_stats(run_contact_stats)
    if combined_contact_stats is not None:
        avg_metrics["contact_timesteps_stats"] = combined_contact_stats

    # Backward-compatibility fallback for legacy avg_metric.json files.
    elif "contact_timesteps" in avg_metrics and _is_numeric(avg_metrics["contact_timesteps"]):
        legacy_values = [
            float(run_metric["contact_timesteps"])
            for run_metric in per_run_metrics
            if isinstance(run_metric, dict)
            and "contact_timesteps" in run_metric
            and _is_numeric(run_metric["contact_timesteps"])
        ]
        if legacy_values:
            fallback_stats = _contact_stats_from_values(legacy_values)
            fallback_stats["source"] = "run_level_means"
            fallback_stats["note"] = (
                "Computed from run-level means because per-run trajectory stats "
                "were unavailable."
            )
            avg_metrics["contact_timesteps_stats"] = fallback_stats

    avg_metrics["_meta"] = {
        "num_runs": len(avg_metric_paths),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "final_dir": str(final_dir),
    }

    out_path = final_dir / "avg_metrics.json"
    _write_json(out_path, avg_metrics)
    return avg_metrics


def _collect_trajectory_metric_samples(
    final_dir: Path,
) -> tuple[dict[str, list[float]], int, int]:
    """Collect flattened trajectory-level samples from one policy final directory."""
    samples: dict[str, list[float]] = {}
    run_count = 0
    traj_count = 0

    for run_dir in sorted(path for path in final_dir.iterdir() if path.is_dir()):
        traj_path = run_dir / "trajectory_metrics.json"
        if not traj_path.exists():
            continue

        payload = _load_json(traj_path)
        trajectories = payload.get("trajectories", []) if isinstance(payload, dict) else []
        if not isinstance(trajectories, list):
            continue

        valid_in_run = 0
        for trajectory in trajectories:
            if not isinstance(trajectory, dict):
                continue
            flattened = _flatten_numeric_metrics(trajectory)
            if not flattened:
                continue

            valid_in_run += 1
            traj_count += 1
            for key, value in flattened.items():
                samples.setdefault(key, []).append(float(value))

        if valid_in_run > 0:
            run_count += 1

    return samples, run_count, traj_count


def compare_policies_by_observation_trajectory_level(
    runs_root: Path,
    obs_filters: set[str] | None,
    policy_a: str,
    policy_b: str,
    significance_test: str,
    min_samples_per_policy: int,
) -> list[Path]:
    """
    Compare two policy types per observation directory using trajectory samples.

    Writes one JSON file per observation dir:
    runs/<env>/<obs>/<policy_a>_vs_<policy_b>_trajectory_significance.json
    """
    if min_samples_per_policy < 1:
        raise ValueError("--min-samples-per-policy must be >= 1")

    out_paths: list[Path] = []
    if policy_a == policy_b:
        print("[warn] policy comparison skipped: compare-policy-a equals compare-policy-b")
        return out_paths

    if significance_test != "none" and not SCIPY_AVAILABLE:
        print("[warn] scipy is unavailable; significance tests will be marked unavailable")

    obs_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    for obs_dir in obs_dirs:
        if obs_filters and obs_dir.name not in obs_filters:
            continue

        final_a = obs_dir / policy_a / "final"
        final_b = obs_dir / policy_b / "final"
        if not final_a.is_dir() or not final_b.is_dir():
            continue

        samples_a, run_count_a, traj_count_a = _collect_trajectory_metric_samples(final_a)
        samples_b, run_count_b, traj_count_b = _collect_trajectory_metric_samples(final_b)
        shared_metrics = sorted(set(samples_a.keys()) & set(samples_b.keys()))

        metric_results: dict[str, Any] = {}
        for metric_key in shared_metrics:
            values_a = samples_a.get(metric_key, [])
            values_b = samples_b.get(metric_key, [])

            summary_a = _sample_summary(values_a)
            summary_b = _sample_summary(values_b)

            metric_payload: dict[str, Any] = {
                policy_a: summary_a,
                policy_b: summary_b,
                "difference_mean": (
                    float(summary_a["mean"] - summary_b["mean"])
                    if _is_numeric(summary_a["mean"]) and _is_numeric(summary_b["mean"])
                    else None
                ),
                "tests": {},
            }

            enough_data = (
                len(values_a) >= min_samples_per_policy
                and len(values_b) >= min_samples_per_policy
            )

            if not enough_data:
                metric_payload["tests"] = {
                    "ok": False,
                    "reason": "insufficient_samples",
                    "required_per_policy": min_samples_per_policy,
                }
            elif significance_test == "none":
                metric_payload["tests"] = {
                    "ok": False,
                    "reason": "disabled",
                }
            elif significance_test == "both":
                metric_payload["tests"] = {
                    "mannwhitney": _compute_significance_test(
                        values_a, values_b, "mannwhitney"
                    ),
                    "welch_t": _compute_significance_test(values_a, values_b, "welch_t"),
                }
            else:
                metric_payload["tests"] = {
                    significance_test: _compute_significance_test(
                        values_a,
                        values_b,
                        significance_test,
                    )
                }

            metric_results[metric_key] = metric_payload

        payload = {
            "_meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "runs_root": str(runs_root),
                "observation_type": obs_dir.name,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "significance_test": significance_test,
                "min_samples_per_policy": min_samples_per_policy,
                "num_runs_policy_a": run_count_a,
                "num_runs_policy_b": run_count_b,
                "num_trajectories_policy_a": traj_count_a,
                "num_trajectories_policy_b": traj_count_b,
                "num_shared_metrics": len(shared_metrics),
            },
            "metrics": metric_results,
        }

        out_path = obs_dir / f"{policy_a}_vs_{policy_b}_trajectory_significance.json"
        _write_json(out_path, payload)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    args = parse_args()
    if args.trajectories < 1:
        raise ValueError("--trajectories must be >= 1")
    if args.seed_stride < 1:
        raise ValueError("--seed-stride must be >= 1")
    if args.min_samples_per_policy < 1:
        raise ValueError("--min-samples-per-policy must be >= 1")

    obs_filters = set(args.obs_types) if args.obs_types else None
    policy_filters = set(args.policy_types) if args.policy_types else None
    run_dirs = discover_run_dirs(args.runs_root, obs_filters, policy_filters)
    final_dirs = discover_final_dirs(args.runs_root, obs_filters, policy_filters)

    if not run_dirs:
        raise RuntimeError(
            "No run directories found with config.yaml and model.pt under "
            f"{args.runs_root}"
        )

    # Best effort consistency check for env naming from dataset string.
    sample_cfg = load_config(run_dirs[0] / "config.yaml")
    dataset_name = sample_cfg.get("train", {}).get("dataset", "")
    inferred_env = resolve_env_name_from_dataset(dataset_name) if dataset_name else ""
    if inferred_env and args.runs_root.name != inferred_env:
        print(
            f"[warn] runs root '{args.runs_root.name}' does not match inferred env "
            f"'{inferred_env}' from dataset '{dataset_name}'."
        )

    print(
        f"Discovered {len(run_dirs)} run directories and {len(final_dirs)} final "
        f"directories under {args.runs_root}"
    )
    for idx, run_dir in enumerate(run_dirs, start=1):
        print(f"[{idx}/{len(run_dirs)}] {run_dir}")
        if args.dry_run:
            continue

        try:
            avg_metrics = evaluate_run_dir(
                run_dir=run_dir,
                trajectories=args.trajectories,
                seed_start=args.seed_start,
                seed_stride=args.seed_stride,
                max_steps=args.max_steps,
                device_override=args.device,
            )
            avg_contact = avg_metrics.get("contact_timesteps", "N/A")
            print(
                f"  wrote {run_dir / 'avg_metric.json'} "
                f"and {run_dir / 'trajectory_metrics.json'} "
                f"(avg contact_timesteps={avg_contact})"
            )
        except Exception as exc:
            print(f"  [error] failed for {run_dir}: {exc}")

    if args.dry_run:
        return

    print(f"Aggregating {len(final_dirs)} final directories")
    for idx, final_dir in enumerate(final_dirs, start=1):
        print(f"[{idx}/{len(final_dirs)}] {final_dir}")
        try:
            avg_metrics = evaluate_final_dir(final_dir)
            avg_contact = avg_metrics.get("contact_timesteps", "N/A")
            print(
                f"  wrote {final_dir / 'avg_metrics.json'} "
                f"(avg contact_timesteps={avg_contact})"
            )
        except Exception as exc:
            print(f"  [error] failed for {final_dir}: {exc}")

    print(
        "Computing trajectory-level policy significance per observation type: "
        f"{args.compare_policy_a} vs {args.compare_policy_b} "
        f"(test={args.significance_test})"
    )
    try:
        out_paths = compare_policies_by_observation_trajectory_level(
            runs_root=args.runs_root,
            obs_filters=obs_filters,
            policy_a=args.compare_policy_a,
            policy_b=args.compare_policy_b,
            significance_test=args.significance_test,
            min_samples_per_policy=args.min_samples_per_policy,
        )
        if out_paths:
            for path in out_paths:
                print(f"  wrote {path}")
        else:
            print("  [warn] no matching observation directories for policy comparison")
    except Exception as exc:
        print(f"  [error] failed trajectory-level significance computation: {exc}")


if __name__ == "__main__":
    main()