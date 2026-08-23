#!/usr/bin/env python3
"""Evaluate CW10 task-prefix checkpoints and summarize CW3/CW6 metrics.

The original continual runs evaluate every learned task only after the complete
CW10 sequence. This script reconstructs the same final-evaluation protocol at
intermediate task checkpoints without retraining the agents. Results are
appended to disk after every task/round so interrupted evaluations can resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from rliable import library as rly
from rliable import metrics as rliable_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import make_cw_env
from evaluation import EvaluationConfig, SACEvaluator, summarize_continual_run
from methods import get_method
from utils import load_sac_checkpoint


METHOD_PATTERNS = {
    "CloneX-SAC": "clonex_sac_cw10_v3_v1_500k_seed{seed}",
    "Adaptive PCGrad": (
        "success_replay_best_adaptive_pcgrad_cw10_v3_v1_500k_seed{seed}"
    ),
    "Frozen Transfer PCGrad": (
        "semantic_routed_frozen_transfer_pcgrad_cw10_v3_v1_500k_seed{seed}"
    ),
}

EVALUATION_FIELDS = (
    "method",
    "method_id",
    "seed",
    "prefix",
    "round_index",
    "task_index",
    "task_name",
    "success_rate",
    "average_return",
    "episodes",
    "checkpoint",
)

METRIC_FIELDS = (
    "method",
    "method_id",
    "seed",
    "prefix",
    "average_performance",
    "average_forgetting",
    "raw_forward_transfer",
    "normalized_forward_transfer",
    "area_forward_transfer",
)

PER_TASK_FIELDS = (
    "method",
    "method_id",
    "seed",
    "prefix",
    "task_index",
    "task_name",
    "final_success",
    "end_of_task_success",
    "forgetting",
    "raw_forward_transfer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/cw10_continual"),
    )
    parser.add_argument(
        "--baseline-curves",
        type=Path,
        default=Path(
            "outputs/single_task_baselines/"
            "cw10_v3_v1_500k_seeds1_2/aggregate/baseline_curves.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/cw_prefix_all_available"),
    )
    parser.add_argument("--prefixes", type=int, nargs="+", default=[3, 6])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_PATTERNS),
        default=list(METHOD_PATTERNS),
        help="Method display names to evaluate; defaults to all methods.",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--episodes-per-round", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=50_000)
    parser.add_argument("--random-seed", type=int, default=20260822)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Recompute tables from the cached checkpoint evaluations only.",
    )
    args = parser.parse_args()
    if not args.prefixes or any(prefix <= 1 or prefix > 10 for prefix in args.prefixes):
        parser.error("prefixes must be integers in [2, 10].")
    if args.rounds <= 0 or args.episodes_per_round <= 0:
        parser.error("rounds and episodes-per-round must be positive.")
    if args.bootstrap_reps <= 0:
        parser.error("bootstrap-reps must be positive.")
    return args


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def discover_runs(
    input_dir: Path,
    selected_methods: list[str],
) -> list[tuple[str, int, Path]]:
    runs: list[tuple[str, int, Path]] = []
    for method_name, pattern in METHOD_PATTERNS.items():
        if method_name not in selected_methods:
            continue
        glob_pattern = pattern.format(seed="*")
        for run_dir in input_dir.glob(glob_pattern):
            match = re.search(r"seed(\d+)$", run_dir.name)
            if match is None or not (run_dir / "summary.json").is_file():
                continue
            runs.append((method_name, int(match.group(1)), run_dir))
    return sorted(runs, key=lambda item: (item[0], item[1]))


def build_agent(config: dict[str, Any], device: str) -> Any:
    # Historical configs predate this default, but the parameter does not
    # alter any checkpoint tensor or evaluation-time actor behavior.
    effective = dict(config)
    effective.setdefault("bc_combination_strategy", "average")
    effective.setdefault("bc_adaptive_target_ratio", 0.2)
    effective.setdefault("bc_adaptive_conflict_ratio", 0.05)
    effective.setdefault("bc_cagrad_alpha", 0.5)
    effective["device"] = device
    args = SimpleNamespace(**effective)
    method = get_method(str(config["method"]))
    tasks = list(config["tasks"])
    env = make_cw_env(
        tasks[0],
        seed=int(config["seed"]),
        max_episode_steps=int(config["max_episode_steps"]),
        append_task_id=method.append_task_id,
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
        num_task_ids=len(tasks),
    )
    try:
        return method.build_agent(
            args=args,
            observation_dim=int(env.observation_space.shape[0]),
            action_dim=int(env.action_space.shape[0]),
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            total_tasks=len(tasks),
        )
    finally:
        env.close()


def load_cached_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def evaluation_key(row: dict[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(row["method_id"]),
        int(row["seed"]),
        int(row["prefix"]),
        int(row["round_index"]),
        int(row["task_index"]),
    )


def evaluate_missing_checkpoints(
    *,
    runs: list[tuple[str, int, Path]],
    prefixes: list[int],
    rounds: int,
    episodes_per_round: int,
    device: str,
    cache_path: Path,
) -> None:
    cached = load_cached_rows(cache_path)
    completed = {evaluation_key(row) for row in cached}
    for method_name, seed, run_dir in runs:
        config = read_json(run_dir / "config.json")
        tasks = list(config["tasks"])
        method_id = str(config["method"])
        method = get_method(method_id)
        agent = build_agent(config, device)
        for prefix in prefixes:
            checkpoint = run_dir / "checkpoints" / f"task_{prefix - 1}.pt"
            if not checkpoint.is_file():
                print(f"[prefix-eval] skip missing checkpoint: {checkpoint}", flush=True)
                continue
            load_sac_checkpoint(
                agent=agent,
                path=checkpoint,
                map_location=device,
                load_optimizer=False,
            )
            for round_index in range(rounds):
                eval_seed = seed + 100_000 + round_index
                for task_index, task_name in enumerate(tasks[:prefix]):
                    key = (method_id, seed, prefix, round_index, task_index)
                    if key in completed:
                        continue
                    agent.on_evaluation_start(task_index)
                    env = make_cw_env(
                        task_name,
                        seed=eval_seed,
                        max_episode_steps=int(config["max_episode_steps"]),
                        append_task_id=method.append_task_id,
                        env_version=str(config["env_version"]),
                        reward_function_version=str(config["reward_function_version"]),
                        num_task_ids=len(tasks),
                    )
                    try:
                        result = SACEvaluator(
                            env,
                            agent,
                            EvaluationConfig(
                                num_episodes=episodes_per_round,
                                max_episode_steps=int(config["max_episode_steps"]),
                                seed=eval_seed,
                            ),
                        ).evaluate(deterministic=False)
                    finally:
                        env.close()
                        agent.on_evaluation_end(task_index)
                    row = {
                        "method": method_name,
                        "method_id": method_id,
                        "seed": seed,
                        "prefix": prefix,
                        "round_index": round_index,
                        "task_index": task_index,
                        "task_name": task_name,
                        "success_rate": result.success_rate,
                        "average_return": result.mean_return,
                        "episodes": episodes_per_round,
                        "checkpoint": str(checkpoint),
                    }
                    append_row(cache_path, row)
                    completed.add(key)
                    print(
                        f"[prefix-eval] method={method_name} seed={seed} "
                        f"cw={prefix} round={round_index + 1}/{rounds} "
                        f"task={task_name} success={result.success_rate:.3f}",
                        flush=True,
                    )


def load_active_curves(run_dir: Path, prefix: int) -> list[list[float]]:
    rows: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with (run_dir / "evaluations.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            active_index = int(row["active_task_index"])
            eval_index = int(row["evaluation_task_index"])
            if active_index == eval_index and active_index < prefix:
                rows[active_index].append(
                    (int(row["active_task_step"]), float(row["stochastic_success_rate"]))
                )
    curves: list[list[float]] = []
    for task_index in range(prefix):
        ordered = sorted(rows[task_index], key=lambda item: item[0])
        # Prefix tasks never include the repeated CW10 final evaluations.
        if len(ordered) != 25:
            raise ValueError(
                f"Expected 25 active evaluations for task {task_index} in "
                f"{run_dir}, found {len(ordered)}."
            )
        curves.append([value for _, value in ordered])
    return curves


def compute_run_metrics(
    *,
    runs: list[tuple[str, int, Path]],
    prefixes: list[int],
    rounds: int,
    cache_rows: list[dict[str, str]],
    baseline_curves: list[list[float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in cache_rows:
        grouped[(row["method_id"], int(row["seed"]), int(row["prefix"]))].append(row)
    metric_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    for method_name, seed, run_dir in runs:
        config = read_json(run_dir / "config.json")
        method_id = str(config["method"])
        for prefix in prefixes:
            rows = grouped.get((method_id, seed, prefix), [])
            expected = rounds * prefix
            if len(rows) != expected:
                continue
            final_rows: list[list[float]] = []
            for round_index in range(rounds):
                round_rows = sorted(
                    (row for row in rows if int(row["round_index"]) == round_index),
                    key=lambda row: int(row["task_index"]),
                )
                if len(round_rows) != prefix:
                    raise ValueError(
                        f"Incomplete cached round: {method_id}, seed={seed}, "
                        f"CW{prefix}, round={round_index}."
                    )
                final_rows.append([float(row["success_rate"]) for row in round_rows])
            active_curves = load_active_curves(run_dir, prefix)
            summary = summarize_continual_run(
                final_success_rows=final_rows,
                active_task_success_curves=active_curves,
                baseline_success_curves=baseline_curves[:prefix],
                tail_size=rounds,
            )
            metric_rows.append(
                {
                    "method": method_name,
                    "method_id": method_id,
                    "seed": seed,
                    "prefix": prefix,
                    "average_performance": summary["average_performance"],
                    "average_forgetting": summary["average_forgetting"],
                    "raw_forward_transfer": summary["raw_forward_transfer"],
                    "normalized_forward_transfer": summary["forward_transfer"],
                    "area_forward_transfer": summary["area_forward_transfer"],
                }
            )
            for task_index, task_name in enumerate(list(config["tasks"])[:prefix]):
                per_task_rows.append(
                    {
                        "method": method_name,
                        "method_id": method_id,
                        "seed": seed,
                        "prefix": prefix,
                        "task_index": task_index,
                        "task_name": task_name,
                        "final_success": summary["final_per_task"][task_index],
                        "end_of_task_success": summary["end_of_task_per_task"][task_index],
                        "forgetting": summary["forgetting_per_task"][task_index],
                        "raw_forward_transfer": summary["raw_forward_transfer_per_task"][task_index],
                    }
                )
    return metric_rows, per_task_rows


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iqm_intervals(
    values_by_method: dict[str, list[float]], *, reps: int, random_seed: int
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    scores = {
        method: np.asarray(values, dtype=np.float64)[:, None]
        for method, values in values_by_method.items()
    }
    points, intervals = rly.get_interval_estimates(
        scores,
        lambda x: np.asarray([rliable_metrics.aggregate_iqm(x)]),
        reps=reps,
        confidence_interval_size=0.95,
        random_state=np.random.RandomState(random_seed),
    )
    point_values = {
        method: float(np.asarray(value)[0]) for method, value in points.items()
    }
    interval_values = {
        method: (
            float(np.asarray(intervals[method])[0, 0]),
            float(np.asarray(intervals[method])[1, 0]),
        )
        for method in points
    }
    return point_values, interval_values


def aggregate_rows(
    metric_rows: list[dict[str, Any]], *, reps: int, random_seed: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = ("average_performance", "raw_forward_transfer", "average_forgetting")
    prefixes = sorted({int(row["prefix"]) for row in metric_rows})
    for scope in ("all_available", "matched"):
        for prefix in prefixes:
            prefix_rows = [row for row in metric_rows if int(row["prefix"]) == prefix]
            seeds_by_method = {
                method: {int(row["seed"]) for row in prefix_rows if row["method"] == method}
                for method in METHOD_PATTERNS
            }
            matched_seeds = set.intersection(*seeds_by_method.values())
            selected_by_method: dict[str, list[dict[str, Any]]] = {}
            for method in METHOD_PATTERNS:
                selected = [row for row in prefix_rows if row["method"] == method]
                if scope == "matched":
                    selected = [row for row in selected if int(row["seed"]) in matched_seeds]
                selected_by_method[method] = selected
            for metric_index, metric in enumerate(metrics):
                values_by_method = {
                    method: [float(row[metric]) for row in selected]
                    for method, selected in selected_by_method.items()
                    if selected
                }
                iqms, intervals = iqm_intervals(
                    values_by_method,
                    reps=reps,
                    random_seed=random_seed + prefix * 100 + metric_index,
                )
                for method in METHOD_PATTERNS:
                    selected = selected_by_method[method]
                    values = values_by_method.get(method, [])
                    if not values:
                        continue
                    ci_low, ci_high = intervals[method]
                    output.append(
                        {
                            "scope": scope,
                            "prefix": prefix,
                            "method": method,
                            "metric": metric,
                            "num_seeds": len(values),
                            "seeds": " ".join(str(int(row["seed"])) for row in selected),
                            "mean": float(np.mean(values)),
                            "iqm": iqms[method],
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                        }
                    )
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "checkpoint_evaluations.csv"
    runs = discover_runs(args.input_dir, args.methods)
    if not runs:
        raise FileNotFoundError(f"No completed runs found in {args.input_dir}")
    if not args.metrics_only:
        evaluate_missing_checkpoints(
            runs=runs,
            prefixes=sorted(set(args.prefixes)),
            rounds=args.rounds,
            episodes_per_round=args.episodes_per_round,
            device=args.device,
            cache_path=cache_path,
        )
    cache_rows = load_cached_rows(cache_path)
    baseline_payload = read_json(args.baseline_curves)
    baseline_curves = baseline_payload.get("stochastic_success_curves")
    if not isinstance(baseline_curves, list):
        raise ValueError("Baseline JSON has no stochastic_success_curves list.")
    metric_rows, per_task_rows = compute_run_metrics(
        runs=runs,
        prefixes=sorted(set(args.prefixes)),
        rounds=args.rounds,
        cache_rows=cache_rows,
        baseline_curves=baseline_curves,
    )
    write_csv(args.output_dir / "prefix_run_metrics.csv", METRIC_FIELDS, metric_rows)
    write_csv(
        args.output_dir / "prefix_per_task_metrics.csv",
        PER_TASK_FIELDS,
        per_task_rows,
    )
    aggregates = aggregate_rows(
        metric_rows,
        reps=args.bootstrap_reps,
        random_seed=args.random_seed,
    )
    aggregate_fields = (
        "scope",
        "prefix",
        "method",
        "metric",
        "num_seeds",
        "seeds",
        "mean",
        "iqm",
        "ci_low",
        "ci_high",
    )
    write_csv(args.output_dir / "prefix_aggregate_metrics.csv", aggregate_fields, aggregates)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "completed_runs": len(metric_rows),
                "discovered_runs": len(runs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
