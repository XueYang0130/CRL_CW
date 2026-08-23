#!/usr/bin/env python3
"""Aggregate the official RECALL CW3 grid with rliable statistics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rliable import library as rly
from rliable import metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from envs import CW3_TASK_SEQUENCES, get_continual_task_sequence  # noqa: E402


METRICS = {
    "average_performance": "average_performance",
    "average_return": "average_return",
    "average_forgetting": "average_forgetting",
    "forward_transfer": "forward_transfer",
    "raw_forward_transfer": "raw_forward_transfer",
    "area_forward_transfer": "area_forward_transfer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("outputs/cw10_continual")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/recall_cw3_v3_v1_500k_seeds1_5"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--sequences", nargs="+", default=list(CW3_TASK_SEQUENCES)
    )
    parser.add_argument("--run-pattern", default="recall_{sequence}_v3_v1_500k_seed{seed}")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260823)
    return parser.parse_args()


def validate_summary(summary: dict, path: Path, sequence: str) -> None:
    if summary.get("method") != "recall":
        raise ValueError(f"Expected method=recall in {path}, got {summary.get('method')!r}")
    expected_tasks = list(get_continual_task_sequence(sequence, "v3"))
    if summary.get("tasks") != expected_tasks:
        raise ValueError(
            f"Task order mismatch in {path}: expected {expected_tasks}, "
            f"got {summary.get('tasks')}"
        )


def load_grid(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[dict]]:
    unknown = sorted(set(args.sequences) - set(CW3_TASK_SEQUENCES))
    if unknown:
        raise ValueError(f"Unknown CW3 sequences: {unknown}")

    values = {
        metric: np.empty((len(args.seeds), len(args.sequences)), dtype=np.float64)
        for metric in METRICS
    }
    rows: list[dict] = []
    for seed_index, seed in enumerate(args.seeds):
        for sequence_index, sequence in enumerate(args.sequences):
            run_name = args.run_pattern.format(sequence=sequence, seed=seed)
            summary_path = args.input_dir / run_name / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing CW3 result: {summary_path}")
            with summary_path.open() as handle:
                summary = json.load(handle)
            validate_summary(summary, summary_path, sequence)

            row = {"seed": seed, "sequence": sequence, "run_name": run_name}
            for metric, field in METRICS.items():
                value = summary.get(field)
                if value is None or not np.isfinite(value):
                    raise ValueError(f"Invalid {field} in {summary_path}: {value}")
                values[metric][seed_index, sequence_index] = float(value)
                row[metric] = float(value)
            rows.append(row)
    return values, rows


def aggregate_grid(
    values: dict[str, np.ndarray], bootstrap_reps: int, random_seed: int
) -> dict:
    aggregate = lambda x: np.asarray(
        [
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_median(x),
        ]
    )
    results = {}
    for index, (metric, score_matrix) in enumerate(values.items()):
        points, intervals = rly.get_interval_estimates(
            {"RECALL": score_matrix},
            aggregate,
            reps=bootstrap_reps,
            confidence_interval_size=0.95,
            random_state=np.random.RandomState(random_seed + index),
        )
        point = np.asarray(points["RECALL"])
        interval = np.asarray(intervals["RECALL"])
        results[metric] = {
            "iqm": float(point[0]),
            "iqm_ci_low": float(interval[0, 0]),
            "iqm_ci_high": float(interval[1, 0]),
            "mean": float(point[1]),
            "median": float(point[2]),
        }
    return results


def save_results(
    args: argparse.Namespace,
    values: dict[str, np.ndarray],
    rows: list[dict],
    aggregate: dict,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_metrics.csv").open("w", newline="") as handle:
        fieldnames = ["seed", "sequence", "run_name", *METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_dir / "aggregate_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "iqm",
                "iqm_ci_low",
                "iqm_ci_high",
                "mean",
                "median",
            ],
        )
        writer.writeheader()
        for metric, result in aggregate.items():
            writer.writerow({"metric": metric, **result})

    report = {
        "method": "recall",
        "seeds": args.seeds,
        "sequences": args.sequences,
        "score_shape": [len(args.seeds), len(args.sequences)],
        "bootstrap_reps": args.bootstrap_reps,
        "confidence_interval": 0.95,
        "aggregation": "rliable stratified bootstrap over the seed-by-sequence grid",
        "aggregate_metrics": aggregate,
        "score_matrices": {metric: matrix.tolist() for metric, matrix in values.items()},
    }
    with (args.output_dir / "rliable_results.json").open("w") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    args = parse_args()
    values, rows = load_grid(args)
    aggregate = aggregate_grid(values, args.bootstrap_reps, args.random_seed)
    save_results(args, values, rows, aggregate)
    print(json.dumps(aggregate, indent=2))
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
