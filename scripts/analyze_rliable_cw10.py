#!/usr/bin/env python3
"""Analyze completed CW10 runs with the official rliable statistics API."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


METHODS = {
    "CloneX-SAC": "clonex_sac_cw10_v3_v1_500k_seed{seed}",
    "Adaptive PCGrad": "success_replay_best_adaptive_pcgrad_cw10_v3_v1_500k_seed{seed}",
    "Frozen Transfer PCGrad": "semantic_routed_frozen_transfer_pcgrad_cw10_v3_v1_500k_seed{seed}",
}

METRICS = {
    "average_performance": "average_performance",
    "forward_transfer": "raw_forward_transfer",
    "forgetting": "average_forgetting",
}

COLORS = {
    "CloneX-SAC": "#C79A20",
    "Adaptive PCGrad": "#A24E9A",
    "Frozen Transfer PCGrad": "#2E8B57",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/cw10_continual"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/rliable_cw10_seeds0_5"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--bootstrap-reps", type=int, default=50_000)
    parser.add_argument("--profile-reps", type=int, default=5_000)
    parser.add_argument("--random-seed", type=int, default=20260818)
    return parser.parse_args()


def load_scores(input_dir: Path, seeds: list[int]) -> dict[str, dict[str, np.ndarray]]:
    scores = {metric: {} for metric in METRICS}
    for method, pattern in METHODS.items():
        values = {metric: [] for metric in METRICS}
        for seed in seeds:
            summary_path = input_dir / pattern.format(seed=seed) / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing matched-seed result: {summary_path}")
            with summary_path.open() as handle:
                summary = json.load(handle)
            for metric, field in METRICS.items():
                value = summary.get(field)
                if value is None or not np.isfinite(value):
                    raise ValueError(f"Invalid {field} in {summary_path}: {value}")
                values[metric].append(float(value))
        for metric in METRICS:
            # rliable expects (num_runs, num_tasks). Here the existing CW10
            # sequence-level metric is one benchmark score per run.
            scores[metric][method] = np.asarray(values[metric], dtype=np.float64)[:, None]
    return scores


def aggregate_statistics(
    score_dict: dict[str, np.ndarray], reps: int, random_seed: int
) -> tuple[dict, dict]:
    aggregate = lambda x: np.asarray(
        [
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_median(x),
        ]
    )
    return rly.get_interval_estimates(
        score_dict,
        aggregate,
        reps=reps,
        confidence_interval_size=0.95,
        random_state=np.random.RandomState(random_seed),
    )


def improvement_statistics(
    score_dict: dict[str, np.ndarray], reps: int, random_seed: int
) -> tuple[dict, dict]:
    pairs = {
        f"{method},CloneX-SAC": (values, score_dict["CloneX-SAC"])
        for method, values in score_dict.items()
        if method != "CloneX-SAC"
    }
    return rly.get_interval_estimates(
        pairs,
        metrics.probability_of_improvement,
        reps=reps,
        confidence_interval_size=0.95,
        random_state=np.random.RandomState(random_seed),
    )


def serializable_array(value) -> list | float:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array.tolist()


def save_tables(output_dir: Path, aggregate_results: dict, improvement_results: dict) -> None:
    with (output_dir / "aggregate_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["metric", "method", "iqm", "iqm_ci_low", "iqm_ci_high", "mean", "median"]
        )
        for metric, result in aggregate_results.items():
            points, intervals = result
            for method in METHODS:
                point = np.asarray(points[method])
                ci = np.asarray(intervals[method])
                writer.writerow(
                    [metric, method, point[0], ci[0, 0], ci[1, 0], point[1], point[2]]
                )

    with (output_dir / "probability_of_improvement.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "comparison", "probability", "ci_low", "ci_high"])
        for metric, result in improvement_results.items():
            points, intervals = result
            for comparison, value in points.items():
                ci = np.asarray(intervals[comparison]).reshape(2, -1)
                writer.writerow(
                    [metric, comparison, float(value), float(ci[0, 0]), float(ci[1, 0])]
                )


def plot_aggregate(output_dir: Path, aggregate_results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=180)
    titles = {
        "average_performance": "Average performance",
        "forward_transfer": "Forward transfer",
        "forgetting": "Forgetting (lower is better)",
    }
    for ax, metric in zip(axes, METRICS):
        points, intervals = aggregate_results[metric]
        for row, method in enumerate(METHODS):
            point = float(np.asarray(points[method])[0])
            ci = np.asarray(intervals[method])[:, 0]
            ax.errorbar(
                point,
                row,
                xerr=[[point - ci[0]], [ci[1] - point]],
                fmt="o",
                color=COLORS[method],
                capsize=4,
                markersize=7,
            )
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(list(METHODS) if ax is axes[0] else [])
        ax.set_title(titles[metric])
        ax.set_xlabel("IQM with 95% CI")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "iqm_interval_estimates.png", bbox_inches="tight")
    plt.close(fig)


def plot_improvement(output_dir: Path, improvement_results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=180)
    titles = {
        "average_performance": "Average performance",
        "forward_transfer": "Forward transfer",
        "forgetting": "Lower forgetting",
    }
    for ax, metric in zip(axes, METRICS):
        points, intervals = improvement_results[metric]
        plot_utils.plot_probability_of_improvement(
            points,
            intervals,
            ax=ax,
            colors=[COLORS[key.split(",")[0]] for key in points],
            xlabel="Probability of improvement over CloneX-SAC",
        )
        ax.set_title(titles[metric])
        ax.axvline(0.5, color="#666666", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "probability_of_improvement.png", bbox_inches="tight")
    plt.close(fig)


def plot_profiles(
    output_dir: Path,
    scores: dict[str, dict[str, np.ndarray]],
    profile_reps: int,
    random_seed: int,
) -> dict:
    np.random.seed(random_seed)
    settings = {
        "average_performance": np.linspace(0.60, 1.00, 81),
        "forward_transfer": np.linspace(0.00, 0.30, 61),
    }
    profiles = {}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180)
    for ax, (metric, thresholds) in zip(axes, settings.items()):
        profile, ci = rly.create_performance_profile(
            scores[metric],
            thresholds,
            reps=profile_reps,
            confidence_interval_size=0.95,
        )
        plot_utils.plot_performance_profiles(
            profile,
            thresholds,
            performance_profile_cis=ci,
            colors=COLORS,
            ax=ax,
            xlabel=metric.replace("_", " ").title() + " threshold",
        )
        profiles[metric] = {
            "thresholds": thresholds.tolist(),
            "point_estimates": {k: serializable_array(v) for k, v in profile.items()},
            "confidence_intervals": {k: serializable_array(v) for k, v in ci.items()},
        }
    fig.tight_layout()
    fig.savefig(output_dir / "performance_profiles.png", bbox_inches="tight")
    plt.close(fig)
    return profiles


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_scores(args.input_dir, args.seeds)

    aggregate_results = {}
    improvement_results = {}
    for index, metric in enumerate(METRICS):
        aggregate_results[metric] = aggregate_statistics(
            scores[metric], args.bootstrap_reps, args.random_seed + index
        )
        comparison_scores = scores[metric]
        if metric == "forgetting":
            comparison_scores = {method: -values for method, values in scores[metric].items()}
        improvement_results[metric] = improvement_statistics(
            comparison_scores, args.bootstrap_reps, args.random_seed + 100 + index
        )

    profiles = plot_profiles(
        args.output_dir, scores, args.profile_reps, args.random_seed + 200
    )
    save_tables(args.output_dir, aggregate_results, improvement_results)
    plot_aggregate(args.output_dir, aggregate_results)
    plot_improvement(args.output_dir, improvement_results)

    report = {
        "methodology": {
            "library": "rliable==1.2.0",
            "seeds": args.seeds,
            "score_shape": [len(args.seeds), 1],
            "bootstrap_reps": args.bootstrap_reps,
            "profile_reps": args.profile_reps,
            "confidence_interval": 0.95,
            "note": "Existing sequence-level CRL metrics are retained without task-level redefinition.",
        },
        "aggregate_metrics": {
            metric: {
                "point_estimates": {k: serializable_array(v) for k, v in result[0].items()},
                "confidence_intervals": {k: serializable_array(v) for k, v in result[1].items()},
            }
            for metric, result in aggregate_results.items()
        },
        "probability_of_improvement": {
            metric: {
                "point_estimates": {k: serializable_array(v) for k, v in result[0].items()},
                "confidence_intervals": {k: serializable_array(v) for k, v in result[1].items()},
            }
            for metric, result in improvement_results.items()
        },
        "performance_profiles": profiles,
    }
    with (args.output_dir / "rliable_results.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report["methodology"], indent=2))
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
