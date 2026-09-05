#!/usr/bin/env python3
"""Create current paper-style result and diagnostic figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "CloneX-SAC": {
        "pattern": "clonex_sac_cw10_v3_v1_500k_seed{seed}",
        "seeds": [1, 2, 3, 4, 5],
        "color": "#C79A20",
    },
    "Adaptive PCGrad": {
        "pattern": "success_replay_best_adaptive_pcgrad_cw10_v3_v1_500k_seed{seed}",
        "seeds": [1, 2, 3, 4, 5],
        "color": "#A24E9A",
    },
    "Frozen Transfer PCGrad": {
        "pattern": "semantic_routed_frozen_transfer_pcgrad_cw10_v3_v1_500k_seed{seed}",
        "seeds": [1, 2, 3, 4, 5],
        "color": "#2E8B57",
    },
    "RECALL (seed1)": {
        "pattern": "recall_cw10_v3_v1_500k_seed{seed}",
        "seeds": [1],
        "color": "#4D4D4D",
    },
}

TASKS = [
    "hammer-v3",
    "push-wall-v3",
    "faucet-close-v3",
    "push-back-v3",
    "stick-pull-v3",
    "handle-press-side-v3",
    "push-v3",
    "shelf-place-v3",
    "window-close-v3",
    "peg-unplug-side-v3",
]

METHOD_ORDER = ["CloneX-SAC", "Adaptive PCGrad", "Frozen Transfer PCGrad", "RECALL (seed1)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/cw10_continual"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/paper_figures_current"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260825)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--font-scale", type=float, default=1.15)
    return parser.parse_args()


def iqm(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    sorted_values = np.sort(values.reshape(-1))
    trim = int(np.floor(0.25 * sorted_values.size))
    if trim == 0:
        return float(sorted_values.mean())
    return float(sorted_values[trim:-trim].mean())


def bootstrap_iqm(values: np.ndarray, reps: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    point = iqm(values)
    if values.size <= 1:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(reps, values.size), replace=True)
    estimates = np.asarray([iqm(sample) for sample in samples], dtype=np.float64)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return point, float(low), float(high)


def run_dir(input_dir: Path, method: str, seed: int) -> Path:
    return input_dir / METHODS[method]["pattern"].format(seed=seed)


def load_summary(input_dir: Path, method: str, seed: int) -> dict:
    path = run_dir(input_dir, method, seed) / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_eval_curve(input_dir: Path, method: str, seed: int) -> dict[str, list[tuple[int, float]]]:
    path = run_dir(input_dir, method, seed) / "evaluations.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    curves = {task: [] for task in TASKS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["active_task_name"] != row["evaluation_task_name"]:
                continue
            task = row["active_task_name"]
            if task in curves:
                curves[task].append(
                    (int(float(row["active_task_step"])), float(row["stochastic_success_rate"]))
                )
    return curves


def time_to_success(
    input_dir: Path, method: str, seed: int, threshold: float, cap: int = 520_000
) -> dict[str, int]:
    curves = load_eval_curve(input_dir, method, seed)
    result: dict[str, int] = {}
    for task, points in curves.items():
        reached = [step for step, success in points if success >= threshold]
        result[task] = min(reached) if reached else cap
    return result


def gradient_by_task(input_dir: Path, method: str, seed: int) -> dict[str, dict[str, float]]:
    path = run_dir(input_dir, method, seed) / "gradient_diagnostics" / "gradient_windows.csv"
    if not path.is_file():
        return {}
    rows_by_task: dict[str, list[dict[str, str]]] = {task: [] for task in TASKS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scope") != "shared_actor_backbone":
                continue
            task = row.get("current_task_name", "")
            if task in rows_by_task:
                rows_by_task[task].append(row)
    stats: dict[str, dict[str, float]] = {}
    for task, rows in rows_by_task.items():
        if not rows:
            continue
        ratios = np.asarray([float(row["bc_to_sac_norm_ratio"]) for row in rows], dtype=np.float64)
        conflicts = np.asarray([1.0 if row["conflict"] == "True" else 0.0 for row in rows])
        cosines = np.asarray([float(row["cosine_similarity"]) for row in rows], dtype=np.float64)
        stats[task] = {
            "bc_sac_ratio": float(np.nanmean(ratios)),
            "conflict_rate": float(np.nanmean(conflicts)),
            "cosine": float(np.nanmean(cosines)),
        }
    return stats


def metric_values(input_dir: Path, metric: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    field = {
        "average_performance": "average_performance",
        "forward_transfer": "raw_forward_transfer",
        "forgetting": "average_forgetting",
        "elapsed_hours": "elapsed_seconds",
    }[metric]
    for method in METHOD_ORDER:
        values[method] = []
        for seed in METHODS[method]["seeds"]:
            summary = load_summary(input_dir, method, seed)
            value = float(summary[field])
            if metric == "elapsed_hours":
                value /= 3600.0
            values[method].append(value)
    return values


def aggregate_table(input_dir: Path, reps: int, seed: int) -> dict[str, dict[str, tuple[float, float, float]]]:
    table: dict[str, dict[str, tuple[float, float, float]]] = {}
    for metric in ("average_performance", "forward_transfer", "forgetting", "elapsed_hours"):
        table[metric] = {}
        for index, (method, values) in enumerate(metric_values(input_dir, metric).items()):
            table[metric][method] = bootstrap_iqm(
                np.asarray(values), reps=reps, seed=seed + 100 * index + len(metric)
            )
    return table


def plot_scatter(
    output_dir: Path,
    table: dict[str, dict[str, tuple[float, float, float]]],
    x_metric: str,
    y_metric: str,
    path_name: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.5), dpi=180)
    label_offsets = {
        "CloneX-SAC": (8, -14),
        "Adaptive PCGrad": (8, 8),
        "Frozen Transfer PCGrad": (8, 14),
        "RECALL (seed1)": (8, 8),
    }
    short_labels = {
        "CloneX-SAC": "CloneX",
        "Adaptive PCGrad": "Adaptive",
        "Frozen Transfer PCGrad": "Frozen",
        "RECALL (seed1)": "RECALL",
    }
    for method in METHOD_ORDER:
        color = METHODS[method]["color"]
        x, x_low, x_high = table[x_metric][method]
        y, y_low, y_high = table[y_metric][method]
        xerr = None if np.isnan(x_low) else [[x - x_low], [x_high - x]]
        yerr = None if np.isnan(y_low) else [[y - y_low], [y_high - y]]
        marker = "D" if method == "RECALL (seed1)" else "o"
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            color=color,
            markersize=8,
            capsize=4,
            linewidth=1.5,
            label=method,
        )
        ax.annotate(
            short_labels[method],
            (x, y),
            xytext=label_offsets[method],
            textcoords="offset points",
            fontsize=10,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / path_name, bbox_inches="tight")
    plt.close(fig)


def plot_time_to_success_heatmap(args: argparse.Namespace) -> None:
    matrix = []
    labels = []
    for method in METHOD_ORDER:
        rows = []
        for task in TASKS:
            values = [
                time_to_success(args.input_dir, method, seed, args.success_threshold)[task] / 1000.0
                for seed in METHODS[method]["seeds"]
            ]
            rows.append(iqm(np.asarray(values)))
        matrix.append(rows)
        labels.append(method)
    matrix_array = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(13.4, 4.9), dpi=180)
    im = ax.imshow(matrix_array, cmap="viridis_r", vmin=20, vmax=520, aspect="auto")
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([task.replace("-v3", "") for task in TASKS], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(f"Time to {args.success_threshold:.1f} success (k steps); lower is better")
    for row in range(matrix_array.shape[0]):
        for col in range(matrix_array.shape[1]):
            value = matrix_array[row, col]
            text = ">500" if value >= 520 else f"{value:.0f}"
            ax.text(col, row, text, ha="center", va="center", color="white" if value > 300 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Task steps (k)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "04_time_to_success_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_gradient_dominance(args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.0), dpi=180)
    x = np.arange(len(TASKS[1:]))
    width = 0.18
    for method_index, method in enumerate(METHOD_ORDER):
        values = []
        for task in TASKS[1:]:
            per_seed = []
            for seed in METHODS[method]["seeds"]:
                stats = gradient_by_task(args.input_dir, method, seed)
                if task in stats:
                    per_seed.append(stats[task]["bc_sac_ratio"])
            values.append(iqm(np.asarray(per_seed)) if per_seed else np.nan)
        offset = (method_index - 1.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            color=METHODS[method]["color"],
            alpha=0.82,
            label=method,
        )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Raw BC / SAC gradient norm ratio (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([task.replace("-v3", "") for task in TASKS[1:]], rotation=35, ha="right")
    ax.set_title("Raw BC gradient dominance by task")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "06_gradient_dominance_by_task.png", bbox_inches="tight")
    plt.close(fig)


def plot_conflict_vs_delay(args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.8), dpi=180)
    for method in METHOD_ORDER:
        xs, ys = [], []
        for seed in METHODS[method]["seeds"]:
            gradients = gradient_by_task(args.input_dir, method, seed)
            delays = time_to_success(args.input_dir, method, seed, args.success_threshold)
            for task in TASKS[1:]:
                if task not in gradients:
                    continue
                xs.append(gradients[task]["conflict_rate"])
                ys.append(delays[task] / 1000.0)
        ax.scatter(
            xs,
            ys,
            color=METHODS[method]["color"],
            alpha=0.62,
            s=42 if method != "RECALL (seed1)" else 58,
            marker="D" if method == "RECALL (seed1)" else "o",
            label=method,
        )
    ax.set_xlabel("BC/SAC conflict rate")
    ax.set_ylabel(f"Time to {args.success_threshold:.1f} success (k steps)")
    ax.set_ylim(0, 540)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    ax.set_title("Gradient conflict vs acquisition delay")
    fig.tight_layout()
    fig.savefig(args.output_dir / "07_conflict_vs_acquisition_delay.png", bbox_inches="tight")
    plt.close(fig)


def plot_runtime_vs_performance(
    output_dir: Path, table: dict[str, dict[str, tuple[float, float, float]]]
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.5), dpi=180)
    for method in METHOD_ORDER:
        runtime, runtime_low, runtime_high = table["elapsed_hours"][method]
        performance, perf_low, perf_high = table["average_performance"][method]
        xerr = None if np.isnan(runtime_low) else [[runtime - runtime_low], [runtime_high - runtime]]
        yerr = None if np.isnan(perf_low) else [[performance - perf_low], [perf_high - performance]]
        ax.errorbar(
            runtime,
            performance,
            xerr=xerr,
            yerr=yerr,
            fmt="D" if method == "RECALL (seed1)" else "o",
            color=METHODS[method]["color"],
            capsize=4,
            markersize=8,
            label=method,
        )
        ax.annotate(method.replace(" PCGrad", ""), (runtime, performance), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Wall-clock time per CW10 seed (hours)")
    ax.set_ylabel("Average performance")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Runtime vs performance")
    fig.tight_layout()
    fig.savefig(output_dir / "08_runtime_vs_performance.png", bbox_inches="tight")
    plt.close(fig)


def final_success_by_task(input_dir: Path, method: str) -> dict[str, float]:
    result = {}
    for task in TASKS:
        values = []
        for seed in METHODS[method]["seeds"]:
            summary = load_summary(input_dir, method, seed)
            values.append(float(summary["final_per_task_success"][task]))
        result[task] = iqm(np.asarray(values))
    return result


def plot_win_loss_matrix(args: argparse.Namespace) -> None:
    baseline = final_success_by_task(args.input_dir, "CloneX-SAC")
    comparisons = ["Adaptive PCGrad", "Frozen Transfer PCGrad", "RECALL (seed1)"]
    matrix = []
    for method in comparisons:
        values = final_success_by_task(args.input_dir, method)
        matrix.append([values[task] - baseline[task] for task in TASKS])
    matrix_array = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(13.2, 3.9), dpi=180)
    im = ax.imshow(matrix_array, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([task.replace("-v3", "") for task in TASKS], rotation=35, ha="right")
    ax.set_yticks(range(len(comparisons)))
    ax.set_yticklabels([method.replace(" PCGrad", "") for method in comparisons])
    ax.set_title("Per-task final success difference vs CloneX-SAC")
    for row in range(matrix_array.shape[0]):
        for col in range(matrix_array.shape[1]):
            value = matrix_array[row, col]
            ax.text(
                col,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else "black",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Success difference")
    fig.tight_layout()
    fig.savefig(args.output_dir / "09_win_loss_matrix_vs_clonex.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10 * args.font_scale,
            "axes.titlesize": 12 * args.font_scale,
            "axes.labelsize": 11 * args.font_scale,
            "legend.fontsize": 9.5 * args.font_scale,
            "xtick.labelsize": 9 * args.font_scale,
            "ytick.labelsize": 9 * args.font_scale,
        }
    )
    table = aggregate_table(args.input_dir, args.bootstrap_reps, args.random_seed)
    plot_scatter(
        args.output_dir,
        table,
        "forward_transfer",
        "average_performance",
        "01_forward_transfer_vs_average_performance.png",
        "Raw forward transfer",
        "Average performance",
    )
    plot_scatter(
        args.output_dir,
        table,
        "forgetting",
        "average_performance",
        "02_forgetting_vs_average_performance.png",
        "Forgetting (lower is better)",
        "Average performance",
    )
    plot_time_to_success_heatmap(args)
    plot_gradient_dominance(args)
    plot_conflict_vs_delay(args)
    plot_runtime_vs_performance(args.output_dir, table)
    plot_win_loss_matrix(args)
    print(args.output_dir)


if __name__ == "__main__":
    main()
