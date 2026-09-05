#!/usr/bin/env python3
"""Plot per-task success learning curves with IQM CIs for matched CW10 runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "CloneX-SAC": "clonex_sac_cw10_v3_v1_500k_seed{seed}",
    "Adaptive PCGrad": "success_replay_best_adaptive_pcgrad_cw10_v3_v1_500k_seed{seed}",
    "Frozen Transfer PCGrad": "semantic_routed_frozen_transfer_pcgrad_cw10_v3_v1_500k_seed{seed}",
}

RECALL_PATTERN = "recall_cw10_v3_v1_500k_seed{seed}"

COLORS = {
    "CloneX-SAC": "#C79A20",
    "Adaptive PCGrad": "#A24E9A",
    "Frozen Transfer PCGrad": "#2E8B57",
    "RECALL (seed1)": "#444444",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/cw10_continual"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/task_learning_curves"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--recall-seed", type=int, default=1)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260825)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--font-scale", type=float, default=1.0)
    return parser.parse_args()


def read_curve(run_dir: Path) -> dict[str, dict[int, float]]:
    path = run_dir / "evaluations.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluations.csv: {path}")
    curves: dict[str, dict[int, float]] = {task: {} for task in TASKS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task_name = row["active_task_name"]
            if row["evaluation_task_name"] != task_name:
                continue
            if task_name not in curves:
                continue
            step = int(float(row["active_task_step"]))
            success = float(row["stochastic_success_rate"])
            curves[task_name][step] = success
    return curves


def matched_steps(method_curves: dict[str, list[dict[str, dict[int, float]]]]) -> list[int]:
    step_sets: list[set[int]] = []
    for curves_by_seed in method_curves.values():
        for task_curves in curves_by_seed:
            for task in TASKS:
                step_sets.append(set(task_curves[task]))
    if not step_sets:
        raise ValueError("No curves loaded.")
    return sorted(set.intersection(*step_sets))


def iqm_curve(
    seed_values: np.ndarray,
    reps: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    num_seeds, num_steps = seed_values.shape

    def iqm(values: np.ndarray) -> np.ndarray:
        sorted_values = np.sort(values, axis=0)
        trim = int(np.floor(0.25 * sorted_values.shape[0]))
        if trim == 0:
            return sorted_values.mean(axis=0)
        return sorted_values[trim:-trim].mean(axis=0)

    point = iqm(seed_values)
    sample_indices = rng.integers(0, num_seeds, size=(reps, num_seeds))
    bootstrap = np.empty((reps, num_steps), dtype=np.float64)
    for index, indices in enumerate(sample_indices):
        bootstrap[index] = iqm(seed_values[indices])
    low, high = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    return point, low, high


def load_method_curves(input_dir: Path, seeds: list[int]) -> dict[str, list[dict[str, dict[int, float]]]]:
    loaded: dict[str, list[dict[str, dict[int, float]]]] = {}
    for method, pattern in METHODS.items():
        loaded[method] = []
        for seed in seeds:
            loaded[method].append(read_curve(input_dir / pattern.format(seed=seed)))
    return loaded


def smooth_values(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    smoothed = np.empty_like(values, dtype=np.float64)
    radius = window // 2
    for index in range(values.shape[-1]):
        start = max(0, index - radius)
        end = min(values.shape[-1], index + radius + 1)
        smoothed[..., index] = values[..., start:end].mean(axis=-1)
    return smoothed


def save_plot(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10 * args.font_scale,
            "axes.titlesize": 11 * args.font_scale,
            "axes.labelsize": 11 * args.font_scale,
            "legend.fontsize": 10 * args.font_scale,
            "xtick.labelsize": 10 * args.font_scale,
            "ytick.labelsize": 10 * args.font_scale,
        }
    )
    method_curves = load_method_curves(args.input_dir, args.seeds)
    steps = matched_steps(method_curves)
    recall_curves = read_curve(args.input_dir / RECALL_PATTERN.format(seed=args.recall_seed))

    fig, axes = plt.subplots(2, 5, figsize=(20, 7.8), dpi=180, sharex=True, sharey=False)
    axes_flat = axes.ravel()

    for task_index, task in enumerate(TASKS):
        ax = axes_flat[task_index]
        for method, curves_by_seed in method_curves.items():
            values = np.asarray(
                [[curves[task][step] for step in steps] for curves in curves_by_seed],
                dtype=np.float64,
            )
            values = smooth_values(values, args.smooth_window)
            point, low, high = iqm_curve(
                values,
                reps=args.bootstrap_reps,
                random_seed=args.random_seed + 1000 * task_index + 17 * len(method),
            )
            ax.plot(
                np.asarray(steps) / 1000.0,
                point,
                label=method,
                color=COLORS[method],
                linewidth=2.0,
            )
            ax.fill_between(
                np.asarray(steps) / 1000.0,
                low,
                high,
                color=COLORS[method],
                alpha=0.16,
                linewidth=0,
            )

        recall_steps = sorted(recall_curves[task])
        recall_values = np.asarray(
            [recall_curves[task][step] for step in recall_steps],
            dtype=np.float64,
        )
        recall_values = smooth_values(recall_values[None, :], args.smooth_window)[0]
        ax.plot(
            np.asarray(recall_steps) / 1000.0,
            recall_values,
            label=f"RECALL (seed{args.recall_seed})",
            color=COLORS["RECALL (seed1)"],
            linewidth=1.7,
            linestyle="--",
        )
        ax.set_title(task)
        if task == "handle-press-side-v3":
            ax.set_ylim(0.8, 1.02)
            ax.set_yticks([0.8, 0.9, 1.0])
        else:
            ax.set_ylim(-0.04, 1.04)
        ax.set_xlim(min(steps) / 1000.0, max(steps) / 1000.0)
        ax.grid(alpha=0.25, linewidth=0.6)
        if task_index % 5 == 0:
            ax.set_ylabel("Success rate")
        if task_index >= 5:
            ax.set_xlabel("Task steps (k)")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        (
        "Per-task success learning curves: IQM over seeds 1-5 with 95% CI; "
        f"RECALL shown as seed1; smoothing window={args.smooth_window}"
        ),
        y=1.08,
        fontsize=13 * args.font_scale,
    )
    fig.tight_layout()
    suffix = (
        "raw" if args.smooth_window <= 1 else f"smooth{args.smooth_window}"
    )
    output_path = (
        args.output_dir
        / f"cw10_task_success_iqm_curves_with_recall_seed1_{suffix}.png"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    output_path = save_plot(parse_args())
    print(output_path)


if __name__ == "__main__":
    main()
