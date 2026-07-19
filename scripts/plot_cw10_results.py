"""Plot CW10 success curves and compute diagnostic continual metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one CW10 run directory.")
    parser.add_argument("run_directory", type=str)
    parser.add_argument("--tail-size", type=int, default=5)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.tail_size <= 0:
        parser.error("--tail-size must be positive.")
    return args


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}.")

    integer_fields = (
        "evaluation_index",
        "global_step",
        "gradient_updates",
        "active_task_index",
        "active_task_step",
        "evaluation_task_index",
    )
    float_fields = (
        "deterministic_average_return",
        "deterministic_success_rate",
        "deterministic_average_episode_length",
        "stochastic_average_return",
        "stochastic_success_rate",
        "stochastic_average_episode_length",
    )

    converted: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for name in integer_fields:
            item[name] = int(item[name])
        for name in float_fields:
            item[name] = float(item[name])
        converted.append(item)
    return converted


def task_boundaries(steps_per_task: int, num_tasks: int) -> list[int]:
    return [steps_per_task * index for index in range(1, num_tasks)]


def add_task_boundaries(axis: Any, boundaries: list[int]) -> None:
    for boundary in boundaries:
        axis.axvline(boundary, linestyle="--", linewidth=0.8, alpha=0.5)


def save_figure(figure: Any, path: Path, show: bool) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def compute_metrics(
    rows: list[dict[str, Any]],
    *,
    tasks: list[str],
    tail_size: int,
) -> dict[str, Any]:
    by_evaluation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_evaluation[row["evaluation_index"]].append(row)

    evaluation_ids = sorted(by_evaluation)
    final_ids = evaluation_ids[-min(tail_size, len(evaluation_ids)):]

    final_per_task: list[float] = []
    end_of_task_per_task: list[float] = []
    forgetting_per_task: list[float] = []

    for task_index, _task_name in enumerate(tasks):
        final_values = [
            next(
                row["stochastic_success_rate"]
                for row in by_evaluation[evaluation_id]
                if row["evaluation_task_index"] == task_index
            )
            for evaluation_id in final_ids
        ]
        final_value = float(np.mean(final_values))
        final_per_task.append(final_value)

        active_rows = [
            row
            for row in rows
            if row["active_task_index"] == task_index
            and row["evaluation_task_index"] == task_index
        ]
        active_rows.sort(key=lambda row: row["global_step"])
        tail_rows = active_rows[-min(tail_size, len(active_rows)):]
        end_value = float(
            np.mean([row["stochastic_success_rate"] for row in tail_rows])
        )
        end_of_task_per_task.append(end_value)
        forgetting_per_task.append(end_value - final_value)

    return {
        "tail_size": tail_size,
        "average_performance": float(np.mean(final_per_task)),
        "average_forgetting": float(np.mean(forgetting_per_task)),
        "final_per_task": dict(zip(tasks, final_per_task, strict=True)),
        "end_of_task_per_task": dict(
            zip(tasks, end_of_task_per_task, strict=True)
        ),
        "forgetting_per_task": dict(
            zip(tasks, forgetting_per_task, strict=True)
        ),
        "forward_transfer": None,
        "forward_transfer_note": (
            "Forward transfer requires matched single-task baseline curves."
        ),
    }


def main() -> None:
    args = parse_args()
    run_directory = Path(args.run_directory).expanduser()
    evaluations_path = run_directory / "evaluations.csv"
    config_path = run_directory / "config.json"

    if not evaluations_path.is_file():
        raise FileNotFoundError(f"Missing {evaluations_path}.")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}.")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    tasks = list(config["tasks"])
    steps_per_task = int(config["steps_per_task"])
    rows = load_rows(evaluations_path)
    plots_directory = run_directory / "plots"
    plots_directory.mkdir(exist_ok=True)
    boundaries = task_boundaries(steps_per_task, len(tasks))

    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[row["global_step"]].append(row)
    global_steps = sorted(by_step)

    average_stochastic = [
        float(np.mean([row["stochastic_success_rate"] for row in by_step[step]]))
        for step in global_steps
    ]
    average_deterministic = [
        float(np.mean([row["deterministic_success_rate"] for row in by_step[step]]))
        for step in global_steps
    ]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(global_steps, average_stochastic, label="Stochastic average success")
    axis.plot(global_steps, average_deterministic, label="Deterministic average success")
    add_task_boundaries(axis, boundaries)
    axis.set_xlabel("Global environment steps")
    axis.set_ylabel("Average success rate")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("CW10 average success")
    axis.legend()
    axis.grid(alpha=0.25)
    save_figure(figure, plots_directory / "average_success_curve.png", args.show)

    figure, axis = plt.subplots(figsize=(12, 7))
    for task_index, task_name in enumerate(tasks):
        task_rows = [
            row for row in rows if row["evaluation_task_index"] == task_index
        ]
        task_rows.sort(key=lambda row: row["global_step"])
        axis.plot(
            [row["global_step"] for row in task_rows],
            [row["stochastic_success_rate"] for row in task_rows],
            label=task_name,
        )
    add_task_boundaries(axis, boundaries)
    axis.set_xlabel("Global environment steps")
    axis.set_ylabel("Stochastic success rate")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("CW10 per-task continual success")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(alpha=0.25)
    save_figure(figure, plots_directory / "per_task_success_curves.png", args.show)

    figure, axis = plt.subplots(figsize=(10, 6))
    for task_index, task_name in enumerate(tasks):
        active_rows = [
            row
            for row in rows
            if row["active_task_index"] == task_index
            and row["evaluation_task_index"] == task_index
        ]
        active_rows.sort(key=lambda row: row["active_task_step"])
        axis.plot(
            [row["active_task_step"] for row in active_rows],
            [row["stochastic_success_rate"] for row in active_rows],
            label=task_name,
        )
    axis.set_xlabel("Environment steps within active task")
    axis.set_ylabel("Stochastic success rate")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Active-task learning curves")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(alpha=0.25)
    save_figure(figure, plots_directory / "active_task_learning_curves.png", args.show)

    metrics = compute_metrics(rows, tasks=tasks, tail_size=args.tail_size)
    metrics_path = run_directory / "continual_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)

    print("Plots created:")
    print(f"  {plots_directory / 'average_success_curve.png'}")
    print(f"  {plots_directory / 'per_task_success_curves.png'}")
    print(f"  {plots_directory / 'active_task_learning_curves.png'}")
    print(f"Metrics: {metrics_path}")
    print(f"Average performance: {metrics['average_performance']:.6f}")
    print(f"Average forgetting : {metrics['average_forgetting']:.6f}")
    print("Forward transfer   : unavailable until single-task baselines exist")


if __name__ == "__main__":
    main()
