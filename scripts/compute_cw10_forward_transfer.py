"""Compute continual-learning metrics from deterministic CW results.

Inputs
------
1. ``baseline_curves.json`` produced by the independent single-task
   baseline aggregation pipeline.
2. ``evaluations.csv`` produced by ``train_cw10_finetune.py``.
3. ``final_evaluation.csv`` from the same continual run directory.

The script extracts the deterministic active-task learning curves from
``evaluations.csv`` and the final all-task performance from
``final_evaluation.csv``. It then computes Average Performance,
Forgetting, raw Forward Transfer, and normalized Forward Transfer.

This script does not train an agent and does not modify checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from crl_cw.evaluation.continual_metrics import (
    compute_continual_metrics,
)


PER_TASK_FIELDS = [
    "task_index",
    "task_name",
    "end_of_task_success",
    "final_success",
    "forgetting",
    "raw_forward_transfer",
    "normalized_forward_transfer",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute deterministic continual metrics using aligned "
            "single-task baseline curves and a sequential CW run."
        )
    )
    parser.add_argument(
        "--baseline-curves",
        type=str,
        required=True,
        help="Path to aggregated baseline_curves.json.",
    )
    parser.add_argument(
        "--continual-evaluations",
        type=str,
        required=True,
        help="Path to the continual run evaluations.csv.",
    )
    parser.add_argument(
        "--final-evaluation",
        type=str,
        default=None,
        help=(
            "Optional path to final_evaluation.csv. By default, the "
            "script uses final_evaluation.csv beside evaluations.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory in which calculated metrics will be saved.",
    )
    parser.add_argument(
        "--tail-size",
        type=int,
        default=5,
        help=(
            "Number of final active-task learning-curve points used "
            "for end-of-task performance. The separate final "
            "all-task CSV may contain only one final result per task."
        ),
    )

    args = parser.parse_args()
    if args.tail_size <= 0:
        parser.error("--tail-size must be positive.")
    return args


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _required_positive_int(
    values: dict[str, Any],
    field_name: str,
) -> int:
    if field_name not in values:
        raise ValueError(
            f"Missing required integer field {field_name!r}."
        )
    try:
        value = int(values[field_name])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Field {field_name!r} must be an integer."
        ) from error

    if value <= 0:
        raise ValueError(
            f"Field {field_name!r} must be positive."
        )
    return value


def _validate_success_curve(
    curve: Any,
    *,
    task_name: str,
    expected_length: int,
) -> list[float]:
    if not isinstance(curve, list):
        raise ValueError(
            f"Baseline curve for {task_name!r} must be a list."
        )
    if len(curve) != expected_length:
        raise ValueError(
            f"Baseline curve length mismatch for {task_name!r}: "
            f"received {len(curve)}, expected {expected_length}."
        )

    result: list[float] = []
    for value in curve:
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(
                f"Baseline success values for {task_name!r} must "
                "be finite and in [0, 1]."
            )
        result.append(numeric)
    return result


def load_baseline_curves(path: Path) -> dict[str, Any]:
    """Load and validate deterministic single-task baseline curves."""
    data = _load_json_object(path)

    required_fields = (
        "tasks",
        "evaluation_steps",
        "deterministic_success_curves",
        "steps_per_task",
        "eval_every",
        "seed",
    )
    missing = [
        field_name
        for field_name in required_fields
        if field_name not in data
    ]
    if missing:
        raise ValueError(
            "Baseline curves JSON is missing fields: "
            f"{missing}"
        )

    tasks_raw = data["tasks"]
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError(
            "Baseline tasks must be a non-empty list."
        )
    tasks = [str(task_name) for task_name in tasks_raw]
    if len(set(tasks)) != len(tasks):
        raise ValueError("Baseline task names must be unique.")

    steps_raw = data["evaluation_steps"]
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError(
            "Baseline evaluation_steps must be a non-empty list."
        )
    evaluation_steps = [int(step) for step in steps_raw]
    if any(step <= 0 for step in evaluation_steps):
        raise ValueError(
            "Baseline evaluation steps must be positive."
        )
    if any(
        current <= previous
        for previous, current in zip(
            evaluation_steps,
            evaluation_steps[1:],
        )
    ):
        raise ValueError(
            "Baseline evaluation steps must be strictly increasing."
        )

    steps_per_task = _required_positive_int(
        data,
        "steps_per_task",
    )
    eval_every = _required_positive_int(
        data,
        "eval_every",
    )
    expected_steps = list(
        range(
            eval_every,
            steps_per_task + 1,
            eval_every,
        )
    )
    if evaluation_steps != expected_steps:
        raise ValueError(
            "Baseline evaluation grid does not match steps_per_task "
            "and eval_every.\n"
            f"Stored  : {evaluation_steps}\n"
            f"Expected: {expected_steps}"
        )

    curves_raw = data["deterministic_success_curves"]
    if not isinstance(curves_raw, list):
        raise ValueError(
            "deterministic_success_curves must be a list."
        )
    if len(curves_raw) != len(tasks):
        raise ValueError(
            "deterministic_success_curves must contain exactly "
            "one curve per task."
        )

    deterministic_curves = [
        _validate_success_curve(
            curve,
            task_name=tasks[task_index],
            expected_length=len(evaluation_steps),
        )
        for task_index, curve in enumerate(curves_raw)
    ]

    # Stochastic fields are intentionally optional. In the current
    # deterministic protocol they are expected to be null.
    for optional_name in (
        "stochastic_success_curves",
        "stochastic_return_curves",
    ):
        optional_value = data.get(optional_name)
        if optional_value is not None and not isinstance(
            optional_value,
            list,
        ):
            raise ValueError(
                f"{optional_name} must be a list or null."
            )

    data["tasks"] = tasks
    data["evaluation_steps"] = evaluation_steps
    data["deterministic_success_curves"] = (
        deterministic_curves
    )
    return data


def load_continual_evaluations(path: Path) -> pd.DataFrame:
    """Load deterministic active-task learning-curve evaluations."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Continual evaluations file does not exist: {path}"
        )

    dataframe = pd.read_csv(path)
    required_columns = {
        "evaluation_index",
        "global_step",
        "active_task_index",
        "active_task_name",
        "active_task_step",
        "evaluation_task_index",
        "evaluation_task_name",
        "deterministic_success_rate",
    }
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(
            "Continual evaluations CSV is missing columns: "
            f"{sorted(missing)}"
        )
    if dataframe.empty:
        raise ValueError(
            "Continual evaluations CSV must not be empty."
        )

    numeric_columns = (
        "evaluation_index",
        "global_step",
        "active_task_index",
        "active_task_step",
        "evaluation_task_index",
        "deterministic_success_rate",
    )
    for column_name in numeric_columns:
        dataframe[column_name] = pd.to_numeric(
            dataframe[column_name],
            errors="raise",
        )

    success = dataframe[
        "deterministic_success_rate"
    ].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(success)):
        raise ValueError(
            "Deterministic continual success contains NaN or infinity."
        )
    if np.any(success < 0.0) or np.any(success > 1.0):
        raise ValueError(
            "Deterministic continual success must be in [0, 1]."
        )

    return dataframe


def extract_active_task_curves(
    *,
    dataframe: pd.DataFrame,
    tasks: Sequence[str],
    evaluation_steps: Sequence[int],
) -> list[list[float]]:
    """Extract one aligned deterministic learning curve per task."""
    expected_steps = [int(step) for step in evaluation_steps]
    curves: list[list[float]] = []

    for task_index, task_name in enumerate(tasks):
        task_rows = dataframe[
            (
                dataframe["active_task_index"]
                == task_index
            )
            & (
                dataframe["active_task_name"]
                == task_name
            )
            & (
                dataframe["evaluation_task_index"]
                == task_index
            )
            & (
                dataframe["evaluation_task_name"]
                == task_name
            )
        ].copy()

        if task_rows.empty:
            raise ValueError(
                "No active-task deterministic evaluations found for "
                f"{task_name!r}."
            )

        task_rows = task_rows.sort_values(
            by=[
                "active_task_step",
                "evaluation_index",
            ]
        )

        duplicated = task_rows[
            "active_task_step"
        ].duplicated(keep=False)
        if duplicated.any():
            duplicate_values = (
                task_rows.loc[
                    duplicated,
                    "active_task_step",
                ]
                .astype(int)
                .tolist()
            )
            raise ValueError(
                "Duplicate active-task evaluation steps found for "
                f"{task_name!r}: {duplicate_values}"
            )

        actual_steps = (
            task_rows["active_task_step"]
            .astype(int)
            .tolist()
        )
        if actual_steps != expected_steps:
            raise ValueError(
                "Continual and baseline evaluation grids do not "
                f"match for {task_name!r}.\n"
                f"Baseline : {expected_steps}\n"
                f"Continual: {actual_steps}"
            )

        curve = (
            task_rows["deterministic_success_rate"]
            .astype(float)
            .tolist()
        )
        curves.append(curve)

    return curves


def _find_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
    *,
    description: str,
) -> str:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    raise ValueError(
        f"Final evaluation CSV has no {description} column. "
        f"Tried: {list(candidates)}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def load_final_evaluation_matrix(
    *,
    path: Path,
    tasks: Sequence[str],
) -> list[list[float]]:
    """Load the separate final all-task deterministic evaluation.

    The current runner stores one row per task in final_evaluation.csv.
    A single matrix row is returned because this file contains one
    final aggregate evaluation result per task.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Final evaluation CSV does not exist: {path}"
        )

    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(
            "Final evaluation CSV must not be empty."
        )

    task_name_column = _find_column(
        dataframe,
        (
            "task_name",
            "evaluation_task_name",
        ),
        description="task-name",
    )
    task_index_column = (
        "task_index"
        if "task_index" in dataframe.columns
        else (
            "evaluation_task_index"
            if "evaluation_task_index"
            in dataframe.columns
            else None
        )
    )
    success_column = _find_column(
        dataframe,
        (
            "success_rate",
            "deterministic_success_rate",
        ),
        description="deterministic success-rate",
    )

    dataframe[task_name_column] = (
        dataframe[task_name_column].astype(str)
    )
    dataframe[success_column] = pd.to_numeric(
        dataframe[success_column],
        errors="raise",
    )

    if task_index_column is not None:
        dataframe[task_index_column] = pd.to_numeric(
            dataframe[task_index_column],
            errors="raise",
        )

    row_values: list[float] = []
    for task_index, task_name in enumerate(tasks):
        matching = dataframe[
            dataframe[task_name_column] == task_name
        ]

        if task_index_column is not None:
            matching = matching[
                matching[task_index_column]
                == task_index
            ]

        if len(matching) != 1:
            raise ValueError(
                "Final evaluation CSV must contain exactly one row "
                f"for task {task_name!r}; found {len(matching)}."
            )

        value = float(
            matching.iloc[0][success_column]
        )
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Final success for {task_name!r} must be finite "
                "and in [0, 1]."
            )
        row_values.append(value)

    unexpected_tasks = set(
        dataframe[task_name_column].astype(str)
    ) - set(tasks)
    if unexpected_tasks:
        raise ValueError(
            "Final evaluation CSV contains unexpected tasks: "
            f"{sorted(unexpected_tasks)}"
        )

    return [row_values]


def save_per_task_metrics(
    *,
    path: Path,
    tasks: Sequence[str],
    result: Any,
) -> None:
    """Save one metrics CSV row per task."""
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=PER_TASK_FIELDS,
        )
        writer.writeheader()

        for task_index, task_name in enumerate(tasks):
            writer.writerow(
                {
                    "task_index": task_index,
                    "task_name": task_name,
                    "end_of_task_success": (
                        result.end_of_task_per_task[
                            task_index
                        ]
                    ),
                    "final_success": (
                        result.final_per_task[
                            task_index
                        ]
                    ),
                    "forgetting": (
                        result.forgetting_per_task[
                            task_index
                        ]
                    ),
                    "raw_forward_transfer": (
                        result.raw_forward_transfer_per_task[
                            task_index
                        ]
                    ),
                    "normalized_forward_transfer": (
                        result
                        .normalized_forward_transfer_per_task[
                            task_index
                        ]
                    ),
                }
            )


def main() -> None:
    """Compute and save deterministic continual-learning metrics."""
    args = parse_args()

    baseline_path = Path(
        args.baseline_curves
    ).expanduser()
    continual_path = Path(
        args.continual_evaluations
    ).expanduser()
    final_path = (
        Path(args.final_evaluation).expanduser()
        if args.final_evaluation is not None
        else continual_path.parent
        / "final_evaluation.csv"
    )
    output_directory = Path(
        args.output_dir
    ).expanduser()
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_data = load_baseline_curves(
        baseline_path
    )
    dataframe = load_continual_evaluations(
        continual_path
    )

    tasks = list(baseline_data["tasks"])
    evaluation_steps = list(
        baseline_data["evaluation_steps"]
    )
    baseline_success_curves = list(
        baseline_data[
            "deterministic_success_curves"
        ]
    )

    active_task_curves = extract_active_task_curves(
        dataframe=dataframe,
        tasks=tasks,
        evaluation_steps=evaluation_steps,
    )
    final_evaluation_matrix = (
        load_final_evaluation_matrix(
            path=final_path,
            tasks=tasks,
        )
    )

    result = compute_continual_metrics(
        final_evaluation_successes=(
            final_evaluation_matrix
        ),
        active_task_success_curves=(
            active_task_curves
        ),
        baseline_success_curves=(
            baseline_success_curves
        ),
        tail_size=args.tail_size,
    )

    metrics_json_path = (
        output_directory
        / "complete_continual_metrics.json"
    )
    per_task_csv_path = (
        output_directory
        / "complete_per_task_metrics.csv"
    )

    output_data = {
        "schema_version": 2,
        "evaluation_protocol": "deterministic",
        "method": (
            "sequential_continual_learning_metrics"
        ),
        "seed": int(baseline_data["seed"]),
        "tasks": tasks,
        "evaluation_steps": evaluation_steps,
        "steps_per_task": int(
            baseline_data["steps_per_task"]
        ),
        "eval_every": int(
            baseline_data["eval_every"]
        ),
        "tail_size": int(args.tail_size),
        "average_performance": (
            result.average_performance
        ),
        "average_forgetting": (
            result.average_forgetting
        ),
        "average_forward_transfer": (
            result.average_forward_transfer
        ),
        "average_raw_forward_transfer": (
            result.average_raw_forward_transfer
        ),
        "final_per_task": {
            task_name: result.final_per_task[
                task_index
            ]
            for task_index, task_name in enumerate(
                tasks
            )
        },
        "end_of_task_per_task": {
            task_name: (
                result.end_of_task_per_task[
                    task_index
                ]
            )
            for task_index, task_name in enumerate(
                tasks
            )
        },
        "forgetting_per_task": {
            task_name: (
                result.forgetting_per_task[
                    task_index
                ]
            )
            for task_index, task_name in enumerate(
                tasks
            )
        },
        "raw_forward_transfer_per_task": {
            task_name: (
                result.raw_forward_transfer_per_task[
                    task_index
                ]
            )
            for task_index, task_name in enumerate(
                tasks
            )
        },
        "normalized_forward_transfer_per_task": {
            task_name: (
                result
                .normalized_forward_transfer_per_task[
                    task_index
                ]
            )
            for task_index, task_name in enumerate(
                tasks
            )
        },
        "baseline_curves_file": str(
            baseline_path
        ),
        "continual_evaluations_file": str(
            continual_path
        ),
        "final_evaluation_file": str(
            final_path
        ),
    }

    with metrics_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            indent=2,
            sort_keys=True,
        )

    save_per_task_metrics(
        path=per_task_csv_path,
        tasks=tasks,
        result=result,
    )

    print("=" * 76)
    print("CW deterministic continual metrics")
    print("=" * 76)
    print(
        "Average Performance       : "
        f"{result.average_performance:.6f}"
    )
    print(
        "Average Forgetting        : "
        f"{result.average_forgetting:.6f}"
    )
    print(
        "Average Forward Transfer  : "
        f"{result.average_forward_transfer:.6f}"
    )
    print(
        "Average raw FT            : "
        f"{result.average_raw_forward_transfer:.6f}"
    )
    print(
        f"Baseline curves           : {baseline_path}"
    )
    print(
        f"Active evaluations        : {continual_path}"
    )
    print(
        f"Final evaluation          : {final_path}"
    )
    print(
        f"Metrics JSON              : {metrics_json_path}"
    )
    print(
        f"Per-task metrics CSV      : {per_task_csv_path}"
    )


if __name__ == "__main__":
    main()
