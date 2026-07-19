"""Aggregate deterministic single-task SAC baseline curves.

This module combines independently trained single-task runs into the
baseline file consumed by the continual-learning Forward Transfer code.

The current experiment protocol uses deterministic evaluation. Stochastic
curves are therefore stored as JSON null when stochastic evaluation is
disabled, rather than copying deterministic values into stochastic fields.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


_REQUIRED_EVALUATION_FIELDS = (
    "environment_step",
    "gradient_updates",
    "deterministic_average_return",
    "deterministic_success_rate",
    "elapsed_seconds",
)

_LONG_CURVE_FIELDS = (
    "task_index",
    "task_name",
    "seed",
    "evaluation_index",
    "environment_step",
    "gradient_updates",
    "deterministic_average_return",
    "deterministic_success_rate",
    "deterministic_average_episode_length",
    "stochastic_average_return",
    "stochastic_success_rate",
    "stochastic_average_episode_length",
    "actor_loss",
    "q1_loss",
    "q2_loss",
    "alpha_loss",
    "alpha",
    "q1_mean",
    "q2_mean",
    "q_target_mean",
    "log_prob_mean",
    "elapsed_seconds",
    "source_run_directory",
)

_TASK_SUMMARY_FIELDS = (
    "task_index",
    "task_name",
    "seed",
    "num_evaluation_points",
    "final_deterministic_success",
    "tail_mean_deterministic_success",
    "maximum_deterministic_success",
    "final_deterministic_return",
    "tail_mean_deterministic_return",
    "final_stochastic_success",
    "final_stochastic_return",
    "elapsed_seconds",
    "source_run_directory",
)


@dataclass(frozen=True)
class BaselineAggregationResult:
    """Paths created by one baseline aggregation."""

    output_directory: Path
    curves_json_path: Path
    long_curves_csv_path: Path
    stochastic_success_csv_path: Path
    deterministic_success_csv_path: Path
    stochastic_return_csv_path: Path
    deterministic_return_csv_path: Path
    task_summaries_csv_path: Path
    summary_json_path: Path


def aggregate_single_task_baselines(
    *,
    batch_directory: str | Path,
    output_directory: str | Path | None = None,
    tail_size: int = 5,
) -> BaselineAggregationResult:
    """Aggregate completed independent-task runs.

    The generated ``baseline_curves.json`` contains deterministic success
    curves used by the current Forward Transfer implementation. If
    stochastic evaluation was disabled, stochastic curve fields are JSON
    ``null``.
    """
    if tail_size <= 0:
        raise ValueError("tail_size must be positive.")

    batch_path = Path(batch_directory).expanduser().resolve()
    if not batch_path.is_dir():
        raise FileNotFoundError(
            f"Batch directory does not exist: {batch_path}"
        )

    batch_config = _read_json_object(batch_path / "batch_config.json")
    manifest_rows = _read_manifest(batch_path / "batch_manifest.csv")

    tasks = _read_task_order(batch_config)
    seed = _required_int(batch_config, "seed")
    steps_per_task = _required_int(batch_config, "steps_per_task")
    eval_every = _required_int(batch_config, "eval_every")
    det_eval_episodes = _required_int(
        batch_config,
        "det_eval_episodes",
    )
    stoch_eval_episodes = _required_int(
        batch_config,
        "stoch_eval_episodes",
    )

    if det_eval_episodes <= 0:
        raise ValueError("det_eval_episodes must be positive.")
    if stoch_eval_episodes < 0:
        raise ValueError("stoch_eval_episodes must be non-negative.")
    if steps_per_task <= 0 or eval_every <= 0:
        raise ValueError("steps_per_task and eval_every must be positive.")
    if steps_per_task % eval_every != 0:
        raise ValueError(
            "steps_per_task must be divisible by eval_every."
        )

    manifest_by_task = _validate_manifest(
        manifest_rows=manifest_rows,
        tasks=tasks,
    )

    destination = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else batch_path / "aggregate"
    )
    destination.mkdir(parents=True, exist_ok=True)

    expected_steps = list(
        range(eval_every, steps_per_task + 1, eval_every)
    )

    source_runs: list[str] = []
    long_rows: list[dict[str, Any]] = []
    task_summary_rows: list[dict[str, Any]] = []

    deterministic_success_curves: list[list[float]] = []
    deterministic_return_curves: list[list[float]] = []
    stochastic_success_curves: list[list[float]] = []
    stochastic_return_curves: list[list[float]] = []
    stochastic_available_for_all = stoch_eval_episodes > 0

    for task_index, task_name in enumerate(tasks):
        manifest_row = manifest_by_task[task_name]
        run_directory = _resolve_run_directory(
            batch_path=batch_path,
            manifest_row=manifest_row,
        )
        source_runs.append(str(run_directory))

        run_config = _read_json_object(run_directory / "config.json")
        run_summary = _read_json_object(run_directory / "summary.json")
        evaluation_rows = _read_evaluation_rows(
            run_directory / "evaluations.csv"
        )

        _validate_run_identity(
            run_config=run_config,
            run_summary=run_summary,
            task_name=task_name,
            seed=seed,
            steps_per_task=steps_per_task,
            eval_every=eval_every,
            det_eval_episodes=det_eval_episodes,
            stoch_eval_episodes=stoch_eval_episodes,
        )

        task_steps = [
            _required_row_int(row, "environment_step")
            for row in evaluation_rows
        ]
        _validate_strictly_increasing(task_steps, task_name=task_name)

        if task_steps != expected_steps:
            raise ValueError(
                f"Task {task_name!r} evaluation grid is {task_steps}, "
                f"expected {expected_steps}."
            )

        deterministic_success = [
            _required_success(row, "deterministic_success_rate")
            for row in evaluation_rows
        ]
        deterministic_return = [
            _required_finite_float(
                row,
                "deterministic_average_return",
            )
            for row in evaluation_rows
        ]

        stochastic_success = _optional_float_curve(
            evaluation_rows,
            "stochastic_success_rate",
            success=True,
        )
        stochastic_return = _optional_float_curve(
            evaluation_rows,
            "stochastic_average_return",
            success=False,
        )

        if stoch_eval_episodes == 0:
            # Deterministic-only protocol. Ignore any stale stochastic
            # values written by older runners and do not expose them in
            # the aggregated baseline file.
            stochastic_success = None
            stochastic_return = None
            stochastic_available_for_all = False
        elif stochastic_success is None or stochastic_return is None:
            stochastic_available_for_all = False
        else:
            stochastic_success_curves.append(stochastic_success)
            stochastic_return_curves.append(stochastic_return)

        deterministic_success_curves.append(deterministic_success)
        deterministic_return_curves.append(deterministic_return)

        for evaluation_index, row in enumerate(evaluation_rows, start=1):
            long_row: dict[str, Any] = {
                "task_index": task_index,
                "task_name": task_name,
                "seed": seed,
                "evaluation_index": evaluation_index,
                "source_run_directory": str(run_directory),
            }
            for field_name in _LONG_CURVE_FIELDS:
                if field_name not in long_row:
                    long_row[field_name] = row.get(field_name, "")
            long_rows.append(long_row)

        selected_count = min(tail_size, len(deterministic_success))
        elapsed_seconds = _required_finite_float(
            evaluation_rows[-1],
            "elapsed_seconds",
        )

        task_summary_rows.append(
            {
                "task_index": task_index,
                "task_name": task_name,
                "seed": seed,
                "num_evaluation_points": len(evaluation_rows),
                "final_deterministic_success": deterministic_success[-1],
                "tail_mean_deterministic_success": _mean(
                    deterministic_success[-selected_count:]
                ),
                "maximum_deterministic_success": max(
                    deterministic_success
                ),
                "final_deterministic_return": deterministic_return[-1],
                "tail_mean_deterministic_return": _mean(
                    deterministic_return[-selected_count:]
                ),
                "final_stochastic_success": (
                    ""
                    if stochastic_success is None
                    else stochastic_success[-1]
                ),
                "final_stochastic_return": (
                    ""
                    if stochastic_return is None
                    else stochastic_return[-1]
                ),
                "elapsed_seconds": elapsed_seconds,
                "source_run_directory": str(run_directory),
            }
        )

    if not stochastic_available_for_all:
        stochastic_success_json: list[list[float]] | None = None
        stochastic_return_json: list[list[float]] | None = None
    else:
        if len(stochastic_success_curves) != len(tasks):
            raise RuntimeError(
                "Stochastic success curves are incomplete."
            )
        if len(stochastic_return_curves) != len(tasks):
            raise RuntimeError(
                "Stochastic return curves are incomplete."
            )
        stochastic_success_json = stochastic_success_curves
        stochastic_return_json = stochastic_return_curves

    curves_json_path = destination / "baseline_curves.json"
    long_curves_csv_path = destination / "baseline_curves_long.csv"
    stochastic_success_csv_path = (
        destination / "stochastic_success_curves.csv"
    )
    deterministic_success_csv_path = (
        destination / "deterministic_success_curves.csv"
    )
    stochastic_return_csv_path = (
        destination / "stochastic_return_curves.csv"
    )
    deterministic_return_csv_path = (
        destination / "deterministic_return_curves.csv"
    )
    task_summaries_csv_path = destination / "task_summaries.csv"
    summary_json_path = destination / "summary.json"

    _write_csv(
        path=long_curves_csv_path,
        fieldnames=list(_LONG_CURVE_FIELDS),
        rows=long_rows,
    )
    _write_wide_curve_csv(
        path=deterministic_success_csv_path,
        evaluation_steps=expected_steps,
        tasks=tasks,
        curves=deterministic_success_curves,
    )
    _write_wide_curve_csv(
        path=deterministic_return_csv_path,
        evaluation_steps=expected_steps,
        tasks=tasks,
        curves=deterministic_return_curves,
    )

    if stochastic_success_json is None:
        _write_disabled_curve_csv(stochastic_success_csv_path)
        _write_disabled_curve_csv(stochastic_return_csv_path)
    else:
        _write_wide_curve_csv(
            path=stochastic_success_csv_path,
            evaluation_steps=expected_steps,
            tasks=tasks,
            curves=stochastic_success_json,
        )
        _write_wide_curve_csv(
            path=stochastic_return_csv_path,
            evaluation_steps=expected_steps,
            tasks=tasks,
            curves=stochastic_return_json,
        )

    _write_csv(
        path=task_summaries_csv_path,
        fieldnames=list(_TASK_SUMMARY_FIELDS),
        rows=task_summary_rows,
    )

    created_at_utc = datetime.now(timezone.utc).isoformat()

    curves_payload: dict[str, Any] = {
        "schema_version": 2,
        "method": "single_task_sac_from_scratch",
        "evaluation_protocol": "deterministic",
        "created_at_utc": created_at_utc,
        "batch_directory": str(batch_path),
        "seed": seed,
        "tasks": tasks,
        "num_tasks": len(tasks),
        "steps_per_task": steps_per_task,
        "eval_every": eval_every,
        "det_eval_episodes": det_eval_episodes,
        "stoch_eval_episodes": stoch_eval_episodes,
        "max_episode_steps": _required_int(
            batch_config,
            "max_episode_steps",
        ),
        "evaluation_steps": expected_steps,
        "deterministic_success_curves": deterministic_success_curves,
        "deterministic_return_curves": deterministic_return_curves,
        "stochastic_success_curves": stochastic_success_json,
        "stochastic_return_curves": stochastic_return_json,
        "source_run_directories": source_runs,
    }
    _write_json(curves_json_path, curves_payload)

    summary_payload: dict[str, Any] = {
        "schema_version": 2,
        "evaluation_protocol": "deterministic",
        "created_at_utc": created_at_utc,
        "batch_directory": str(batch_path),
        "output_directory": str(destination),
        "seed": seed,
        "tasks": tasks,
        "num_tasks": len(tasks),
        "steps_per_task": steps_per_task,
        "eval_every": eval_every,
        "det_eval_episodes": det_eval_episodes,
        "stoch_eval_episodes": stoch_eval_episodes,
        "num_evaluation_points": len(expected_steps),
        "evaluation_steps": expected_steps,
        "tail_size": tail_size,
        "mean_final_deterministic_success": _mean(
            [curve[-1] for curve in deterministic_success_curves]
        ),
        "mean_tail_deterministic_success": _mean(
            [
                _mean(curve[-min(tail_size, len(curve)):])
                for curve in deterministic_success_curves
            ]
        ),
        "mean_final_deterministic_return": _mean(
            [curve[-1] for curve in deterministic_return_curves]
        ),
        "stochastic_evaluation_available": (
            stochastic_success_json is not None
        ),
        "forward_transfer_ready": True,
        "forward_transfer_curve_key": (
            "deterministic_success_curves"
        ),
        "curves_json": str(curves_json_path),
        "long_curves_csv": str(long_curves_csv_path),
        "deterministic_success_csv": str(
            deterministic_success_csv_path
        ),
        "deterministic_return_csv": str(
            deterministic_return_csv_path
        ),
        "stochastic_success_csv": str(
            stochastic_success_csv_path
        ),
        "stochastic_return_csv": str(
            stochastic_return_csv_path
        ),
        "task_summaries_csv": str(task_summaries_csv_path),
    }
    _write_json(summary_json_path, summary_payload)

    return BaselineAggregationResult(
        output_directory=destination,
        curves_json_path=curves_json_path,
        long_curves_csv_path=long_curves_csv_path,
        stochastic_success_csv_path=stochastic_success_csv_path,
        deterministic_success_csv_path=deterministic_success_csv_path,
        stochastic_return_csv_path=stochastic_return_csv_path,
        deterministic_return_csv_path=deterministic_return_csv_path,
        task_summaries_csv_path=task_summaries_csv_path,
        summary_json_path=summary_json_path,
    )


def _optional_float_curve(
    rows: Sequence[dict[str, str]],
    field: str,
    *,
    success: bool,
) -> list[float] | None:
    values: list[float] = []
    for row in rows:
        raw = row.get(field, "").strip()
        if raw == "":
            return None
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(
                f"Field {field!r} must be numeric when present."
            ) from error
        if not math.isfinite(value):
            raise ValueError(f"Field {field!r} must be finite.")
        if success and not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Success field {field!r} must be in [0, 1]."
            )
        values.append(value)
    return values


def _read_task_order(configuration: dict[str, Any]) -> list[str]:
    raw_tasks = configuration.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError(
            "batch_config.json must contain a non-empty tasks list."
        )
    tasks = [str(task) for task in raw_tasks]
    if len(set(tasks)) != len(tasks):
        raise ValueError("Task names must be unique.")
    return tasks


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = [dict(row) for row in csv.DictReader(file)]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _validate_manifest(
    *,
    manifest_rows: Sequence[dict[str, str]],
    tasks: Sequence[str],
) -> dict[str, dict[str, str]]:
    by_task: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        task_name = row.get("task_name", "").strip()
        if not task_name:
            raise ValueError("Every manifest row requires task_name.")
        if task_name in by_task:
            raise ValueError(
                f"Duplicate manifest entry for {task_name!r}."
            )
        by_task[task_name] = row

    if set(by_task) != set(tasks):
        raise ValueError("Manifest tasks do not match batch tasks.")

    for task_name in tasks:
        status = by_task[task_name].get("status", "").strip().lower()
        if status != "completed":
            raise ValueError(
                f"Task {task_name!r} is not completed: {status!r}."
            )
    return by_task


def _resolve_run_directory(
    *,
    batch_path: Path,
    manifest_row: dict[str, str],
) -> Path:
    raw_path = manifest_row.get("run_directory", "").strip()
    if not raw_path:
        raise ValueError("Manifest row is missing run_directory.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = batch_path / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Task run directory does not exist: {path}"
        )
    return path


def _validate_run_identity(
    *,
    run_config: dict[str, Any],
    run_summary: dict[str, Any],
    task_name: str,
    seed: int,
    steps_per_task: int,
    eval_every: int,
    det_eval_episodes: int,
    stoch_eval_episodes: int,
) -> None:
    checks: tuple[tuple[str, Any, Any], ...] = (
        ("task", str(run_config.get("task")), task_name),
        ("summary task", str(run_summary.get("task_name")), task_name),
        ("seed", _required_int(run_config, "seed"), seed),
        (
            "total_steps",
            _required_int(run_config, "total_steps"),
            steps_per_task,
        ),
        (
            "eval_every",
            _required_int(run_config, "eval_every"),
            eval_every,
        ),
        (
            "det_eval_episodes",
            _required_int(run_config, "det_eval_episodes"),
            det_eval_episodes,
        ),
        (
            "stoch_eval_episodes",
            _required_int(run_config, "stoch_eval_episodes"),
            stoch_eval_episodes,
        ),
    )
    for label, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"{label} mismatch for {task_name!r}: "
                f"actual={actual!r}, expected={expected!r}."
            )


def _read_evaluation_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation CSV does not exist: {path}"
        )
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing = [
            field
            for field in _REQUIRED_EVALUATION_FIELDS
            if field not in fieldnames
        ]
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Evaluation CSV is empty: {path}")
    return rows


def _validate_strictly_increasing(
    values: Sequence[int],
    *,
    task_name: str,
) -> None:
    for previous, current in zip(values, values[1:]):
        if current <= previous:
            raise ValueError(
                f"Evaluation steps for {task_name!r} are not "
                "strictly increasing."
            )


def _required_int(values: dict[str, Any], name: str) -> int:
    if name not in values:
        raise ValueError(f"Missing required integer field {name!r}.")
    try:
        return int(values[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Field {name!r} must be an integer.") from error


def _required_row_int(values: dict[str, str], name: str) -> int:
    return _required_int(dict(values), name)


def _required_finite_float(
    values: dict[str, Any],
    name: str,
) -> float:
    if name not in values:
        raise ValueError(f"Missing required float field {name!r}.")
    try:
        value = float(values[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Field {name!r} must be numeric.") from error
    if not math.isfinite(value):
        raise ValueError(f"Field {name!r} must be finite.")
    return value


def _required_success(values: dict[str, Any], name: str) -> float:
    value = _required_finite_float(values, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Success field {name!r} must be in [0, 1]."
        )
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)


def _write_csv(
    *,
    path: Path,
    fieldnames: list[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_disabled_curve_csv(path: Path) -> None:
    _write_csv(
        path=path,
        fieldnames=["status"],
        rows=[{"status": "disabled"}],
    )


def _write_wide_curve_csv(
    *,
    path: Path,
    evaluation_steps: Sequence[int],
    tasks: Sequence[str],
    curves: Sequence[Sequence[float]],
) -> None:
    if len(tasks) != len(curves):
        raise ValueError("Tasks and curves must have equal length.")

    fieldnames = ["environment_step", *tasks]
    rows: list[dict[str, Any]] = []
    for evaluation_index, environment_step in enumerate(evaluation_steps):
        row: dict[str, Any] = {"environment_step": environment_step}
        for task_name, curve in zip(tasks, curves, strict=True):
            if len(curve) != len(evaluation_steps):
                raise ValueError(
                    f"Curve length mismatch for task {task_name!r}."
                )
            row[task_name] = curve[evaluation_index]
        rows.append(row)

    _write_csv(path=path, fieldnames=fieldnames, rows=rows)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate a mean from no values.")
    return float(sum(float(value) for value in values) / len(values))
