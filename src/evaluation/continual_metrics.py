"""Continual-learning metrics for CW10 experiments.

This module implements the standard benchmark definitions for:

1. final average performance;
2. forgetting;
3. raw forward transfer;
4. normalized forward transfer;
5. area-based normalized forward transfer.

The module contains only metric calculations. It does not run environments,
train agents, read CSV files, or manage task switching.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


_SUCCESS_TOLERANCE = 1e-9
_DENOMINATOR_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ContinualMetricsResult:
    """Aggregated and per-task continual-learning metrics.

    Attributes:
        average_performance:
            Mean final success rate over all tasks.

        average_forgetting:
            Mean difference between each task's performance immediately
            after learning that task and its performance at the end of
            the complete task sequence.

            Positive values mean forgetting.
            Negative values mean later improvement.

        average_forward_transfer:
            Mean normalized forward transfer over tasks.

        average_raw_forward_transfer:
            Mean unnormalized forward transfer over tasks.

        average_area_forward_transfer:
            Mean area-based normalized forward transfer over tasks.

        final_per_task:
            Final success rate for every task.

        end_of_task_per_task:
            Success rate of each task at the end of training that task.

        forgetting_per_task:
            `end_of_task_per_task - final_per_task`.

        raw_forward_transfer_per_task:
            Difference between the mean continual-learning curve and
            the mean single-task baseline curve.

        normalized_forward_transfer_per_task:
            Raw forward transfer divided by the remaining performance
            headroom of the single-task baseline. The value is NaN for
            a task whose baseline mean success is 1.0, because the
            normalized metric is mathematically undefined.

        area_forward_transfer_per_task:
            Area-based normalized forward transfer for each task. This
            uses the trapezoidal integral of the continual minus
            baseline success curve, normalized by the baseline's
            remaining headroom area over the same evaluation horizon.
    """

    average_performance: float
    average_forgetting: float
    average_forward_transfer: float
    average_raw_forward_transfer: float
    average_area_forward_transfer: float

    final_per_task: tuple[float, ...]
    end_of_task_per_task: tuple[float, ...]
    forgetting_per_task: tuple[float, ...]

    raw_forward_transfer_per_task: tuple[float, ...]
    normalized_forward_transfer_per_task: tuple[float, ...]
    area_forward_transfer_per_task: tuple[float, ...]

    @property
    def num_tasks(self) -> int:
        """Return the number of tasks represented by the metrics."""
        return len(self.final_per_task)

    def as_dict(self) -> dict[str, float]:
        """Return the four main scalar metrics."""
        return {
            "average_performance": self.average_performance,
            "average_forgetting": self.average_forgetting,
            "forward_transfer": self.average_forward_transfer,
            "raw_forward_transfer": (
                self.average_raw_forward_transfer
            ),
            "area_forward_transfer": (
                self.average_area_forward_transfer
            ),
        }


def mean_last_success(
    values: Sequence[float] | np.ndarray,
    *,
    count: int = 5,
) -> float:
    """Return the mean of the last `count` success measurements.

    When fewer than `count` measurements are available, all available
    measurements are used. This mirrors the behavior of selecting the
    final five logged evaluations from a shorter run.

    Args:
        values:
            One non-empty sequence of success rates.

        count:
            Maximum number of final measurements to average.

    Returns:
        Mean success rate over the selected tail.
    """
    if count <= 0:
        raise ValueError(
            "count must be positive."
        )

    success_values = _as_success_vector(
        values,
        name="values",
    )

    selected_values = success_values[
        -min(count, success_values.size):
    ]

    return float(
        np.mean(selected_values)
    )


def compute_final_task_performance(
    final_evaluation_successes: (
        Sequence[Sequence[float]]
        | np.ndarray
    ),
    *,
    tail_size: int = 5,
) -> tuple[float, ...]:
    """Compute final success for every task.

    Args:
        final_evaluation_successes:
            Matrix with shape:

                (evaluation_points, number_of_tasks)

            Each row contains an evaluation of all tasks near or at the
            end of the full continual-learning sequence.

        tail_size:
            Maximum number of final evaluation rows to average.

    Returns:
        One final success value per task.
    """
    if tail_size <= 0:
        raise ValueError(
            "tail_size must be positive."
        )

    success_matrix = _as_success_matrix(
        final_evaluation_successes,
        name="final_evaluation_successes",
    )

    selected_rows = success_matrix[
        -min(tail_size, success_matrix.shape[0]):
    ]

    final_per_task = np.mean(
        selected_rows,
        axis=0,
    )

    return tuple(
        float(value)
        for value in final_per_task
    )


def compute_end_of_task_performance(
    active_task_success_curves: (
        Sequence[Sequence[float] | np.ndarray]
    ),
    *,
    tail_size: int = 5,
) -> tuple[float, ...]:
    """Compute each task's performance when its training period ends.

    Args:
        active_task_success_curves:
            One success-rate curve per task.

            Curve i must contain evaluations of task i while task i was
            the active training task.

        tail_size:
            Maximum number of final points from each task curve to
            average.

    Returns:
        One end-of-task success value per task.
    """
    if tail_size <= 0:
        raise ValueError(
            "tail_size must be positive."
        )

    curves = _validate_curve_collection(
        active_task_success_curves,
        name="active_task_success_curves",
    )

    return tuple(
        mean_last_success(
            curve,
            count=tail_size,
        )
        for curve in curves
    )


def compute_forward_transfer(
    active_task_success_curves: (
        Sequence[Sequence[float] | np.ndarray]
    ),
    baseline_success_curves: (
        Sequence[Sequence[float] | np.ndarray]
    ),
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
]:
    """Compute raw and normalized forward transfer for every task.

    For task i:

        raw_FT_i =
            mean(continual_curve_i)
            - mean(single_task_baseline_curve_i)

        normalized_FT_i =
            raw_FT_i
            / (
                1
                - mean(single_task_baseline_curve_i)
            )

        A baseline mean success of 1.0 is rejected because normalized
        forward transfer has zero remaining headroom.

    Args:
        active_task_success_curves:
            Success curve for each task while learned in the continual
            sequence.

        baseline_success_curves:
            Corresponding success curve from SAC trained on each task
            independently from scratch.

    Returns:
        Tuple containing:

        1. per-task raw forward transfer;
        2. per-task normalized forward transfer.
    """
    continual_curves = _validate_curve_collection(
        active_task_success_curves,
        name="active_task_success_curves",
    )

    baseline_curves = _validate_curve_collection(
        baseline_success_curves,
        name="baseline_success_curves",
    )

    if len(continual_curves) != len(
        baseline_curves
    ):
        raise ValueError(
            "Continual and baseline curve collections must "
            "contain the same number of tasks."
        )

    raw_transfer: list[float] = []
    normalized_transfer: list[float] = []

    for task_index, (
        continual_curve,
        baseline_curve,
    ) in enumerate(
        zip(
            continual_curves,
            baseline_curves,
            strict=True,
        )
    ):
        if (
            continual_curve.size
            != baseline_curve.size
        ):
            raise ValueError(
                "Continual and baseline curves must contain the "
                "same number of evaluation points for each task: "
                f"task={task_index}, "
                f"continual={continual_curve.size}, "
                f"baseline={baseline_curve.size}."
            )

        continual_mean = float(
            np.mean(continual_curve)
        )

        baseline_mean = float(
            np.mean(baseline_curve)
        )

        raw_value = (
            continual_mean
            - baseline_mean
        )

        remaining_headroom = (
            1.0
            - baseline_mean
        )

        raw_transfer.append(
            float(raw_value)
        )

        if (
            remaining_headroom
            <= _DENOMINATOR_TOLERANCE
        ):
            raise ValueError(
                "Normalized forward transfer is undefined when the "
                "single-task baseline mean is 1.0."
            )
        else:
            normalized_value = (
                raw_value
                / remaining_headroom
            )

            normalized_transfer.append(
                float(normalized_value)
            )

    return (
        tuple(raw_transfer),
        tuple(normalized_transfer),
    )


def compute_area_forward_transfer(
    active_task_success_curves: (
        Sequence[Sequence[float] | np.ndarray]
    ),
    baseline_success_curves: (
        Sequence[Sequence[float] | np.ndarray]
    ),
) -> tuple[float, ...]:
    """Compute area-based normalized forward transfer for every task.

    For task i:

        area_FT_i =
            integral(continual_curve_i - baseline_curve_i)
            / integral(1 - baseline_curve_i)

    The integrals are approximated with the trapezoidal rule over the
    shared evaluation grid index. A baseline curve with zero remaining
    headroom area is rejected because the normalized metric is
    undefined.
    """
    continual_curves = _validate_curve_collection(
        active_task_success_curves,
        name="active_task_success_curves",
    )

    baseline_curves = _validate_curve_collection(
        baseline_success_curves,
        name="baseline_success_curves",
    )

    if len(continual_curves) != len(
        baseline_curves
    ):
        raise ValueError(
            "Continual and baseline curve collections must "
            "contain the same number of tasks."
        )

    area_transfer: list[float] = []

    for task_index, (
        continual_curve,
        baseline_curve,
    ) in enumerate(
        zip(
            continual_curves,
            baseline_curves,
            strict=True,
        )
    ):
        if (
            continual_curve.size
            != baseline_curve.size
        ):
            raise ValueError(
                "Continual and baseline curves must contain the "
                "same number of evaluation points for each task: "
                f"task={task_index}, "
                f"continual={continual_curve.size}, "
                f"baseline={baseline_curve.size}."
            )

        x_axis = np.arange(
            continual_curve.size,
            dtype=np.float64,
        )
        numerator = float(
            np.trapz(
                continual_curve - baseline_curve,
                x=x_axis,
            )
        )
        denominator = float(
            np.trapz(
                1.0 - baseline_curve,
                x=x_axis,
            )
        )

        if denominator <= _DENOMINATOR_TOLERANCE:
            raise ValueError(
                "Area forward transfer is undefined when the "
                "single-task baseline headroom area is zero."
            )

        area_transfer.append(
            numerator / denominator
        )

    return tuple(
        float(value)
        for value in area_transfer
    )


def compute_continual_metrics(
    *,
    final_evaluation_successes: (
        Sequence[Sequence[float]]
        | np.ndarray
    ),
    active_task_success_curves: (
        Sequence[Sequence[float] | np.ndarray]
    ),
    baseline_success_curves: (
        Sequence[Sequence[float] | np.ndarray]
    ),
    tail_size: int = 5,
) -> ContinualMetricsResult:
    """Compute the complete continual-learning metric summary.

    Args:
        final_evaluation_successes:
            Evaluations of all tasks at the end of the full task
            sequence.

            Shape:

                (evaluation_points, number_of_tasks)

        active_task_success_curves:
            For each task, the task's success curve while that task was
            actively being trained in the continual sequence.

        baseline_success_curves:
            For each task, the learning curve of SAC trained on that task
            independently from scratch.

        tail_size:
            Number of final evaluation points used for performance and
            forgetting calculations.

    Returns:
        ContinualMetricsResult.
    """
    final_per_task = (
        compute_final_task_performance(
            final_evaluation_successes,
            tail_size=tail_size,
        )
    )

    end_of_task_per_task = (
        compute_end_of_task_performance(
            active_task_success_curves,
            tail_size=tail_size,
        )
    )

    task_count = len(
        final_per_task
    )

    if (
        len(end_of_task_per_task)
        != task_count
    ):
        raise ValueError(
            "Final evaluations and active-task curves must "
            "represent the same number of tasks."
        )

    if len(
        baseline_success_curves
    ) != task_count:
        raise ValueError(
            "Final evaluations and baseline curves must "
            "represent the same number of tasks."
        )

    forgetting_per_task = tuple(
        float(
            end_of_task
            - final
        )
        for end_of_task, final in zip(
            end_of_task_per_task,
            final_per_task,
            strict=True,
        )
    )

    (
        raw_forward_transfer,
        normalized_forward_transfer,
    ) = compute_forward_transfer(
        active_task_success_curves,
        baseline_success_curves,
    )

    area_forward_transfer = (
        compute_area_forward_transfer(
            active_task_success_curves,
            baseline_success_curves,
        )
    )

    average_performance = float(
        np.mean(final_per_task)
    )

    average_forgetting = float(
        np.mean(forgetting_per_task)
    )

    average_raw_forward_transfer = float(
        np.mean(raw_forward_transfer)
    )

    average_area_forward_transfer = float(
        np.mean(area_forward_transfer)
    )

    normalized_transfer_array = np.asarray(
        normalized_forward_transfer,
        dtype=np.float64,
    )

    if np.all(
        np.isnan(
            normalized_transfer_array
        )
    ):
        average_forward_transfer = float("nan")
    else:
        average_forward_transfer = float(
            np.nanmean(
                normalized_transfer_array
            )
        )

    result = ContinualMetricsResult(
        average_performance=(
            average_performance
        ),
        average_forgetting=(
            average_forgetting
        ),
        average_forward_transfer=(
            average_forward_transfer
        ),
        average_raw_forward_transfer=(
            average_raw_forward_transfer
        ),
        average_area_forward_transfer=(
            average_area_forward_transfer
        ),
        final_per_task=(
            final_per_task
        ),
        end_of_task_per_task=(
            end_of_task_per_task
        ),
        forgetting_per_task=(
            forgetting_per_task
        ),
        raw_forward_transfer_per_task=(
            raw_forward_transfer
        ),
        normalized_forward_transfer_per_task=(
            normalized_forward_transfer
        ),
        area_forward_transfer_per_task=(
            area_forward_transfer
        ),
    )

    _validate_result_is_finite(
        result
    )

    return result


def _as_success_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Convert and validate one non-empty success vector."""
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, "
            f"got shape {array.shape}."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    _validate_success_values(
        array,
        name=name,
    )

    return np.clip(
        array,
        0.0,
        1.0,
    )


def _as_success_matrix(
    values: (
        Sequence[Sequence[float]]
        | np.ndarray
    ),
    *,
    name: str,
) -> np.ndarray:
    """Convert and validate one non-empty success matrix."""
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional, "
            f"got shape {array.shape}."
        )

    if (
        array.shape[0] == 0
        or array.shape[1] == 0
    ):
        raise ValueError(
            f"{name} must not be empty."
        )

    _validate_success_values(
        array,
        name=name,
    )

    return np.clip(
        array,
        0.0,
        1.0,
    )


def _validate_curve_collection(
    curves: (
        Sequence[Sequence[float] | np.ndarray]
    ),
    *,
    name: str,
) -> tuple[np.ndarray, ...]:
    """Validate one collection containing one curve per task."""
    if len(curves) == 0:
        raise ValueError(
            f"{name} must contain at least one task."
        )

    return tuple(
        _as_success_vector(
            curve,
            name=f"{name}[{task_index}]",
        )
        for task_index, curve in enumerate(
            curves
        )
    )


def _validate_success_values(
    values: np.ndarray,
    *,
    name: str,
) -> None:
    """Validate finite success rates in the interval [0, 1]."""
    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    if np.any(
        values < -_SUCCESS_TOLERANCE
    ) or np.any(
        values > 1.0 + _SUCCESS_TOLERANCE
    ):
        raise ValueError(
            f"{name} must contain success rates in [0, 1]."
        )


def _validate_result_is_finite(
    result: ContinualMetricsResult,
) -> None:
    """Validate required metrics and permitted undefined normalized FT.

    Average Performance, Forgetting, raw FT, and all corresponding
    per-task values must remain finite. Normalized FT may be NaN only
    when a task's independent baseline mean success is 1.0.
    """
    required_scalar_values = (
        result.average_performance,
        result.average_forgetting,
        result.average_raw_forward_transfer,
        result.average_area_forward_transfer,
    )

    if not all(
        math.isfinite(value)
        for value in required_scalar_values
    ):
        raise RuntimeError(
            "A required continual-learning scalar metric is "
            "non-finite."
        )

    if not (
        math.isfinite(
            result.average_forward_transfer
        )
        or math.isnan(
            result.average_forward_transfer
        )
    ):
        raise RuntimeError(
            "Average normalized forward transfer must be finite "
            "or NaN."
        )

    required_sequence_values = (
        result.final_per_task,
        result.end_of_task_per_task,
        result.forgetting_per_task,
        result.raw_forward_transfer_per_task,
        result.area_forward_transfer_per_task,
    )

    if not all(
        math.isfinite(value)
        for sequence in required_sequence_values
        for value in sequence
    ):
        raise RuntimeError(
            "A required continual-learning per-task metric is "
            "non-finite."
        )

    if not all(
        math.isfinite(value)
        or math.isnan(value)
        for value in (
            result.normalized_forward_transfer_per_task
        )
    ):
        raise RuntimeError(
            "Normalized forward transfer values must be finite "
            "or NaN."
        )
