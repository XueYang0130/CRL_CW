from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evaluation.continual_metrics import compute_continual_metrics


def summarize_continual_run(
    *,
    final_success_rows: Sequence[Sequence[float]],
    active_task_success_curves: Sequence[Sequence[float]],
    baseline_success_curves: Sequence[Sequence[float]] | None,
    tail_size: int = 5,
) -> dict[str, Any]:
    if baseline_success_curves is None:
        result = compute_continual_metrics(
            final_evaluation_successes=final_success_rows,
            active_task_success_curves=active_task_success_curves,
            baseline_success_curves=active_task_success_curves,
            tail_size=tail_size,
        )
        summary = result.as_dict()
        summary["forward_transfer"] = None
        summary["raw_forward_transfer"] = None
        summary["area_forward_transfer"] = None
        summary["forward_transfer_available"] = False
        summary["raw_forward_transfer_per_task"] = [None] * len(result.raw_forward_transfer_per_task)
        summary["normalized_forward_transfer_per_task"] = [None] * len(
            result.normalized_forward_transfer_per_task
        )
        summary["area_forward_transfer_per_task"] = [None] * len(
            result.area_forward_transfer_per_task
        )
    else:
        result = compute_continual_metrics(
            final_evaluation_successes=final_success_rows,
            active_task_success_curves=active_task_success_curves,
            baseline_success_curves=baseline_success_curves,
            tail_size=tail_size,
        )
        summary = result.as_dict()
        summary["forward_transfer_available"] = True
        summary["raw_forward_transfer_per_task"] = list(result.raw_forward_transfer_per_task)
        summary["normalized_forward_transfer_per_task"] = list(
            result.normalized_forward_transfer_per_task
        )
        summary["area_forward_transfer_per_task"] = list(
            result.area_forward_transfer_per_task
        )

    summary["final_per_task"] = list(result.final_per_task)
    summary["end_of_task_per_task"] = list(result.end_of_task_per_task)
    summary["forgetting_per_task"] = list(result.forgetting_per_task)
    return summary
