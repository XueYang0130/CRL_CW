"""Evaluation components for CRL-CW."""

from .continual_metrics import (
    ContinualMetricsResult,
    compute_continual_metrics,
    compute_end_of_task_performance,
    compute_final_task_performance,
    compute_forward_transfer,
    mean_last_success,
)
from .evaluator import (
    EvaluationConfig,
    EvaluationResult,
    SACEvaluator,
)

__all__ = [
    "ContinualMetricsResult",
    "EvaluationConfig",
    "EvaluationResult",
    "SACEvaluator",
    "compute_continual_metrics",
    "compute_end_of_task_performance",
    "compute_final_task_performance",
    "compute_forward_transfer",
    "mean_last_success",
]