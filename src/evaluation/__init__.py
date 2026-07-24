"""Evaluation components for CRL-CW."""

from .continual_metrics import (
    ContinualMetricsResult,
    compute_area_forward_transfer,
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
from .single_task_baselines import (
    BaselineAggregationResult,
    aggregate_single_task_baselines,
)
from .summary import summarize_continual_run

__all__ = [
    "BaselineAggregationResult",
    "ContinualMetricsResult",
    "EvaluationConfig",
    "EvaluationResult",
    "SACEvaluator",
    "aggregate_single_task_baselines",
    "compute_area_forward_transfer",
    "compute_continual_metrics",
    "compute_end_of_task_performance",
    "compute_final_task_performance",
    "compute_forward_transfer",
    "mean_last_success",
    "summarize_continual_run",
]
