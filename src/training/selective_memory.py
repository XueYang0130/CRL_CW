from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SegmentScore:
    old_task_name: str
    compared_task_name: str
    event_segment: str
    mean_drift_l2: float
    num_steps: int
    mean_progress_ratio: float
    score: float
    source_path: str


@dataclass(frozen=True)
class SelectedSegment:
    old_task_name: str
    compared_task_name: str
    event_segment: str
    score: float
    reason: str
    source_path: str


def load_event_segment_summary(path: str | Path) -> list[dict[str, Any]]:
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def _parse_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _parse_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def score_event_segments(
    rows: list[dict[str, Any]],
    *,
    score_mode: str = "drift_mass",
) -> list[SegmentScore]:
    scores: list[SegmentScore] = []
    for row in rows:
        mean_drift_l2 = _parse_float(row.get("mean_drift_l2"))
        num_steps = _parse_int(row.get("num_steps"))
        mean_progress_ratio = _parse_float(row.get("mean_progress_ratio"))
        if mean_drift_l2 is None or num_steps is None or num_steps <= 0:
            continue
        if score_mode == "drift_mass":
            score = mean_drift_l2 * num_steps
        elif score_mode == "mean_drift":
            score = mean_drift_l2
        elif score_mode == "late_bias":
            if mean_progress_ratio is None:
                raise ValueError("late_bias scoring requires mean_progress_ratio.")
            score = mean_drift_l2 * (0.5 + mean_progress_ratio) * num_steps
        else:
            raise ValueError(f"Unsupported score_mode: {score_mode}")
        scores.append(
            SegmentScore(
                old_task_name=str(row.get("task_name", "")),
                compared_task_name=str(row.get("compared_task_name", "")),
                event_segment=str(row.get("event_segment", "")),
                mean_drift_l2=mean_drift_l2,
                num_steps=num_steps,
                mean_progress_ratio=0.0 if mean_progress_ratio is None else mean_progress_ratio,
                score=float(score),
                source_path=str(row.get("source_path", "")),
            )
        )
    return scores


def select_top_segments(
    segment_scores: list[SegmentScore],
    *,
    top_k: int = 1,
) -> list[SelectedSegment]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    selected: list[SelectedSegment] = []
    grouped: dict[tuple[str, str], list[SegmentScore]] = {}
    for score in segment_scores:
        key = (score.old_task_name, score.compared_task_name)
        grouped.setdefault(key, []).append(score)
    for (_, _), group_scores in grouped.items():
        ranked = sorted(group_scores, key=lambda item: item.score, reverse=True)[:top_k]
        for rank, score in enumerate(ranked, start=1):
            selected.append(
                SelectedSegment(
                    old_task_name=score.old_task_name,
                    compared_task_name=score.compared_task_name,
                    event_segment=score.event_segment,
                    score=score.score,
                    reason=f"rank_{rank}_by_score",
                    source_path=score.source_path,
                )
            )
    return selected
