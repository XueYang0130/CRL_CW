from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.semantic_segments import SEGMENT_ORDER


@dataclass(frozen=True)
class SegmentSelection:
    selected_segments: tuple[str, ...]
    priority: tuple[str, ...]
    weights: tuple[tuple[str, float], ...]
    reason: str
    selection_source: str


def normalize_segment_weights(
    weights: dict[str, Any] | None,
    *,
    selected_segments: tuple[str, ...],
    field_name: str = "weights",
) -> tuple[tuple[str, float], ...]:
    if not selected_segments:
        raise ValueError(f"{field_name} requires at least one selected segment.")
    if len(set(selected_segments)) != len(selected_segments):
        raise ValueError(f"{field_name} selected segments must be unique.")
    if weights is None:
        uniform = 1.0 / len(selected_segments)
        return tuple((segment, uniform) for segment in selected_segments)
    if not isinstance(weights, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    unknown = sorted(set(weights).difference(selected_segments))
    if unknown:
        raise ValueError(f"{field_name} contains unselected segments: {', '.join(unknown)}.")
    raw: list[float] = []
    for segment in selected_segments:
        if segment not in weights:
            raise ValueError(f"{field_name} is missing selected segment {segment}.")
        value = float(weights[segment])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field_name}.{segment} must be finite and positive.")
        raw.append(value)
    total = sum(raw)
    normalized = [value / total for value in raw]
    normalized[-1] = 1.0 - sum(normalized[:-1])
    return tuple(zip(selected_segments, normalized, strict=True))


def _validate_segments(segments: list[str] | tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not segments:
        raise ValueError(f"{field_name} must contain at least one segment.")
    if len(set(segments)) != len(segments):
        raise ValueError(f"{field_name} must not contain duplicate segments.")
    invalid = sorted(set(segments).difference(SEGMENT_ORDER))
    if invalid:
        raise ValueError(
            f"Unsupported semantic segments in {field_name}: {', '.join(invalid)}. "
            f"Supported: {', '.join(SEGMENT_ORDER)}."
        )
    return tuple(segments)


def _validate_weights(
    weights: dict[str, Any] | None,
    *,
    selected_segments: tuple[str, ...],
    field_name: str,
) -> tuple[tuple[str, float], ...]:
    return normalize_segment_weights(
        weights,
        selected_segments=selected_segments,
        field_name=field_name,
    )


def load_segment_selection_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("segment selection manifest must be a JSON object.")
    return payload


def select_segments_for_task_pair(
    *,
    previous_task_name: str | None,
    new_task_name: str,
    manifest: dict[str, Any] | None,
    fallback_segments: list[str] | tuple[str, ...],
) -> SegmentSelection:
    fallback = _validate_segments(list(fallback_segments), field_name="fallback_segments")
    default_priority = fallback
    if manifest is None:
        return SegmentSelection(
            selected_segments=fallback,
            priority=default_priority,
            weights=_validate_weights(None, selected_segments=fallback, field_name="weights"),
            reason="No segment selection manifest provided.",
            selection_source="default",
        )

    default_segments = manifest.get("default_segments", list(fallback))
    default_segments = _validate_segments(default_segments, field_name="default_segments")

    if previous_task_name is None:
        return SegmentSelection(
            selected_segments=default_segments,
            priority=default_segments,
            weights=_validate_weights(
                manifest.get("default_weights"),
                selected_segments=default_segments,
                field_name="default_weights",
            ),
            reason="First task uses default semantic segment selection.",
            selection_source="default",
        )

    pair_key = f"{previous_task_name}->{new_task_name}"
    pair_entry = manifest.get("pairs", {}).get(pair_key)
    if pair_entry is None:
        return SegmentSelection(
            selected_segments=default_segments,
            priority=default_segments,
            weights=_validate_weights(
                manifest.get("default_weights"),
                selected_segments=default_segments,
                field_name="default_weights",
            ),
            reason=f"No pair-specific selection found for {pair_key}.",
            selection_source="default",
        )
    if not isinstance(pair_entry, dict):
        raise ValueError(f"Pair entry for {pair_key} must be a JSON object.")

    selected_segments = _validate_segments(
        pair_entry.get("selected_segments", list(default_segments)),
        field_name=f"{pair_key}.selected_segments",
    )
    priority = _validate_segments(
        pair_entry.get("priority", list(selected_segments)),
        field_name=f"{pair_key}.priority",
    )
    invalid_priority = sorted(set(priority).difference(selected_segments))
    if invalid_priority:
        raise ValueError(
            f"{pair_key}.priority contains unselected segments: "
            f"{', '.join(invalid_priority)}."
        )
    reason = str(pair_entry.get("reason", f"Pair-specific semantic selection for {pair_key}."))
    weights = _validate_weights(
        pair_entry.get("weights"),
        selected_segments=selected_segments,
        field_name=f"{pair_key}.weights",
    )

    return SegmentSelection(
        selected_segments=selected_segments,
        priority=priority,
        weights=weights,
        reason=reason,
        selection_source="manifest",
    )


def select_task_specific_segments_for_task(
    *,
    task_name: str,
    manifest: dict[str, Any] | None,
) -> SegmentSelection | None:
    if manifest is None:
        return None
    task_entry = manifest.get("tasks", {}).get(task_name)
    if task_entry is None:
        return None
    if not isinstance(task_entry, dict):
        raise ValueError(f"Task-specific entry for {task_name} must be a JSON object.")
    selected_segments = _validate_segments(
        task_entry.get("selected_segments", []),
        field_name=f"{task_name}.selected_segments",
    )
    priority = _validate_segments(
        task_entry.get("priority", list(selected_segments)),
        field_name=f"{task_name}.priority",
    )
    invalid_priority = sorted(set(priority).difference(selected_segments))
    if invalid_priority:
        raise ValueError(
            f"{task_name}.priority contains unselected segments: "
            f"{', '.join(invalid_priority)}."
        )
    reason = str(task_entry.get("reason", f"Task-specific semantic selection for {task_name}."))
    weights = _validate_weights(
        task_entry.get("weights"),
        selected_segments=selected_segments,
        field_name=f"{task_name}.weights",
    )
    return SegmentSelection(
        selected_segments=selected_segments,
        priority=priority,
        weights=weights,
        reason=reason,
        selection_source="task_specific_manifest",
    )
