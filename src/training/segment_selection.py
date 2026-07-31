from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.semantic_segments import SEGMENT_ORDER


@dataclass(frozen=True)
class SegmentSelection:
    selected_segments: tuple[str, ...]
    priority: tuple[str, ...]
    reason: str
    selection_source: str


def _validate_segments(segments: list[str] | tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not segments:
        raise ValueError(f"{field_name} must contain at least one segment.")
    invalid = sorted(set(segments).difference(SEGMENT_ORDER))
    if invalid:
        raise ValueError(
            f"Unsupported semantic segments in {field_name}: {', '.join(invalid)}. "
            f"Supported: {', '.join(SEGMENT_ORDER)}."
        )
    return tuple(segments)


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
            reason="No segment selection manifest provided.",
            selection_source="default",
        )

    default_segments = manifest.get("default_segments", list(fallback))
    default_segments = _validate_segments(default_segments, field_name="default_segments")

    if previous_task_name is None:
        return SegmentSelection(
            selected_segments=default_segments,
            priority=default_segments,
            reason="First task uses default semantic segment selection.",
            selection_source="default",
        )

    pair_key = f"{previous_task_name}->{new_task_name}"
    pair_entry = manifest.get("pairs", {}).get(pair_key)
    if pair_entry is None:
        return SegmentSelection(
            selected_segments=default_segments,
            priority=default_segments,
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
    reason = str(pair_entry.get("reason", f"Pair-specific semantic selection for {pair_key}."))

    return SegmentSelection(
        selected_segments=selected_segments,
        priority=priority,
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
    reason = str(task_entry.get("reason", f"Task-specific semantic selection for {task_name}."))
    return SegmentSelection(
        selected_segments=selected_segments,
        priority=priority,
        reason=reason,
        selection_source="task_specific_manifest",
    )
