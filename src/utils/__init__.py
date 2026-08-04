"""General utilities for CRL-CW."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointInfo,
    load_sac_checkpoint,
    save_sac_checkpoint,
)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(dict(payload), file, indent=2, sort_keys=True)
    return destination


def write_csv(
    path: str | Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return destination


def append_csv_rows(
    path: str | Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized_rows = [dict(row) for row in rows]
    if not materialized_rows:
        return destination
    write_header = not destination.exists() or destination.stat().st_size == 0
    with destination.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerows(materialized_rows)
        file.flush()
    return destination

__all__ = [
    "CHECKPOINT_VERSION",
    "CheckpointInfo",
    "load_sac_checkpoint",
    "save_sac_checkpoint",
    "append_csv_rows",
    "write_csv",
    "write_json",
]
