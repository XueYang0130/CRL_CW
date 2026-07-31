from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.selective_memory import (
    load_event_segment_summary,
    score_event_segments,
    select_top_segments,
)
from utils import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-summaries", nargs="+", required=True, type=str)
    parser.add_argument("--score-mode", choices=("drift_mass", "mean_drift", "late_bias"), default="drift_mass")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="outputs/selective_memory")
    parser.add_argument("--run-name", type=str, default="segment_manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    merged_rows: list[dict[str, object]] = []
    for summary_path in args.event_summaries:
        rows = load_event_segment_summary(summary_path)
        for row in rows:
            row = dict(row)
            row["source_path"] = str(Path(summary_path).resolve())
            merged_rows.append(row)

    segment_scores = score_event_segments(merged_rows, score_mode=args.score_mode)
    selected_segments = select_top_segments(segment_scores, top_k=args.top_k)

    if merged_rows:
        write_csv(output_dir / "merged_event_segments.csv", fieldnames=merged_rows[0].keys(), rows=merged_rows)
    if segment_scores:
        write_csv(
            output_dir / "segment_scores.csv",
            fieldnames=segment_scores[0].__dict__.keys(),
            rows=[score.__dict__ for score in segment_scores],
        )
    if selected_segments:
        write_csv(
            output_dir / "selected_segments.csv",
            fieldnames=selected_segments[0].__dict__.keys(),
            rows=[segment.__dict__ for segment in selected_segments],
        )

    write_json(
        output_dir / "manifest.json",
        {
            "score_mode": args.score_mode,
            "top_k": args.top_k,
            "event_summaries": [str(Path(path).resolve()) for path in args.event_summaries],
            "selected_segments": [segment.__dict__ for segment in selected_segments],
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "selected_segments": len(selected_segments)}, indent=2))


if __name__ == "__main__":
    main()
