from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from envs import get_cw10_tasks
from training.semantic_segments import SEGMENT_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-version", choices=("v2", "v3"), default="v3")
    parser.add_argument("--sequence-task-count", type=int, default=10)
    parser.add_argument(
        "--default-segments",
        nargs="+",
        default=["contact_or_alignment", "manipulation"],
    )
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = get_cw10_tasks(args.env_version)[: args.sequence_task_count]
    pairs: dict[str, dict[str, object]] = {}
    for index in range(1, len(tasks)):
        previous_task = tasks[index - 1]
        current_task = tasks[index]
        pairs[f"{previous_task}->{current_task}"] = {
            "selected_segments": list(args.default_segments),
            "priority": list(args.default_segments),
            "reason": "Placeholder selection. Replace with task-adaptive semantic choice.",
        }

    payload = {
        "segment_order": list(SEGMENT_ORDER),
        "default_segments": list(args.default_segments),
        "pairs": pairs,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
