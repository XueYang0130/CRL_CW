from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import get_cw10_tasks
from training.semantic_segments import SEGMENT_ORDER, STICKPULL_TRANSFER_SEGMENT_ORDER
from utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-version", choices=("v2", "v3"), default="v3")
    parser.add_argument("--sequence-task-count", type=int, default=10)
    parser.add_argument(
        "--segment-scheme",
        choices=("default", "stickpull_transfer"),
        default="default",
    )
    parser.add_argument("--memory-manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="llm_segment_selection")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def build_task_pairs(tasks: list[str]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index in range(1, len(tasks)):
        previous_task = tasks[index - 1]
        current_task = tasks[index]
        pairs.append(
            {
                "pair_index": index,
                "previous_task": previous_task,
                "new_task": current_task,
                "pair_key": f"{previous_task}->{current_task}",
            }
        )
    return pairs


def memory_summary_rows(memory_manifest_path: Path | None) -> list[dict[str, Any]]:
    if memory_manifest_path is None:
        return []
    manifest = load_json(memory_manifest_path)
    rows_path = Path(manifest["files"]["task_summary"]).expanduser().resolve()
    if not rows_path.exists():
        return []
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    with rows_path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if line_index == 0:
                header = parts
                continue
            assert header is not None
            row = {key: value for key, value in zip(header, parts, strict=True)}
            rows.append(row)
    return rows


def build_prompt(
    *,
    tasks: list[str],
    pairs: list[dict[str, Any]],
    segment_order: tuple[str, ...],
    memory_rows: list[dict[str, Any]],
    segment_scheme: str,
) -> str:
    pair_lines = "\n".join(
        f"- {entry['pair_key']}" for entry in pairs
    )
    if memory_rows:
        summary_lines = "\n".join(
            "- "
            + row["source_task_name"]
            + ": "
            + ", ".join(
                f"{key}={row[key]}"
                for key in row
                if key not in {"source_task_index", "source_task_name"}
            )
            for row in memory_rows
        )
    else:
        summary_lines = "- No memory summary attached."

    return f"""You are helping select semantic replay segments for continual reinforcement learning.

Task sequence:
{chr(10).join(f"- {task}" for task in tasks)}

Consecutive task pairs:
{pair_lines}

Segment scheme:
- {segment_scheme}

Allowed segment labels:
{chr(10).join(f"- {segment}" for segment in segment_order)}

Observed source-memory summary:
{summary_lines}

Output requirements:
1. Produce a JSON manifest with keys:
   - "segment_order"
   - "default_segments"
   - "pairs"
   - optional "tasks"
2. For each pair key under "pairs", provide:
   - "selected_segments": subset of allowed labels
   - "priority": ordered subset of allowed labels
   - "reason": short explanation
3. If you want task-specific always-keep segments, add them under "tasks".
4. Return valid JSON only.

Selection heuristic:
- Prefer narrower selections when a specific transition clearly needs specialized transfer.
- Prefer broader selections when multiple source tasks likely contribute complementary knowledge.
- For stick-pull related transfer, prioritize contact-preserving, alignment, hooking, and terminal control semantics over generic approach states.
"""


def main() -> None:
    args = parse_args()
    tasks = get_cw10_tasks(args.env_version)[: args.sequence_task_count]
    pairs = build_task_pairs(tasks)
    segment_order = (
        SEGMENT_ORDER
        if args.segment_scheme == "default"
        else STICKPULL_TRANSFER_SEGMENT_ORDER
    )

    memory_manifest_path = (
        Path(args.memory_manifest).expanduser().resolve()
        if args.memory_manifest is not None
        else None
    )
    memory_rows = memory_summary_rows(memory_manifest_path)

    output_dir = Path(args.output_dir).expanduser().resolve() / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    starter_manifest = {
        "segment_order": list(segment_order),
        "default_segments": list(segment_order[: min(2, len(segment_order))]),
        "pairs": {
            entry["pair_key"]: {
                "selected_segments": list(segment_order[: min(2, len(segment_order))]),
                "priority": list(segment_order[: min(2, len(segment_order))]),
                "reason": "Placeholder. Replace with LLM-selected segments.",
            }
            for entry in pairs
        },
        "tasks": {},
    }

    prompt_text = build_prompt(
        tasks=tasks,
        pairs=pairs,
        segment_order=segment_order,
        memory_rows=memory_rows,
        segment_scheme=args.segment_scheme,
    )

    write_json(
        output_dir / "task_context.json",
        {
            "tasks": tasks,
            "pairs": pairs,
            "segment_scheme": args.segment_scheme,
            "segment_order": list(segment_order),
            "memory_summary_rows": memory_rows,
        },
    )
    write_json(output_dir / "starter_manifest.json", starter_manifest)
    (output_dir / "prompt.md").write_text(prompt_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "pair_count": len(pairs),
                "segment_scheme": args.segment_scheme,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
