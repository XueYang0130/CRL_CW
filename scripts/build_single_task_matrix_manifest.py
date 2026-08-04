from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-summaries",
        type=str,
        default=(
            "outputs/single_task_baselines/cw10_v3_v1_500k_seeds1_2/"
            "aggregate/task_summaries.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=("final", "best_return", "best_success"),
        default="best_return",
    )
    parser.add_argument("--expected-total-steps", type=int, default=500000)
    parser.add_argument("--include-self-pairs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=str, default="outputs/diagnostic_matrices")
    parser.add_argument("--run-name", type=str, default="single_task_all_pairs")
    return parser.parse_args()


def checkpoint_name(metric: str) -> str:
    mapping = {
        "final": "final.pt",
        "best_return": "best_return.pt",
        "best_success": "best_success.pt",
    }
    return mapping[metric]


def load_run_summary(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    return json.loads(summary_path.read_text())


def preferred_checkpoint_for_runs(
    *,
    run_dirs: list[Path],
    checkpoint_metric: str,
) -> str:
    if len(run_dirs) == 1:
        return str(run_dirs[0] / "checkpoints" / checkpoint_name(checkpoint_metric))
    ranked = []
    for run_dir in run_dirs:
        summary = load_run_summary(run_dir)
        ranked.append(
            (
                float(summary.get("final_stochastic_success_rate", 0.0)),
                float(summary.get("best_success_rate", 0.0)),
                float(summary.get("final_stochastic_average_return", float("-inf"))),
                float(summary.get("best_return", float("-inf"))),
                str(run_dir),
            )
        )
    ranked.sort(reverse=True)
    return str(Path(ranked[0][-1]) / "checkpoints" / checkpoint_name(checkpoint_metric))


def main() -> None:
    args = parse_args()
    task_summaries_path = Path(args.task_summaries).expanduser().resolve()
    with task_summaries_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    task_entries: list[dict[str, object]] = []
    for row in rows:
        source_runs = [segment for segment in str(row["source_run_directories"]).split("|") if segment]
        run_paths = [Path(run_dir) for run_dir in source_runs]
        candidate_checkpoints = [
            str(Path(run_dir) / "checkpoints" / checkpoint_name(args.checkpoint_metric))
            for run_dir in source_runs
        ]
        run_summaries = [load_run_summary(run_path) for run_path in run_paths]
        total_steps = [int(summary["total_steps"]) for summary in run_summaries]
        reward_versions = [str(summary["reward_function_version"]) for summary in run_summaries]
        if len(set(total_steps)) != 1:
            raise ValueError(
                f"Task {row['task_name']} mixes total_steps across source runs: {total_steps}"
            )
        if len(set(reward_versions)) != 1:
            raise ValueError(
                f"Task {row['task_name']} mixes reward protocols across source runs: {reward_versions}"
            )
        if args.expected_total_steps is not None and total_steps[0] != args.expected_total_steps:
            raise ValueError(
                f"Task {row['task_name']} uses total_steps={total_steps[0]}, "
                f"expected {args.expected_total_steps}."
            )
        task_entries.append(
            {
                "task_index": int(row["task_index"]),
                "task_name": row["task_name"],
                "source_run_directories": source_runs,
                "candidate_checkpoints": candidate_checkpoints,
                "preferred_checkpoint": preferred_checkpoint_for_runs(
                    run_dirs=run_paths,
                    checkpoint_metric=args.checkpoint_metric,
                ),
                "seed": row.get("seed", ""),
                "num_runs_averaged": int(row.get("num_runs_averaged", "1")),
                "tail_mean_success": float(row.get("tail_mean_stochastic_success", "0.0")),
                "best_success": float(row.get("maximum_stochastic_success", "0.0")),
                "total_steps": total_steps[0],
                "reward_function_version": reward_versions[0],
            }
        )

    pairs: list[dict[str, object]] = []
    for old_entry in task_entries:
        for new_entry in task_entries:
            if not args.include_self_pairs and old_entry["task_index"] == new_entry["task_index"]:
                continue
            pairs.append(
                {
                    "old_task_index": old_entry["task_index"],
                    "old_task_name": old_entry["task_name"],
                    "new_task_index": new_entry["task_index"],
                    "new_task_name": new_entry["task_name"],
                    "preferred_checkpoint": old_entry["preferred_checkpoint"],
                    "candidate_checkpoints": "|".join(old_entry["candidate_checkpoints"]),
                }
            )

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "pairs.csv", fieldnames=pairs[0].keys(), rows=pairs)
    write_json(
        output_dir / "manifest.json",
        {
            "task_summaries_path": str(task_summaries_path),
            "checkpoint_metric": args.checkpoint_metric,
            "expected_total_steps": args.expected_total_steps,
            "include_self_pairs": args.include_self_pairs,
            "tasks": task_entries,
            "pairs": pairs,
        },
    )
    print({"output_dir": str(output_dir), "tasks": len(task_entries), "pairs": len(pairs)})


if __name__ == "__main__":
    main()
