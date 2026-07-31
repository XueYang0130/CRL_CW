from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SIGNAL_FIELDS = (
    "tcp_to_obj",
    "object_displacement",
    "object_motion",
    "near_object",
    "grasp_success",
    "grasp_reward",
    "in_place_reward",
    "progress_ratio",
    "success_so_far",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--task-indices", nargs="+", type=int, required=True)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--output-name", default="boundary_analysis")
    args = parser.parse_args()
    if args.window < 0:
        parser.error("--window must be non-negative.")
    return args


def read_rows(path: Path, task_indices: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for raw in csv.DictReader(file):
            task_index = int(raw["task_index"])
            if task_index not in task_indices:
                continue
            row: dict[str, Any] = dict(raw)
            row["task_index"] = task_index
            row["episode_index"] = int(raw["episode_index"])
            row["step_index"] = int(raw["step_index"])
            for field in SIGNAL_FIELDS:
                row[field] = float(raw[field])
            rows.append(row)
    return rows


def contiguous_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(rows) + 1):
        if index < len(rows) and rows[index]["task_aware_v3"] == rows[start]["task_aware_v3"]:
            continue
        run = rows[start:index]
        runs.append(
            {
                "label": run[0]["task_aware_v3"],
                "specific_label": run[0]["task_specific_v3"],
                "start_step": run[0]["step_index"],
                "end_step": run[-1]["step_index"],
                "length": len(run),
            }
        )
        start = index
    return runs


def transition_windows(
    rows: list[dict[str, Any]], window: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index in range(1, len(rows)):
        before = rows[index - 1]
        after = rows[index]
        if before["task_aware_v3"] == after["task_aware_v3"]:
            continue
        transition_id = f"{before['task_aware_v3']}->{after['task_aware_v3']}"
        for offset_index in range(max(0, index - window), min(len(rows), index + window + 1)):
            source = rows[offset_index]
            record = {
                "task_index": source["task_index"],
                "task_name": source["task_name"],
                "episode_index": source["episode_index"],
                "transition_step": after["step_index"],
                "transition": transition_id,
                "relative_step": offset_index - index,
                "step_index": source["step_index"],
                "general_label": source["task_aware_v3"],
                "specific_label": source["task_specific_v3"],
            }
            record.update({field: source[field] for field in SIGNAL_FIELDS})
            output.append(record)
    return output


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    rows = read_rows(benchmark_dir / "steps.csv", set(args.task_indices))
    if not rows:
        raise ValueError("No matching rows found in benchmark steps.csv.")

    episodes: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episodes[(row["task_index"], row["episode_index"])].append(row)

    run_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    per_task_runs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    per_task_transitions: dict[int, Counter[str]] = defaultdict(Counter)

    for (task_index, episode_index), episode_rows in sorted(episodes.items()):
        episode_rows.sort(key=lambda row: row["step_index"])
        runs = contiguous_runs(episode_rows)
        for run_index, run in enumerate(runs):
            record = {
                "task_index": task_index,
                "task_name": episode_rows[0]["task_name"],
                "episode_index": episode_index,
                "run_index": run_index,
                **run,
            }
            run_rows.append(record)
            per_task_runs[task_index].append(record)
        for left, right in zip(runs, runs[1:]):
            per_task_transitions[task_index][f"{left['label']}->{right['label']}"] += 1
        boundary_rows.extend(transition_windows(episode_rows, args.window))

    for task_index in sorted(per_task_runs):
        task_runs = per_task_runs[task_index]
        lengths: dict[str, list[int]] = defaultdict(list)
        episodes_with_label: dict[str, set[int]] = defaultdict(set)
        for run in task_runs:
            lengths[run["label"]].append(run["length"])
            episodes_with_label[run["label"]].add(run["episode_index"])
        episode_count = len({run["episode_index"] for run in task_runs})
        summary.append(
            {
                "task_index": task_index,
                "task_name": task_runs[0]["task_name"],
                "episodes": episode_count,
                "label_coverage": {
                    label: len(indices) / episode_count
                    for label, indices in episodes_with_label.items()
                },
                "mean_run_length": {
                    label: sum(values) / len(values) for label, values in lengths.items()
                },
                "transition_counts": dict(per_task_transitions[task_index]),
            }
        )

    output_dir = benchmark_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=run_rows[0].keys())
        writer.writeheader()
        writer.writerows(run_rows)
    with (output_dir / "transition_windows.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=boundary_rows[0].keys())
        writer.writeheader()
        writer.writerows(boundary_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"tasks": summary}, file, indent=2)

    for task in summary:
        coverage = ", ".join(
            f"{label}={value:.0%}" for label, value in task["label_coverage"].items()
        )
        transitions = ", ".join(
            f"{name}:{count}" for name, count in task["transition_counts"].items()
        )
        print(f"[boundary] {task['task_name']} coverage: {coverage}")
        print(f"[boundary] {task['task_name']} transitions: {transitions}")
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
