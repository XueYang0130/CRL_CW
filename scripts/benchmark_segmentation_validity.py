from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import make_cw_env
from envs.cw_env import extract_success
from methods import get_method
from training.semantic_segments import (
    SEGMENT_ORDER,
    TaskAwareSegmenter,
    TaskAwareV3Segmenter,
    extract_features,
    segment_label,
)
from utils import load_sac_checkpoint, write_json


STEP_FIELDS = (
    "task_index",
    "task_name",
    "episode_index",
    "step_index",
    "success_so_far",
    "heuristic_v1",
    "task_aware_v2",
    "task_aware_v3",
    "task_specific_v3",
    "tcp_to_obj",
    "obj_to_target",
    "progress_ratio",
    "object_displacement",
    "object_motion",
    "gripper_open",
    "action_motion",
    "gripper_action",
    "near_object",
    "grasp_success",
    "grasp_reward",
    "in_place_reward",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task-indices", nargs="+", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--post-success-steps", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/segmentation_validity")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_attempts < args.episodes:
        parser.error("Require episodes > 0 and max-attempts >= episodes.")
    if args.post_success_steps < 0:
        parser.error("post-success-steps must be non-negative.")
    return args


def load_config(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict) or not isinstance(config.get("tasks"), list):
        raise ValueError("Run config must contain a tasks list.")
    return config


def build_agent(config: dict[str, Any], device: str) -> Any:
    effective = dict(config)
    effective["device"] = device
    args = SimpleNamespace(**effective)
    method = get_method(str(config["method"]))
    tasks = list(config["tasks"])
    env = make_cw_env(
        tasks[0],
        seed=int(config["seed"]),
        max_episode_steps=int(config["max_episode_steps"]),
        append_task_id=method.append_task_id,
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
        num_task_ids=len(tasks),
    )
    try:
        return method.build_agent(
            args=args,
            observation_dim=int(env.observation_space.shape[0]),
            action_dim=int(env.action_space.shape[0]),
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            total_tasks=len(tasks),
        )
    finally:
        env.close()


def transition_metrics(labels: list[str]) -> tuple[int, int, int]:
    if not labels:
        return 0, 0, 0
    ranks = {label: index for index, label in enumerate(SEGMENT_ORDER)}
    switches = sum(left != right for left, right in zip(labels, labels[1:]))
    regressions = sum(
        ranks[right] < ranks[left]
        for left, right in zip(labels, labels[1:])
    )
    runs: list[int] = []
    run_length = 1
    for left, right in zip(labels, labels[1:]):
        if left == right:
            run_length += 1
        else:
            runs.append(run_length)
            run_length = 1
    runs.append(run_length)
    short_runs = sum(length < 3 for length in runs)
    return switches, regressions, short_runs


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    tasks = list(config["tasks"])
    task_indices = args.task_indices or list(range(len(tasks)))
    if any(index < 0 or index >= len(tasks) for index in task_indices):
        raise ValueError("task-indices contains an out-of-range task index.")
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    agent = build_agent(config, args.device)
    method = get_method(str(config["method"]))
    step_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for task_index in task_indices:
        task_name = tasks[task_index]
        checkpoint = run_dir / "checkpoints" / f"task_{task_index}.pt"
        load_sac_checkpoint(
            agent=agent,
            path=checkpoint,
            map_location=args.device,
            load_optimizer=False,
        )
        env = make_cw_env(
            task_name,
            seed=int(config["seed"]) + 70_000 + task_index,
            max_episode_steps=int(config["max_episode_steps"]),
            append_task_id=method.append_task_id,
            env_version=str(config["env_version"]),
            reward_function_version=str(config["reward_function_version"]),
            num_task_ids=len(tasks),
        )
        successful = 0
        attempts = 0
        task_rows: list[dict[str, Any]] = []
        episode_metrics: list[dict[str, Any]] = []
        try:
            while successful < args.episodes and attempts < args.max_attempts:
                reset_result = env.reset(seed=int(config["seed"]) + 80_000 + task_index * 1000 + attempts)
                observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                # Initialize episode-relative geometry before the first action.
                extract_features(env, task_name, False)
                segmenter = TaskAwareSegmenter()
                segmenter_v3 = TaskAwareV3Segmenter()
                episode_rows: list[dict[str, Any]] = []
                labels_v1: list[str] = []
                labels_v2: list[str] = []
                labels_v3: list[str] = []
                success_so_far = False
                success_step: int | None = None
                previous_object_position: np.ndarray | None = None
                for step_index in range(int(config["max_episode_steps"])):
                    action = agent.select_action(observation, deterministic=True)
                    object_position = np.asarray(
                        env.unwrapped._get_pos_objects(), dtype=np.float64
                    ).reshape(-1)[:3].copy()
                    step_result = env.step(action)
                    next_observation, _, terminated, truncated, info = step_result
                    step_success = bool(extract_success(info))
                    success_so_far = success_so_far or step_success
                    features = extract_features(
                        env,
                        task_name,
                        success_so_far,
                        previous_object_position=object_position,
                        action=action,
                        info=info,
                    )
                    previous_object_position = object_position
                    label_v1 = segment_label(features)
                    label_v2 = segmenter.label(features)
                    semantic_v3 = segmenter_v3.label(features)
                    labels_v1.append(label_v1)
                    labels_v2.append(label_v2)
                    labels_v3.append(semantic_v3.general)
                    row = {
                        "task_index": task_index,
                        "task_name": task_name,
                        "episode_index": successful,
                        "step_index": step_index,
                        "success_so_far": float(success_so_far),
                        "heuristic_v1": label_v1,
                        "task_aware_v2": label_v2,
                        "task_aware_v3": semantic_v3.general,
                        "task_specific_v3": semantic_v3.task_specific,
                        "tcp_to_obj": features.tcp_to_obj,
                        "obj_to_target": features.obj_to_target,
                        "progress_ratio": features.progress_ratio,
                        "object_displacement": features.object_displacement,
                        "object_motion": features.object_motion,
                        "gripper_open": features.gripper_open,
                        "action_motion": features.action_motion,
                        "gripper_action": features.gripper_action,
                        "near_object": features.near_object,
                        "grasp_success": features.grasp_success,
                        "grasp_reward": features.grasp_reward,
                        "in_place_reward": features.in_place_reward,
                    }
                    episode_rows.append(row)
                    observation = next_observation
                    if step_success and success_step is None:
                        success_step = step_index
                    if (
                        success_step is not None
                        and step_index >= success_step + args.post_success_steps
                    ):
                        break
                    if terminated or truncated:
                        break
                attempts += 1
                if not success_so_far:
                    continue
                successful += 1
                for row in episode_rows:
                    row["episode_index"] = successful - 1
                task_rows.extend(episode_rows)
                v1_switch, v1_regress, v1_short = transition_metrics(labels_v1)
                v2_switch, v2_regress, v2_short = transition_metrics(labels_v2)
                v3_switch, v3_regress, v3_short = transition_metrics(labels_v3)
                episode_metrics.append(
                    {
                        "v1_switches": v1_switch,
                        "v1_regressions": v1_regress,
                        "v1_short_runs": v1_short,
                        "v2_switches": v2_switch,
                        "v2_regressions": v2_regress,
                        "v2_short_runs": v2_short,
                        "v3_switches": v3_switch,
                        "v3_regressions": v3_regress,
                        "v3_short_runs": v3_short,
                    }
                )
        finally:
            env.close()

        step_rows.extend(task_rows)
        counts_v1 = Counter(row["heuristic_v1"] for row in task_rows)
        counts_v2 = Counter(row["task_aware_v2"] for row in task_rows)
        counts_v3 = Counter(row["task_aware_v3"] for row in task_rows)
        specific_v3 = Counter(row["task_specific_v3"] for row in task_rows)
        summaries.append(
            {
                "task_index": task_index,
                "task_name": task_name,
                "requested_successful_episodes": args.episodes,
                "successful_episodes": successful,
                "attempts": attempts,
                "v1_segment_counts": dict(counts_v1),
                "v2_segment_counts": dict(counts_v2),
                "v3_segment_counts": dict(counts_v3),
                "v3_task_specific_counts": dict(specific_v3),
                "v1_mean_switches": float(np.mean([m["v1_switches"] for m in episode_metrics])) if episode_metrics else None,
                "v2_mean_switches": float(np.mean([m["v2_switches"] for m in episode_metrics])) if episode_metrics else None,
                "v1_mean_regressions": float(np.mean([m["v1_regressions"] for m in episode_metrics])) if episode_metrics else None,
                "v2_mean_regressions": float(np.mean([m["v2_regressions"] for m in episode_metrics])) if episode_metrics else None,
                "v1_mean_short_runs": float(np.mean([m["v1_short_runs"] for m in episode_metrics])) if episode_metrics else None,
                "v2_mean_short_runs": float(np.mean([m["v2_short_runs"] for m in episode_metrics])) if episode_metrics else None,
                "v3_mean_switches": float(np.mean([m["v3_switches"] for m in episode_metrics])) if episode_metrics else None,
                "v3_mean_regressions": float(np.mean([m["v3_regressions"] for m in episode_metrics])) if episode_metrics else None,
                "v3_mean_short_runs": float(np.mean([m["v3_short_runs"] for m in episode_metrics])) if episode_metrics else None,
            }
        )
        print(
            f"[segmentation] task={task_name} successes={successful}/{args.episodes} "
            f"attempts={attempts} v1_switches={summaries[-1]['v1_mean_switches']} "
            f"v2_switches={summaries[-1]['v2_mean_switches']} "
            f"v3_switches={summaries[-1]['v3_mean_switches']}"
        )

    with (output_dir / "steps.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=STEP_FIELDS)
        writer.writeheader()
        writer.writerows(step_rows)
    write_json(
        output_dir / "summary.json",
        {
            "source_run": str(run_dir),
            "schemes": ["heuristic_v1", "task_aware_v2", "task_aware_v3"],
            "post_success_steps": args.post_success_steps,
            "tasks": summaries,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "tasks": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
