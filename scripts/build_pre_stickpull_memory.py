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

from agents import FullBehaviorCloningSACAgent
from envs import DEFAULT_EPISODE_LENGTH, make_cw_env, extract_success
from methods import get_method
from training.semantic_segments import (
    SEGMENT_ORDER,
    STICKPULL_TRANSFER_SEGMENT_ORDER,
    extract_features,
    segment_label_for_scheme,
)
from utils import load_sac_checkpoint, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--max-source-task-index", type=int, default=3)
    parser.add_argument("--reference-episodes", type=int, default=20)
    parser.add_argument("--reference-max-attempts", type=int, default=80)
    parser.add_argument("--checkpoint-kind", choices=("task_end",), default="task_end")
    parser.add_argument(
        "--segment-scheme",
        choices=("default", "stickpull_transfer"),
        default="default",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def namespace_from_mapping(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**payload)


def build_agent_from_config(
    *,
    config: dict[str, Any],
    device: str,
) -> FullBehaviorCloningSACAgent:
    args = namespace_from_mapping(config)
    method = get_method(str(config["method"]))
    task_name = str(config["tasks"][0])
    env = make_cw_env(
        task_name,
        seed=int(config["seed"]),
        max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
        append_task_id=bool(method.append_task_id),
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
    )
    try:
        agent = method.build_agent(
            args=args,
            observation_dim=int(env.observation_space.shape[0]),
            action_dim=int(env.action_space.shape[0]),
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            total_tasks=len(config["tasks"]),
        )
    finally:
        env.close()
    if not isinstance(agent, FullBehaviorCloningSACAgent):
        raise TypeError("This script requires a FullBehaviorCloningSACAgent-compatible method.")
    return agent.to(device)


def make_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    base_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "memory_libraries"
    )
    run_name = args.run_name or f"pre_stickpull_{run_dir.name}"
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def collect_task_memory(
    *,
    agent: FullBehaviorCloningSACAgent,
    task_name: str,
    task_index: int,
    env_version: str,
    reward_function_version: str,
    append_task_id: bool,
    max_episode_steps: int,
    reference_episodes: int,
    reference_max_attempts: int,
    seed: int,
    segment_scheme: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, int]]:
    env = make_cw_env(
        task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        append_task_id=append_task_id,
        env_version=env_version,
        reward_function_version=reward_function_version,
    )
    kept_observations: list[np.ndarray] = []
    kept_target_means: list[np.ndarray] = []
    kept_target_log_stds: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    segment_counter: Counter[str] = Counter()
    successful_episodes = 0
    attempts = 0
    try:
        while successful_episodes < reference_episodes and attempts < reference_max_attempts:
            observation, _ = env.reset(seed=seed + attempts)
            episode_observations: list[np.ndarray] = []
            episode_success = False
            episode_segments: list[str] = []
            for step_index in range(max_episode_steps):
                action = agent.select_action_with_head(
                    observation,
                    head_index=task_index,
                    deterministic=True,
                )
                next_observation, _, terminated, truncated, info = env.step(action)
                episode_success = episode_success or bool(extract_success(info))
                features = extract_features(env, task_name, success_so_far=episode_success)
                label = segment_label_for_scheme(features, scheme=segment_scheme)
                episode_observations.append(np.asarray(observation, dtype=np.float32).copy())
                episode_segments.append(label)
                observation = next_observation
                if terminated or truncated:
                    break
            attempts += 1
            if not episode_success:
                continue
            successful_episodes += 1
            observations_array = np.stack(episode_observations).astype(np.float32)
            target_means, target_log_stds = agent.compute_reference_targets(observations_array)
            target_means_array = target_means.detach().cpu().numpy().astype(np.float32)
            target_log_stds_array = target_log_stds.detach().cpu().numpy().astype(np.float32)
            for step_index, segment in enumerate(episode_segments):
                kept_observations.append(observations_array[step_index])
                kept_target_means.append(target_means_array[step_index])
                kept_target_log_stds.append(target_log_stds_array[step_index])
                metadata_rows.append(
                    {
                        "source_task_index": task_index,
                        "source_task_name": task_name,
                        "successful_episode_index": successful_episodes - 1,
                        "step_index": step_index,
                        "segment": segment,
                    }
                )
                segment_counter[segment] += 1
    finally:
        env.close()

    if kept_observations:
        payload = {
            "observations": np.stack(kept_observations).astype(np.float32),
            "target_means": np.stack(kept_target_means).astype(np.float32),
            "target_log_stds": np.stack(kept_target_log_stds).astype(np.float32),
        }
    else:
        payload = {
            "observations": np.empty((0, agent.observation_dim), dtype=np.float32),
            "target_means": np.empty((0, agent.action_dim), dtype=np.float32),
            "target_log_stds": np.empty((0, agent.action_dim), dtype=np.float32),
        }
    return payload, metadata_rows, dict(segment_counter)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = load_json(run_dir / "config.json")
    output_dir = make_output_dir(args, run_dir)

    tasks = list(config["tasks"])
    if not 0 <= args.max_source_task_index < len(tasks):
        raise ValueError("max-source-task-index out of range.")
    if args.max_source_task_index >= 4:
        # task index 4 is stick-pull in CW10 order; this script is meant to build history before it
        raise ValueError("max-source-task-index must be <= 3 for the pre-stick-pull library.")

    agent = build_agent_from_config(config=config, device=args.device)
    append_task_id = bool(get_method(str(config["method"])).append_task_id)

    payloads: list[dict[str, np.ndarray]] = []
    metadata_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    summary_segment_order = (
        SEGMENT_ORDER
        if args.segment_scheme == "default"
        else STICKPULL_TRANSFER_SEGMENT_ORDER
    )

    for task_index in range(args.max_source_task_index + 1):
        checkpoint_path = run_dir / "checkpoints" / f"task_{task_index}.pt"
        load_sac_checkpoint(
            agent=agent,
            path=checkpoint_path,
            map_location=args.device,
            load_optimizer=False,
        )
        task_name = tasks[task_index]
        payload, task_metadata_rows, segment_counts = collect_task_memory(
            agent=agent,
            task_name=task_name,
            task_index=task_index,
            env_version=str(config["env_version"]),
            reward_function_version=str(config["reward_function_version"]),
            append_task_id=append_task_id,
            max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
            reference_episodes=args.reference_episodes,
            reference_max_attempts=args.reference_max_attempts,
            seed=args.seed + task_index * 10_000,
            segment_scheme=args.segment_scheme,
        )
        payloads.append(payload)
        metadata_rows.extend(task_metadata_rows)
        summary_row: dict[str, Any] = {
            "source_task_index": task_index,
            "source_task_name": task_name,
            "reference_states": int(payload["observations"].shape[0]),
        }
        for segment_name in summary_segment_order:
            summary_row[segment_name] = int(segment_counts.get(segment_name, 0))
        summary_rows.append(summary_row)

    observations = np.concatenate([payload["observations"] for payload in payloads], axis=0)
    target_means = np.concatenate([payload["target_means"] for payload in payloads], axis=0)
    target_log_stds = np.concatenate([payload["target_log_stds"] for payload in payloads], axis=0)

    np.savez_compressed(
        output_dir / "library.npz",
        observations=observations,
        target_means=target_means,
        target_log_stds=target_log_stds,
    )
    if metadata_rows:
        write_csv(output_dir / "library_rows.csv", metadata_rows[0].keys(), metadata_rows)
    if summary_rows:
        write_csv(output_dir / "task_summary.csv", summary_rows[0].keys(), summary_rows)
    write_json(
        output_dir / "manifest.json",
        {
            "run_dir": str(run_dir),
            "max_source_task_index": args.max_source_task_index,
            "source_tasks": tasks[: args.max_source_task_index + 1],
            "reference_episodes": args.reference_episodes,
            "reference_max_attempts": args.reference_max_attempts,
            "segment_scheme": args.segment_scheme,
            "observation_count": int(observations.shape[0]),
            "files": {
                "library": str(output_dir / "library.npz"),
                "rows": str(output_dir / "library_rows.csv"),
                "task_summary": str(output_dir / "task_summary.csv"),
            },
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "reference_states": int(observations.shape[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
