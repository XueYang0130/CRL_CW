from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import DEFAULT_EPISODE_LENGTH, make_cw_env
from envs.cw_env import extract_success
from evaluation import EvaluationConfig, SACEvaluator
from methods import get_method
from utils import load_sac_checkpoint, write_csv, write_json


SEGMENT_ORDER = (
    "approach",
    "contact_or_alignment",
    "manipulation",
    "finish_or_stabilize",
)


@dataclass(frozen=True)
class FeatureSnapshot:
    task_name: str
    tcp_to_obj: float
    obj_to_target: float
    progress_ratio: float
    object_displacement: float
    gripper_open: float
    success: float


@dataclass(frozen=True)
class EventStepRecord:
    task_name: str
    old_task_index: int
    old_task_name: str
    compared_task_index: int
    compared_task_name: str
    episode_index: int
    step_index: int
    temporal_phase_index: int
    event_segment: str
    reward: float
    success_so_far: float
    tcp_to_obj: float
    obj_to_target: float
    progress_ratio: float
    object_displacement: float
    gripper_open: float
    old_action_0: float
    old_action_1: float
    old_action_2: float
    old_action_3: float
    compared_action_0: float
    compared_action_1: float
    compared_action_2: float
    compared_action_3: float
    drift_l2: float
    drift_dim_0: float
    drift_dim_1: float
    drift_dim_2: float
    drift_dim_3: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--compared-task-indices", nargs="+", required=True, type=int)
    parser.add_argument("--reference-episodes", type=int, default=20)
    parser.add_argument("--reference-max-attempts", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=25)
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


def require_reward_function_version(config: dict[str, Any]) -> str:
    reward_function_version = config.get("reward_function_version")
    if reward_function_version is None:
        raise ValueError(
            "Run config is missing 'reward_function_version'. "
            "This diagnostic script refuses to guess the evaluation protocol."
        )
    return str(reward_function_version)


def build_agent_from_config(config: dict[str, Any]) -> Any:
    args = namespace_from_mapping(config)
    method = get_method(str(config["method"]))
    task_name = str(config["tasks"][0])
    env = make_cw_env(
        task_name,
        seed=int(config["seed"]),
        max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
        append_task_id=bool(method.append_task_id),
        env_version=str(config["env_version"]),
        reward_function_version=require_reward_function_version(config),
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
    return agent


def temporal_phase_index(step_index: int, episode_length: int) -> int:
    progress = (step_index + 1) / max(episode_length, 1)
    return min(4, int(math.floor(progress * 5)))


def make_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    base_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "diagnostics"
    )
    run_name = args.run_name or f"event_segments_{run_dir.name}"
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def evaluate_checkpoint(
    *,
    agent: Any,
    checkpoint_path: Path,
    task_name: str,
    config: dict[str, Any],
    eval_episodes: int,
    eval_seed: int,
) -> dict[str, float | int | str]:
    load_sac_checkpoint(
        agent=agent,
        path=checkpoint_path,
        map_location=config["device"],
        load_optimizer=False,
    )
    method = get_method(str(config["method"]))
    env = make_cw_env(
        task_name,
        seed=eval_seed,
        max_episode_steps=int(config["max_episode_steps"]),
        append_task_id=bool(method.append_task_id),
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
    )
    try:
        evaluator = SACEvaluator(
            env=env,
            agent=agent,
            config=EvaluationConfig(
                num_episodes=eval_episodes,
                max_episode_steps=int(config["max_episode_steps"]),
                seed=eval_seed,
            ),
        )
        result = evaluator.evaluate(deterministic=True)
    finally:
        env.close()
    return {
        "checkpoint": str(checkpoint_path),
        "task_name": task_name,
        "episodes": eval_episodes,
        "success_rate": result.success_rate,
        "average_return": result.mean_return,
        "average_episode_length": result.mean_episode_length,
    }


def _progress_ratio(current: float, initial: float) -> float:
    if initial <= 1e-8:
        return 1.0
    return float(np.clip(1.0 - current / initial, 0.0, 1.0))


def extract_features(env: Any, task_name: str, success_so_far: bool) -> FeatureSnapshot:
    base_env = env.unwrapped
    tcp = np.asarray(base_env.tcp_center, dtype=np.float64)
    obj = np.asarray(base_env._get_pos_objects(), dtype=np.float64)
    target = np.asarray(base_env._target_pos, dtype=np.float64)

    if task_name == "faucet-close-v3":
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        obj_to_target = float(np.linalg.norm(obj - target))
        initial = float(base_env.maxPullDist)
        object_displacement = float(np.linalg.norm(obj - np.asarray(base_env.obj_init_pos, dtype=np.float64)))
        gripper_open = 0.0
        progress_ratio = _progress_ratio(obj_to_target, initial)
    elif task_name == "push-wall-v3":
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        obj_to_target = float(np.linalg.norm(obj - target))
        initial = float(base_env.maxPushDist)
        object_displacement = float(
            np.linalg.norm(obj[:2] - np.asarray(base_env.obj_init_pos, dtype=np.float64)[:2])
        )
        gripper_open = float(env.unwrapped._get_obs()[3])
        progress_ratio = _progress_ratio(obj_to_target, initial)
    elif task_name == "window-close-v3":
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        obj_to_target = float(abs(obj[0] - target[0]))
        initial = float(abs(base_env.window_handle_pos_init[0] - target[0]))
        object_displacement = float(abs(obj[0] - base_env.window_handle_pos_init[0]))
        gripper_open = 0.0
        progress_ratio = _progress_ratio(obj_to_target, initial)
    elif task_name == "handle-press-side-v3":
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        obj_to_target = float(abs(obj[2] - target[2]))
        initial = float(abs(base_env._handle_init_pos[2] - target[2]))
        object_displacement = float(abs(obj[2] - base_env._handle_init_pos[2]))
        gripper_open = 0.0
        progress_ratio = _progress_ratio(obj_to_target, initial)
    else:
        raise ValueError(f"Unsupported event segmentation task: {task_name}")

    return FeatureSnapshot(
        task_name=task_name,
        tcp_to_obj=tcp_to_obj,
        obj_to_target=obj_to_target,
        progress_ratio=progress_ratio,
        object_displacement=object_displacement,
        gripper_open=gripper_open,
        success=float(success_so_far),
    )


def segment_label(features: FeatureSnapshot) -> str:
    task_name = features.task_name
    if task_name == "faucet-close-v3":
        if features.success > 0.0 or features.progress_ratio >= 0.85:
            return "finish_or_stabilize"
        if features.progress_ratio >= 0.20:
            return "manipulation"
        if features.tcp_to_obj <= 0.05:
            return "contact_or_alignment"
        return "approach"

    if task_name == "push-wall-v3":
        if features.success > 0.0 or features.progress_ratio >= 0.85:
            return "finish_or_stabilize"
        if features.progress_ratio >= 0.20 or features.object_displacement >= 0.03:
            return "manipulation"
        if features.tcp_to_obj <= 0.04:
            return "contact_or_alignment"
        return "approach"

    if task_name == "window-close-v3":
        if features.success > 0.0 or features.progress_ratio >= 0.85:
            return "finish_or_stabilize"
        if features.progress_ratio >= 0.25:
            return "manipulation"
        if features.tcp_to_obj <= 0.04:
            return "contact_or_alignment"
        return "approach"

    if task_name == "handle-press-side-v3":
        if features.success > 0.0 or features.progress_ratio >= 0.85:
            return "finish_or_stabilize"
        if features.progress_ratio >= 0.20:
            return "manipulation"
        if features.tcp_to_obj <= 0.04:
            return "contact_or_alignment"
        return "approach"

    raise ValueError(f"Unsupported event segmentation task: {task_name}")


def collect_event_records(
    *,
    old_agent: Any,
    compared_agent: Any,
    old_task_index: int,
    old_task_name: str,
    compared_task_index: int,
    compared_task_name: str,
    config: dict[str, Any],
    reference_episodes: int,
    reference_max_attempts: int,
    rollout_seed: int,
) -> tuple[list[EventStepRecord], dict[str, int]]:
    method = get_method(str(config["method"]))
    env = make_cw_env(
        old_task_name,
        seed=rollout_seed,
        max_episode_steps=int(config["max_episode_steps"]),
        append_task_id=bool(method.append_task_id),
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
    )

    records: list[EventStepRecord] = []
    success_episode_count = 0
    attempt_count = 0

    try:
        while success_episode_count < reference_episodes and attempt_count < reference_max_attempts:
            observation, _ = env.reset(seed=rollout_seed + attempt_count)
            episode_rows: list[dict[str, Any]] = []
            episode_success = False

            for step_index in range(int(config["max_episode_steps"])):
                old_action = old_agent.select_action(observation, deterministic=True)
                compared_action = compared_agent.select_action(observation, deterministic=True)

                features = extract_features(env, old_task_name, episode_success)
                next_observation, reward, terminated, truncated, info = env.step(old_action)
                episode_success = episode_success or bool(extract_success(info))
                features_after = FeatureSnapshot(
                    task_name=features.task_name,
                    tcp_to_obj=features.tcp_to_obj,
                    obj_to_target=features.obj_to_target,
                    progress_ratio=features.progress_ratio,
                    object_displacement=features.object_displacement,
                    gripper_open=features.gripper_open,
                    success=float(episode_success),
                )
                label = segment_label(features_after)
                drift_vector = np.asarray(compared_action, dtype=np.float32) - np.asarray(old_action, dtype=np.float32)
                episode_rows.append(
                    {
                        "step_index": step_index,
                        "event_segment": label,
                        "reward": float(reward),
                        "success_so_far": float(episode_success),
                        "features": features_after,
                        "old_action": np.asarray(old_action, dtype=np.float32),
                        "compared_action": np.asarray(compared_action, dtype=np.float32),
                        "drift_vector": drift_vector,
                        "drift_l2": float(np.linalg.norm(drift_vector, ord=2)),
                    }
                )
                observation = next_observation
                if terminated or truncated:
                    break

            attempt_count += 1
            if not episode_success:
                continue

            episode_length = len(episode_rows)
            for row in episode_rows:
                features_after = row["features"]
                old_action = row["old_action"]
                compared_action = row["compared_action"]
                drift_vector = row["drift_vector"]
                records.append(
                    EventStepRecord(
                        task_name=old_task_name,
                        old_task_index=old_task_index,
                        old_task_name=old_task_name,
                        compared_task_index=compared_task_index,
                        compared_task_name=compared_task_name,
                        episode_index=success_episode_count,
                        step_index=int(row["step_index"]),
                        temporal_phase_index=temporal_phase_index(int(row["step_index"]), episode_length),
                        event_segment=str(row["event_segment"]),
                        reward=float(row["reward"]),
                        success_so_far=float(row["success_so_far"]),
                        tcp_to_obj=features_after.tcp_to_obj,
                        obj_to_target=features_after.obj_to_target,
                        progress_ratio=features_after.progress_ratio,
                        object_displacement=features_after.object_displacement,
                        gripper_open=features_after.gripper_open,
                        old_action_0=float(old_action[0]),
                        old_action_1=float(old_action[1]),
                        old_action_2=float(old_action[2]),
                        old_action_3=float(old_action[3]),
                        compared_action_0=float(compared_action[0]),
                        compared_action_1=float(compared_action[1]),
                        compared_action_2=float(compared_action[2]),
                        compared_action_3=float(compared_action[3]),
                        drift_l2=float(row["drift_l2"]),
                        drift_dim_0=float(abs(drift_vector[0])),
                        drift_dim_1=float(abs(drift_vector[1])),
                        drift_dim_2=float(abs(drift_vector[2])),
                        drift_dim_3=float(abs(drift_vector[3])),
                    )
                )
            success_episode_count += 1
    finally:
        env.close()

    return records, {
        "attempt_count": attempt_count,
        "successful_reference_episodes": success_episode_count,
        "reference_steps": len(records),
    }


def summarize_event_segments(records: list[EventStepRecord]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for label in SEGMENT_ORDER:
        segment_rows = [record for record in records if record.event_segment == label]
        if not segment_rows:
            rows.append(
                {
                    "task_name": records[0].task_name if records else "",
                    "compared_task_index": records[0].compared_task_index if records else "",
                    "compared_task_name": records[0].compared_task_name if records else "",
                    "event_segment": label,
                    "num_steps": 0,
                    "mean_drift_l2": "",
                    "std_drift_l2": "",
                    "mean_progress_ratio": "",
                    "mean_tcp_to_obj": "",
                    "mean_obj_to_target": "",
                }
            )
            continue
        drift = np.asarray([record.drift_l2 for record in segment_rows], dtype=np.float64)
        rows.append(
            {
                "task_name": segment_rows[0].task_name,
                "compared_task_index": segment_rows[0].compared_task_index,
                "compared_task_name": segment_rows[0].compared_task_name,
                "event_segment": label,
                "num_steps": len(segment_rows),
                "mean_drift_l2": float(drift.mean()),
                "std_drift_l2": float(drift.std()),
                "mean_progress_ratio": float(np.mean([record.progress_ratio for record in segment_rows])),
                "mean_tcp_to_obj": float(np.mean([record.tcp_to_obj for record in segment_rows])),
                "mean_obj_to_target": float(np.mean([record.obj_to_target for record in segment_rows])),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = load_json(run_dir / "config.json")
    config["device"] = args.device
    tasks = list(config["tasks"])

    if not 0 <= args.task_index < len(tasks):
        raise ValueError(f"task_index must be in [0, {len(tasks)}), got {args.task_index}.")
    old_task_name = str(tasks[args.task_index])
    supported = {"faucet-close-v3", "push-wall-v3", "window-close-v3", "handle-press-side-v3"}
    if old_task_name not in supported:
        raise ValueError(
            f"Event-based segmentation currently supports only {sorted(supported)}; got {old_task_name}."
        )

    output_dir = make_output_dir(args, run_dir)
    old_checkpoint = run_dir / "checkpoints" / f"task_{args.task_index}.pt"

    old_agent = build_agent_from_config(config)
    compared_agent = build_agent_from_config(config)
    load_sac_checkpoint(
        agent=old_agent,
        path=old_checkpoint,
        map_location=args.device,
        load_optimizer=False,
    )

    task_dir = output_dir / f"task_{args.task_index}_{old_task_name}"
    task_dir.mkdir()

    comparison_summaries: list[dict[str, Any]] = []
    eval_rows: list[dict[str, float | int | str]] = []
    segment_summary_rows: list[dict[str, float | int | str]] = []
    record_rows: list[dict[str, Any]] = []

    old_eval = evaluate_checkpoint(
        agent=old_agent,
        checkpoint_path=old_checkpoint,
        task_name=old_task_name,
        config=config,
        eval_episodes=args.eval_episodes,
        eval_seed=int(config["seed"]) + 1000 + args.task_index,
    )
    old_eval["checkpoint_role"] = "task_end"
    old_eval["task_index"] = args.task_index
    old_eval["compared_task_index"] = args.task_index
    old_eval["compared_task_name"] = old_task_name
    eval_rows.append(old_eval)

    for compared_index in args.compared_task_indices:
        if compared_index <= args.task_index:
            raise ValueError("compared_task_indices must be strictly later than task_index.")
        if compared_index >= len(tasks):
            raise ValueError(f"compared_task_index {compared_index} out of range.")
        compared_task_name = str(tasks[compared_index])
        compared_checkpoint = run_dir / "checkpoints" / f"task_{compared_index}.pt"
        load_sac_checkpoint(
            agent=compared_agent,
            path=compared_checkpoint,
            map_location=args.device,
            load_optimizer=False,
        )
        compared_eval = evaluate_checkpoint(
            agent=compared_agent,
            checkpoint_path=compared_checkpoint,
            task_name=old_task_name,
            config=config,
            eval_episodes=args.eval_episodes,
            eval_seed=int(config["seed"]) + 2000 + args.task_index * 100 + compared_index,
        )
        compared_eval["checkpoint_role"] = "compared"
        compared_eval["task_index"] = args.task_index
        compared_eval["compared_task_index"] = compared_index
        compared_eval["compared_task_name"] = compared_task_name
        eval_rows.append(compared_eval)

        records, diagnostics = collect_event_records(
            old_agent=old_agent,
            compared_agent=compared_agent,
            old_task_index=args.task_index,
            old_task_name=old_task_name,
            compared_task_index=compared_index,
            compared_task_name=compared_task_name,
            config=config,
            reference_episodes=args.reference_episodes,
            reference_max_attempts=args.reference_max_attempts,
            rollout_seed=int(config["seed"]) + 3000 + args.task_index,
        )
        record_rows.extend(record.__dict__ for record in records)
        event_rows = summarize_event_segments(records)
        segment_summary_rows.extend(event_rows)
        comparison_summaries.append(
            {
                "compared_task_index": compared_index,
                "compared_task_name": compared_task_name,
                "compared_success_rate": compared_eval["success_rate"],
                "compared_average_return": compared_eval["average_return"],
                "task_end_success_rate": old_eval["success_rate"],
                "task_end_average_return": old_eval["average_return"],
                **diagnostics,
            }
        )
        print(
            f"[event] task={old_task_name} compare={compared_task_name} "
            f"task_end_success={float(old_eval['success_rate']):.3f} "
            f"compared_success={float(compared_eval['success_rate']):.3f} "
            f"reference_success_episodes={diagnostics['successful_reference_episodes']}"
        )

    if record_rows:
        write_csv(task_dir / "event_reference_steps.csv", fieldnames=record_rows[0].keys(), rows=record_rows)
    if segment_summary_rows:
        write_csv(
            task_dir / "event_segment_summary.csv",
            fieldnames=segment_summary_rows[0].keys(),
            rows=segment_summary_rows,
        )
    write_csv(output_dir / "checkpoint_evaluations.csv", fieldnames=eval_rows[0].keys(), rows=eval_rows)
    write_json(
        task_dir / "summary.json",
        {
            "task_index": args.task_index,
            "task_name": old_task_name,
            "task_end_checkpoint": str(old_checkpoint),
            "task_end_success_rate": old_eval["success_rate"],
            "task_end_average_return": old_eval["average_return"],
            "comparisons": comparison_summaries,
        },
    )
    write_json(
        output_dir / "manifest.json",
        {
            "source_run_dir": str(run_dir),
            "task_index": args.task_index,
            "task_name": old_task_name,
            "compared_task_indices": args.compared_task_indices,
            "reference_episodes": args.reference_episodes,
            "reference_max_attempts": args.reference_max_attempts,
            "eval_episodes": args.eval_episodes,
            "device": args.device,
        },
    )


if __name__ == "__main__":
    main()
