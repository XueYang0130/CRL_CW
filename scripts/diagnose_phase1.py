from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
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

from envs import DEFAULT_EPISODE_LENGTH, make_cw_env
from envs.cw_env import extract_success
from evaluation import EvaluationConfig, SACEvaluator
from methods import get_method
from utils import load_sac_checkpoint, write_csv, write_json


NUM_PHASES = 5


@dataclass(frozen=True)
class StepRecord:
    compared_task_index: int
    compared_task_name: str
    episode_index: int
    step_index: int
    phase_index: int
    old_return_prefix: float
    reward: float
    success_so_far: float
    old_action_0: float
    old_action_1: float
    old_action_2: float
    old_action_3: float
    final_action_0: float
    final_action_1: float
    final_action_2: float
    final_action_3: float
    drift_l2: float
    drift_dim_0: float
    drift_dim_1: float
    drift_dim_2: float
    drift_dim_3: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--task-indices", nargs="+", required=True, type=int)
    parser.add_argument("--reference-episodes", type=int, default=20)
    parser.add_argument("--reference-max-attempts", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument(
        "--compare-mode",
        choices=("final", "subsequent", "all"),
        default="final",
    )
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


def phase_index_for_step(step_index: int, episode_length: int) -> int:
    if episode_length <= 0:
        return 0
    progress = (step_index + 1) / episode_length
    return min(NUM_PHASES - 1, int(math.floor(progress * NUM_PHASES)))


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


def collect_reference_steps(
    *,
    old_agent: Any,
    compared_agent: Any,
    compared_task_index: int,
    compared_task_name: str,
    task_name: str,
    config: dict[str, Any],
    reference_episodes: int,
    reference_max_attempts: int,
    rollout_seed: int,
) -> tuple[list[StepRecord], dict[str, float | int]]:
    method = get_method(str(config["method"]))
    env = make_cw_env(
        task_name,
        seed=rollout_seed,
        max_episode_steps=int(config["max_episode_steps"]),
        append_task_id=bool(method.append_task_id),
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
    )

    records: list[StepRecord] = []
    attempt_count = 0
    success_episode_count = 0

    try:
        while (
            success_episode_count < reference_episodes
            and attempt_count < reference_max_attempts
        ):
            observation, _ = env.reset(seed=rollout_seed + attempt_count)
            episode_steps: list[dict[str, Any]] = []
            episode_return = 0.0
            episode_success = False

            for step_index in range(int(config["max_episode_steps"])):
                old_action = old_agent.select_action(
                    observation,
                    deterministic=True,
                )
                compared_action = compared_agent.select_action(
                    observation,
                    deterministic=True,
                )

                next_observation, reward, terminated, truncated, info = env.step(old_action)
                episode_return += float(reward)
                episode_success = episode_success or bool(extract_success(info))

                drift_vector = np.asarray(compared_action, dtype=np.float32) - np.asarray(old_action, dtype=np.float32)
                episode_steps.append(
                    {
                        "step_index": step_index,
                        "reward": float(reward),
                        "success_so_far": float(episode_success),
                        "old_return_prefix": float(episode_return),
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

            episode_length = len(episode_steps)
            for step in episode_steps:
                phase_index = phase_index_for_step(
                    step_index=int(step["step_index"]),
                    episode_length=episode_length,
                )
                drift_vector = np.asarray(step["drift_vector"], dtype=np.float32)
                old_action = np.asarray(step["old_action"], dtype=np.float32)
                compared_action = np.asarray(step["compared_action"], dtype=np.float32)
                records.append(
                    StepRecord(
                        compared_task_index=compared_task_index,
                        compared_task_name=compared_task_name,
                        episode_index=success_episode_count,
                        step_index=int(step["step_index"]),
                        phase_index=phase_index,
                        old_return_prefix=float(step["old_return_prefix"]),
                        reward=float(step["reward"]),
                        success_so_far=float(step["success_so_far"]),
                        old_action_0=float(old_action[0]),
                        old_action_1=float(old_action[1]),
                        old_action_2=float(old_action[2]),
                        old_action_3=float(old_action[3]),
                        final_action_0=float(compared_action[0]),
                        final_action_1=float(compared_action[1]),
                        final_action_2=float(compared_action[2]),
                        final_action_3=float(compared_action[3]),
                        drift_l2=float(step["drift_l2"]),
                        drift_dim_0=float(abs(drift_vector[0])),
                        drift_dim_1=float(abs(drift_vector[1])),
                        drift_dim_2=float(abs(drift_vector[2])),
                        drift_dim_3=float(abs(drift_vector[3])),
                    )
                )
            success_episode_count += 1
    finally:
        env.close()

    diagnostics = {
        "attempt_count": attempt_count,
        "successful_reference_episodes": success_episode_count,
        "reference_steps": len(records),
    }
    return records, diagnostics


def summarize_phase_records(
    task_name: str,
    records: list[StepRecord],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for phase_index in range(NUM_PHASES):
        phase_records = [record for record in records if record.phase_index == phase_index]
        if not phase_records:
            rows.append(
                {
                    "task_name": task_name,
                    "compared_task_index": records[0].compared_task_index if records else "",
                    "compared_task_name": records[0].compared_task_name if records else "",
                    "phase_index": phase_index,
                    "phase_label": f"phase_{phase_index + 1}",
                    "num_steps": 0,
                    "mean_drift_l2": "",
                    "std_drift_l2": "",
                    "mean_drift_dim_0": "",
                    "mean_drift_dim_1": "",
                    "mean_drift_dim_2": "",
                    "mean_drift_dim_3": "",
                }
            )
            continue
        drift_l2 = np.asarray([record.drift_l2 for record in phase_records], dtype=np.float64)
        rows.append(
            {
                "task_name": task_name,
                "compared_task_index": phase_records[0].compared_task_index,
                "compared_task_name": phase_records[0].compared_task_name,
                "phase_index": phase_index,
                "phase_label": f"phase_{phase_index + 1}",
                "num_steps": len(phase_records),
                "mean_drift_l2": float(drift_l2.mean()),
                "std_drift_l2": float(drift_l2.std()),
                "mean_drift_dim_0": float(np.mean([record.drift_dim_0 for record in phase_records])),
                "mean_drift_dim_1": float(np.mean([record.drift_dim_1 for record in phase_records])),
                "mean_drift_dim_2": float(np.mean([record.drift_dim_2 for record in phase_records])),
                "mean_drift_dim_3": float(np.mean([record.drift_dim_3 for record in phase_records])),
            }
        )
    return rows


def make_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    base_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "diagnostics"
    )
    run_name = (
        args.run_name
        if args.run_name is not None
        else f"phase1_{run_dir.name}"
    )
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def comparison_indices(
    *,
    task_index: int,
    total_tasks: int,
    compare_mode: str,
) -> list[int]:
    if compare_mode == "final":
        return [total_tasks - 1]
    if compare_mode == "subsequent":
        return list(range(task_index + 1, total_tasks))
    if compare_mode == "all":
        return list(range(task_index, total_tasks))
    raise ValueError(f"Unsupported compare_mode: {compare_mode}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = load_json(run_dir / "config.json")
    config["device"] = args.device
    tasks = list(config["tasks"])
    output_dir = make_output_dir(args, run_dir)

    evaluation_rows: list[dict[str, float | int | str]] = []
    phase_rows: list[dict[str, float | int | str]] = []
    manifest: list[dict[str, Any]] = []

    for task_index in args.task_indices:
        if not 0 <= task_index < len(tasks):
            raise ValueError(f"task_index must be in [0, {len(tasks)}), got {task_index}.")
        task_name = str(tasks[task_index])
        old_checkpoint = run_dir / "checkpoints" / f"task_{task_index}.pt"

        old_agent = build_agent_from_config(config)
        compared_agent = build_agent_from_config(config)

        load_sac_checkpoint(
            agent=old_agent,
            path=old_checkpoint,
            map_location=args.device,
            load_optimizer=False,
        )
        old_eval = evaluate_checkpoint(
            agent=old_agent,
            checkpoint_path=old_checkpoint,
            task_name=task_name,
            config=config,
            eval_episodes=args.eval_episodes,
            eval_seed=int(config["seed"]) + 1000 + task_index,
        )
        old_eval["checkpoint_role"] = "task_end"
        old_eval["task_index"] = task_index
        old_eval["compared_task_index"] = task_index
        old_eval["compared_task_name"] = task_name
        evaluation_rows.append(old_eval)

        task_dir = output_dir / f"task_{task_index}_{task_name}"
        task_dir.mkdir()
        comparison_summaries: list[dict[str, Any]] = []
        reference_steps_rows: list[dict[str, Any]] = []
        task_phase_rows_all: list[dict[str, float | int | str]] = []

        for compared_index in comparison_indices(
            task_index=task_index,
            total_tasks=len(tasks),
            compare_mode=args.compare_mode,
        ):
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
                task_name=task_name,
                config=config,
                eval_episodes=args.eval_episodes,
                eval_seed=int(config["seed"]) + 2000 + task_index * 100 + compared_index,
            )
            compared_eval["checkpoint_role"] = (
                "task_end" if compared_index == task_index else "compared"
            )
            compared_eval["task_index"] = task_index
            compared_eval["compared_task_index"] = compared_index
            compared_eval["compared_task_name"] = compared_task_name
            evaluation_rows.append(compared_eval)

            if compared_index == task_index:
                continue

            records, diagnostics = collect_reference_steps(
                old_agent=old_agent,
                compared_agent=compared_agent,
                compared_task_index=compared_index,
                compared_task_name=compared_task_name,
                task_name=task_name,
                config=config,
                reference_episodes=args.reference_episodes,
                reference_max_attempts=args.reference_max_attempts,
                rollout_seed=int(config["seed"]) + 3000 + task_index,
            )
            reference_steps_rows.extend(record.__dict__ for record in records)

            task_phase_rows = summarize_phase_records(task_name, records)
            task_phase_rows_all.extend(task_phase_rows)
            phase_rows.extend(task_phase_rows)

            comparison_summary = {
                "task_index": task_index,
                "task_name": task_name,
                "task_end_checkpoint": str(old_checkpoint),
                "compared_task_index": compared_index,
                "compared_task_name": compared_task_name,
                "compared_checkpoint": str(compared_checkpoint),
                "task_end_success_rate": old_eval["success_rate"],
                "compared_success_rate": compared_eval["success_rate"],
                "task_end_average_return": old_eval["average_return"],
                "compared_average_return": compared_eval["average_return"],
                "reference_episode_target": args.reference_episodes,
                **diagnostics,
            }
            comparison_summaries.append(comparison_summary)

            print(
                f"[phase1] task={task_name} compare={compared_task_name} "
                f"task_end_success={float(old_eval['success_rate']):.3f} "
                f"compared_success={float(compared_eval['success_rate']):.3f} "
                f"reference_success_episodes={diagnostics['successful_reference_episodes']}"
            )

        if reference_steps_rows:
            write_csv(
                task_dir / "reference_steps.csv",
                fieldnames=reference_steps_rows[0].keys(),
                rows=reference_steps_rows,
            )
        if task_phase_rows_all:
            write_csv(
                task_dir / "phase_summary.csv",
                fieldnames=task_phase_rows_all[0].keys(),
                rows=task_phase_rows_all,
            )

        task_summary = {
            "task_index": task_index,
            "task_name": task_name,
            "task_end_checkpoint": str(old_checkpoint),
            "task_end_success_rate": old_eval["success_rate"],
            "task_end_average_return": old_eval["average_return"],
            "reference_episode_target": args.reference_episodes,
            "comparisons": comparison_summaries,
        }
        write_json(task_dir / "summary.json", task_summary)
        manifest.append(task_summary)

    write_csv(
        output_dir / "checkpoint_evaluations.csv",
        fieldnames=evaluation_rows[0].keys() if evaluation_rows else [],
        rows=evaluation_rows,
    )
    write_csv(
        output_dir / "phase_summary_all_tasks.csv",
        fieldnames=phase_rows[0].keys() if phase_rows else [],
        rows=phase_rows,
    )
    write_json(
        output_dir / "manifest.json",
        {
            "source_run_dir": str(run_dir),
            "task_indices": args.task_indices,
            "reference_episodes": args.reference_episodes,
            "reference_max_attempts": args.reference_max_attempts,
            "eval_episodes": args.eval_episodes,
            "compare_mode": args.compare_mode,
            "device": args.device,
            "tasks": manifest,
        },
    )


if __name__ == "__main__":
    main()
