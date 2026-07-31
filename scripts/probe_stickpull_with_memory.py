from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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

from agents import FullBehaviorCloningSACAgent, ReplayBuffer
from envs import DEFAULT_EPISODE_LENGTH, make_cw_env
from evaluation import EvaluationConfig, SACEvaluator
from methods import get_method
from training.sac_trainer import SACTrainer, SACTrainerConfig
from utils import load_sac_checkpoint, save_sac_checkpoint, write_csv, write_json


EVAL_FIELDS = [
    "environment_step",
    "gradient_updates",
    "stochastic_average_return",
    "stochastic_success_rate",
    "stochastic_average_episode_length",
    "alpha",
    "elapsed_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--memory-manifest", required=True, type=str)
    parser.add_argument("--new-task-index", type=int, default=4)
    parser.add_argument("--source-task-indices", nargs="*", type=int, default=None)
    parser.add_argument("--segments", nargs="*", type=str, default=None)
    parser.add_argument("--boost-segments", nargs="*", type=str, default=None)
    parser.add_argument("--boost-source-task-indices", nargs="*", type=int, default=None)
    parser.add_argument("--boost-multiplier", type=int, default=1)
    parser.add_argument("--variant", choices=("no_memory", "full_memory", "filtered_memory"), default="filtered_memory")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--stoch-eval-episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--reset-buffer-on-task-change", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-optimizer-on-task-change", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
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
        else PROJECT_ROOT / "outputs" / "stickpull_memory_probe"
    )
    run_name = args.run_name or f"{args.variant}_{run_dir.name}"
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    return output_dir


def load_filtered_memory(
    *,
    manifest_path: Path,
    source_task_indices: list[int] | None,
    allowed_segments: set[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]], int]:
    manifest = load_json(manifest_path)
    base_dir = manifest_path.parent
    library_npz = np.load(base_dir / "library.npz")
    rows_path = base_dir / "library_rows.csv"
    with rows_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    keep_indices: list[int] = []
    for index, row in enumerate(rows):
        task_ok = source_task_indices is None or int(row["source_task_index"]) in source_task_indices
        segment_ok = allowed_segments is None or row["segment"] in allowed_segments
        if task_ok and segment_ok:
            keep_indices.append(index)
    observations = library_npz["observations"][keep_indices]
    target_means = library_npz["target_means"][keep_indices]
    target_log_stds = library_npz["target_log_stds"][keep_indices]
    kept_rows = [rows[index] for index in keep_indices]
    return observations, target_means, target_log_stds, kept_rows, len(keep_indices)


def apply_boosted_memory(
    *,
    observations: np.ndarray,
    target_means: np.ndarray,
    target_log_stds: np.ndarray,
    rows: list[dict[str, str]],
    boost_segments: set[str] | None,
    boost_source_task_indices: set[int] | None,
    boost_multiplier: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if boost_multiplier <= 1:
        return observations, target_means, target_log_stds, 0
    if observations.shape[0] == 0:
        return observations, target_means, target_log_stds, 0

    boosted_indices: list[int] = []
    for index, row in enumerate(rows):
        task_ok = (
            boost_source_task_indices is None
            or int(row["source_task_index"]) in boost_source_task_indices
        )
        segment_ok = (
            boost_segments is None
            or row["segment"] in boost_segments
        )
        if task_ok and segment_ok:
            boosted_indices.append(index)

    if not boosted_indices:
        return observations, target_means, target_log_stds, 0

    repeated_indices = np.repeat(
        np.asarray(boosted_indices, dtype=np.int64),
        boost_multiplier - 1,
    )
    boosted_observations = np.concatenate(
        (observations, observations[repeated_indices]),
        axis=0,
    )
    boosted_target_means = np.concatenate(
        (target_means, target_means[repeated_indices]),
        axis=0,
    )
    boosted_target_log_stds = np.concatenate(
        (target_log_stds, target_log_stds[repeated_indices]),
        axis=0,
    )
    return (
        boosted_observations,
        boosted_target_means,
        boosted_target_log_stds,
        int(repeated_indices.shape[0]),
    )


def evaluate_task(
    *,
    agent: FullBehaviorCloningSACAgent,
    task_name: str,
    env_version: str,
    reward_function_version: str,
    episodes: int,
    deterministic: bool,
    max_episode_steps: int,
    seed: int,
    append_task_id: bool,
) -> dict[str, float]:
    env = make_cw_env(
        task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        append_task_id=append_task_id,
        env_version=env_version,
        reward_function_version=reward_function_version,
    )
    try:
        evaluator = SACEvaluator(
            env,
            agent,
            EvaluationConfig(
                num_episodes=episodes,
                max_episode_steps=max_episode_steps,
                seed=seed,
            ),
        )
        result = evaluator.evaluate(deterministic=deterministic)
    finally:
        env.close()
    return {
        "average_return": result.mean_return,
        "success_rate": result.success_rate,
        "average_episode_length": result.mean_episode_length,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = load_json(run_dir / "config.json")
    output_dir = make_output_dir(args, run_dir)

    tasks = list(config["tasks"])
    if not 0 <= args.new_task_index < len(tasks):
        raise ValueError("new-task-index out of range.")
    if args.new_task_index == 0:
        raise ValueError("new-task-index must be later than at least one previous task.")

    method = get_method(str(config["method"]))
    append_task_id = bool(method.append_task_id)
    new_task_name = tasks[args.new_task_index]
    previous_task_index = args.new_task_index - 1

    agent = build_agent_from_config(config=config, device=args.device)
    load_sac_checkpoint(
        agent=agent,
        path=run_dir / "checkpoints" / f"task_{previous_task_index}.pt",
        map_location=args.device,
        load_optimizer=False,
    )
    replay_buffer = ReplayBuffer(
        observation_dim=agent.observation_dim,
        action_dim=agent.action_dim,
        capacity=int(config.get("replay_size", 1_000_000)),
        seed=args.seed,
    )
    agent.on_task_start(task_index=args.new_task_index, replay_buffer=replay_buffer)
    if args.reset_buffer_on_task_change:
        replay_buffer.clear()
    if args.reset_optimizer_on_task_change:
        agent.rebuild_optimizer()
    if args.copy_alpha:
        agent.copy_alpha(
            source_task_index=previous_task_index,
            target_task_index=args.new_task_index,
        )

    reference_state_count = 0
    boosted_reference_state_count = 0
    if args.variant != "no_memory":
        observations, target_means, target_log_stds, rows, reference_state_count = load_filtered_memory(
            manifest_path=Path(args.memory_manifest).expanduser().resolve(),
            source_task_indices=args.source_task_indices,
            allowed_segments=None if args.variant == "full_memory" or args.segments is None else set(args.segments),
        )
        observations, target_means, target_log_stds, boosted_reference_state_count = apply_boosted_memory(
            observations=observations,
            target_means=target_means,
            target_log_stds=target_log_stds,
            rows=rows,
            boost_segments=None if args.boost_segments is None else set(args.boost_segments),
            boost_source_task_indices=None if args.boost_source_task_indices is None else set(args.boost_source_task_indices),
            boost_multiplier=args.boost_multiplier,
        )
        agent.set_reference_memory_with_targets(
            observations=observations,
            target_means=target_means,
            target_log_stds=target_log_stds,
        )

    train_env = make_cw_env(
        new_task_name,
        seed=args.seed,
        max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
        append_task_id=append_task_id,
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
    )
    trainer = SACTrainer(
        env=train_env,
        agent=agent,
        replay_buffer=replay_buffer,
        config=SACTrainerConfig(
            total_steps=args.steps,
            batch_size=args.batch_size,
            start_steps=args.start_steps,
            update_after=args.update_after,
            update_every=args.update_every,
            max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
            callback_every_steps=args.eval_every,
            seed=args.seed + args.new_task_index,
            reseed_global_rng=False,
        ),
    )

    started_at = time.time()
    eval_rows: list[dict[str, float | int | str]] = []

    run_config = {
        **config,
        "probe_variant": args.variant,
        "probe_source_task_indices": args.source_task_indices,
        "probe_segments": args.segments,
        "probe_reference_state_count": reference_state_count,
        "probe_boost_segments": args.boost_segments,
        "probe_boost_source_task_indices": args.boost_source_task_indices,
        "probe_boost_multiplier": args.boost_multiplier,
        "probe_boosted_reference_state_count": boosted_reference_state_count,
        "probe_steps": args.steps,
        "probe_seed": args.seed,
        "probe_new_task_index": args.new_task_index,
        "probe_new_task_name": new_task_name,
    }
    write_json(output_dir / "config.json", run_config)

    best_success = float("-inf")
    best_return = float("-inf")

    def evaluate(step: int, gradient_updates: int, update_metrics: dict[str, float] | None) -> None:
        del update_metrics
        result = evaluate_task(
            agent=agent,
            task_name=new_task_name,
            env_version=str(config["env_version"]),
            reward_function_version=str(config["reward_function_version"]),
            episodes=args.stoch_eval_episodes,
            deterministic=False,
            max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
            seed=args.seed + 50_000 + step,
            append_task_id=append_task_id,
        )
        eval_rows.append(
            {
                "environment_step": step,
                "gradient_updates": gradient_updates,
                "stochastic_average_return": result["average_return"],
                "stochastic_success_rate": result["success_rate"],
                "stochastic_average_episode_length": result["average_episode_length"],
                "alpha": agent.diagnostic_alpha_value(args.new_task_index),
                "elapsed_seconds": time.time() - started_at,
            }
        )
        nonlocal best_success, best_return
        if result["success_rate"] > best_success:
            best_success = result["success_rate"]
            save_sac_checkpoint(
                agent=agent,
                path=output_dir / "checkpoints" / "best_success.pt",
                environment_step=step,
                metadata={"task_name": new_task_name, "metric": "success"},
            )
        if result["average_return"] > best_return:
            best_return = result["average_return"]
            save_sac_checkpoint(
                agent=agent,
                path=output_dir / "checkpoints" / "best_return.pt",
                environment_step=step,
                metadata={"task_name": new_task_name, "metric": "return"},
            )
        print(
            f"[stickpull-probe] variant={args.variant} step={step:,}/{args.steps:,} "
            f"success={result['success_rate']:.3f} return={result['average_return']:.3f}"
        )

    summary = trainer.train(step_callback=evaluate)
    save_sac_checkpoint(
        agent=agent,
        path=output_dir / "checkpoints" / "final.pt",
        environment_step=summary.total_env_steps,
        metadata={"task_name": new_task_name, "metric": "final"},
    )
    write_csv(output_dir / "evaluations.csv", EVAL_FIELDS, eval_rows)

    final_eval = eval_rows[-1]
    summary_payload = {
        "variant": args.variant,
        "new_task_name": new_task_name,
        "source_task_indices": args.source_task_indices,
        "segments": args.segments,
        "reference_state_count": reference_state_count,
        "boost_segments": args.boost_segments,
        "boost_source_task_indices": args.boost_source_task_indices,
        "boost_multiplier": args.boost_multiplier,
        "boosted_reference_state_count": boosted_reference_state_count,
        "final_success_rate": final_eval["stochastic_success_rate"],
        "final_average_return": final_eval["stochastic_average_return"],
        "best_success_rate": best_success,
        "best_average_return": best_return,
        "completed_episodes": summary.completed_episodes,
        "gradient_updates": summary.gradient_updates,
        "mean_training_return": summary.mean_episode_return,
        "elapsed_seconds": time.time() - started_at,
        "run_directory": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary_payload)


if __name__ == "__main__":
    main()
