from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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

from agents import LocalBehaviorCloningSACAgent, ReplayBuffer
from envs import DEFAULT_EPISODE_LENGTH, make_cw_env, extract_success
from evaluation import EvaluationConfig, SACEvaluator
from methods import get_method
from training.sac_trainer import SACTrainer, SACTrainerConfig
from training.selective_memory import SelectedSegment
from utils import load_sac_checkpoint, save_sac_checkpoint, write_csv, write_json


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
    success: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str)
    parser.add_argument("--old-task-index", required=True, type=int)
    parser.add_argument("--new-task-index", required=True, type=int)
    parser.add_argument("--segments", nargs="+", default=None, choices=SEGMENT_ORDER)
    parser.add_argument("--segment-manifest", type=str, default=None)
    parser.add_argument("--variant", choices=("local_bc", "full_bc", "no_bc"), default="local_bc")
    parser.add_argument("--reference-episodes", type=int, default=20)
    parser.add_argument("--reference-max-attempts", type=int, default=80)
    parser.add_argument("--max-reference-states", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--stoch-eval-episodes", type=int, default=10)
    parser.add_argument("--det-eval-episodes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--local-bc-batch-size", type=int, default=128)
    parser.add_argument("--local-bc-coefficient", type=float, default=10.0)
    parser.add_argument("--use-selective-replay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--selective-replay-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def resolve_segments(
    *,
    args: argparse.Namespace,
    old_task_name: str,
    new_task_name: str,
) -> list[str]:
    if args.segment_manifest is None:
        if args.segments is None:
            raise ValueError("Either --segments or --segment-manifest must be provided.")
        return list(args.segments)

    manifest = load_json(Path(args.segment_manifest).expanduser().resolve())
    raw_segments = manifest.get("selected_segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError("segment manifest must contain a list field 'selected_segments'.")

    matched_segments: list[str] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        segment = SelectedSegment(
            old_task_name=str(raw_segment["old_task_name"]),
            compared_task_name=str(raw_segment["compared_task_name"]),
            event_segment=str(raw_segment["event_segment"]),
            score=float(raw_segment["score"]),
            reason=str(raw_segment["reason"]),
            source_path=str(raw_segment["source_path"]),
        )
        if (
            segment.old_task_name == old_task_name
            and segment.compared_task_name == new_task_name
            and segment.event_segment in SEGMENT_ORDER
        ):
            matched_segments.append(segment.event_segment)
    if not matched_segments:
        raise ValueError(
            "No selected segments in the manifest match "
            f"old_task={old_task_name!r}, new_task={new_task_name!r}."
        )
    unique_segments: list[str] = []
    for segment in matched_segments:
        if segment not in unique_segments:
            unique_segments.append(segment)
    return unique_segments


def require_reward_function_version(config: dict[str, Any]) -> str:
    reward_function_version = config.get("reward_function_version")
    if reward_function_version is None:
        raise ValueError(
            "Run config is missing 'reward_function_version'. "
            "Refusing to guess the environment reward protocol for pairwise BC."
        )
    return str(reward_function_version)


def namespace_from_mapping(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**payload)


def build_local_bc_agent(
    *,
    config: dict[str, Any],
    device: str,
) -> LocalBehaviorCloningSACAgent:
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
        args = namespace_from_mapping(config)
        kwargs = {
            "observation_dim": int(env.observation_space.shape[0]),
            "action_dim": int(env.action_space.shape[0]),
            "action_low": env.action_space.low,
            "action_high": env.action_space.high,
            "num_tasks": 1,
            "task_id_dim": 0,
            "learning_rate": float(args.learning_rate),
            "gamma": float(args.gamma),
            "polyak": float(args.polyak),
            "target_entropy": int(env.action_space.shape[0])
            * math.log(
                float(args.target_output_std) * math.sqrt(2.0 * math.pi * math.e)
            ),
            "initial_log_alpha": float(args.initial_log_alpha),
            "device": device,
            "gradient_clip_norm": getattr(args, "gradient_clip_norm", None),
            "local_bc_coefficient": float(config.get("local_bc_coefficient", 0.0)),
            "local_bc_batch_size": int(config.get("local_bc_batch_size", 128)),
        }
        return LocalBehaviorCloningSACAgent(**kwargs)
    finally:
        env.close()


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
        progress_ratio = _progress_ratio(obj_to_target, initial)
    elif task_name == "push-wall-v3":
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        obj_to_target = float(np.linalg.norm(obj - target))
        initial = float(base_env.maxPushDist)
        object_displacement = float(
            np.linalg.norm(obj[:2] - np.asarray(base_env.obj_init_pos, dtype=np.float64)[:2])
        )
        progress_ratio = _progress_ratio(obj_to_target, initial)
    elif task_name == "window-close-v3":
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        obj_to_target = float(abs(obj[0] - target[0]))
        initial = float(abs(base_env.window_handle_pos_init[0] - target[0]))
        object_displacement = float(abs(obj[0] - base_env.window_handle_pos_init[0]))
        progress_ratio = _progress_ratio(obj_to_target, initial)
    elif task_name == "handle-press-side-v3":
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        obj_to_target = float(abs(obj[2] - target[2]))
        initial = float(abs(base_env._handle_init_pos[2] - target[2]))
        object_displacement = float(abs(obj[2] - base_env._handle_init_pos[2]))
        progress_ratio = _progress_ratio(obj_to_target, initial)
    else:
        raise ValueError(f"Unsupported task for local BC segmentation: {task_name}")
    return FeatureSnapshot(
        task_name=task_name,
        tcp_to_obj=tcp_to_obj,
        obj_to_target=obj_to_target,
        progress_ratio=progress_ratio,
        object_displacement=object_displacement,
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
    raise ValueError(f"Unsupported task for local BC segmentation: {task_name}")


def collect_reference_memory(
    *,
    agent: LocalBehaviorCloningSACAgent,
    task_name: str,
    env_version: str,
    reward_function_version: str,
    max_episode_steps: int,
    seed: int,
    reference_episodes: int,
    reference_max_attempts: int,
    max_reference_states: int | None,
    segments: set[str],
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    env = make_cw_env(
        task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        append_task_id=False,
        env_version=env_version,
        reward_function_version=reward_function_version,
    )
    kept_observations: list[np.ndarray] = []
    kept_target_means: list[np.ndarray] = []
    kept_target_log_stds: list[np.ndarray] = []
    kept_rewards: list[float] = []
    kept_next_observations: list[np.ndarray] = []
    kept_terminated: list[float] = []
    segment_rows: list[dict[str, Any]] = []
    successful_episodes = 0
    attempts = 0
    try:
        while successful_episodes < reference_episodes and attempts < reference_max_attempts:
            observation, _ = env.reset(seed=seed + attempts)
            episode_rows: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, str]] = []
            episode_success = False
            for _ in range(max_episode_steps):
                action = agent.select_action(observation, deterministic=True)
                features = extract_features(env, task_name, episode_success)
                next_observation, reward, terminated, truncated, info = env.step(action)
                episode_success = episode_success or bool(extract_success(info))
                label = segment_label(
                    FeatureSnapshot(
                        task_name=features.task_name,
                        tcp_to_obj=features.tcp_to_obj,
                        obj_to_target=features.obj_to_target,
                        progress_ratio=features.progress_ratio,
                        object_displacement=features.object_displacement,
                        success=float(episode_success),
                    )
                )
                episode_rows.append(
                    (
                        np.asarray(observation, dtype=np.float32).copy(),
                        np.asarray(action, dtype=np.float32).copy(),
                        float(reward),
                        np.asarray(next_observation, dtype=np.float32).copy(),
                        bool(terminated),
                        label,
                    )
                )
                observation = next_observation
                if terminated or truncated:
                    break
            attempts += 1
            if not episode_success:
                continue
            successful_episodes += 1
            for (
                observation_row,
                action_row,
                reward_row,
                next_observation_row,
                terminated_row,
                label,
            ) in episode_rows:
                keep = variant == "full_bc" or (variant == "local_bc" and label in segments)
                if keep:
                    kept_observations.append(observation_row)
                    observation_tensor = torch.as_tensor(
                        observation_row,
                        dtype=torch.float32,
                        device=agent.device,
                    ).unsqueeze(0)
                    target_means_tensor, target_log_stds_tensor = agent._cloning_distribution_parameters(
                        observation_tensor
                    )
                    kept_target_means.append(
                        target_means_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    )
                    kept_target_log_stds.append(
                        target_log_stds_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    )
                    kept_rewards.append(reward_row)
                    kept_next_observations.append(next_observation_row)
                    kept_terminated.append(float(terminated_row))
                segment_rows.append(
                    {
                        "successful_episode_index": successful_episodes - 1,
                        "segment": label,
                        "kept": int(keep),
                    }
                )
    finally:
        env.close()
    if variant != "no_bc" and not kept_observations:
        raise RuntimeError("No reference states were collected for the requested BC setting.")
    if max_reference_states is not None:
        if max_reference_states <= 0:
            raise ValueError("max_reference_states must be positive when provided.")
        if len(kept_observations) > max_reference_states:
            indices = np.linspace(
                0,
                len(kept_observations) - 1,
                num=max_reference_states,
                dtype=np.int64,
            )
            kept_observations = [kept_observations[index] for index in indices]
            kept_target_means = [kept_target_means[index] for index in indices]
            kept_target_log_stds = [kept_target_log_stds[index] for index in indices]
            kept_rewards = [kept_rewards[index] for index in indices]
            kept_next_observations = [kept_next_observations[index] for index in indices]
            kept_terminated = [kept_terminated[index] for index in indices]
    if kept_observations:
        return (
            np.stack(kept_observations).astype(np.float32),
            np.stack(kept_target_means).astype(np.float32),
            np.stack(kept_target_log_stds).astype(np.float32),
            {
                "observations": np.stack(kept_observations).astype(np.float32),
                "actions": np.zeros((len(kept_observations), agent.action_dim), dtype=np.float32),
                "rewards": np.asarray(kept_rewards, dtype=np.float32).reshape(-1, 1),
                "next_observations": np.stack(kept_next_observations).astype(np.float32),
                "terminated": np.asarray(kept_terminated, dtype=np.float32).reshape(-1, 1),
            },
            segment_rows,
        )
    return (
        np.empty((0, agent.observation_dim), dtype=np.float32),
        np.empty((0, agent.action_dim), dtype=np.float32),
        np.empty((0, agent.action_dim), dtype=np.float32),
        {
            "observations": np.empty((0, agent.observation_dim), dtype=np.float32),
            "actions": np.empty((0, agent.action_dim), dtype=np.float32),
            "rewards": np.empty((0, 1), dtype=np.float32),
            "next_observations": np.empty((0, agent.observation_dim), dtype=np.float32),
            "terminated": np.empty((0, 1), dtype=np.float32),
        },
        segment_rows,
    )


def build_reference_replay_buffer(
    *,
    agent: LocalBehaviorCloningSACAgent,
    transitions: dict[str, np.ndarray],
    seed: int,
) -> ReplayBuffer | None:
    transition_count = int(transitions["observations"].shape[0])
    if transition_count == 0:
        return None
    replay_buffer = ReplayBuffer(
        observation_dim=agent.observation_dim,
        action_dim=agent.action_dim,
        capacity=transition_count,
        seed=seed,
    )
    for index in range(transition_count):
        replay_buffer.add(
            observation=transitions["observations"][index],
            action=transitions["actions"][index],
            reward=float(transitions["rewards"][index, 0]),
            next_observation=transitions["next_observations"][index],
            terminated=bool(transitions["terminated"][index, 0]),
        )
    return replay_buffer


def evaluate_task(
    *,
    agent: LocalBehaviorCloningSACAgent,
    task_name: str,
    env_version: str,
    reward_function_version: str,
    episodes: int,
    deterministic: bool,
    max_episode_steps: int,
    seed: int,
) -> dict[str, float]:
    env = make_cw_env(
        task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        append_task_id=False,
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


def make_output_dir(args: argparse.Namespace) -> Path:
    base_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "local_bc_pairs"
    )
    run_name = args.run_name or (
        f"{args.variant}_old{args.old_task_index}_new{args.new_task_index}_seed{args.seed}"
    )
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    return output_dir


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    source_config = load_json(run_dir / "config.json")
    tasks = list(source_config["tasks"])
    if not 0 <= args.old_task_index < len(tasks):
        raise ValueError("old-task-index out of range.")
    if not 0 <= args.new_task_index < len(tasks):
        raise ValueError("new-task-index out of range.")
    if args.new_task_index <= args.old_task_index:
        raise ValueError("new-task-index must be strictly later than old-task-index.")

    old_task_name = str(tasks[args.old_task_index])
    new_task_name = str(tasks[args.new_task_index])
    selected_segments = resolve_segments(
        args=args,
        old_task_name=old_task_name,
        new_task_name=new_task_name,
    )
    output_dir = make_output_dir(args)

    config = dict(source_config)
    config["device"] = args.device
    config["seed"] = args.seed
    config["local_bc_batch_size"] = args.local_bc_batch_size
    config["local_bc_coefficient"] = 0.0 if args.variant == "no_bc" else args.local_bc_coefficient

    agent = build_local_bc_agent(config=config, device=args.device)
    old_checkpoint = run_dir / "checkpoints" / f"task_{args.old_task_index}.pt"
    load_sac_checkpoint(
        agent=agent,
        path=old_checkpoint,
        map_location=args.device,
        load_optimizer=False,
    )
    agent.rebuild_optimizer()

    (
        reference_observations,
        reference_target_means,
        reference_target_log_stds,
        reference_transitions,
        segment_rows,
    ) = collect_reference_memory(
        agent=agent,
        task_name=old_task_name,
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
        max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
        seed=args.seed + 10_000,
        reference_episodes=args.reference_episodes,
        reference_max_attempts=args.reference_max_attempts,
        max_reference_states=args.max_reference_states,
        segments=set(selected_segments),
        variant=args.variant,
    )
    agent.set_reference_memory(
        observations=reference_observations,
        target_means=reference_target_means,
        target_log_stds=reference_target_log_stds,
    )
    np.savez_compressed(
        output_dir / "reference_memory.npz",
        observations=reference_observations,
        target_means=reference_target_means,
        target_log_stds=reference_target_log_stds,
    )
    np.savez_compressed(
        output_dir / "reference_transitions.npz",
        observations=reference_transitions["observations"],
        actions=reference_transitions["actions"],
        rewards=reference_transitions["rewards"],
        next_observations=reference_transitions["next_observations"],
        terminated=reference_transitions["terminated"],
    )
    if segment_rows:
        write_csv(
            output_dir / "reference_segments.csv",
            fieldnames=segment_rows[0].keys(),
            rows=segment_rows,
        )
    reference_replay_buffer = None
    if args.use_selective_replay:
        if args.variant == "no_bc":
            raise ValueError("selective replay requires variant local_bc or full_bc.")
        reference_replay_buffer = build_reference_replay_buffer(
            agent=agent,
            transitions=reference_transitions,
            seed=args.seed + 60_000,
        )

    train_env = make_cw_env(
        new_task_name,
        seed=args.seed,
        max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
        append_task_id=False,
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
    )
    replay_buffer = ReplayBuffer(
        observation_dim=agent.observation_dim,
        action_dim=agent.action_dim,
        capacity=args.replay_size,
        seed=args.seed,
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
            seed=args.seed,
            reseed_global_rng=True,
        ),
        reference_replay_buffer=reference_replay_buffer,
        reference_batch_size=args.selective_replay_batch_size if args.use_selective_replay else 0,
    )

    started_at = time.time()
    evaluation_rows: list[dict[str, Any]] = []

    def evaluate(task_step: int, gradient_updates: int, update_metrics: dict[str, float] | None) -> None:
        del update_metrics
        old_metrics = evaluate_task(
            agent=agent,
            task_name=old_task_name,
            env_version=str(config["env_version"]),
            reward_function_version=str(config["reward_function_version"]),
            episodes=args.stoch_eval_episodes,
            deterministic=False,
            max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
            seed=args.seed + 20_000 + task_step,
        )
        new_metrics = evaluate_task(
            agent=agent,
            task_name=new_task_name,
            env_version=str(config["env_version"]),
            reward_function_version=str(config["reward_function_version"]),
            episodes=args.stoch_eval_episodes,
            deterministic=False,
            max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
            seed=args.seed + 30_000 + task_step,
        )
        evaluation_rows.append(
            {
                "task_step": task_step,
                "gradient_updates": gradient_updates,
                "old_task_name": old_task_name,
                "old_success_rate": old_metrics["success_rate"],
                "old_average_return": old_metrics["average_return"],
                "new_task_name": new_task_name,
                "new_success_rate": new_metrics["success_rate"],
                "new_average_return": new_metrics["average_return"],
                "elapsed_seconds": time.time() - started_at,
            }
        )
        print(
            f"[local-bc] old={old_task_name} new={new_task_name} "
            f"step={task_step:,}/{args.steps:,} "
            f"old_success={old_metrics['success_rate']:.3f} "
            f"new_success={new_metrics['success_rate']:.3f}"
        )

    training_summary = trainer.train(step_callback=evaluate)
    train_env.close()

    final_old = evaluate_task(
        agent=agent,
        task_name=old_task_name,
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
        episodes=max(args.stoch_eval_episodes, 10),
        deterministic=False,
        max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
        seed=args.seed + 40_000,
    )
    final_new = evaluate_task(
        agent=agent,
        task_name=new_task_name,
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
        episodes=max(args.stoch_eval_episodes, 10),
        deterministic=False,
        max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
        seed=args.seed + 50_000,
    )
    save_sac_checkpoint(
        agent=agent,
        path=output_dir / "checkpoints" / "final.pt",
        environment_step=args.steps,
        metadata={
            "old_task_index": args.old_task_index,
            "new_task_index": args.new_task_index,
            "variant": args.variant,
            "segments": list(selected_segments),
            "segment_manifest": args.segment_manifest,
            "max_reference_states": args.max_reference_states,
            "use_selective_replay": args.use_selective_replay,
            "selective_replay_batch_size": args.selective_replay_batch_size,
        },
    )

    if evaluation_rows:
        write_csv(output_dir / "evaluations.csv", fieldnames=evaluation_rows[0].keys(), rows=evaluation_rows)
    write_json(
        output_dir / "summary.json",
        {
            "source_run_dir": str(run_dir),
            "old_task_index": args.old_task_index,
            "old_task_name": old_task_name,
            "new_task_index": args.new_task_index,
            "new_task_name": new_task_name,
            "variant": args.variant,
            "segments": list(selected_segments),
            "segment_manifest": args.segment_manifest,
            "max_reference_states": args.max_reference_states,
            "reference_states": int(reference_observations.shape[0]),
            "reference_transitions": int(reference_transitions["observations"].shape[0]),
            "use_selective_replay": args.use_selective_replay,
            "selective_replay_batch_size": (
                args.selective_replay_batch_size if args.use_selective_replay else 0
            ),
            "training_steps": args.steps,
            "gradient_updates": training_summary.gradient_updates,
            "old_final_success": final_old["success_rate"],
            "old_final_return": final_old["average_return"],
            "new_final_success": final_new["success_rate"],
            "new_final_return": final_new["average_return"],
            "run_directory": str(output_dir),
            "elapsed_seconds": time.time() - started_at,
        },
    )


if __name__ == "__main__":
    main()
