from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import DEFAULT_EPISODE_LENGTH, extract_success, make_cw_env
from methods import get_method, method_defaults
from scripts.probe_stickpull_with_memory import load_json
from training.sac_trainer import seed_global_rngs
from utils import load_sac_checkpoint, write_csv, write_json


SUMMARY_FIELDS = (
    "source_task_index",
    "source_task_name",
    "attempted_episodes",
    "random_episodes",
    "successful_episodes",
    "pool_states",
    "successful_pool_states",
    "random_pool_states",
    "broad_memory_states",
    "broad_random_states",
    "success_memory_states",
)


def build_agent_from_config(
    *,
    config: dict[str, Any],
    device: str,
) -> Any:
    method = get_method(str(config["method"]))
    merged_config = method_defaults(method.method_id)
    merged_config.update(config)
    merged_config["device"] = device
    args = SimpleNamespace(**merged_config)
    task_name = str(config["tasks"][0])
    env = make_cw_env(
        task_name,
        seed=int(config["seed"]),
        max_episode_steps=int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH)),
        append_task_id=bool(method.append_task_id),
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
        num_task_ids=len(config["tasks"]),
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
    return agent.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build matched broad and success-only old-task memories from the "
            "same stochastic rollout pool, with both memories relabelled by "
            "the same per-task best actor snapshot."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-source-task-index", type=int, default=3)
    parser.add_argument("--pool-episodes-per-task", type=int, default=80)
    parser.add_argument("--random-episode-ratio", type=float, default=0.25)
    parser.add_argument("--memory-states-per-task", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def make_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    base = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "memory_libraries"
    )
    name = args.run_name or f"stickpull_matched_{run_dir.name}"
    output_dir = base / name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def sample_without_replacement(
    observations: np.ndarray,
    *,
    capacity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if observations.shape[0] <= capacity:
        return observations.copy()
    indices = rng.choice(observations.shape[0], size=capacity, replace=False)
    return observations[np.sort(indices)].copy()


def collect_rollout_pool(
    *,
    agent: Any,
    task_name: str,
    task_index: int,
    env_version: str,
    reward_function_version: str,
    append_task_id: bool,
    max_episode_steps: int,
    episodes: int,
    random_episode_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    env = make_cw_env(
        task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        append_task_id=append_task_id,
        env_version=env_version,
        reward_function_version=reward_function_version,
    )
    all_observations: list[np.ndarray] = []
    successful_observations: list[np.ndarray] = []
    random_observations: list[np.ndarray] = []
    policy_observations: list[np.ndarray] = []
    successful_episodes = 0
    random_episodes = 0
    rng = np.random.default_rng(seed + 70_000)
    random_episode_count = int(round(episodes * random_episode_ratio))
    random_episode_indices = set(
        int(index)
        for index in rng.choice(
            episodes,
            size=random_episode_count,
            replace=False,
        ).tolist()
    )
    try:
        for episode_index in range(episodes):
            observation, _ = env.reset(seed=seed + episode_index)
            episode_observations: list[np.ndarray] = []
            succeeded = False
            use_random_policy = episode_index in random_episode_indices
            random_episodes += int(use_random_policy)
            for _ in range(max_episode_steps):
                episode_observations.append(
                    np.asarray(observation, dtype=np.float32).copy()
                )
                if use_random_policy:
                    action = rng.uniform(
                        low=env.action_space.low,
                        high=env.action_space.high,
                    ).astype(np.float32)
                else:
                    action = agent.select_action_with_head(
                        observation,
                        head_index=task_index,
                        deterministic=False,
                    )
                observation, _, terminated, truncated, info = env.step(action)
                succeeded = succeeded or bool(extract_success(info))
                if terminated or truncated:
                    break
            episode_array = np.stack(episode_observations).astype(np.float32)
            all_observations.append(episode_array)
            if use_random_policy:
                random_observations.append(episode_array)
            else:
                policy_observations.append(episode_array)
            if succeeded:
                successful_episodes += 1
                successful_observations.append(episode_array)
    finally:
        env.close()

    all_array = np.concatenate(all_observations, axis=0)
    success_array = (
        np.concatenate(successful_observations, axis=0)
        if successful_observations
        else np.empty((0, agent.observation_dim), dtype=np.float32)
    )
    random_array = (
        np.concatenate(random_observations, axis=0)
        if random_observations
        else np.empty((0, agent.observation_dim), dtype=np.float32)
    )
    policy_array = (
        np.concatenate(policy_observations, axis=0)
        if policy_observations
        else np.empty((0, agent.observation_dim), dtype=np.float32)
    )
    return (
        all_array,
        success_array,
        policy_array,
        random_array,
        successful_episodes,
        random_episodes,
    )


def sample_stratified_broad_memory(
    *,
    policy_pool: np.ndarray,
    random_pool: np.ndarray,
    capacity: int,
    random_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    random_count = int(round(capacity * random_ratio))
    policy_count = capacity - random_count
    if random_pool.shape[0] < random_count:
        raise RuntimeError(
            f"Random rollout pool is underfilled: need {random_count} states, "
            f"collected {random_pool.shape[0]}."
        )
    random_memory = sample_without_replacement(
        random_pool,
        capacity=random_count,
        rng=rng,
    )

    if policy_pool.shape[0] < policy_count:
        raise RuntimeError(
            f"Old-policy rollout pool is underfilled: need {policy_count} states, "
            f"collected {policy_pool.shape[0]}."
        )
    policy_memory = sample_without_replacement(
        policy_pool,
        capacity=policy_count,
        rng=rng,
    )
    memory = np.concatenate((policy_memory, random_memory), axis=0)
    permutation = rng.permutation(memory.shape[0])
    return memory[permutation].copy(), random_count


@torch.no_grad()
def relabel_with_best_teacher(
    *,
    agent: Any,
    teacher_path: Path,
    observations: np.ndarray,
    expected_task_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(teacher_path, map_location=agent.device, weights_only=False)
    if int(payload["task_index"]) != expected_task_index:
        raise ValueError(
            f"Teacher task mismatch at {teacher_path}: expected "
            f"{expected_task_index}, found {payload['task_index']}."
        )
    agent.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    means, log_stds = agent.compute_reference_targets(observations)
    return (
        means.detach().cpu().numpy().astype(np.float32),
        log_stds.detach().cpu().numpy().astype(np.float32),
    )


def validate_memory(
    *,
    observations: np.ndarray,
    target_means: np.ndarray,
    target_log_stds: np.ndarray,
    expected_observation_dim: int,
    expected_action_dim: int,
) -> None:
    if observations.ndim != 2 or observations.shape[1] != expected_observation_dim:
        raise ValueError("Generated observations have the wrong shape.")
    if target_means.shape != (observations.shape[0], expected_action_dim):
        raise ValueError("Generated target means have the wrong shape.")
    if target_log_stds.shape != target_means.shape:
        raise ValueError("Generated target log stds have the wrong shape.")
    for name, values in (
        ("observations", observations),
        ("target_means", target_means),
        ("target_log_stds", target_log_stds),
    ):
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"Generated {name} contain non-finite values.")


def main() -> None:
    args = parse_args()
    if args.pool_episodes_per_task <= 0:
        raise ValueError("--pool-episodes-per-task must be positive.")
    if not 0.0 <= args.random_episode_ratio < 1.0:
        raise ValueError("--random-episode-ratio must be in [0, 1).")
    if args.memory_states_per_task <= 0:
        raise ValueError("--memory-states-per-task must be positive.")

    seed_global_rngs(args.seed)
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = load_json(run_dir / "config.json")
    tasks = list(config["tasks"])
    if not 0 <= args.max_source_task_index < 4:
        raise ValueError("Pre-stick-pull source tasks must end at an index in [0, 3].")

    output_dir = make_output_dir(args, run_dir)
    agent = build_agent_from_config(config=config, device=args.device)
    method = get_method(str(config["method"]))
    append_task_id = bool(method.append_task_id)
    max_episode_steps = int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH))

    broad_parts: list[dict[str, np.ndarray]] = []
    success_parts: list[dict[str, np.ndarray]] = []
    summary_rows: list[dict[str, int | str]] = []

    for task_index in range(args.max_source_task_index + 1):
        task_name = tasks[task_index]
        checkpoint_path = run_dir / "checkpoints" / f"task_{task_index}.pt"
        teacher_path = run_dir / "teacher_snapshots" / f"task_{task_index}_best_actor.pt"
        if not teacher_path.is_file():
            raise FileNotFoundError(
                f"Best teacher snapshot is required but missing: {teacher_path}"
            )
        load_sac_checkpoint(
            agent=agent,
            path=checkpoint_path,
            map_location=args.device,
            load_optimizer=False,
        )
        task_seed = args.seed + 100_000 * (task_index + 1)
        seed_global_rngs(task_seed)
        (
            pool,
            successful_pool,
            policy_pool,
            random_pool,
            successful_episodes,
            random_episodes,
        ) = collect_rollout_pool(
            agent=agent,
            task_name=task_name,
            task_index=task_index,
            env_version=str(config["env_version"]),
            reward_function_version=str(config["reward_function_version"]),
            append_task_id=append_task_id,
            max_episode_steps=max_episode_steps,
            episodes=args.pool_episodes_per_task,
            random_episode_ratio=args.random_episode_ratio,
            seed=task_seed,
        )
        rng = np.random.default_rng(task_seed + 50_000)
        broad_observations, broad_random_states = sample_stratified_broad_memory(
            policy_pool=policy_pool,
            random_pool=random_pool,
            capacity=args.memory_states_per_task,
            random_ratio=args.random_episode_ratio,
            rng=rng,
        )
        success_observations = sample_without_replacement(
            successful_pool,
            capacity=args.memory_states_per_task,
            rng=rng,
        )
        broad_means, broad_log_stds = relabel_with_best_teacher(
            agent=agent,
            teacher_path=teacher_path,
            observations=broad_observations,
            expected_task_index=task_index,
        )
        if success_observations.shape[0] == 0:
            success_means = np.empty((0, agent.action_dim), dtype=np.float32)
            success_log_stds = np.empty((0, agent.action_dim), dtype=np.float32)
        else:
            success_means, success_log_stds = relabel_with_best_teacher(
                agent=agent,
                teacher_path=teacher_path,
                observations=success_observations,
                expected_task_index=task_index,
            )
        broad_parts.append(
            {
                "observations": broad_observations,
                "target_means": broad_means,
                "target_log_stds": broad_log_stds,
            }
        )
        success_parts.append(
            {
                "observations": success_observations,
                "target_means": success_means,
                "target_log_stds": success_log_stds,
            }
        )
        summary_rows.append(
            {
                "source_task_index": task_index,
                "source_task_name": task_name,
                "attempted_episodes": args.pool_episodes_per_task,
                "random_episodes": random_episodes,
                "successful_episodes": successful_episodes,
                "pool_states": int(pool.shape[0]),
                "successful_pool_states": int(successful_pool.shape[0]),
                "random_pool_states": int(random_pool.shape[0]),
                "broad_memory_states": int(broad_observations.shape[0]),
                "broad_random_states": broad_random_states,
                "success_memory_states": int(success_observations.shape[0]),
            }
        )
        print(
            f"[matched-memory] task={task_name} successes="
            f"{successful_episodes}/{args.pool_episodes_per_task} "
            f"random_episodes={random_episodes} "
            f"broad={broad_observations.shape[0]:,} "
            f"success={success_observations.shape[0]:,}"
        )

    files: dict[str, str] = {}
    digests: dict[str, str] = {}
    counts: dict[str, int] = {}
    for memory_name, parts in (
        ("broad", broad_parts),
        ("success", success_parts),
    ):
        payload = {
            key: np.concatenate([part[key] for part in parts], axis=0)
            for key in ("observations", "target_means", "target_log_stds")
        }
        validate_memory(
            **payload,
            expected_observation_dim=agent.observation_dim,
            expected_action_dim=agent.action_dim,
        )
        path = output_dir / f"{memory_name}_memory.npz"
        np.savez_compressed(path, **payload)
        files[memory_name] = str(path)
        counts[memory_name] = int(payload["observations"].shape[0])
        digests[memory_name] = array_digest(payload["observations"])

    write_csv(output_dir / "task_summary.csv", SUMMARY_FIELDS, summary_rows)
    manifest = {
        "schema_version": 1,
        "purpose": "matched_stickpull_memory_causal_ablation",
        "run_dir": str(run_dir),
        "source_tasks": tasks[: args.max_source_task_index + 1],
        "max_source_task_index": args.max_source_task_index,
        "pool_policy": "task_end_stochastic_actor",
        "teacher": "per_task_best_actor_snapshot",
        "pool_episodes_per_task": args.pool_episodes_per_task,
        "random_episode_ratio": args.random_episode_ratio,
        "memory_states_per_task": args.memory_states_per_task,
        "seed": args.seed,
        "files": files,
        "counts": counts,
        "observation_sha256": digests,
        "task_summary": str(output_dir / "task_summary.csv"),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
