"""Quick A/B test: transferred critic vs reset critic on actual training.

Loads a checkpoint at a task boundary, trains for a short period on the
next task with two conditions, and compares final success rate.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import ReplayBuffer, SACAgent
from envs import get_cw10_tasks, make_cw_env
from evaluation import SACEvaluator, EvaluationConfig
from training.sac_trainer import SACTrainer, SACTrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--env-version", default="v3")
    parser.add_argument("--reward-function-version", default="cw10_v1")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--target-tasks", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_agent(checkpoint_path: Path, device: str) -> SACAgent:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = payload["agent_config"]
    agent = SACAgent(
        observation_dim=config["observation_dim"],
        action_dim=config["action_dim"],
        action_low=[-1.0] * config["action_dim"],
        action_high=[1.0] * config["action_dim"],
        num_tasks=config["num_tasks"],
        task_id_dim=config["task_id_dim"],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        polyak=config["polyak"],
        device=device,
        hide_task_id=True,
    )
    agent.load_state_dict(payload["agent_state_dict"], strict=True)
    return agent


def evaluate_agent(agent: SACAgent, env, episodes: int, max_steps: int) -> float:
    """Evaluate stochastic success rate."""
    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                output = agent.actor.sample(obs_t)
            action = output.sampled_action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("success", 0) > 0:
                successes += 1
                break
            if terminated or truncated:
                break
    return successes / episodes


def train_on_task(
    agent: SACAgent,
    task_name: str,
    task_index: int,
    train_steps: int,
    max_episode_steps: int,
    env_version: str,
    reward_function_version: str,
    seed: int,
    tasks: list[str],
) -> float:
    """Train agent on a task and return final success rate."""
    env = make_cw_env(
        task_name=task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        append_task_id=True,
        env_version=env_version,
        reward_function_version=reward_function_version,
        num_task_ids=len(tasks),
    )
    replay_buffer = ReplayBuffer(
        observation_dim=agent.observation_dim,
        action_dim=agent.action_dim,
        capacity=1_000_000,
    )
    agent.rebuild_optimizer()

    config = SACTrainerConfig(
        total_steps=train_steps,
        batch_size=128,
        start_steps=10_000,
        update_after=1_000,
        update_every=50,
        max_episode_steps=max_episode_steps,
        seed=seed,
        reseed_global_rng=False,
    )
    trainer = SACTrainer(env, agent, replay_buffer, config)
    trainer.train()

    # Evaluate
    eval_env = make_cw_env(
        task_name=task_name,
        seed=seed + 999,
        max_episode_steps=max_episode_steps,
        append_task_id=True,
        env_version=env_version,
        reward_function_version=reward_function_version,
        num_task_ids=len(tasks),
    )
    success = evaluate_agent(agent, eval_env, episodes=10, max_steps=max_episode_steps)
    env.close()
    eval_env.close()
    return success


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    tasks = get_cw10_tasks(args.env_version)
    target_tasks = args.target_tasks or [1, 2, 3, 4, 5]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = []
    print(f"{'Task':<20} {'Transfer SR':<15} {'Reset SR':<15} {'Diff':<10}")
    print("-" * 60)

    for target_idx in target_tasks:
        task_name = tasks[target_idx]
        prev_ckpt = checkpoint_dir / f"task_{target_idx - 1}.pt"
        if not prev_ckpt.exists():
            print(f"{task_name:<20} SKIP (no checkpoint)")
            continue

        # Condition A: transferred critic (default behavior)
        torch.manual_seed(args.seed + target_idx)
        np.random.seed(args.seed + target_idx)
        agent_transfer = load_agent(prev_ckpt, args.device)
        t0 = time.time()
        sr_transfer = train_on_task(
            agent_transfer, task_name, target_idx,
            train_steps=args.train_steps,
            max_episode_steps=args.max_episode_steps,
            env_version=args.env_version,
            reward_function_version=args.reward_function_version,
            seed=args.seed + target_idx,
            tasks=tasks,
        )
        t_transfer = time.time() - t0

        # Condition B: reset critic
        torch.manual_seed(args.seed + target_idx)
        np.random.seed(args.seed + target_idx)
        agent_reset = load_agent(prev_ckpt, args.device)
        agent_reset.reset_critics()
        t0 = time.time()
        sr_reset = train_on_task(
            agent_reset, task_name, target_idx,
            train_steps=args.train_steps,
            max_episode_steps=args.max_episode_steps,
            env_version=args.env_version,
            reward_function_version=args.reward_function_version,
            seed=args.seed + target_idx,
            tasks=tasks,
        )
        t_reset = time.time() - t0

        diff = sr_reset - sr_transfer
        results.append({
            "task": task_name,
            "task_index": target_idx,
            "transfer_success": sr_transfer,
            "reset_success": sr_reset,
            "diff": diff,
            "time_transfer": t_transfer,
            "time_reset": t_reset,
        })
        marker = "+++" if diff > 0.05 else ("---" if diff < -0.05 else "   ")
        print(f"{task_name:<20} {sr_transfer:<15.3f} {sr_reset:<15.3f} {diff:+.3f} {marker}")

    print("\n" + "=" * 60)
    diffs = [r["diff"] for r in results]
    print(f"Mean diff (reset - transfer): {np.mean(diffs):+.3f}")
    print(f"Tasks where reset better: {sum(1 for d in diffs if d > 0)}/{len(diffs)}")
    print(f"Tasks where transfer better: {sum(1 for d in diffs if d < 0)}/{len(diffs)}")

    output_path = checkpoint_dir.parent / "critic_reset_ab_test.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
