from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents import ReplayBuffer
from envs import extract_success, make_cw_env
from evaluation import EvaluationConfig, SACEvaluator
from methods import get_method
from training.sac_trainer import SACTrainer, SACTrainerConfig, seed_global_rngs
from utils import save_sac_checkpoint, write_csv, write_json


SINGLE_TASK_EVAL_FIELDS = [
    "environment_step",
    "gradient_updates",
    "deterministic_average_return",
    "deterministic_success_rate",
    "deterministic_average_episode_length",
    "stochastic_average_return",
    "stochastic_success_rate",
    "stochastic_average_episode_length",
    "actor_loss",
    "q1_loss",
    "q2_loss",
    "alpha_loss",
    "alpha",
    "q1_mean",
    "q2_mean",
    "q_target_mean",
    "log_prob_mean",
    "elapsed_seconds",
    "near_object_gripper_action_mean",
    "near_object_gripper_action_positive_ratio",
    "near_object_steps",
]


@dataclass(frozen=True)
class DebugRolloutStats:
    mean_return: float
    success_rate: float
    mean_episode_length: float
    near_object_gripper_action_mean: float | None
    near_object_gripper_action_positive_ratio: float | None
    near_object_steps: int


def _run_debug_rollout(
    *,
    env: Any,
    agent: Any,
    config: EvaluationConfig,
    deterministic: bool,
) -> DebugRolloutStats:
    episode_returns: list[float] = []
    episode_successes: list[bool] = []
    episode_lengths: list[int] = []
    near_object_gripper_actions: list[float] = []

    for episode_index in range(config.num_episodes):
        observation, _ = env.reset(seed=config.seed + episode_index)
        episode_return = 0.0
        episode_success = False
        episode_length = 0

        for _ in range(config.max_episode_steps):
            action = agent.select_action(
                observation,
                deterministic=deterministic,
            )
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            episode_success = episode_success or bool(extract_success(info))

            if float(info.get("near_object", 0.0)) > 0.0:
                near_object_gripper_actions.append(float(action[3]))

            observation = next_observation
            if terminated or truncated:
                break

        episode_returns.append(float(episode_return))
        episode_successes.append(bool(episode_success))
        episode_lengths.append(int(episode_length))

    near_object_mean = None
    near_object_positive_ratio = None
    if near_object_gripper_actions:
        near_object_mean = sum(near_object_gripper_actions) / len(near_object_gripper_actions)
        near_object_positive_ratio = (
            sum(value > 0.0 for value in near_object_gripper_actions)
            / len(near_object_gripper_actions)
        )

    return DebugRolloutStats(
        mean_return=sum(episode_returns) / len(episode_returns),
        success_rate=sum(episode_successes) / len(episode_successes),
        mean_episode_length=sum(episode_lengths) / len(episode_lengths),
        near_object_gripper_action_mean=near_object_mean,
        near_object_gripper_action_positive_ratio=near_object_positive_ratio,
        near_object_steps=len(near_object_gripper_actions),
    )


def run_single_task_experiment(args: Any, run_dir: Path) -> dict[str, Any]:
    seed_global_rngs(args.seed)
    started_at = time.time()
    train_env = make_cw_env(
        args.task,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        append_task_id=False,
        env_version=args.env_version,
        reward_function_version=args.reward_function_version,
    )
    eval_env = make_cw_env(
        args.task,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        append_task_id=False,
        env_version=args.env_version,
        reward_function_version=args.reward_function_version,
    )

    observation_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    method = get_method(args.method)
    agent = method.build_agent(
        args=args,
        observation_dim=observation_dim,
        action_dim=action_dim,
        action_low=train_env.action_space.low,
        action_high=train_env.action_space.high,
        total_tasks=1,
    )
    replay_buffer = ReplayBuffer(
        observation_dim=observation_dim,
        action_dim=action_dim,
        capacity=args.replay_size,
        seed=args.seed,
    )
    trainer = SACTrainer(
        env=train_env,
        agent=agent,
        replay_buffer=replay_buffer,
        config=SACTrainerConfig(
            total_steps=args.total_steps,
            batch_size=args.batch_size,
            start_steps=args.start_steps,
            update_after=args.update_after,
            update_every=args.update_every,
            max_episode_steps=args.max_episode_steps,
            callback_every_steps=args.eval_every,
            seed=args.seed,
        ),
    )

    deterministic_eval = None if args.det_eval_episodes <= 0 else SACEvaluator(
        eval_env,
        agent,
        EvaluationConfig(
            num_episodes=args.det_eval_episodes,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
        ),
    )
    stochastic_eval = SACEvaluator(
        eval_env,
        agent,
        EvaluationConfig(
            num_episodes=args.stoch_eval_episodes,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
        ),
    )

    eval_rows: list[dict[str, float | int | str]] = []
    config = vars(args).copy()
    config["observation_dim"] = observation_dim
    config["action_dim"] = action_dim
    write_json(run_dir / "config.json", config)
    best_success_rate = float("-inf")
    best_return = float("-inf")
    best_summary: dict[str, float | int | None] | None = None

    def evaluate(step: int, gradient_updates: int, update_metrics: dict[str, float] | None) -> None:
        deterministic = None
        if deterministic_eval is not None:
            deterministic_result = deterministic_eval.evaluate(deterministic=True)
            deterministic = deterministic_result
        stochastic_debug = _run_debug_rollout(
            env=eval_env,
            agent=agent,
            config=EvaluationConfig(
                num_episodes=args.stoch_eval_episodes,
                max_episode_steps=args.max_episode_steps,
                seed=args.seed,
            ),
            deterministic=False,
        )
        eval_rows.append(
            {
                "environment_step": step,
                "gradient_updates": gradient_updates,
                "deterministic_average_return": "" if deterministic is None else deterministic.mean_return,
                "deterministic_success_rate": "" if deterministic is None else deterministic.success_rate,
                "deterministic_average_episode_length": "" if deterministic is None else deterministic.mean_episode_length,
                "stochastic_average_return": stochastic_debug.mean_return,
                "stochastic_success_rate": stochastic_debug.success_rate,
                "stochastic_average_episode_length": stochastic_debug.mean_episode_length,
                "actor_loss": "" if update_metrics is None else update_metrics.get("actor_loss", ""),
                "q1_loss": "" if update_metrics is None else update_metrics.get("q1_loss", ""),
                "q2_loss": "" if update_metrics is None else update_metrics.get("q2_loss", ""),
                "alpha_loss": "" if update_metrics is None else update_metrics.get("alpha_loss", ""),
                "alpha": agent.alpha_value,
                "q1_mean": "" if update_metrics is None else update_metrics.get("q1_mean", ""),
                "q2_mean": "" if update_metrics is None else update_metrics.get("q2_mean", ""),
                "q_target_mean": "" if update_metrics is None else update_metrics.get("q_target_mean", ""),
                "log_prob_mean": "" if update_metrics is None else update_metrics.get("log_prob_mean", ""),
                "elapsed_seconds": time.time() - started_at,
                "near_object_gripper_action_mean": (
                    ""
                    if stochastic_debug.near_object_gripper_action_mean is None
                    else stochastic_debug.near_object_gripper_action_mean
                ),
                "near_object_gripper_action_positive_ratio": (
                    ""
                    if stochastic_debug.near_object_gripper_action_positive_ratio is None
                    else stochastic_debug.near_object_gripper_action_positive_ratio
                ),
                "near_object_steps": stochastic_debug.near_object_steps,
            }
        )
        nonlocal best_success_rate, best_return, best_summary
        if stochastic_debug.success_rate > best_success_rate:
            best_success_rate = stochastic_debug.success_rate
            save_sac_checkpoint(
                agent=agent,
                path=run_dir / "checkpoints" / "best_success.pt",
                environment_step=step,
                metadata={"task_name": args.task, "seed": args.seed, "metric": "success"},
            )
        if stochastic_debug.mean_return > best_return:
            best_return = stochastic_debug.mean_return
            save_sac_checkpoint(
                agent=agent,
                path=run_dir / "checkpoints" / "best_return.pt",
                environment_step=step,
                metadata={"task_name": args.task, "seed": args.seed, "metric": "return"},
            )
        best_summary = {
            "best_success_rate": best_success_rate,
            "best_return": best_return,
            "latest_near_object_gripper_action_mean": stochastic_debug.near_object_gripper_action_mean,
            "latest_near_object_gripper_action_positive_ratio": stochastic_debug.near_object_gripper_action_positive_ratio,
            "latest_near_object_steps": stochastic_debug.near_object_steps,
        }
        print(
            f"[single] task={args.task} step={step:,} "
            f"stoch_success={stochastic_debug.success_rate:.3f} "
            f"stoch_return={stochastic_debug.mean_return:.3f}"
        )

    summary = trainer.train(step_callback=evaluate)
    save_sac_checkpoint(
        agent=agent,
        path=run_dir / "checkpoints" / "final.pt",
        environment_step=summary.total_env_steps,
        metadata={"task_name": args.task, "seed": args.seed},
    )
    write_csv(run_dir / "evaluations.csv", SINGLE_TASK_EVAL_FIELDS, eval_rows)

    final_row = eval_rows[-1]
    summary_payload = {
        "task_name": args.task,
        "seed": args.seed,
        "total_steps": summary.total_env_steps,
        "gradient_updates": summary.gradient_updates,
        "completed_episodes": summary.completed_episodes,
        "mean_training_return": summary.mean_episode_return,
        "mean_training_episode_length": summary.mean_episode_length,
        "final_stochastic_success_rate": final_row["stochastic_success_rate"],
        "final_stochastic_average_return": final_row["stochastic_average_return"],
        "final_deterministic_success_rate": final_row["deterministic_success_rate"],
        "final_deterministic_average_return": final_row["deterministic_average_return"],
        "reward_function_version": args.reward_function_version,
        "best_success_rate": None if best_summary is None else best_summary["best_success_rate"],
        "best_return": None if best_summary is None else best_summary["best_return"],
        "near_object_gripper_action_mean": final_row["near_object_gripper_action_mean"],
        "near_object_gripper_action_positive_ratio": final_row["near_object_gripper_action_positive_ratio"],
        "near_object_steps": final_row["near_object_steps"],
        "elapsed_seconds": time.time() - started_at,
        "run_directory": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary_payload)
    return {
        "summary": summary_payload,
        "config": config,
        "evaluations": eval_rows,
    }
