from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agents import ReplayBuffer
from envs import make_cw_env
from evaluation import EvaluationConfig, SACEvaluator
from methods import get_method
from training.sac_trainer import SACTrainer, SACTrainerConfig
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
]


def run_single_task_experiment(args: Any, run_dir: Path) -> dict[str, Any]:
    started_at = time.time()
    train_env = make_cw_env(
        args.task,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        append_task_id=False,
        env_version=args.env_version,
    )
    eval_env = make_cw_env(
        args.task,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        append_task_id=False,
        env_version=args.env_version,
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

    def evaluate(step: int, gradient_updates: int, update_metrics: dict[str, float] | None) -> None:
        deterministic = None if deterministic_eval is None else deterministic_eval.evaluate(
            deterministic=True
        )
        stochastic = stochastic_eval.evaluate(deterministic=False)
        eval_rows.append(
            {
                "environment_step": step,
                "gradient_updates": gradient_updates,
                "deterministic_average_return": "" if deterministic is None else deterministic.mean_return,
                "deterministic_success_rate": "" if deterministic is None else deterministic.success_rate,
                "deterministic_average_episode_length": "" if deterministic is None else deterministic.mean_episode_length,
                "stochastic_average_return": stochastic.mean_return,
                "stochastic_success_rate": stochastic.success_rate,
                "stochastic_average_episode_length": stochastic.mean_episode_length,
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
            }
        )
        print(
            f"[single] task={args.task} step={step:,} "
            f"stoch_success={stochastic.success_rate:.3f} "
            f"stoch_return={stochastic.mean_return:.3f}"
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
        "elapsed_seconds": time.time() - started_at,
        "run_directory": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary_payload)
    return {
        "summary": summary_payload,
        "config": config,
        "evaluations": eval_rows,
    }
