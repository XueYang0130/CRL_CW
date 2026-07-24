from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from agents import ReplayBuffer, SACAgent
from envs import get_cw10_tasks, make_cw_env
from evaluation import EvaluationConfig, SACEvaluator, summarize_continual_run
from methods import get_method
from training.sac_trainer import SACTrainer, SACTrainerConfig
from utils import save_sac_checkpoint, write_csv, write_json


CW10_EVAL_FIELDS = [
    "evaluation_index",
    "global_step",
    "gradient_updates",
    "active_task_index",
    "active_task_name",
    "active_task_step",
    "evaluation_task_index",
    "evaluation_task_name",
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

CW10_TASK_SUMMARY_FIELDS = [
    "task_index",
    "task_name",
    "exploration_strategy",
    "selected_exploration_head",
    "selected_exploration_source_task",
    "selected_exploration_mean_return",
    "global_step_end",
    "task_gradient_updates",
    "cumulative_gradient_updates",
    "completed_training_episodes",
    "mean_training_return",
    "final_alpha",
    "replay_buffer_size",
    "elapsed_seconds",
    "final_guide_steps",
    "curriculum_transitions",
]


def normalize_exploration_strategy(strategy: str) -> str:
    aliases = {
        "best-return": "best_return",
    }
    return aliases.get(strategy, strategy)


def load_baseline_curves(path: str | None) -> list[list[float]] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload["stochastic_success_curves"]


def select_best_return_guide(
    *,
    agent: SACAgent,
    env: Any,
    task_index: int,
    episodes_per_head: int,
    max_episode_steps: int,
    seed: int,
) -> tuple[int, float]:
    if episodes_per_head <= 0:
        raise ValueError("best_return_eval_episodes must be positive.")
    mean_returns: list[float] = []
    for head_index in range(task_index):
        returns = []
        for episode_index in range(episodes_per_head):
            reset_result = env.reset(seed=seed + head_index * episodes_per_head + episode_index)
            observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            episode_return = 0.0
            for _ in range(max_episode_steps):
                action = agent.select_guide_action(
                    observation,
                    guide_task_index=head_index,
                    deterministic=False,
                )
                step_result = env.step(action)
                if len(step_result) == 5:
                    observation, reward, terminated, truncated, _ = step_result
                    done = bool(terminated or truncated)
                else:
                    observation, reward, done, _ = step_result
                episode_return += float(reward)
                if done:
                    break
            returns.append(episode_return)
        mean_returns.append(sum(returns) / len(returns))
    selected = max(range(task_index), key=mean_returns.__getitem__)
    return selected, mean_returns[selected]


def evaluate_mixed_policy_return(
    *,
    agent: SACAgent,
    env: Any,
    guide_head_index: int,
    guide_steps: int,
    episodes: int,
    max_episode_steps: int,
    seed: int,
) -> float:
    returns = []
    for episode_index in range(episodes):
        reset_result = env.reset(seed=seed + episode_index)
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        episode_return = 0.0
        for episode_step in range(max_episode_steps):
            if episode_step < guide_steps:
                action = agent.select_guide_action(
                    observation,
                    guide_task_index=guide_head_index,
                    deterministic=False,
                )
            else:
                action = agent.select_action(observation, deterministic=False)
            step_result = env.step(action)
            if len(step_result) == 5:
                observation, reward, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                observation, reward, done, _ = step_result
            episode_return += float(reward)
            if done:
                break
        returns.append(episode_return)
    return sum(returns) / len(returns)


def jsrl_horizons(*, initial_steps: int, stages: int) -> list[int]:
    if initial_steps <= 0:
        raise ValueError("jsrl_initial_guide_steps must be positive.")
    if stages < 2:
        raise ValueError("jsrl_curriculum_stages must be at least 2.")
    return [
        round(initial_steps * remaining / (stages - 1))
        for remaining in range(stages - 1, -1, -1)
    ]


def reached_relative_threshold(
    *, current: float, reference: float, tolerance: float
) -> bool:
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("jsrl_stage_tolerance must be in [0, 1).")
    if abs(reference) < 1e-8:
        return current > reference + 1e-8
    return current >= reference - tolerance * abs(reference)


def jsrl_advance_reason(
    *,
    moving_average: float,
    reference: float,
    tolerance: float,
    stage_evaluations: int,
    min_evaluations: int,
    max_evaluations_without_advance: int,
    has_next_stage: bool,
) -> str | None:
    if not has_next_stage:
        return None
    if stage_evaluations >= min_evaluations and reached_relative_threshold(
        current=moving_average,
        reference=reference,
        tolerance=tolerance,
    ):
        return "performance"
    if stage_evaluations >= max_evaluations_without_advance:
        return "patience"
    return None


def evaluate_all_tasks(
    *,
    agent: SACAgent,
    tasks: list[str],
    env_version: str,
    eval_seed: int,
    det_eval_episodes: int,
    stoch_eval_episodes: int,
    max_episode_steps: int,
    active_task_index: int,
    active_task_step: int,
    global_step: int,
    gradient_updates: int,
    evaluation_index: int,
    started_at: float,
    update_metrics: dict[str, float] | None,
    append_task_id: bool,
    evaluation_task_indices: list[int] | None = None,
) -> list[dict[str, float | int | str]]:
    rows = []
    task_indices = (
        list(range(len(tasks)))
        if evaluation_task_indices is None
        else evaluation_task_indices
    )
    for eval_task_index in task_indices:
        task_name = tasks[eval_task_index]
        agent.on_evaluation_start(eval_task_index)
        env = make_cw_env(
            task_name,
            seed=eval_seed,
            max_episode_steps=max_episode_steps,
            append_task_id=append_task_id,
            env_version=env_version,
        )
        deterministic = None
        if det_eval_episodes > 0:
            deterministic = SACEvaluator(
                env,
                agent,
                EvaluationConfig(
                    num_episodes=det_eval_episodes,
                    max_episode_steps=max_episode_steps,
                    seed=eval_seed,
                ),
            ).evaluate(deterministic=True)
        stochastic = SACEvaluator(
            env,
            agent,
            EvaluationConfig(
                num_episodes=stoch_eval_episodes,
                max_episode_steps=max_episode_steps,
                seed=eval_seed,
            ),
        ).evaluate(deterministic=False)
        rows.append(
            {
                "evaluation_index": evaluation_index,
                "global_step": global_step,
                "gradient_updates": gradient_updates,
                "active_task_index": active_task_index,
                "active_task_name": tasks[active_task_index],
                "active_task_step": active_task_step,
                "evaluation_task_index": eval_task_index,
                "evaluation_task_name": task_name,
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
                "alpha": (
                    agent.diagnostic_alpha_value(active_task_index)
                ),
                "q1_mean": "" if update_metrics is None else update_metrics.get("q1_mean", ""),
                "q2_mean": "" if update_metrics is None else update_metrics.get("q2_mean", ""),
                "q_target_mean": "" if update_metrics is None else update_metrics.get("q_target_mean", ""),
                "log_prob_mean": "" if update_metrics is None else update_metrics.get("log_prob_mean", ""),
                "elapsed_seconds": time.time() - started_at,
            }
        )
        env.close()
        agent.on_evaluation_end(eval_task_index)
    return rows


def run_cw10_experiment(args: Any, run_dir: Path) -> dict[str, Any]:
    tasks = get_cw10_tasks(args.env_version)[: args.sequence_task_count]
    started_at = time.time()
    method = get_method(args.method)
    append_task_id = method.append_task_id

    first_env = make_cw_env(
        tasks[0],
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        append_task_id=append_task_id,
        env_version=args.env_version,
    )
    observation_dim = first_env.observation_space.shape[0]
    action_dim = first_env.action_space.shape[0]
    first_env.close()

    bounds_env = make_cw_env(
        tasks[0],
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        append_task_id=append_task_id,
        env_version=args.env_version,
    )
    agent = method.build_agent(
        args=args,
        observation_dim=observation_dim,
        action_dim=action_dim,
        action_low=bounds_env.action_space.low,
        action_high=bounds_env.action_space.high,
        total_tasks=len(tasks),
    )
    bounds_env.close()
    replay_buffer = ReplayBuffer(
        observation_dim=observation_dim,
        action_dim=action_dim,
        capacity=args.replay_size,
        seed=args.seed,
    )

    config = vars(args).copy()
    config["tasks"] = tasks
    config["observation_dim"] = observation_dim
    config["action_dim"] = action_dim
    write_json(run_dir / "config.json", config)

    all_eval_rows: list[dict[str, float | int | str]] = []
    task_summary_rows: list[dict[str, float | int | str | None]] = []
    active_task_success_curves: list[list[float]] = []
    global_step = 0
    cumulative_gradient_updates = 0
    evaluation_index = 0
    last_update_metrics: dict[str, float] | None = None
    baseline_curves = load_baseline_curves(args.baseline_curves)

    for task_index, task_name in enumerate(tasks):
        effective_exploration_strategy = normalize_exploration_strategy(
            args.exploration_strategy
        )
        agent.on_task_start(
            task_index=task_index,
            replay_buffer=replay_buffer,
        )
        if task_index > 0 and args.reset_buffer_on_task_change:
            replay_buffer.clear()
        if task_index > 0 and args.reset_optimizer_on_task_change:
            agent.rebuild_optimizer()
        if task_index > 0 and method.transfer_alpha:
            if agent.num_tasks == 1:
                raise RuntimeError("Alpha transfer requires multi-task alpha parameters.")
            agent.copy_alpha(
                source_task_index=task_index - 1,
                target_task_index=task_index,
            )

        exploration_strategy, exploration_available_heads = method.exploration_config(
            task_index=task_index,
            strategy=effective_exploration_strategy,
        )

        env = make_cw_env(
            task_name,
            seed=args.seed + task_index,
            max_episode_steps=args.max_episode_steps,
            append_task_id=append_task_id,
            env_version=args.env_version,
        )
        selected_guide: int | None = None
        selected_guide_return: float | None = None
        if agent.requires_guide_selection(task_index):
            selected_guide, selected_guide_return = select_best_return_guide(
                agent=agent,
                env=env,
                task_index=task_index,
                episodes_per_head=args.best_return_eval_episodes,
                max_episode_steps=args.max_episode_steps,
                seed=args.seed + 10_000 + task_index * 1_000,
            )
            agent.initialize_task_from_guide(
                task_index=task_index,
                guide_task_index=selected_guide,
            )

        trainer_start_steps = args.start_steps
        trainer_update_after = args.update_after
        exploration_head_index = None
        curriculum_horizons = [0]
        curriculum_stage = 0
        curriculum_returns: deque[float] = deque(
            maxlen=args.jsrl_moving_average_window
        )
        curriculum_reference = selected_guide_return
        curriculum_stage_best = float("-inf")
        curriculum_stage_evaluations = 0
        curriculum_transitions: list[dict[str, float | int | str]] = []
        guide_head_index = None
        guide_steps = 0
        if selected_guide is not None and method.guide_mode == "fixed_warmup":
            if args.wsrl_warmup_steps <= 0:
                raise ValueError("wsrl_warmup_steps must be positive.")
            # SACTrainer follows the reference's inclusive start-step boundary.
            trainer_start_steps = args.wsrl_warmup_steps - 1
            trainer_update_after = max(args.update_after, args.wsrl_warmup_steps)
            exploration_head_index = selected_guide
        elif selected_guide is not None and method.guide_mode == "curriculum":
            if args.jsrl_moving_average_window <= 0:
                raise ValueError("jsrl_moving_average_window must be positive.")
            if args.jsrl_min_evaluations_per_stage <= 0:
                raise ValueError("jsrl_min_evaluations_per_stage must be positive.")
            if args.jsrl_max_evaluations_without_advance <= 0:
                raise ValueError(
                    "jsrl_max_evaluations_without_advance must be positive."
                )
            if (
                args.jsrl_max_evaluations_without_advance
                < args.jsrl_moving_average_window
            ):
                raise ValueError(
                    "jsrl_max_evaluations_without_advance must be at least "
                    "jsrl_moving_average_window."
                )
            if (
                args.jsrl_min_evaluations_per_stage
                > args.jsrl_max_evaluations_without_advance
            ):
                raise ValueError(
                    "jsrl_min_evaluations_per_stage cannot exceed "
                    "jsrl_max_evaluations_without_advance."
                )
            if args.jsrl_evaluation_interval % args.eval_every != 0:
                raise ValueError("jsrl_evaluation_interval must be divisible by eval_every.")
            curriculum_horizons = jsrl_horizons(
                initial_steps=args.jsrl_initial_guide_steps,
                stages=args.jsrl_curriculum_stages,
            )
            if curriculum_horizons[0] >= args.max_episode_steps:
                raise ValueError("Initial JSRL guide steps must leave learner-controlled steps.")
            guide_head_index = selected_guide
            guide_steps = curriculum_horizons[0]
        trainer = SACTrainer(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            config=SACTrainerConfig(
                total_steps=args.steps_per_task,
                batch_size=args.batch_size,
                start_steps=trainer_start_steps,
                exploration_head_index=exploration_head_index,
                exploration_strategy=exploration_strategy,
                exploration_available_heads=exploration_available_heads,
                update_after=trainer_update_after,
                update_every=args.update_every,
                max_episode_steps=args.max_episode_steps,
                callback_every_steps=args.eval_every,
                seed=args.seed + task_index,
                reseed_global_rng=(task_index == 0),
                guide_head_index=guide_head_index,
                guide_steps=guide_steps,
            ),
        )
        curriculum_eval_env = (
            make_cw_env(
                task_name,
                seed=args.seed + 20_000 + task_index,
                max_episode_steps=args.max_episode_steps,
                append_task_id=append_task_id,
                env_version=args.env_version,
            )
            if guide_head_index is not None
            else None
        )

        task_curve_success: list[float] = []
        gradient_updates_before_task = cumulative_gradient_updates

        def evaluate(task_step: int, gradient_updates: int, update_metrics: dict[str, float] | None) -> None:
            nonlocal global_step, evaluation_index, last_update_metrics
            nonlocal curriculum_stage, curriculum_reference
            nonlocal curriculum_stage_best, curriculum_stage_evaluations
            evaluation_index += 1
            last_update_metrics = update_metrics
            global_step = task_index * args.steps_per_task + task_step
            rows = evaluate_all_tasks(
                agent=agent,
                tasks=tasks,
                env_version=args.env_version,
                eval_seed=args.seed + evaluation_index,
                det_eval_episodes=args.det_eval_episodes,
                stoch_eval_episodes=args.stoch_eval_episodes,
                max_episode_steps=args.max_episode_steps,
                active_task_index=task_index,
                active_task_step=task_step,
                global_step=global_step,
                gradient_updates=gradient_updates_before_task + gradient_updates,
                evaluation_index=evaluation_index,
                started_at=started_at,
                update_metrics=update_metrics,
                append_task_id=append_task_id,
                evaluation_task_indices=[task_index],
            )
            all_eval_rows.extend(rows)
            active_row = rows[0]
            task_curve_success.append(float(active_row["stochastic_success_rate"]))
            print(
                f"[cw10] task={task_name} step={task_step:,}/{args.steps_per_task:,} "
                f"success={active_row['stochastic_success_rate']:.3f} "
                f"return={active_row['stochastic_average_return']:.3f}"
            )
            if (
                guide_head_index is not None
                and task_step % args.jsrl_evaluation_interval == 0
            ):
                if curriculum_eval_env is None:
                    raise RuntimeError("Missing JSRL curriculum evaluation environment.")
                mixed_return = evaluate_mixed_policy_return(
                    agent=agent,
                    env=curriculum_eval_env,
                    guide_head_index=guide_head_index,
                    guide_steps=trainer.guide_steps,
                    episodes=args.stoch_eval_episodes,
                    max_episode_steps=args.max_episode_steps,
                    seed=args.seed + 30_000 + evaluation_index,
                )
                curriculum_returns.append(mixed_return)
                curriculum_stage_evaluations += 1
                if len(curriculum_returns) == args.jsrl_moving_average_window:
                    moving_average = sum(curriculum_returns) / len(curriculum_returns)
                    curriculum_stage_best = max(curriculum_stage_best, moving_average)
                    advance_reason = (
                        None
                        if curriculum_reference is None
                        else jsrl_advance_reason(
                            moving_average=moving_average,
                            reference=curriculum_reference,
                            tolerance=args.jsrl_stage_tolerance,
                            stage_evaluations=curriculum_stage_evaluations,
                            min_evaluations=args.jsrl_min_evaluations_per_stage,
                            max_evaluations_without_advance=(
                                args.jsrl_max_evaluations_without_advance
                            ),
                            has_next_stage=(
                                curriculum_stage
                                < len(curriculum_horizons) - 1
                            ),
                        )
                    )
                    if advance_reason is not None:
                        old_h = trainer.guide_steps
                        threshold_reference = curriculum_reference
                        curriculum_reference = max(
                            curriculum_reference,
                            curriculum_stage_best,
                        )
                        curriculum_stage += 1
                        trainer.set_guide_steps(curriculum_horizons[curriculum_stage])
                        curriculum_stage_evaluations = 0
                        curriculum_stage_best = float("-inf")
                        curriculum_transitions.append(
                            {
                                "task_step": task_step,
                                "from_h": old_h,
                                "to_h": trainer.guide_steps,
                                "reason": advance_reason,
                                "moving_average_return": moving_average,
                                "reference_return": threshold_reference,
                            }
                        )

        training_summary = trainer.train(step_callback=evaluate)
        if curriculum_eval_env is not None:
            curriculum_eval_env.close()
        cumulative_gradient_updates += training_summary.gradient_updates
        agent.on_task_end(
            task_index=task_index,
            replay_buffer=replay_buffer,
            batch_size=args.batch_size,
        )
        active_task_success_curves.append(task_curve_success)
        save_sac_checkpoint(
            agent=agent,
            path=run_dir / "checkpoints" / f"task_{task_index}.pt",
            environment_step=(task_index + 1) * args.steps_per_task,
            metadata={"task_name": task_name, "task_index": task_index},
        )
        task_summary_rows.append(
            {
                "task_index": task_index,
                "task_name": task_name,
                "exploration_strategy": (
                    "best_return"
                    if selected_guide is not None
                    else effective_exploration_strategy
                ),
                "selected_exploration_head": selected_guide,
                "selected_exploration_source_task": (
                    None if selected_guide is None else tasks[selected_guide]
                ),
                "selected_exploration_mean_return": selected_guide_return,
                "global_step_end": (task_index + 1) * args.steps_per_task,
                "task_gradient_updates": training_summary.gradient_updates,
                "cumulative_gradient_updates": cumulative_gradient_updates,
                "completed_training_episodes": training_summary.completed_episodes,
                "mean_training_return": training_summary.mean_episode_return,
                "final_alpha": (
                    agent.diagnostic_alpha_value(task_index)
                ),
                "replay_buffer_size": len(replay_buffer),
                "elapsed_seconds": time.time() - started_at,
                "final_guide_steps": trainer.guide_steps,
                "curriculum_transitions": json.dumps(curriculum_transitions),
            }
        )
        env.close()

    final_global_step = len(tasks) * args.steps_per_task
    final_task_index = len(tasks) - 1
    for final_round in range(args.tail_size):
        evaluation_index += 1
        final_rows = evaluate_all_tasks(
            agent=agent,
            tasks=tasks,
            env_version=args.env_version,
            eval_seed=args.seed + 100_000 + final_round,
            det_eval_episodes=args.det_eval_episodes,
            stoch_eval_episodes=args.stoch_eval_episodes,
            max_episode_steps=args.max_episode_steps,
            active_task_index=final_task_index,
            active_task_step=args.steps_per_task,
            global_step=final_global_step,
            gradient_updates=cumulative_gradient_updates,
            evaluation_index=evaluation_index,
            started_at=started_at,
            update_metrics=last_update_metrics,
            append_task_id=append_task_id,
        )
        all_eval_rows.extend(final_rows)

    write_csv(run_dir / "evaluations.csv", CW10_EVAL_FIELDS, all_eval_rows)
    write_csv(run_dir / "task_summaries.csv", CW10_TASK_SUMMARY_FIELDS, task_summary_rows)

    evaluation_grid = []
    for evaluation_id in sorted({int(row["evaluation_index"]) for row in all_eval_rows}):
        evaluation_rows = [row for row in all_eval_rows if int(row["evaluation_index"]) == evaluation_id]
        evaluation_grid.append([float(row["stochastic_success_rate"]) for row in sorted(evaluation_rows, key=lambda item: int(item["evaluation_task_index"]))])

    final_success_rows = evaluation_grid[-min(args.tail_size, len(evaluation_grid)) :]
    final_return_rows = []
    for evaluation_id in sorted({int(row["evaluation_index"]) for row in all_eval_rows})[-min(args.tail_size, len(evaluation_grid)):]:
        evaluation_rows = [row for row in all_eval_rows if int(row["evaluation_index"]) == evaluation_id]
        final_return_rows.append([float(row["stochastic_average_return"]) for row in sorted(evaluation_rows, key=lambda item: int(item["evaluation_task_index"]))])

    metrics = summarize_continual_run(
        final_success_rows=final_success_rows,
        active_task_success_curves=active_task_success_curves,
        baseline_success_curves=baseline_curves,
        tail_size=args.tail_size,
    )
    average_return = sum(sum(row) / len(row) for row in final_return_rows) / len(final_return_rows)
    summary_payload = {
        "tasks": tasks,
        "average_performance": metrics["average_performance"],
        "average_return": average_return,
        "average_forgetting": metrics["average_forgetting"],
        "forward_transfer": metrics["forward_transfer"],
        "raw_forward_transfer": metrics["raw_forward_transfer"],
        "area_forward_transfer": metrics["area_forward_transfer"],
        "forward_transfer_available": metrics["forward_transfer_available"],
        "final_per_task_success": dict(zip(tasks, metrics["final_per_task"], strict=True)),
        "end_of_task_per_task": dict(zip(tasks, metrics["end_of_task_per_task"], strict=True)),
        "forgetting_per_task": dict(zip(tasks, metrics["forgetting_per_task"], strict=True)),
        "raw_forward_transfer_per_task": (
            None
            if metrics["raw_forward_transfer_per_task"][0] is None
            else dict(zip(tasks, metrics["raw_forward_transfer_per_task"], strict=True))
        ),
        "normalized_forward_transfer_per_task": (
            None
            if metrics["normalized_forward_transfer_per_task"][0] is None
            else dict(zip(tasks, metrics["normalized_forward_transfer_per_task"], strict=True))
        ),
        "area_forward_transfer_per_task": (
            None
            if metrics["area_forward_transfer_per_task"][0] is None
            else dict(zip(tasks, metrics["area_forward_transfer_per_task"], strict=True))
        ),
        "run_directory": str(run_dir),
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(run_dir / "summary.json", summary_payload)
    return {
        "summary": summary_payload,
        "config": config,
        "evaluations": all_eval_rows,
        "task_summaries": task_summary_rows,
    }
