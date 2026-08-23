from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents import FullBehaviorCloningSACAgent, ReplayBuffer, SACAgent
from agents.gradient_diagnostics import (
    GRADIENT_LAYER_FIELDS,
    GRADIENT_TASK_PAIR_FIELDS,
    GRADIENT_WINDOW_FIELDS,
)
from envs import canonical_cw10_task_name, get_continual_task_sequence, make_cw_env
from evaluation import EvaluationConfig, SACEvaluator, summarize_continual_run
from methods import (
    bc_gradient_strategy_for_method,
    get_method,
    required_bc_combination_strategy,
)
from training.sac_trainer import SACTrainer, SACTrainerConfig, seed_global_rngs
from training.semantic_segments import (
    SEGMENT_ORDER,
    TaskAwareSegmenter,
    TaskAwareV3Segmenter,
    extract_features,
    segment_label,
)
from training.segment_selection import (
    SegmentSelection,
    load_segment_selection_manifest,
    normalize_segment_weights,
    select_task_specific_segments_for_task,
    select_segments_for_task_pair,
)
from training.llm_controller import (
    append_controller_log,
    build_llm_gate_prompt,
    call_openai_json_controller,
    read_api_key,
)
from training.llm_bc_schedule_controller import (
    LLMBCScheduleDecision,
    SCHEDULE_DELAYED,
    SCHEDULE_STANDARD,
    build_task_schedule_prompt,
    call_task_schedule_controller,
    standard_fallback_decision,
)
from training.llm_policy_prior_controller import (
    LLMPolicyPriorDecision,
    build_policy_prior_prompt,
    call_policy_prior_controller,
    no_transfer_decision,
    resolve_prior_exploration,
)
from training.implicit_policy_prior import (
    PriorInitializationResult,
    collect_reset_observations,
    initialize_actor_head_from_source,
)
from training.success_replay import SuccessfulStateReservoir
from training.bc_progress_gate import BCProgressGate
from utils import append_csv_rows, save_sac_checkpoint, write_csv, write_json


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
    "ensemble_critic_loss",
    "q_ensemble_std_mean",
    "td_error_abs_mean",
    "priority_beta",
    "importance_weight_mean",
    "bc_progress_score",
    "bc_progress_multiplier",
    "bc_progress_next_multiplier",
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
    "reference_success_episodes",
    "reference_states",
    "success_replay_seen_states",
    "background_reference_states",
    "task_specific_reference_states",
    "semantic_segment_scheme",
    "collected_segment_states",
    "selected_segments",
    "task_specific_segments",
    "segment_priority",
    "segment_weights",
    "segment_selection_reason",
    "segment_selection_source",
    "success_replay_teacher",
    "best_teacher_step",
    "best_teacher_success",
    "best_teacher_return",
    "bc_task_schedule",
    "bc_task_schedule_source",
    "bc_task_schedule_confidence",
    "policy_prior_source_task",
    "policy_prior_confidence",
    "policy_prior_decision_source",
    "policy_prior_initial_kl",
    "policy_prior_final_kl",
    "bc_task_schedule_reason_codes",
    "bc_task_schedule_reason",
]


def clone_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def add_relabelled_reference_memory(
    *,
    agent: FullBehaviorCloningSACAgent,
    observations: np.ndarray,
    teacher_actor_state: dict[str, torch.Tensor] | None,
) -> None:
    """Label states with one frozen teacher without changing the live actor."""
    if observations.shape[0] == 0:
        return
    final_actor_state = clone_module_state(agent.actor)
    try:
        if teacher_actor_state is not None:
            agent.actor.load_state_dict(teacher_actor_state, strict=True)
        agent.add_reference_memory(observations=observations)
    finally:
        agent.actor.load_state_dict(final_actor_state, strict=True)

STATIC_SEMANTIC_METHODS = {
    "semantic_local_bc",
    "adaptive_semantic_bc",
    "general_task_specific_bc",
    "semantic_hybrid_bc",
}

def stage_aware_segment_groups(progress_ratio: float) -> tuple[str, ...]:
    if progress_ratio < 0.30:
        return ("approach", "contact_or_alignment")
    if progress_ratio < 0.70:
        return ("contact_or_alignment", "manipulation")
    return ("manipulation", "finish_or_stabilize")


def combine_reference_payloads(
    payloads: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    non_empty_payloads = [
        payload
        for payload in payloads
        if payload["observations"].shape[0] > 0
    ]
    if not non_empty_payloads:
        obs_dim = payloads[0]["observations"].shape[1] if payloads else 1
        act_dim = payloads[0]["target_means"].shape[1] if payloads else 1
        return (
            np.empty((0, obs_dim), dtype=np.float32),
            np.empty((0, act_dim), dtype=np.float32),
            np.empty((0, act_dim), dtype=np.float32),
        )
    observations = np.concatenate(
        [payload["observations"] for payload in non_empty_payloads],
        axis=0,
    )
    target_means = np.concatenate(
        [payload["target_means"] for payload in non_empty_payloads],
        axis=0,
    )
    target_log_stds = np.concatenate(
        [payload["target_log_stds"] for payload in non_empty_payloads],
        axis=0,
    )
    return observations, target_means, target_log_stds


def segment_payload_counts(
    payload_store: dict[str, list[dict[str, np.ndarray]]],
    labels: tuple[str, ...] | None = None,
) -> dict[str, int]:
    labels = SEGMENT_ORDER if labels is None else labels
    return {
        segment: int(
            sum(
                payload["observations"].shape[0]
                for payload in payload_store.get(segment, [])
            )
        )
        for segment in labels
    }


def selected_payloads_from_store(
    *,
    payload_store: dict[str, list[dict[str, np.ndarray]]],
    selected_segments: tuple[str, ...],
) -> list[dict[str, np.ndarray]]:
    return [
        payload
        for segment in selected_segments
        for payload in payload_store.get(segment, [])
    ]


def _sample_payload_states(
    payloads: list[dict[str, np.ndarray]],
    count: int,
) -> dict[str, np.ndarray] | None:
    observations, target_means, target_log_stds = combine_reference_payloads(payloads)
    available = observations.shape[0]
    if available == 0 or count <= 0:
        return None
    kept = min(count, available)
    indices = np.linspace(0, available - 1, num=kept, dtype=int)
    return {
        "observations": observations[indices],
        "target_means": target_means[indices],
        "target_log_stds": target_log_stds[indices],
    }


def _resample_payload_states(
    payloads: list[dict[str, np.ndarray]],
    count: int,
) -> dict[str, np.ndarray] | None:
    observations, target_means, target_log_stds = combine_reference_payloads(payloads)
    available = observations.shape[0]
    if available == 0 or count <= 0:
        return None
    if count <= available:
        indices = np.linspace(0, available - 1, num=count, dtype=int)
    else:
        indices = np.arange(count, dtype=int) % available
    return {
        "observations": observations[indices],
        "target_means": target_means[indices],
        "target_log_stds": target_log_stds[indices],
    }


def build_hybrid_memory_from_store(
    *,
    payload_store: dict[str, list[dict[str, np.ndarray]]],
    selected_segments: tuple[str, ...],
    segment_weights: dict[str, float] | None,
    background_ratio: float,
    task_specific_segments: tuple[str, ...] = (),
    task_specific_ratio: float = 0.0,
    task_specific_payload_store: dict[str, list[dict[str, np.ndarray]]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if (
        not np.isfinite(background_ratio)
        or not np.isfinite(task_specific_ratio)
        or background_ratio < 0.0
        or task_specific_ratio < 0.0
    ):
        raise ValueError("Hybrid memory ratios must be finite and non-negative.")
    selected = set(selected_segments)
    task_specific = set(task_specific_segments).difference(selected)
    general_payloads: list[dict[str, np.ndarray]] = []
    available_counts = {
        segment: int(combine_reference_payloads(payload_store.get(segment, []))[0].shape[0])
        for segment in selected_segments
    }
    nonempty_segments = tuple(
        segment for segment in selected_segments if available_counts[segment] > 0
    )
    if nonempty_segments:
        requested_weights = None if segment_weights is None else {
            segment: segment_weights[segment] for segment in nonempty_segments
        }
        probabilities = dict(
            normalize_segment_weights(
                requested_weights,
                selected_segments=nonempty_segments,
                field_name="segment_weights",
            )
        )
        general_budget = sum(available_counts.values())
        exact_counts = {
            segment: general_budget * probabilities[segment]
            for segment in nonempty_segments
        }
        allocated_counts = {
            segment: int(np.floor(exact_counts[segment]))
            for segment in nonempty_segments
        }
        remainder = general_budget - sum(allocated_counts.values())
        remainder_order = sorted(
            nonempty_segments,
            key=lambda segment: exact_counts[segment] - allocated_counts[segment],
            reverse=True,
        )
        for segment in remainder_order[:remainder]:
            allocated_counts[segment] += 1
        for segment in nonempty_segments:
            weighted_payload = _resample_payload_states(
                payload_store.get(segment, []),
                allocated_counts[segment],
            )
            if weighted_payload is not None:
                general_payloads.append(weighted_payload)
    general_count = sum(payload["observations"].shape[0] for payload in general_payloads)

    background_segments = set(SEGMENT_ORDER).difference(selected).difference(task_specific)
    background_payload = _sample_payload_states(
        [payload for segment in SEGMENT_ORDER if segment in background_segments for payload in payload_store[segment]],
        int(round(general_count * background_ratio)),
    )
    if task_specific_payload_store is None:
        task_specific_candidates = [
            payload
            for segment in SEGMENT_ORDER
            if segment in task_specific
            for payload in payload_store[segment]
        ]
    else:
        task_specific_candidates = [
            payload
            for label_payloads in task_specific_payload_store.values()
            for payload in label_payloads
        ]
    task_specific_payload = _sample_payload_states(
        task_specific_candidates,
        int(round(general_count * task_specific_ratio)),
    )
    payloads = list(general_payloads)
    if background_payload is not None:
        payloads.append(background_payload)
    if task_specific_payload is not None:
        payloads.append(task_specific_payload)
    observations, target_means, target_log_stds = combine_reference_payloads(payloads)
    return (
        observations,
        target_means,
        target_log_stds,
        0 if background_payload is None else int(background_payload["observations"].shape[0]),
        0 if task_specific_payload is None else int(task_specific_payload["observations"].shape[0]),
    )


def normalize_exploration_strategy(strategy: str) -> str:
    aliases = {
        "best-return": "best_return",
    }
    return aliases.get(strategy, strategy)


def load_baseline_curves(
    path: str | None,
    *,
    tasks: list[str] | None = None,
) -> list[list[float]] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Baseline curve file must contain a JSON object.")
    curves = payload.get("stochastic_success_curves")
    if not isinstance(curves, list) or not all(
        isinstance(curve, list) for curve in curves
    ):
        raise ValueError(
            "Baseline curve file must contain a list-valued "
            "stochastic_success_curves field."
        )
    baseline_tasks = payload.get("tasks")
    if tasks is None or baseline_tasks is None:
        return curves
    if not isinstance(baseline_tasks, list) or len(baseline_tasks) != len(curves):
        raise ValueError(
            "Baseline curve tasks must be a list aligned with stochastic_success_curves."
        )
    curves_by_task: dict[str, list[float]] = {}
    for task_name, curve in zip(baseline_tasks, curves, strict=True):
        if not isinstance(task_name, str):
            raise ValueError("Baseline curve task names must be strings.")
        canonical_name = canonical_cw10_task_name(task_name)
        if canonical_name in curves_by_task:
            raise ValueError(f"Duplicate baseline curve task {task_name!r}.")
        curves_by_task[canonical_name] = curve
    try:
        return [curves_by_task[canonical_cw10_task_name(task)] for task in tasks]
    except KeyError as exc:
        raise ValueError(
            f"Baseline curves do not contain continual task {exc.args[0]!r}."
        ) from exc


def validate_baseline_curves_for_run(
    curves: list[list[float]] | None,
    *,
    task_count: int,
    steps_per_task: int,
    eval_every: int,
) -> None:
    if curves is None:
        return
    if len(curves) != task_count:
        raise ValueError(
            "Baseline curves must match the continual task count: "
            f"expected {task_count}, got {len(curves)}."
        )
    expected_points = (steps_per_task + eval_every - 1) // eval_every
    for task_index, curve in enumerate(curves):
        if len(curve) != expected_points:
            raise ValueError(
                "Baseline curve length does not match the evaluation schedule: "
                f"task={task_index}, expected={expected_points}, got={len(curve)}."
            )
        values = np.asarray(curve, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Baseline curve {task_index} contains non-finite values.")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(
                f"Baseline curve {task_index} must contain success rates in [0, 1]."
            )
        if 1.0 - float(np.mean(values)) <= 1e-12:
            raise ValueError(
                "Normalized forward transfer is undefined because baseline "
                f"curve {task_index} has no remaining success headroom."
            )


def unique_task_metric_labels(tasks: list[str]) -> list[str]:
    """Keep legacy names unless a repeated environment needs position identity."""

    counts = {task: tasks.count(task) for task in set(tasks)}
    return [
        task if counts[task] == 1 else f"{task}@position_{index}"
        for index, task in enumerate(tasks)
    ]


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
            if getattr(agent, "official_recall_cumulative_guide_mask", False):
                reset_result = env.reset()
            else:
                reset_result = env.reset(
                    seed=seed + head_index * episodes_per_head + episode_index
                )
            observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            episode_return = 0.0
            for _ in range(max_episode_steps):
                if getattr(agent, "official_recall_cumulative_guide_mask", False):
                    action = agent.select_official_recall_guide_action(
                        observation,
                        head_index=head_index,
                    )
                else:
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
    reward_function_version: str,
    num_task_ids: int,
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
            reward_function_version=reward_function_version,
            num_task_ids=num_task_ids,
            task_id_index=eval_task_index,
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
                "ensemble_critic_loss": "" if update_metrics is None else update_metrics.get("ensemble_critic_loss", ""),
                "q_ensemble_std_mean": "" if update_metrics is None else update_metrics.get("q_ensemble_std_mean", ""),
                "td_error_abs_mean": "" if update_metrics is None else update_metrics.get("td_error_abs_mean", ""),
                "priority_beta": "" if update_metrics is None else update_metrics.get("priority_beta", ""),
                "importance_weight_mean": "" if update_metrics is None else update_metrics.get("importance_weight_mean", ""),
                "elapsed_seconds": time.time() - started_at,
            }
        )
        env.close()
        agent.on_evaluation_end(eval_task_index)
    return rows


def collect_successful_reference_observations(
    *,
    agent: SACAgent,
    task_name: str,
    env_version: str,
    reward_function_version: str,
    append_task_id: bool,
    max_episode_steps: int,
    episodes: int,
    max_attempts: int,
    seed: int,
    num_task_ids: int,
    task_id_index: int,
) -> tuple[np.ndarray, int]:
    if episodes <= 0:
        raise ValueError("full_bc_reference_episodes must be positive.")
    if max_attempts <= 0:
        raise ValueError("full_bc_reference_max_attempts must be positive.")
    env = make_cw_env(
        task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        append_task_id=append_task_id,
        env_version=env_version,
        reward_function_version=reward_function_version,
        num_task_ids=num_task_ids,
        task_id_index=task_id_index,
    )
    kept_observations: list[np.ndarray] = []
    successful_episodes = 0
    attempts = 0
    try:
        while successful_episodes < episodes and attempts < max_attempts:
            reset_result = env.reset(seed=seed + attempts)
            observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            episode_observations: list[np.ndarray] = []
            episode_success = False
            for _ in range(max_episode_steps):
                episode_observations.append(np.asarray(observation, dtype=np.float32).copy())
                step_result = env.step(agent.select_action(observation, deterministic=True))
                if len(step_result) == 5:
                    observation, _, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                else:
                    observation, _, done, info = step_result
                episode_success = episode_success or bool(info.get("success", 0.0))
                if done:
                    break
            attempts += 1
            if not episode_success:
                continue
            successful_episodes += 1
            kept_observations.extend(episode_observations)
    finally:
        env.close()

    if kept_observations:
        return np.stack(kept_observations).astype(np.float32), successful_episodes
    empty_env = make_cw_env(
        task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        append_task_id=append_task_id,
        env_version=env_version,
        reward_function_version=reward_function_version,
        num_task_ids=num_task_ids,
        task_id_index=task_id_index,
    )
    try:
        observation_dim = int(empty_env.observation_space.shape[0])
    finally:
        empty_env.close()
    return np.empty((0, observation_dim), dtype=np.float32), successful_episodes


def collect_segmented_reference_observations(
    *,
    agent: SACAgent,
    task_name: str,
    env_version: str,
    reward_function_version: str,
    append_task_id: bool,
    max_episode_steps: int,
    episodes: int,
    max_attempts: int,
    seed: int,
    allowed_segments: set[str],
    background_segment_ratio: float = 0.0,
    task_specific_segments: set[str] | None = None,
    task_specific_segment_ratio: float = 0.0,
    segment_scheme: str = "heuristic_v1",
    post_success_steps: int = 10,
    num_task_ids: int = 10,
    task_id_index: int | None = None,
) -> tuple[np.ndarray, int, int, int]:
    if not allowed_segments:
        raise ValueError("semantic_local_bc requires at least one semantic segment.")
    invalid_segments = sorted(allowed_segments.difference(SEGMENT_ORDER))
    if invalid_segments:
        raise ValueError(
            f"Unsupported semantic segments: {', '.join(invalid_segments)}. "
            f"Supported: {', '.join(SEGMENT_ORDER)}."
        )
    if not np.isfinite(background_segment_ratio) or background_segment_ratio < 0.0:
        raise ValueError(
            "background_segment_ratio must be finite and non-negative."
        )
    if (
        not np.isfinite(task_specific_segment_ratio)
        or task_specific_segment_ratio < 0.0
    ):
        raise ValueError(
            "task_specific_segment_ratio must be finite and non-negative."
        )
    if post_success_steps < 0:
        raise ValueError("post_success_steps must be non-negative.")
    if task_specific_segments is None:
        task_specific_segments = set()
    invalid_task_specific = sorted(task_specific_segments.difference(SEGMENT_ORDER))
    if invalid_task_specific:
        raise ValueError(
            f"Unsupported task-specific semantic segments: {', '.join(invalid_task_specific)}. "
            f"Supported: {', '.join(SEGMENT_ORDER)}."
        )
    env = make_cw_env(
        task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        append_task_id=append_task_id,
        env_version=env_version,
        reward_function_version=reward_function_version,
        num_task_ids=num_task_ids,
        task_id_index=task_id_index,
    )
    selected_observations: list[np.ndarray] = []
    background_observations: list[np.ndarray] = []
    task_specific_observations: list[np.ndarray] = []
    successful_episodes = 0
    attempts = 0
    try:
        while successful_episodes < episodes and attempts < max_attempts:
            reset_result = env.reset(seed=seed + attempts)
            observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            episode_rows: list[tuple[np.ndarray, str]] = []
            episode_success = False
            segmenter_v2 = TaskAwareSegmenter()
            segmenter_v3 = TaskAwareV3Segmenter()
            previous_object_position: np.ndarray | None = None
            previous_action: np.ndarray | None = None
            previous_info: dict[str, Any] | None = None
            success_detected = False
            collected_post_success_steps = 0
            for _ in range(max_episode_steps):
                if success_detected and post_success_steps == 0:
                    break
                features = extract_features(
                    env,
                    task_name,
                    episode_success,
                    previous_object_position=previous_object_position,
                    action=previous_action,
                    info=previous_info,
                )
                if segment_scheme == "heuristic_v1":
                    label = segment_label(features)
                elif segment_scheme == "task_aware_v2":
                    label = segmenter_v2.label(features)
                elif segment_scheme == "task_aware_v3":
                    label = segmenter_v3.label(features).general
                else:
                    raise ValueError(f"Unsupported semantic segment scheme: {segment_scheme}")
                episode_rows.append(
                    (
                        np.asarray(observation, dtype=np.float32).copy(),
                        label,
                    )
                )
                if success_detected:
                    collected_post_success_steps += 1
                    if collected_post_success_steps >= post_success_steps:
                        break
                previous_object_position = np.asarray(
                    env.unwrapped._get_pos_objects(), dtype=np.float64
                ).reshape(-1)[:3].copy()
                previous_action = agent.select_action(observation, deterministic=True)
                step_result = env.step(previous_action)
                if len(step_result) == 5:
                    observation, _, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                else:
                    observation, _, done, info = step_result
                previous_info = info
                episode_success = episode_success or bool(info.get("success", 0.0))
                success_detected = success_detected or bool(info.get("success", 0.0))
                if done:
                    break
            attempts += 1
            if not episode_success:
                continue
            successful_episodes += 1
            for observation_row, label in episode_rows:
                if label in allowed_segments:
                    selected_observations.append(observation_row)
                elif label in task_specific_segments:
                    task_specific_observations.append(observation_row)
                else:
                    background_observations.append(observation_row)
    finally:
        env.close()

    kept_observations = list(selected_observations)
    num_background_kept = 0
    num_task_specific_kept = 0
    if selected_observations and background_observations and background_segment_ratio > 0.0:
        max_background = int(round(len(selected_observations) * background_segment_ratio))
        if max_background > 0:
            num_background_kept = min(max_background, len(background_observations))
            sample_indices = np.linspace(
                0,
                len(background_observations) - 1,
                num=num_background_kept,
                dtype=int,
            )
            kept_observations.extend(background_observations[index] for index in sample_indices)
    if selected_observations and task_specific_observations and task_specific_segment_ratio > 0.0:
        max_task_specific = int(round(len(selected_observations) * task_specific_segment_ratio))
        if max_task_specific > 0:
            num_task_specific_kept = min(max_task_specific, len(task_specific_observations))
            sample_indices = np.linspace(
                0,
                len(task_specific_observations) - 1,
                num=num_task_specific_kept,
                dtype=int,
            )
            kept_observations.extend(task_specific_observations[index] for index in sample_indices)

    if kept_observations:
        return (
            np.stack(kept_observations).astype(np.float32),
            successful_episodes,
            num_background_kept,
            num_task_specific_kept,
        )
    empty_env = make_cw_env(
        task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        append_task_id=append_task_id,
        env_version=env_version,
        reward_function_version=reward_function_version,
        num_task_ids=num_task_ids,
        task_id_index=task_id_index,
    )
    try:
        observation_dim = int(empty_env.observation_space.shape[0])
    finally:
        empty_env.close()
    return (
        np.empty((0, observation_dim), dtype=np.float32),
        successful_episodes,
        num_background_kept,
        num_task_specific_kept,
    )


def collect_reference_payloads_by_segment(
    *,
    agent: FullBehaviorCloningSACAgent,
    task_name: str,
    env_version: str,
    reward_function_version: str,
    append_task_id: bool,
    max_episode_steps: int,
    episodes: int,
    max_attempts: int,
    seed: int,
    segment_scheme: str,
    post_success_steps: int,
    num_task_ids: int,
    task_id_index: int,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    int,
]:
    env = make_cw_env(
        task_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        append_task_id=append_task_id,
        env_version=env_version,
        reward_function_version=reward_function_version,
        num_task_ids=num_task_ids,
        task_id_index=task_id_index,
    )
    segment_observations: dict[str, list[np.ndarray]] = {
        segment: [] for segment in SEGMENT_ORDER
    }
    task_specific_observations: dict[str, list[np.ndarray]] = {}
    successful_episodes = 0
    attempts = 0
    if post_success_steps < 0:
        raise ValueError("post_success_steps must be non-negative.")
    try:
        while successful_episodes < episodes and attempts < max_attempts:
            reset_result = env.reset(seed=seed + attempts)
            observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            episode_rows: list[tuple[np.ndarray, str, str | None]] = []
            episode_success = False
            segmenter_v2 = TaskAwareSegmenter()
            segmenter_v3 = TaskAwareV3Segmenter()
            previous_object_position: np.ndarray | None = None
            previous_action: np.ndarray | None = None
            previous_info: dict[str, Any] | None = None
            success_detected = False
            collected_post_success_steps = 0
            for _ in range(max_episode_steps):
                if success_detected and post_success_steps == 0:
                    break
                features = extract_features(
                    env,
                    task_name,
                    episode_success,
                    previous_object_position=previous_object_position,
                    action=previous_action,
                    info=previous_info,
                )
                if segment_scheme == "heuristic_v1":
                    label = segment_label(features)
                elif segment_scheme == "task_aware_v2":
                    label = segmenter_v2.label(features)
                elif segment_scheme == "task_aware_v3":
                    semantic_label = segmenter_v3.label(features)
                    label = semantic_label.general
                    task_specific_label = semantic_label.task_specific
                else:
                    raise ValueError(f"Unsupported semantic segment scheme: {segment_scheme}")
                if segment_scheme != "task_aware_v3":
                    task_specific_label = None
                episode_rows.append(
                    (
                        np.asarray(observation, dtype=np.float32).copy(),
                        label,
                        task_specific_label,
                    )
                )
                if success_detected:
                    collected_post_success_steps += 1
                    if collected_post_success_steps >= post_success_steps:
                        break
                previous_object_position = np.asarray(
                    env.unwrapped._get_pos_objects(), dtype=np.float64
                ).reshape(-1)[:3].copy()
                previous_action = agent.select_action(observation, deterministic=True)
                step_result = env.step(previous_action)
                if len(step_result) == 5:
                    observation, _, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                else:
                    observation, _, done, info = step_result
                previous_info = info
                episode_success = episode_success or bool(info.get("success", 0.0))
                success_detected = success_detected or bool(info.get("success", 0.0))
                if done:
                    break
            attempts += 1
            if not episode_success:
                continue
            successful_episodes += 1
            for observation_row, label, task_specific_label in episode_rows:
                segment_observations[label].append(observation_row)
                if task_specific_label is not None:
                    task_specific_observations.setdefault(task_specific_label, []).append(
                        observation_row
                    )
    finally:
        env.close()

    payloads: dict[str, dict[str, np.ndarray]] = {}
    for segment in SEGMENT_ORDER:
        observations_list = segment_observations[segment]
        if observations_list:
            observations = np.stack(observations_list).astype(np.float32)
            target_means, target_log_stds = agent.compute_reference_targets(observations)
            payloads[segment] = {
                "observations": observations,
                "target_means": target_means.detach().cpu().numpy().astype(np.float32),
                "target_log_stds": target_log_stds.detach().cpu().numpy().astype(np.float32),
            }
        else:
            payloads[segment] = {
                "observations": np.empty((0, agent.observation_dim), dtype=np.float32),
                "target_means": np.empty((0, agent.action_dim), dtype=np.float32),
                "target_log_stds": np.empty((0, agent.action_dim), dtype=np.float32),
            }
    task_specific_payloads: dict[str, dict[str, np.ndarray]] = {}
    for label, observations_list in task_specific_observations.items():
        observations = np.stack(observations_list).astype(np.float32)
        target_means, target_log_stds = agent.compute_reference_targets(observations)
        task_specific_payloads[label] = {
            "observations": observations,
            "target_means": target_means.detach().cpu().numpy().astype(np.float32),
            "target_log_stds": target_log_stds.detach().cpu().numpy().astype(np.float32),
        }
    return payloads, task_specific_payloads, successful_episodes


def run_cw10_experiment(args: Any, run_dir: Path) -> dict[str, Any]:
    seed_global_rngs(args.seed)
    tasks = get_continual_task_sequence(args.task_sequence, args.env_version)
    if args.task_sequence == "cw10":
        tasks = tasks[: args.sequence_task_count]
    started_at = time.time()
    method = get_method(args.method)
    append_task_id = method.append_task_id

    first_env = make_cw_env(
        tasks[0],
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        append_task_id=append_task_id,
        env_version=args.env_version,
        reward_function_version=args.reward_function_version,
        num_task_ids=len(tasks),
        task_id_index=0,
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
        reward_function_version=args.reward_function_version,
        num_task_ids=len(tasks),
        task_id_index=0,
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
    if method.complete_reference_memory and not isinstance(
        agent,
        FullBehaviorCloningSACAgent,
    ):
        raise RuntimeError(
            f"Method {args.method} declares complete reference memory but its "
            "agent cannot store behavior-cloning reference observations."
        )
    if hasattr(agent, "bc_gradient_strategy"):
        expected_strategy = bc_gradient_strategy_for_method(args.method)
        actual_strategy = str(agent.bc_gradient_strategy)
        if actual_strategy != expected_strategy:
            raise RuntimeError(
                f"Method-agent gradient strategy mismatch: method={args.method}, "
                f"expected={expected_strategy}, actual={actual_strategy}."
            )
        required_combination = required_bc_combination_strategy(args.method)
        actual_combination = str(agent.bc_combination_strategy)
        if (
            required_combination is not None
            and actual_combination != required_combination
        ):
            raise RuntimeError(
                "Method-agent BC combination strategy mismatch: "
                f"method={args.method}, expected={required_combination}, "
                f"actual={actual_combination}."
            )
    if hasattr(agent, "gradient_diagnostics"):
        actual_diagnostics = bool(agent.gradient_diagnostics.enabled)
        if actual_diagnostics != bool(args.gradient_diagnostics):
            raise RuntimeError(
                "Method-agent gradient diagnostics mismatch: "
                f"requested={bool(args.gradient_diagnostics)}, "
                f"actual={actual_diagnostics}."
            )
    elif args.gradient_diagnostics:
        raise RuntimeError(
            f"Method {args.method} does not support actor gradient diagnostics."
        )
    replay_buffer = method.build_replay_buffer(
        args=args,
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    for attribute, expected in (
        ("observation_dim", observation_dim),
        ("action_dim", action_dim),
        ("capacity", args.replay_size),
    ):
        actual = getattr(replay_buffer, attribute, None)
        if actual != expected:
            raise RuntimeError(
                f"Method replay buffer {attribute} mismatch: "
                f"expected={expected}, actual={actual}."
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
    segment_selection_manifest = (
        load_segment_selection_manifest(args.segment_selection_manifest)
        if args.segment_selection_manifest
        else None
    )
    task_specific_segment_manifest = (
        load_segment_selection_manifest(args.task_specific_segment_manifest)
        if args.task_specific_segment_manifest
        else None
    )
    last_update_metrics: dict[str, float] | None = None
    global_progress_gate = (
        BCProgressGate(
            thresholds=tuple(args.bc_progress_thresholds),
            multipliers=tuple(args.bc_progress_multipliers),
            window=args.bc_progress_window,
        )
        if args.bc_progress_gate
        else None
    )

    def flush_gradient_diagnostics(
        *,
        diagnostic_evaluation_index: int | None = None,
        diagnostic_global_step: int | None = None,
        diagnostic_active_task_step: int | None = None,
    ) -> None:
        if not hasattr(agent, "drain_gradient_diagnostics"):
            return
        diagnostic_rows = agent.drain_gradient_diagnostics()
        for category, rows in diagnostic_rows.items():
            for row in rows:
                row["evaluation_index"] = diagnostic_evaluation_index
                row["global_step"] = diagnostic_global_step
                row["active_task_step"] = diagnostic_active_task_step
                current_index = int(row["current_task_index"])
                row["current_task_name"] = tasks[current_index]
                if category == "task_pairs":
                    source_index = int(row["source_task_index"])
                    row["source_task_name"] = tasks[source_index]
        append_csv_rows(
            run_dir / "gradient_diagnostics" / "gradient_windows.csv",
            GRADIENT_WINDOW_FIELDS,
            diagnostic_rows["windows"],
        )
        append_csv_rows(
            run_dir / "gradient_diagnostics" / "gradient_layers.csv",
            GRADIENT_LAYER_FIELDS,
            diagnostic_rows["layers"],
        )
        append_csv_rows(
            run_dir / "gradient_diagnostics" / "gradient_task_pairs.csv",
            GRADIENT_TASK_PAIR_FIELDS,
            diagnostic_rows["task_pairs"],
        )
    baseline_curves = load_baseline_curves(args.baseline_curves, tasks=tasks)
    validate_baseline_curves_for_run(
        baseline_curves,
        task_count=len(tasks),
        steps_per_task=args.steps_per_task,
        eval_every=args.eval_every,
    )
    stage_aware_memory_store: dict[str, list[dict[str, np.ndarray]]] = {
        segment: [] for segment in SEGMENT_ORDER
    }
    static_semantic_memory_store: dict[str, list[dict[str, np.ndarray]]] = {
        segment: [] for segment in SEGMENT_ORDER
    }
    llm_online_memory_store: dict[str, list[dict[str, np.ndarray]]] = {
        segment: [] for segment in SEGMENT_ORDER
    }
    llm_task_specific_memory_store: dict[str, list[dict[str, np.ndarray]]] = {}
    llm_memory_source_task_indices: set[int] = set()
    llm_api_key = None
    if args.segment_selection_mode == "llm_online":
        llm_api_key = read_api_key(args.llm_controller_api_key_env)
    schedule_api_key = None
    schedule_api_key_error: str | None = None
    if args.llm_task_bc_schedule:
        try:
            schedule_api_key = read_api_key(args.llm_controller_api_key_env)
        except Exception as error:
            schedule_api_key_error = f"{type(error).__name__}: {error}"
    prior_api_key = None
    prior_api_key_error: str | None = None
    if method.llm_prior_initialization:
        try:
            prior_api_key = read_api_key(args.llm_controller_api_key_env)
        except Exception as error:
            prior_api_key_error = f"{type(error).__name__}: {error}"

    for task_index, task_name in enumerate(tasks):
        prior_decision: LLMPolicyPriorDecision | None = None
        prior_result: PriorInitializationResult | None = None
        active_progress_gate = global_progress_gate
        schedule_decision: LLMBCScheduleDecision | None = None
        if args.llm_task_bc_schedule:
            if task_index == 0:
                schedule_decision = standard_fallback_decision(
                    "No old-task memory exists for the first task."
                )
                schedule_decision = LLMBCScheduleDecision(
                    **{**schedule_decision.__dict__, "source": "first_task"}
                )
                append_controller_log(
                    run_dir / "bc_schedule_decisions.json",
                    {
                        "task_index": task_index,
                        "task_name": task_name,
                        "visible_previous_task_indices": [],
                        "visible_previous_task_names": [],
                        "model": args.llm_controller_model,
                        "prompt_path": None,
                        "schedule": schedule_decision.schedule,
                        "reason_codes": list(schedule_decision.reason_codes),
                        "confidence": schedule_decision.confidence,
                        "reason": schedule_decision.reason,
                        "source": schedule_decision.source,
                        "raw_response_text": "",
                    },
                )
            else:
                prompt = build_task_schedule_prompt(
                    current_task_name=task_name,
                    previous_task_names=list(tasks[:task_index]),
                    prompt_dir=args.llm_bc_schedule_prompt_dir,
                )
                prompt_dir = run_dir / "bc_schedule_prompts"
                prompt_dir.mkdir(parents=True, exist_ok=True)
                prompt_path = prompt_dir / f"task_{task_index:02d}_{task_name}.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                try:
                    if schedule_api_key is None:
                        raise RuntimeError(
                            "LLM schedule API key was not loaded: "
                            f"{schedule_api_key_error or 'unknown error'}"
                        )
                    schedule_decision = call_task_schedule_controller(
                        api_key=schedule_api_key,
                        model=args.llm_controller_model,
                        prompt=prompt,
                        max_output_tokens=args.llm_controller_max_output_tokens,
                    )
                except Exception as error:
                    schedule_decision = standard_fallback_decision(
                        f"Controller failure: {type(error).__name__}: {error}"
                    )
                append_controller_log(
                    run_dir / "bc_schedule_decisions.json",
                    {
                        "task_index": task_index,
                        "task_name": task_name,
                        "visible_previous_task_indices": list(range(task_index)),
                        "visible_previous_task_names": list(tasks[:task_index]),
                        "model": args.llm_controller_model,
                        "prompt_path": str(prompt_path.relative_to(run_dir)),
                        "schedule": schedule_decision.schedule,
                        "reason_codes": list(schedule_decision.reason_codes),
                        "confidence": schedule_decision.confidence,
                        "reason": schedule_decision.reason,
                        "source": schedule_decision.source,
                        "raw_response_text": schedule_decision.raw_response_text,
                    },
                )
            active_progress_gate = (
                BCProgressGate(
                    thresholds=tuple(args.bc_progress_thresholds),
                    multipliers=tuple(args.bc_progress_multipliers),
                    window=args.bc_progress_window,
                )
                if schedule_decision.schedule == SCHEDULE_DELAYED
                else None
            )
            print(
                f"[bc-schedule] task={task_name} schedule={schedule_decision.schedule} "
                f"source={schedule_decision.source} confidence={schedule_decision.confidence:.3f}"
            )
        if active_progress_gate is not None:
            if not isinstance(agent, FullBehaviorCloningSACAgent):
                raise RuntimeError("BC progress gate requires FullBehaviorCloningSACAgent.")
            active_progress_gate.reset()
            agent.set_bc_progress_multiplier(active_progress_gate.multiplier)
        elif isinstance(agent, FullBehaviorCloningSACAgent):
            agent.set_bc_progress_multiplier(1.0)
        leaked_sources = sorted(
            source_index
            for source_index in llm_memory_source_task_indices
            if source_index >= task_index
        )
        if leaked_sources:
            raise RuntimeError(
                "LLM memory leakage detected before task start: "
                f"current_task_index={task_index}, invalid_sources={leaked_sources}."
            )
        segment_selection: SegmentSelection | None = None
        task_specific_selection: SegmentSelection | None = None
        active_reference_states = 0
        active_background_reference_states = 0
        active_task_specific_reference_states = 0
        success_replay = (
            SuccessfulStateReservoir(
                capacity=args.episodic_memory_per_task,
                observation_dim=observation_dim,
                seed=args.seed + 600_000 + task_index,
            )
            if method.success_replay_teacher is not None
            else None
        )
        best_teacher_state: dict[str, torch.Tensor] | None = None
        best_teacher_score = (float("-inf"), float("-inf"))
        best_teacher_step: int | None = None
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
        if args.method == "stage_aware_semantic_bc":
            if not isinstance(agent, FullBehaviorCloningSACAgent):
                raise RuntimeError("stage_aware_semantic_bc requires FullBehaviorCloningSACAgent.")
            initial_segments = stage_aware_segment_groups(0.0)
            observations, target_means, target_log_stds = combine_reference_payloads(
                [
                    payload
                    for segment in initial_segments
                    for payload in stage_aware_memory_store[segment]
                ]
            )
            if observations.shape[0] > 0:
                agent.set_reference_memory_with_targets(
                    observations=observations,
                    target_means=target_means,
                    target_log_stds=target_log_stds,
                )
            else:
                agent.clear_reference_memory()
        elif args.method in STATIC_SEMANTIC_METHODS and args.segment_selection_mode != "llm_online":
            if not isinstance(agent, FullBehaviorCloningSACAgent):
                raise RuntimeError(f"{args.method} requires FullBehaviorCloningSACAgent.")
            previous_task_name = None if task_index == 0 else tasks[task_index - 1]
            if (
                args.method in {"adaptive_semantic_bc", "general_task_specific_bc", "semantic_hybrid_bc"}
                or args.segment_selection_mode == "task_adaptive"
            ):
                segment_selection = select_segments_for_task_pair(
                    previous_task_name=previous_task_name,
                    new_task_name=task_name,
                    manifest=segment_selection_manifest,
                    fallback_segments=args.semantic_segments,
                )
            else:
                fixed_weights = dict(
                    normalize_segment_weights(
                        args.semantic_segment_weights,
                        selected_segments=tuple(args.semantic_segments),
                        field_name="semantic_segment_weights",
                    )
                )
                segment_selection = SegmentSelection(
                    selected_segments=tuple(args.semantic_segments),
                    priority=tuple(args.semantic_segments),
                    weights=tuple(fixed_weights.items()),
                    reason="Fixed semantic segment selection from CLI/config.",
                    selection_source="fixed",
                )
            if args.method in {"general_task_specific_bc", "semantic_hybrid_bc"}:
                task_specific_selection = select_task_specific_segments_for_task(
                    task_name=task_name,
                    manifest=task_specific_segment_manifest,
                )
            (
                observations,
                target_means,
                target_log_stds,
                active_background_reference_states,
                active_task_specific_reference_states,
            ) = build_hybrid_memory_from_store(
                payload_store=static_semantic_memory_store,
                selected_segments=segment_selection.selected_segments,
                segment_weights=dict(segment_selection.weights),
                background_ratio=args.background_segment_ratio,
                task_specific_segments=(
                    ()
                    if task_specific_selection is None
                    else task_specific_selection.selected_segments
                ),
                task_specific_ratio=args.task_specific_segment_ratio,
            )
            active_reference_states = int(observations.shape[0])
            if active_reference_states > 0:
                agent.set_reference_memory_with_targets(
                    observations=observations,
                    target_means=target_means,
                    target_log_stds=target_log_stds,
                )
            else:
                agent.clear_reference_memory()
        elif (
            args.segment_selection_mode == "llm_online"
            and task_index > 0
            and isinstance(agent, FullBehaviorCloningSACAgent)
        ):
            initial_selected_segments = tuple(args.semantic_segments)
            initial_task_specific = select_task_specific_segments_for_task(
                task_name=task_name,
                manifest=task_specific_segment_manifest,
            )
            observations, target_means, target_log_stds, _, _ = build_hybrid_memory_from_store(
                payload_store=llm_online_memory_store,
                selected_segments=initial_selected_segments,
                segment_weights={
                    segment: float((args.semantic_segment_weights or {}).get(segment, 1.0))
                    for segment in initial_selected_segments
                },
                background_ratio=args.background_segment_ratio,
                task_specific_segments=(
                    () if initial_task_specific is None else initial_task_specific.selected_segments
                ),
                task_specific_ratio=args.task_specific_segment_ratio,
                task_specific_payload_store=llm_task_specific_memory_store,
            )
            if observations.shape[0] > 0:
                agent.set_reference_memory_with_targets(
                    observations=observations,
                    target_means=target_means,
                    target_log_stds=target_log_stds,
                )
            else:
                agent.clear_reference_memory()

        exploration_strategy, exploration_available_heads = method.exploration_config(
            task_index=task_index,
            strategy=effective_exploration_strategy,
        )
        actual_exploration_strategy = (
            "random" if exploration_strategy is None else exploration_strategy
        )

        env = make_cw_env(
            task_name,
            seed=args.seed + task_index,
            max_episode_steps=args.max_episode_steps,
            append_task_id=append_task_id,
            env_version=args.env_version,
            reward_function_version=args.reward_function_version,
            num_task_ids=len(tasks),
            task_id_index=task_index,
        )
        if method.llm_prior_initialization:
            if task_index == 0:
                prior_decision = no_transfer_decision(
                    "No previous policy head exists for the first task.",
                    source="first_task",
                )
                prior_prompt_path = None
            else:
                prior_prompt = build_policy_prior_prompt(
                    current_task_name=task_name,
                    previous_task_names=list(tasks[:task_index]),
                    prompt_dir=args.llm_prior_prompt_dir,
                )
                prior_prompt_dir = run_dir / "policy_prior_prompts"
                prior_prompt_dir.mkdir(parents=True, exist_ok=True)
                prior_prompt_file = prior_prompt_dir / f"task_{task_index:02d}_{task_name}.txt"
                prior_prompt_file.write_text(prior_prompt, encoding="utf-8")
                prior_prompt_path = str(prior_prompt_file.relative_to(run_dir))
                try:
                    if prior_api_key is None:
                        raise RuntimeError(
                            "LLM prior API key was not loaded: "
                            f"{prior_api_key_error or 'unknown error'}"
                        )
                    prior_decision = call_policy_prior_controller(
                        api_key=prior_api_key,
                        model=args.llm_controller_model,
                        prompt=prior_prompt,
                        max_output_tokens=args.llm_controller_max_output_tokens,
                        previous_task_names=list(tasks[:task_index]),
                        minimum_confidence=args.llm_prior_min_confidence,
                    )
                except Exception as error:
                    prior_decision = no_transfer_decision(
                        f"Controller failure: {type(error).__name__}: {error}"
                    )
            if prior_decision.source_task_name is not None:
                source_task_index = tasks.index(prior_decision.source_task_name)
                if source_task_index >= task_index:
                    raise RuntimeError("Policy-prior selector leaked current or future task data.")
                reset_observations = collect_reset_observations(
                    env,
                    count=args.implicit_prior_reset_states,
                    seed=args.seed + 80_000 + task_index * 1_000,
                )
                prior_result = initialize_actor_head_from_source(
                    agent,
                    observations=reset_observations,
                    source_task_index=source_task_index,
                    target_task_index=task_index,
                    updates=args.implicit_prior_updates,
                    learning_rate=args.implicit_prior_learning_rate,
                )
            append_controller_log(
                run_dir / "policy_prior_decisions.json",
                {
                    "task_index": task_index,
                    "task_name": task_name,
                    "visible_previous_task_indices": list(range(task_index)),
                    "visible_previous_task_names": list(tasks[:task_index]),
                    "model": args.llm_controller_model,
                    "prompt_path": prior_prompt_path,
                    "source_task_name": prior_decision.source_task_name,
                    "confidence": prior_decision.confidence,
                    "reason": prior_decision.reason,
                    "source": prior_decision.source,
                    "raw_response_text": prior_decision.raw_response_text,
                    "reset_states": None if prior_result is None else prior_result.reset_states,
                    "updates": None if prior_result is None else prior_result.updates,
                    "initial_kl": None if prior_result is None else prior_result.initial_kl,
                    "final_kl": None if prior_result is None else prior_result.final_kl,
                },
            )
            print(
                f"[policy-prior] task={task_name} source="
                f"{prior_decision.source_task_name or 'none'} "
                f"confidence={prior_decision.confidence:.3f} "
                f"decision_source={prior_decision.source}"
            )
            exploration_strategy, exploration_available_heads, actual_exploration_strategy = (
                resolve_prior_exploration(
                    task_index=task_index,
                    decision=prior_decision,
                )
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
                policy_from_start=bool(
                    selected_guide is not None
                    and method.policy_from_start_after_guide
                ),
            ),
        )
        curriculum_eval_env = (
            make_cw_env(
                task_name,
                seed=args.seed + 20_000 + task_index,
                max_episode_steps=args.max_episode_steps,
                append_task_id=append_task_id,
                env_version=args.env_version,
                reward_function_version=args.reward_function_version,
                num_task_ids=len(tasks),
                task_id_index=task_index,
            )
            if guide_head_index is not None
            else None
        )

        task_curve_success: list[float] = []
        task_curve_returns: list[float] = []
        gradient_updates_before_task = cumulative_gradient_updates
        current_stage_segments = (
            stage_aware_segment_groups(0.0)
            if args.method == "stage_aware_semantic_bc"
            else tuple()
        )
        current_llm_segments = (
            tuple(args.semantic_segments)
            if args.segment_selection_mode == "llm_online"
            else tuple()
        )
        current_llm_priority = current_llm_segments
        current_llm_weights = dict(
            normalize_segment_weights(
                args.semantic_segment_weights,
                selected_segments=current_llm_segments,
                field_name="semantic_segment_weights",
            )
        ) if current_llm_segments else {}

        def evaluate(task_step: int, gradient_updates: int, update_metrics: dict[str, float] | None) -> None:
            nonlocal global_step, evaluation_index, last_update_metrics
            nonlocal curriculum_stage, curriculum_reference
            nonlocal curriculum_stage_best, curriculum_stage_evaluations
            nonlocal current_stage_segments, current_llm_segments, current_llm_priority, current_llm_weights
            nonlocal best_teacher_state, best_teacher_score, best_teacher_step
            evaluation_index += 1
            flush_gradient_diagnostics(
                diagnostic_evaluation_index=evaluation_index,
                diagnostic_global_step=task_index * args.steps_per_task + task_step,
                diagnostic_active_task_step=task_step,
            )
            last_update_metrics = update_metrics
            global_step = task_index * args.steps_per_task + task_step
            if args.method == "stage_aware_semantic_bc":
                progress_ratio = float(task_step) / float(args.steps_per_task)
                next_stage_segments = stage_aware_segment_groups(progress_ratio)
                if next_stage_segments != current_stage_segments:
                    if not isinstance(agent, FullBehaviorCloningSACAgent):
                        raise RuntimeError("stage_aware_semantic_bc requires FullBehaviorCloningSACAgent.")
                    observations, target_means, target_log_stds = combine_reference_payloads(
                        [
                            payload
                            for segment in next_stage_segments
                            for payload in stage_aware_memory_store[segment]
                        ]
                    )
                    if observations.shape[0] > 0:
                        agent.set_reference_memory_with_targets(
                            observations=observations,
                            target_means=target_means,
                            target_log_stds=target_log_stds,
                        )
                    else:
                        agent.clear_reference_memory()
                    current_stage_segments = next_stage_segments
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
                reward_function_version=args.reward_function_version,
                num_task_ids=len(tasks),
                evaluation_task_indices=[task_index],
            )
            all_eval_rows.extend(rows)
            active_row = rows[0]
            if active_progress_gate is None:
                active_row["bc_progress_score"] = ""
                active_row["bc_progress_multiplier"] = ""
                active_row["bc_progress_next_multiplier"] = ""
            else:
                applied_multiplier = agent.bc_progress_multiplier
                next_multiplier = active_progress_gate.update(
                    float(active_row["stochastic_success_rate"])
                )
                agent.set_bc_progress_multiplier(next_multiplier)
                active_row["bc_progress_score"] = active_progress_gate.score
                active_row["bc_progress_multiplier"] = applied_multiplier
                active_row["bc_progress_next_multiplier"] = next_multiplier
                print(
                    f"[bc-progress] task={task_name} step={task_step:,} "
                    f"score={active_progress_gate.score:.3f} "
                    f"applied={applied_multiplier:.3f} next={next_multiplier:.3f}"
                )
            task_curve_success.append(float(active_row["stochastic_success_rate"]))
            task_curve_returns.append(float(active_row["stochastic_average_return"]))
            if method.success_replay_teacher == "best":
                candidate_score = (
                    float(active_row["stochastic_success_rate"]),
                    float(active_row["stochastic_average_return"]),
                )
                if candidate_score > best_teacher_score:
                    best_teacher_score = candidate_score
                    best_teacher_state = clone_module_state(agent.actor)
                    best_teacher_step = task_step
            print(
                f"[{args.task_sequence}] task={task_name} "
                f"step={task_step:,}/{args.steps_per_task:,} "
                f"success={active_row['stochastic_success_rate']:.3f} "
                f"return={active_row['stochastic_average_return']:.3f}"
            )
            if (
                args.segment_selection_mode == "llm_online"
                and task_index > 0
                and isinstance(agent, FullBehaviorCloningSACAgent)
                and llm_api_key is not None
                and len(task_curve_success) % args.llm_controller_update_every_evals == 0
            ):
                leaked_sources = sorted(
                    source_index
                    for source_index in llm_memory_source_task_indices
                    if source_index >= task_index
                )
                if leaked_sources:
                    raise RuntimeError(
                        "LLM memory leakage detected before controller call: "
                        f"current_task_index={task_index}, invalid_sources={leaked_sources}."
                    )
                available_counts = segment_payload_counts(llm_online_memory_store)
                available_task_specific_counts = segment_payload_counts(
                    llm_task_specific_memory_store,
                    labels=tuple(sorted(llm_task_specific_memory_store)),
                )
                prompt = build_llm_gate_prompt(
                    previous_task_name=tasks[task_index - 1],
                    current_task_name=task_name,
                    evaluation_index_within_task=len(task_curve_success),
                    task_step=task_step,
                    task_total_steps=args.steps_per_task,
                    recent_success_curve=task_curve_success[-5:],
                    recent_return_curve=task_curve_returns[-5:],
                    available_segment_counts=available_counts,
                    available_task_specific_counts=available_task_specific_counts,
                    current_selected_segments=current_llm_segments,
                    current_segment_weights=current_llm_weights,
                    prompt_dir=args.llm_controller_prompt_dir,
                )
                controller_log = {
                    "task_index": task_index,
                    "task_name": task_name,
                    "task_step": task_step,
                    "evaluation_index_within_task": len(task_curve_success),
                    "available_segment_counts": available_counts,
                    "available_task_specific_counts": available_task_specific_counts,
                    "memory_source_task_indices": sorted(llm_memory_source_task_indices),
                    "memory_source_task_names": [
                        tasks[source_index]
                        for source_index in sorted(llm_memory_source_task_indices)
                    ],
                    "background_segment_ratio": args.background_segment_ratio,
                    "task_specific_segment_ratio": args.task_specific_segment_ratio,
                    "model": args.llm_controller_model,
                }
                prompt_dir = run_dir / "controller_prompts"
                prompt_dir.mkdir(parents=True, exist_ok=True)
                prompt_path = prompt_dir / (
                    f"task_{task_index:02d}_eval_{len(task_curve_success):03d}.txt"
                )
                prompt_path.write_text(prompt, encoding="utf-8")
                controller_log["prompt_path"] = str(prompt_path.relative_to(run_dir))
                raw_response_text = ""
                try:
                    controller_payload, raw_response_text = call_openai_json_controller(
                        api_key=llm_api_key,
                        model=args.llm_controller_model,
                        prompt=prompt,
                        max_output_tokens=args.llm_controller_max_output_tokens,
                    )
                    selected_value = controller_payload.get("selected_segments")
                    if not isinstance(selected_value, list) or not selected_value:
                        raise ValueError("Controller must select at least one segment.")
                    if not all(isinstance(segment, str) for segment in selected_value):
                        raise ValueError("Controller segment names must be strings.")
                    selected_segments = tuple(selected_value)
                    priority_value = controller_payload.get("priority", list(selected_segments))
                    if not isinstance(priority_value, list) or not all(
                        isinstance(segment, str) for segment in priority_value
                    ):
                        raise ValueError("Controller priority must be a list of segment names.")
                    priority = tuple(priority_value)
                    weights_value = controller_payload.get("weights")
                    if not isinstance(weights_value, dict):
                        raise ValueError("Controller weights must be an object.")
                    weights = {segment: float(weights_value[segment]) for segment in selected_segments}
                    weights = dict(
                        normalize_segment_weights(
                            weights,
                            selected_segments=selected_segments,
                            field_name="Controller weights",
                        )
                    )
                    if not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
                        raise ValueError("Controller weights did not normalize to 1.0.")
                    invalid_selected = sorted(set(selected_segments).difference(SEGMENT_ORDER))
                    invalid_priority = sorted(set(priority).difference(selected_segments))
                    empty_selected = sorted(
                        segment
                        for segment in selected_segments
                        if available_counts.get(segment, 0) <= 0
                    )
                    if invalid_selected:
                        raise ValueError(
                            "Controller returned unsupported segments: "
                            + ", ".join(invalid_selected)
                        )
                    if invalid_priority:
                        raise ValueError(
                            "Controller priority is outside its selected set: "
                            + ", ".join(invalid_priority)
                        )
                    if empty_selected:
                        raise ValueError(
                            "Controller selected empty memory segments: "
                            + ", ".join(empty_selected)
                        )
                    current_task_specific = select_task_specific_segments_for_task(
                        task_name=task_name,
                        manifest=task_specific_segment_manifest,
                    )
                    (
                        observations,
                        target_means,
                        target_log_stds,
                        applied_background_states,
                        applied_task_specific_states,
                    ) = build_hybrid_memory_from_store(
                        payload_store=llm_online_memory_store,
                        selected_segments=selected_segments,
                        segment_weights=weights,
                        background_ratio=args.background_segment_ratio,
                        task_specific_segments=(
                            ()
                            if current_task_specific is None
                            else current_task_specific.selected_segments
                        ),
                        task_specific_ratio=args.task_specific_segment_ratio,
                        task_specific_payload_store=llm_task_specific_memory_store,
                    )
                    if observations.shape[0] <= 0:
                        raise ValueError("Controller selection produced no reference states.")
                    agent.set_reference_memory_with_targets(
                        observations=observations,
                        target_means=target_means,
                        target_log_stds=target_log_stds,
                    )
                    current_llm_segments = selected_segments
                    current_llm_priority = priority
                    current_llm_weights = weights
                    controller_log.update(
                        {
                            "status": "applied",
                            "selected_segments": list(selected_segments),
                            "priority": list(priority),
                            "weights": weights,
                            "background_reference_states": applied_background_states,
                            "task_specific_reference_states": applied_task_specific_states,
                            "reason": str(controller_payload.get("reason", "")),
                            "raw_response_text": raw_response_text,
                        }
                    )
                except Exception as error:
                    controller_log.update(
                        {
                            "status": "fallback_keep_previous",
                            "selected_segments": list(current_llm_segments),
                            "priority": list(current_llm_priority),
                            "weights": current_llm_weights,
                            "reason": "controller_error",
                            "error": f"{type(error).__name__}: {error}",
                            "raw_response_text": raw_response_text,
                            "controller_error_detail": str(error),
                        }
                    )
                    print(
                        "[llm-controller] warning: keeping previous segments "
                        f"{list(current_llm_segments)} after {type(error).__name__}: {error}"
                    )
                append_controller_log(
                    run_dir / "controller_decisions.json",
                    controller_log,
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

        training_summary = trainer.train(
            step_callback=evaluate,
            completed_episode_callback=(
                None if success_replay is None else success_replay.add_episode
            ),
        )
        if curriculum_eval_env is not None:
            curriculum_eval_env.close()
        cumulative_gradient_updates += training_summary.gradient_updates
        agent.on_task_end(
            task_index=task_index,
            replay_buffer=replay_buffer,
            batch_size=args.batch_size,
        )
        reference_success_episodes = None
        reference_states = None
        background_reference_states = None
        task_specific_reference_states = None
        collected_segment_states = None
        if success_replay is not None:
            if not isinstance(agent, FullBehaviorCloningSACAgent):
                raise RuntimeError(
                    "Successful replay relabeling requires FullBehaviorCloningSACAgent."
                )
            reference_observations = success_replay.observations()
            reference_success_episodes = success_replay.successful_episodes
            reference_states = int(reference_observations.shape[0])
            if reference_states > 0:
                if (
                    method.success_replay_teacher == "best"
                    and best_teacher_state is None
                ):
                    raise RuntimeError(
                        "Best-teacher relabeling has no validation snapshot."
                    )
                add_relabelled_reference_memory(
                    agent=agent,
                    observations=reference_observations,
                    teacher_actor_state=(
                        best_teacher_state
                        if method.success_replay_teacher == "best"
                        else None
                    ),
                )
            if method.success_replay_teacher == "best":
                if best_teacher_state is None or best_teacher_step is None:
                    raise RuntimeError("Best-teacher snapshot is missing at task end.")
                teacher_dir = run_dir / "teacher_snapshots"
                teacher_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "actor_state_dict": best_teacher_state,
                        "task_index": task_index,
                        "task_name": task_name,
                        "validation_step": best_teacher_step,
                        "validation_success": best_teacher_score[0],
                        "validation_return": best_teacher_score[1],
                    },
                    teacher_dir / f"task_{task_index}_best_actor.pt",
                )
        if method.complete_reference_memory or args.method in {
            "semantic_local_bc",
            "adaptive_semantic_bc",
            "general_task_specific_bc",
            "semantic_hybrid_bc",
            "stage_aware_semantic_bc",
        }:
            if method.complete_reference_memory:
                reference_observations, reference_success_episodes = collect_successful_reference_observations(
                    agent=agent,
                    task_name=task_name,
                    env_version=args.env_version,
                    reward_function_version=args.reward_function_version,
                    append_task_id=append_task_id,
                    max_episode_steps=args.max_episode_steps,
                    episodes=args.full_bc_reference_episodes,
                    max_attempts=args.full_bc_reference_max_attempts,
                    seed=args.seed + 40_000 + task_index * 1_000,
                    num_task_ids=len(tasks),
                    task_id_index=task_index,
                )
                background_reference_states = 0
                task_specific_reference_states = 0
            elif args.method == "stage_aware_semantic_bc":
                if not isinstance(agent, FullBehaviorCloningSACAgent):
                    raise RuntimeError("stage_aware_semantic_bc requires FullBehaviorCloningSACAgent.")
                per_segment_payloads, _, reference_success_episodes = collect_reference_payloads_by_segment(
                    agent=agent,
                    task_name=task_name,
                    env_version=args.env_version,
                    reward_function_version=args.reward_function_version,
                    append_task_id=append_task_id,
                    max_episode_steps=args.max_episode_steps,
                    episodes=args.full_bc_reference_episodes,
                    max_attempts=args.full_bc_reference_max_attempts,
                    seed=args.seed + 40_000 + task_index * 1_000,
                    segment_scheme=args.semantic_segment_scheme,
                    post_success_steps=args.semantic_post_success_steps,
                    num_task_ids=len(tasks),
                    task_id_index=task_index,
                )
                for segment, payload in per_segment_payloads.items():
                    if payload["observations"].shape[0] > 0:
                        stage_aware_memory_store[segment].append(payload)
                collected_segment_states = {
                    segment: int(payload["observations"].shape[0])
                    for segment, payload in per_segment_payloads.items()
                }
                current_stage_segments = stage_aware_segment_groups(1.0)
                reference_observations, target_means, target_log_stds = combine_reference_payloads(
                    [
                        payload
                        for segment in current_stage_segments
                        for payload in stage_aware_memory_store[segment]
                    ]
                )
                reference_states = int(reference_observations.shape[0])
                if reference_observations.shape[0] > 0:
                    agent.set_reference_memory_with_targets(
                        observations=reference_observations,
                        target_means=target_means,
                        target_log_stds=target_log_stds,
                    )
                else:
                    agent.clear_reference_memory()
                background_reference_states = 0
                task_specific_reference_states = 0
            elif args.segment_selection_mode == "llm_online":
                if not isinstance(agent, FullBehaviorCloningSACAgent):
                    raise RuntimeError("llm_online segment selection requires FullBehaviorCloningSACAgent.")
                (
                    per_segment_payloads,
                    per_task_specific_payloads,
                    reference_success_episodes,
                ) = collect_reference_payloads_by_segment(
                    agent=agent,
                    task_name=task_name,
                    env_version=args.env_version,
                    reward_function_version=args.reward_function_version,
                    append_task_id=append_task_id,
                    max_episode_steps=args.max_episode_steps,
                    episodes=args.full_bc_reference_episodes,
                    max_attempts=args.full_bc_reference_max_attempts,
                    seed=args.seed + 40_000 + task_index * 1_000,
                    segment_scheme=args.semantic_segment_scheme,
                    post_success_steps=args.semantic_post_success_steps,
                    num_task_ids=len(tasks),
                    task_id_index=task_index,
                )
                for segment, payload in per_segment_payloads.items():
                    if payload["observations"].shape[0] > 0:
                        llm_online_memory_store[segment].append(payload)
                for label, payload in per_task_specific_payloads.items():
                    if payload["observations"].shape[0] > 0:
                        llm_task_specific_memory_store.setdefault(label, []).append(payload)
                llm_memory_source_task_indices.add(task_index)
                collected_segment_states = {
                    segment: int(payload["observations"].shape[0])
                    for segment, payload in per_segment_payloads.items()
                }
                current_segments = current_llm_segments
                current_task_specific = select_task_specific_segments_for_task(
                    task_name=task_name,
                    manifest=task_specific_segment_manifest,
                )
                (
                    reference_observations,
                    target_means,
                    target_log_stds,
                    background_reference_states,
                    task_specific_reference_states,
                ) = build_hybrid_memory_from_store(
                    payload_store=llm_online_memory_store,
                    selected_segments=current_segments,
                    segment_weights=current_llm_weights,
                    background_ratio=args.background_segment_ratio,
                    task_specific_segments=(
                        ()
                        if current_task_specific is None
                        else current_task_specific.selected_segments
                    ),
                    task_specific_ratio=args.task_specific_segment_ratio,
                    task_specific_payload_store=llm_task_specific_memory_store,
                )
                reference_states = int(reference_observations.shape[0])
                if reference_observations.shape[0] > 0:
                    agent.set_reference_memory_with_targets(
                        observations=reference_observations,
                        target_means=target_means,
                        target_log_stds=target_log_stds,
                    )
                else:
                    agent.clear_reference_memory()
                segment_selection = SegmentSelection(
                    selected_segments=current_segments,
                    priority=current_llm_priority,
                    weights=tuple(current_llm_weights.items()),
                    reason="Final LLM-online selection used for accumulated memory.",
                    selection_source="llm_online",
                )
            else:
                if args.method not in STATIC_SEMANTIC_METHODS:
                    raise RuntimeError(f"Unsupported semantic method: {args.method}")
                if not isinstance(agent, FullBehaviorCloningSACAgent):
                    raise RuntimeError(f"{args.method} requires FullBehaviorCloningSACAgent.")
                per_segment_payloads, _, reference_success_episodes = collect_reference_payloads_by_segment(
                    agent=agent,
                    task_name=task_name,
                    env_version=args.env_version,
                    reward_function_version=args.reward_function_version,
                    append_task_id=append_task_id,
                    max_episode_steps=args.max_episode_steps,
                    episodes=args.full_bc_reference_episodes,
                    max_attempts=args.full_bc_reference_max_attempts,
                    seed=args.seed + 40_000 + task_index * 1_000,
                    segment_scheme=args.semantic_segment_scheme,
                    post_success_steps=args.semantic_post_success_steps,
                    num_task_ids=len(tasks),
                    task_id_index=task_index,
                )
                for segment, payload in per_segment_payloads.items():
                    if payload["observations"].shape[0] > 0:
                        static_semantic_memory_store[segment].append(payload)
                collected_segment_states = {
                    segment: int(payload["observations"].shape[0])
                    for segment, payload in per_segment_payloads.items()
                }
                reference_states = active_reference_states
                background_reference_states = active_background_reference_states
                task_specific_reference_states = active_task_specific_reference_states
            if method.complete_reference_memory:
                reference_states = int(reference_observations.shape[0])
                agent.add_reference_memory(observations=reference_observations)
        else:
            segment_selection = None
            task_specific_selection = None
        active_task_success_curves.append(task_curve_success)
        save_sac_checkpoint(
            agent=agent,
            path=run_dir / "checkpoints" / f"task_{task_index}.pt",
            environment_step=(task_index + 1) * args.steps_per_task,
            metadata={
                "task_name": task_name,
                "task_index": task_index,
                "method": args.method,
                "bc_gradient_strategy": getattr(
                    agent,
                    "bc_gradient_strategy",
                    None,
                ),
                "bc_combination_strategy": getattr(
                    agent,
                    "bc_combination_strategy",
                    None,
                ),
                "bc_cagrad_alpha": getattr(agent, "bc_cagrad_alpha", None),
            },
        )
        task_summary_rows.append(
            {
                "task_index": task_index,
                "task_name": task_name,
                "exploration_strategy": (
                    "best_return"
                    if selected_guide is not None
                    else actual_exploration_strategy
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
                "reference_success_episodes": reference_success_episodes,
                "reference_states": reference_states,
                "success_replay_seen_states": (
                    None
                    if success_replay is None
                    else success_replay.seen_successful_states
                ),
                "background_reference_states": background_reference_states,
                "task_specific_reference_states": task_specific_reference_states,
                "semantic_segment_scheme": (
                    args.semantic_segment_scheme
                    if args.method in {
                        "semantic_local_bc",
                        "adaptive_semantic_bc",
                        "general_task_specific_bc",
                        "semantic_hybrid_bc",
                        "stage_aware_semantic_bc",
                    }
                    else None
                ),
                "collected_segment_states": collected_segment_states,
                "selected_segments": (
                    None
                    if segment_selection is None
                    else json.dumps(list(segment_selection.selected_segments))
                ),
                "task_specific_segments": (
                    None
                    if task_specific_selection is None
                    else json.dumps(list(task_specific_selection.selected_segments))
                ),
                "segment_priority": (
                    None
                    if segment_selection is None
                    else json.dumps(list(segment_selection.priority))
                ),
                "segment_weights": (
                    None
                    if segment_selection is None
                    else json.dumps(dict(segment_selection.weights), sort_keys=True)
                ),
                "segment_selection_reason": (
                    None if segment_selection is None else segment_selection.reason
                ),
                "segment_selection_source": (
                    None if segment_selection is None else segment_selection.selection_source
                ),
                "success_replay_teacher": method.success_replay_teacher,
                "best_teacher_step": best_teacher_step,
                "best_teacher_success": (
                    None if best_teacher_step is None else best_teacher_score[0]
                ),
                "best_teacher_return": (
                    None if best_teacher_step is None else best_teacher_score[1]
                ),
                "bc_task_schedule": (
                    None if schedule_decision is None else schedule_decision.schedule
                ),
                "bc_task_schedule_source": (
                    None if schedule_decision is None else schedule_decision.source
                ),
                "bc_task_schedule_confidence": (
                    None if schedule_decision is None else schedule_decision.confidence
                ),
                "policy_prior_source_task": (
                    None if prior_decision is None else prior_decision.source_task_name
                ),
                "policy_prior_confidence": (
                    None if prior_decision is None else prior_decision.confidence
                ),
                "policy_prior_decision_source": (
                    None if prior_decision is None else prior_decision.source
                ),
                "policy_prior_initial_kl": (
                    None if prior_result is None else prior_result.initial_kl
                ),
                "policy_prior_final_kl": (
                    None if prior_result is None else prior_result.final_kl
                ),
                "bc_task_schedule_reason_codes": (
                    None
                    if schedule_decision is None
                    else json.dumps(list(schedule_decision.reason_codes))
                ),
                "bc_task_schedule_reason": (
                    None if schedule_decision is None else schedule_decision.reason
                ),
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
            reward_function_version=args.reward_function_version,
            num_task_ids=len(tasks),
        )
        all_eval_rows.extend(final_rows)

    flush_gradient_diagnostics()
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
    task_metric_labels = unique_task_metric_labels(tasks)
    summary_payload = {
        "method": args.method,
        "tasks": tasks,
        "average_performance": metrics["average_performance"],
        "average_return": average_return,
        "average_forgetting": metrics["average_forgetting"],
        "forward_transfer": metrics["forward_transfer"],
        "raw_forward_transfer": metrics["raw_forward_transfer"],
        "area_forward_transfer": metrics["area_forward_transfer"],
        "forward_transfer_available": metrics["forward_transfer_available"],
        "final_per_task_success": dict(zip(task_metric_labels, metrics["final_per_task"], strict=True)),
        "end_of_task_per_task": dict(zip(task_metric_labels, metrics["end_of_task_per_task"], strict=True)),
        "forgetting_per_task": dict(zip(task_metric_labels, metrics["forgetting_per_task"], strict=True)),
        "raw_forward_transfer_per_task": (
            None
            if metrics["raw_forward_transfer_per_task"][0] is None
            else dict(zip(task_metric_labels, metrics["raw_forward_transfer_per_task"], strict=True))
        ),
        "normalized_forward_transfer_per_task": (
            None
            if metrics["normalized_forward_transfer_per_task"][0] is None
            else dict(zip(task_metric_labels, metrics["normalized_forward_transfer_per_task"], strict=True))
        ),
        "area_forward_transfer_per_task": (
            None
            if metrics["area_forward_transfer_per_task"][0] is None
            else dict(zip(task_metric_labels, metrics["area_forward_transfer_per_task"], strict=True))
        ),
        "run_directory": str(run_dir),
        "elapsed_seconds": time.time() - started_at,
    }
    if hasattr(agent, "gradient_diagnostics_summary"):
        gradient_summary = agent.gradient_diagnostics_summary()
        write_json(
            run_dir / "gradient_diagnostics" / "summary.json",
            gradient_summary,
        )
        summary_payload["gradient_diagnostics"] = gradient_summary
    write_json(run_dir / "summary.json", summary_payload)
    return {
        "summary": summary_payload,
        "config": config,
        "evaluations": all_eval_rows,
        "task_summaries": task_summary_rows,
    }
