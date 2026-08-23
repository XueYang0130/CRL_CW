"""Mixed80 Adaptive PCGrad with discovery-focused critic/replay updates."""

from __future__ import annotations

from typing import Any

from agents import (
    OptimisticEnsembleFullBCAgent,
    PrioritizedNStepReplayBuffer,
    SACAgent,
)
from methods.success_replay import make_success_replay_method


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return OptimisticEnsembleFullBCAgent(
        **agent_kwargs,
        episodic_batch_size=args.episodic_batch_size,
        actor_cloning_coefficient=args.actor_cloning_coefficient,
        bc_gradient_strategy=args.bc_gradient_strategy,
        bc_max_norm_ratio=args.bc_max_norm_ratio,
        bc_combination_strategy=args.bc_combination_strategy,
        bc_adaptive_target_ratio=args.bc_adaptive_target_ratio,
        bc_adaptive_conflict_ratio=args.bc_adaptive_conflict_ratio,
        bc_cagrad_alpha=args.bc_cagrad_alpha,
        gradient_diagnostics=args.gradient_diagnostics,
        gradient_diagnostics_interval=args.gradient_diagnostics_interval,
        gradient_diagnostics_source_batch_size=(
            args.gradient_diagnostics_source_batch_size
        ),
        gradient_diagnostics_seed=args.seed + 700_000,
        critic_ensemble_size=args.critic_ensemble_size,
        critic_bootstrap_probability=args.critic_bootstrap_probability,
        optimistic_ucb_beta=args.optimistic_ucb_beta,
        optimistic_action_candidates=args.optimistic_action_candidates,
        ensemble_seed=args.seed + 710_000,
    )


def build_replay_buffer(
    observation_dim: int,
    action_dim: int,
    args: Any,
) -> PrioritizedNStepReplayBuffer:
    beta_steps = (
        args.steps_per_task
        if args.priority_beta_steps == 0
        else args.priority_beta_steps
    )
    return PrioritizedNStepReplayBuffer(
        observation_dim=observation_dim,
        action_dim=action_dim,
        capacity=args.replay_size,
        seed=args.seed,
        gamma=args.gamma,
        n_step=args.n_step_return,
        prioritized_fraction=args.prioritized_replay_fraction,
        priority_alpha=args.priority_alpha,
        priority_beta_start=args.priority_beta_start,
        priority_beta_end=args.priority_beta_end,
        priority_beta_steps=beta_steps,
        priority_epsilon=args.priority_epsilon,
    )


METHOD = make_success_replay_method(
    "success_replay_mixed80_optimistic_nstep_pcgrad",
    teacher="best",
    gradient_strategy="pcgrad_sac_priority",
    broader_replay_selector="random",
    broader_replay_ratio=0.2,
    dynamic_broader_replay_fill=True,
    defaults={
        "bc_combination_strategy": "adaptive_additive",
        "bc_adaptive_target_ratio": 0.2,
        "bc_adaptive_conflict_ratio": 0.05,
        "critic_ensemble_size": 4,
        "critic_bootstrap_probability": 0.8,
        "optimistic_ucb_beta": 0.5,
        "optimistic_action_candidates": 8,
        "n_step_return": 3,
        "prioritized_replay_fraction": 0.2,
        "priority_alpha": 0.6,
        "priority_beta_start": 0.4,
        "priority_beta_end": 1.0,
        "priority_beta_steps": 0,
        "priority_epsilon": 1e-6,
    },
    agent_factory=build_agent,
    replay_factory=build_replay_buffer,
)
