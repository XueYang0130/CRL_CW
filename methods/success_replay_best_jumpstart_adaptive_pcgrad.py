from __future__ import annotations

from typing import Any

from agents import DemonstrationGuidedFullBCAgent, SACAgent
from methods.success_replay import make_success_replay_method


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return DemonstrationGuidedFullBCAgent(
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
        gradient_diagnostics_source_batch_size=args.gradient_diagnostics_source_batch_size,
        gradient_diagnostics_seed=args.seed + 700_000,
    )


METHOD = make_success_replay_method(
    "success_replay_best_jumpstart_adaptive_pcgrad",
    teacher="best",
    gradient_strategy="pcgrad_sac_priority",
    defaults={
        "bc_combination_strategy": "adaptive_additive",
        "bc_adaptive_target_ratio": 0.2,
        "bc_adaptive_conflict_ratio": 0.05,
        "jsrl_initial_guide_steps": 100,
        "jsrl_curriculum_stages": 5,
        "jsrl_evaluation_interval": 20_000,
        "jsrl_moving_average_window": 3,
        "jsrl_stage_tolerance": 0.10,
        "jsrl_min_evaluations_per_stage": 1,
        "jsrl_max_evaluations_without_advance": 3,
        "dg_min_guide_improvement": 0.0,
    },
    guide_mode="demonstration_curriculum",
    agent_factory=build_agent,
)
