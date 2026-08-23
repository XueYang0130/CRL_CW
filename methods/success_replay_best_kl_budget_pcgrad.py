from __future__ import annotations

from typing import Any

from agents import KLBudgetFullBehaviorCloningSACAgent, SACAgent
from methods.success_replay import make_success_replay_method


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return KLBudgetFullBehaviorCloningSACAgent(
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
        kl_budget_low=args.kl_budget_low,
        kl_budget_high=args.kl_budget_high,
        kl_budget_ema_beta=args.kl_budget_ema_beta,
    )


METHOD = make_success_replay_method(
    "success_replay_best_kl_budget_pcgrad",
    teacher="best",
    gradient_strategy="pcgrad_sac_priority",
    defaults={
        "bc_combination_strategy": "adaptive_additive",
        "bc_adaptive_target_ratio": 0.2,
        "bc_adaptive_conflict_ratio": 0.05,
        "kl_budget_low": 0.05,
        "kl_budget_high": 0.5,
        "kl_budget_ema_beta": 0.99,
    },
    agent_factory=build_agent,
)
