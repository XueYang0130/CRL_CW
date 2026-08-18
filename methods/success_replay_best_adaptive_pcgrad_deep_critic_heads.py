from __future__ import annotations

from typing import Any

from agents import DeepHeadCriticFullBCAgent, SACAgent
from methods.success_replay import make_success_replay_method


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return DeepHeadCriticFullBCAgent(
        **agent_kwargs,
        critic_head_hidden_size=args.critic_head_hidden_size,
        episodic_batch_size=args.episodic_batch_size,
        actor_cloning_coefficient=args.actor_cloning_coefficient,
        bc_gradient_strategy=args.bc_gradient_strategy,
        bc_max_norm_ratio=args.bc_max_norm_ratio,
        bc_combination_strategy=args.bc_combination_strategy,
        bc_adaptive_target_ratio=args.bc_adaptive_target_ratio,
        bc_adaptive_conflict_ratio=args.bc_adaptive_conflict_ratio,
        gradient_diagnostics=args.gradient_diagnostics,
        gradient_diagnostics_interval=args.gradient_diagnostics_interval,
        gradient_diagnostics_source_batch_size=args.gradient_diagnostics_source_batch_size,
        gradient_diagnostics_seed=args.seed + 700_000,
    )


METHOD = make_success_replay_method(
    "success_replay_best_adaptive_pcgrad_deep_critic_heads",
    teacher="best",
    gradient_strategy="pcgrad_sac_priority",
    defaults={
        "bc_combination_strategy": "adaptive_additive",
        "bc_adaptive_target_ratio": 0.2,
        "bc_adaptive_conflict_ratio": 0.05,
        "critic_head_hidden_size": 64,
    },
    agent_factory=build_agent,
)
