from __future__ import annotations

from typing import Any

from agents import ConflictLoRAFullBCAgent, SACAgent
from methods.success_replay import make_success_replay_method


def build_conflict_lora_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return ConflictLoRAFullBCAgent(
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
        conflict_lora_rank=args.conflict_lora_rank,
        conflict_lora_coefficient=args.conflict_lora_coefficient,
        conflict_lora_max_norm_ratio=args.conflict_lora_max_norm_ratio,
    )


METHOD = make_success_replay_method(
    "success_replay_best_conflict_lora",
    teacher="best",
    gradient_strategy="pcgrad_sac_priority",
    defaults={
        "bc_combination_strategy": "adaptive_additive",
        "bc_adaptive_target_ratio": 0.2,
        "bc_adaptive_conflict_ratio": 0.05,
        "conflict_lora_rank": 4,
        "conflict_lora_coefficient": 1.0,
        "conflict_lora_max_norm_ratio": 0.25,
    },
    agent_factory=build_conflict_lora_agent,
)
