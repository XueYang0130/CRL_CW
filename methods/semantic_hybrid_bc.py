from typing import Any

from agents import FullBehaviorCloningSACAgent, SACAgent
from methods.base import MethodSpec


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return FullBehaviorCloningSACAgent(
        **agent_kwargs,
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


METHOD = MethodSpec(
    method_id="semantic_hybrid_bc",
    modes=frozenset({"continual"}),
    append_task_id=True,
    multi_head=True,
    hide_task_id=True,
    reference_exploration=True,
    defaults={
        "exploration_strategy": "best_return",
        "reset_buffer_on_task_change": True,
        "reset_optimizer_on_task_change": True,
        "episodic_batch_size": 128,
        "actor_cloning_coefficient": 100.0,
        "gradient_clip_norm": 0.1,
        "full_bc_reference_episodes": 20,
        "full_bc_reference_max_attempts": 80,
        "semantic_segments": ["contact_or_alignment", "manipulation"],
        "semantic_segment_scheme": "task_aware_v3",
        "background_segment_ratio": 0.2,
        "segment_selection_mode": "task_adaptive",
        "task_specific_segment_ratio": 0.2,
    },
    agent_factory=build_agent,
)
