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
    )


METHOD = MethodSpec(
    method_id="general_task_specific_bc",
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
        "background_segment_ratio": 0.0,
        "segment_selection_mode": "task_adaptive",
        "task_specific_segment_ratio": 0.2,
    },
    agent_factory=build_agent,
)
