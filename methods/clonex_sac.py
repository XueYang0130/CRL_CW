from typing import Any

from agents import ClonExSACAgent, SACAgent
from methods.base import MethodSpec


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return ClonExSACAgent(
        **agent_kwargs,
        episodic_memory_per_task=args.episodic_memory_per_task,
        episodic_batch_size=args.episodic_batch_size,
        actor_cloning_coefficient=args.actor_cloning_coefficient,
    )


METHOD = MethodSpec(
    method_id="clonex_sac",
    modes=frozenset({"continual"}),
    append_task_id=True,
    multi_head=True,
    hide_task_id=True,
    reference_exploration=True,
    defaults={
        "exploration_strategy": "best_return",
        "reset_buffer_on_task_change": True,
        "reset_optimizer_on_task_change": True,
        "episodic_memory_per_task": 10_000,
        "episodic_batch_size": 128,
        "actor_cloning_coefficient": 100.0,
        "gradient_clip_norm": 0.1,
    },
    agent_factory=build_agent,
)
