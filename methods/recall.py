from typing import Any

from agents import RECALLAgent, SACAgent
from methods.base import MethodSpec


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return RECALLAgent(
        **agent_kwargs,
        episodic_memory_per_task=args.episodic_memory_per_task,
        episodic_batch_size=args.episodic_batch_size,
        policy_reg_coef=args.actor_cloning_coefficient,
        value_reg_coef=args.recall_value_reg_coef,
        regularize_critic=args.recall_regularize_critic,
        recall_seed=args.seed,
        gradient_diagnostics=args.gradient_diagnostics,
        gradient_diagnostics_interval=args.gradient_diagnostics_interval,
        gradient_diagnostics_source_batch_size=args.gradient_diagnostics_source_batch_size,
        gradient_diagnostics_seed=args.seed + 700_000,
    )


METHOD = MethodSpec(
    method_id="recall",
    modes=frozenset({"continual"}),
    append_task_id=True,
    multi_head=True,
    hide_task_id=True,
    reference_exploration=False,
    policy_from_start_after_guide=True,
    defaults={
        "exploration_strategy": "best_return",
        "best_return_eval_episodes": 10,
        "reset_buffer_on_task_change": True,
        "reset_optimizer_on_task_change": True,
        "episodic_memory_per_task": 10_000,
        "episodic_batch_size": 128,
        "actor_cloning_coefficient": 10.0,
        "gradient_clip_norm": 0.1,
        "recall_value_reg_coef": 1.0,
        "recall_regularize_critic": False,
        "gradient_diagnostics": True,
    },
    agent_factory=build_agent,
)
