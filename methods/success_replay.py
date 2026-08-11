from __future__ import annotations

from typing import Any, Callable

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
        bc_cagrad_alpha=args.bc_cagrad_alpha,
        gradient_diagnostics=args.gradient_diagnostics,
        gradient_diagnostics_interval=args.gradient_diagnostics_interval,
        gradient_diagnostics_source_batch_size=args.gradient_diagnostics_source_batch_size,
        gradient_diagnostics_seed=args.seed + 700_000,
    )


def make_success_replay_method(
    method_id: str,
    *,
    teacher: str,
    gradient_strategy: str = "standard",
    defaults: dict[str, object] | None = None,
    broader_replay_selector: str | None = None,
    broader_replay_ratio: float = 0.0,
    agent_factory: Callable[[dict[str, Any], Any, int], SACAgent] = build_agent,
) -> MethodSpec:
    method_defaults = {
        "exploration_strategy": "best_return",
        "reset_buffer_on_task_change": True,
        "reset_optimizer_on_task_change": True,
        "episodic_memory_per_task": 10_000,
        "episodic_batch_size": 128,
        "actor_cloning_coefficient": 100.0,
        "bc_gradient_strategy": gradient_strategy,
        "bc_max_norm_ratio": 1.0,
        "bc_combination_strategy": "average",
        "bc_adaptive_target_ratio": 0.2,
        "bc_adaptive_conflict_ratio": 0.05,
        "bc_cagrad_alpha": 0.5,
        "gradient_clip_norm": 0.1,
        "gradient_diagnostics": True,
    }
    if defaults is not None:
        method_defaults.update(defaults)
    return MethodSpec(
        method_id=method_id,
        modes=frozenset({"continual"}),
        append_task_id=True,
        multi_head=True,
        hide_task_id=True,
        reference_exploration=True,
        success_replay_teacher=teacher,
        broader_replay_selector=broader_replay_selector,
        broader_replay_ratio=broader_replay_ratio,
        defaults=method_defaults,
        agent_factory=agent_factory,
    )
