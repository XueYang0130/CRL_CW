from typing import Any

from agents import EWCSACAgent, SACAgent
from methods.base import MethodSpec


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return EWCSACAgent(
        **agent_kwargs,
        cl_reg_coef=args.cl_reg_coef,
        fisher_batches=args.fisher_batches,
    )


METHOD = MethodSpec(
    method_id="ewc",
    modes=frozenset({"continual"}),
    append_task_id=True,
    multi_head=True,
    hide_task_id=True,
    reference_exploration=True,
    defaults={
        "cl_reg_coef": 1_000.0,
        "fisher_batches": 10,
        "gradient_clip_norm": 0.1,
    },
    agent_factory=build_agent,
)
