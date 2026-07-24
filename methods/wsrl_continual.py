from typing import Any

from agents import SACAgent, WSRLContinualAgent
from methods.base import MethodSpec


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del total_tasks
    return WSRLContinualAgent(
        **agent_kwargs,
        backbone_source=args.wsrl_backbone_source,
        head_source=args.wsrl_head_source,
    )


METHOD = MethodSpec(
    method_id="wsrl_continual",
    modes=frozenset({"continual"}),
    append_task_id=True,
    multi_head=True,
    hide_task_id=True,
    defaults={
        "reset_buffer_on_task_change": True,
        "reset_optimizer_on_task_change": True,
        "wsrl_backbone_source": "current",
        "wsrl_head_source": "reset",
        "wsrl_warmup_steps": 10_000,
    },
    agent_factory=build_agent,
    guide_mode="fixed_warmup",
)
