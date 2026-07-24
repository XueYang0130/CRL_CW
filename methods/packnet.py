from typing import Any

from agents import PackNetSACAgent, SACAgent
from methods.base import MethodSpec


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    return PackNetSACAgent(
        **agent_kwargs,
        total_tasks=total_tasks,
        retrain_steps=args.packnet_retrain_steps,
    )


METHOD = MethodSpec(
    method_id="packnet",
    modes=frozenset({"continual"}),
    defaults={
        "packnet_retrain_steps": 100_000,
        "gradient_clip_norm": 2e-5,
    },
    agent_factory=build_agent,
)
