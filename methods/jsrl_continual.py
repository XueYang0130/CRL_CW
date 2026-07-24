from typing import Any

from agents import SACAgent, WSRLContinualAgent
from methods.base import MethodSpec


def build_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del args, total_tasks
    return WSRLContinualAgent(
        **agent_kwargs,
        backbone_source="current",
        head_source="reset",
    )


METHOD = MethodSpec(
    method_id="jsrl_continual",
    modes=frozenset({"continual"}),
    append_task_id=True,
    multi_head=True,
    hide_task_id=True,
    defaults={
        "reset_buffer_on_task_change": True,
        "reset_optimizer_on_task_change": True,
        "jsrl_initial_guide_steps": 180,
        "jsrl_curriculum_stages": 10,
        "jsrl_evaluation_interval": 20_000,
        "jsrl_moving_average_window": 5,
        "jsrl_stage_tolerance": 0.10,
        "jsrl_min_evaluations_per_stage": 1,
        "jsrl_max_evaluations_without_advance": 5,
    },
    agent_factory=build_agent,
    guide_mode="curriculum",
)
