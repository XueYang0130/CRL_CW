from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

from agents import SACAgent


AgentFactory = Callable[[dict[str, Any], Any, int], SACAgent]


def build_sac_agent(
    agent_kwargs: dict[str, Any],
    args: Any,
    total_tasks: int,
) -> SACAgent:
    del args, total_tasks
    return SACAgent(**agent_kwargs)


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    modes: frozenset[str]
    append_task_id: bool = False
    multi_head: bool = False
    hide_task_id: bool = False
    transfer_alpha: bool = False
    defaults: dict[str, object] = field(default_factory=dict)
    agent_factory: AgentFactory = build_sac_agent
    reference_exploration: bool = False
    guide_mode: str = "none"

    def build_agent(
        self,
        *,
        args: Any,
        observation_dim: int,
        action_dim: int,
        action_low: Any,
        action_high: Any,
        total_tasks: int,
    ) -> SACAgent:
        num_tasks = total_tasks if self.multi_head else 1
        task_id_dim = total_tasks if self.append_task_id else 0
        kwargs = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "action_low": action_low,
            "action_high": action_high,
            "num_tasks": num_tasks,
            "task_id_dim": task_id_dim,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "polyak": args.polyak,
            "target_entropy": action_dim
            * math.log(
                args.target_output_std * math.sqrt(2.0 * math.pi * math.e)
            ),
            "initial_log_alpha": args.initial_log_alpha,
            "device": args.device,
            "hide_task_id": self.hide_task_id,
            "gradient_clip_norm": args.gradient_clip_norm,
        }
        return self.agent_factory(kwargs, args, total_tasks)

    def exploration_config(
        self,
        *,
        task_index: int,
        strategy: str,
    ) -> tuple[str | None, int | None]:
        if not self.reference_exploration or task_index == 0 or strategy == "random":
            return None, None
        return strategy, task_index + 1
