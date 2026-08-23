from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

from agents import ReplayBuffer, SACAgent


AgentFactory = Callable[[dict[str, Any], Any, int], SACAgent]
ReplayFactory = Callable[[int, int, Any], object]


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
    complete_reference_memory: bool = False
    success_replay_teacher: str | None = None
    broader_replay_selector: str | None = None
    broader_replay_ratio: float = 0.0
    dynamic_broader_replay_fill: bool = False
    guide_mode: str = "none"
    llm_prior_initialization: bool = False
    policy_from_start_after_guide: bool = False
    replay_factory: ReplayFactory | None = None

    def __post_init__(self) -> None:
        if self.success_replay_teacher not in {None, "final", "best"}:
            raise ValueError(
                "success_replay_teacher must be None, 'final', or 'best'."
            )
        if self.broader_replay_selector not in {None, "random", "llm"}:
            raise ValueError(
                "broader_replay_selector must be None, 'random', or 'llm'."
            )
        if not 0.0 <= self.broader_replay_ratio < 1.0:
            raise ValueError("broader_replay_ratio must be in [0, 1).")
        if self.broader_replay_selector is None and self.broader_replay_ratio != 0.0:
            raise ValueError("broader_replay_ratio requires a broader replay selector.")
        if self.broader_replay_selector is not None and self.success_replay_teacher is None:
            raise ValueError("Broader replay requires a success-replay teacher.")
        if self.dynamic_broader_replay_fill and self.broader_replay_selector is None:
            raise ValueError(
                "dynamic_broader_replay_fill requires a broader replay selector."
            )

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

    def build_replay_buffer(
        self,
        *,
        args: Any,
        observation_dim: int,
        action_dim: int,
    ) -> object:
        """Build the online replay used by this method.

        Existing methods retain the exact standard ReplayBuffer constructor.
        A method must explicitly provide ``replay_factory`` to opt into a
        different sampling/update contract.
        """
        if self.replay_factory is None:
            return ReplayBuffer(
                observation_dim=observation_dim,
                action_dim=action_dim,
                capacity=args.replay_size,
                seed=args.seed,
            )
        return self.replay_factory(observation_dim, action_dim, args)
