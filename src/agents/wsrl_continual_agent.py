from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .sac_agent import SACAgent


class WSRLContinualAgent(SACAgent):
    """Multi-head SAC with configurable warm-start parameter transfer."""

    VALID_BACKBONE_SOURCES = frozenset({"current", "best_return"})
    VALID_HEAD_SOURCES = frozenset({"reset", "current", "best_return"})

    def __init__(
        self,
        *args: object,
        backbone_source: str = "current",
        head_source: str = "reset",
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if backbone_source not in self.VALID_BACKBONE_SOURCES:
            raise ValueError(f"Unsupported WSRL backbone source: {backbone_source}")
        if head_source not in self.VALID_HEAD_SOURCES:
            raise ValueError(f"Unsupported WSRL head source: {head_source}")
        if backbone_source == "current" and head_source == "best_return":
            raise ValueError("best_return head requires best_return backbone.")
        if backbone_source == "best_return" and head_source == "current":
            raise ValueError("current head cannot be paired with best_return backbone.")
        self.backbone_source = backbone_source
        self.head_source = head_source
        self._task_snapshots: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
        self._guide_actors: dict[int, nn.Module] = {}

    def requires_guide_selection(self, task_index: int) -> bool:
        return task_index > 0

    def on_task_end(self, *, task_index: int, replay_buffer: object, batch_size: int) -> None:
        del replay_buffer, batch_size
        self._task_snapshots[task_index] = {
            "actor": self._cpu_state(self.actor),
            "critic1": self._cpu_state(self.critic1),
            "critic2": self._cpu_state(self.critic2),
        }
        guide_actor = copy.deepcopy(self.actor).eval()
        guide_actor.requires_grad_(False)
        self._guide_actors[task_index] = guide_actor

    @torch.inference_mode()
    def select_guide_action(
        self,
        observation: np.ndarray,
        *,
        guide_task_index: int,
        deterministic: bool = False,
    ) -> np.ndarray:
        self._validate_task_index(guide_task_index)
        if guide_task_index not in self._guide_actors:
            raise RuntimeError(
                f"Missing frozen guide policy for task {guide_task_index}."
            )
        overridden_observation = self._observation_for_actor_head(
            observation=observation,
            head_index=guide_task_index,
        )
        guide_actor = self._guide_actors[guide_task_index]
        return guide_actor.act(
            overridden_observation,
            deterministic=deterministic,
            device=next(guide_actor.parameters()).device,
        )

    @torch.no_grad()
    def initialize_task_from_guide(
        self,
        *,
        task_index: int,
        guide_task_index: int,
    ) -> None:
        self._validate_task_index(task_index)
        self._validate_task_index(guide_task_index)
        if task_index == 0:
            return
        if guide_task_index not in self._task_snapshots:
            raise RuntimeError(f"Missing WSRL snapshot for task {guide_task_index}.")

        previous_task_index = task_index - 1
        if self.backbone_source == "best_return":
            snapshot = self._task_snapshots[guide_task_index]
            self.actor.backbone.load_state_dict(self._backbone_state(snapshot["actor"]))
            self.critic1.backbone.load_state_dict(self._backbone_state(snapshot["critic1"]))
            self.critic2.backbone.load_state_dict(self._backbone_state(snapshot["critic2"]))

        if self.head_source == "current":
            source_task_index = previous_task_index
            self._copy_all_heads(source_task_index, task_index)
        elif self.head_source == "best_return":
            self._load_all_heads_from_snapshot(guide_task_index, task_index)

        self.hard_update_targets()

    def _copy_all_heads(self, source: int, target: int) -> None:
        self._copy_actor_head(self.actor, source, target)
        self._copy_critic_head(self.critic1, source, target)
        self._copy_critic_head(self.critic2, source, target)

    def _load_all_heads_from_snapshot(self, source: int, target: int) -> None:
        snapshot = self._task_snapshots[source]
        self._load_actor_head(self.actor, snapshot["actor"], source, target)
        self._load_critic_head(self.critic1, snapshot["critic1"], source, target)
        self._load_critic_head(self.critic2, snapshot["critic2"], source, target)

    def _copy_actor_head(self, module: nn.Module, source: int, target: int) -> None:
        self._copy_linear_head(module.mean_head, source, target)
        self._copy_linear_head(module.log_std_head, source, target)

    def _copy_critic_head(self, module: nn.Module, source: int, target: int) -> None:
        self._copy_linear_head(module.q_head, source, target)

    def _load_actor_head(self, module: nn.Module, state: dict[str, torch.Tensor], source: int, target: int) -> None:
        self._load_linear_head(module.mean_head, state, "mean_head", source, target)
        self._load_linear_head(module.log_std_head, state, "log_std_head", source, target)

    def _load_critic_head(self, module: nn.Module, state: dict[str, torch.Tensor], source: int, target: int) -> None:
        self._load_linear_head(module.q_head, state, "q_head", source, target)

    def _copy_linear_head(self, layer: nn.Linear, source: int, target: int) -> None:
        layer.weight[target::self.num_tasks].copy_(layer.weight[source::self.num_tasks])
        layer.bias[target::self.num_tasks].copy_(layer.bias[source::self.num_tasks])

    def _load_linear_head(self, layer: nn.Linear, state: dict[str, torch.Tensor], prefix: str, source: int, target: int) -> None:
        layer.weight[target::self.num_tasks].copy_(state[f"{prefix}.weight"][source::self.num_tasks].to(self.device))
        layer.bias[target::self.num_tasks].copy_(state[f"{prefix}.bias"][source::self.num_tasks].to(self.device))

    @staticmethod
    def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

    @staticmethod
    def _backbone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key.removeprefix("backbone."): value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
