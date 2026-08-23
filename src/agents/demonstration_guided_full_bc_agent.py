from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from .full_bc_agent import FullBehaviorCloningSACAgent


class DemonstrationGuidedFullBCAgent(FullBehaviorCloningSACAgent):
    """Adaptive-PCGrad agent with frozen policies used only as rollout guides."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._task_guide_actors: dict[int, nn.Module] = {}
        self._active_task_index: int | None = None
        self._selected_old_guide_index: int | None = None
        self._self_evolving_guide: nn.Module | None = None
        self._self_evolving_score = (float("-inf"), float("-inf"))

    def requires_guide_selection(self, task_index: int) -> bool:
        return task_index > 0

    def on_task_start(self, *, task_index: int, replay_buffer: object) -> None:
        super().on_task_start(task_index=task_index, replay_buffer=replay_buffer)
        self._active_task_index = task_index
        self._selected_old_guide_index = None
        self._self_evolving_guide = None
        self._self_evolving_score = (float("-inf"), float("-inf"))

    def on_task_end(
        self,
        *,
        task_index: int,
        replay_buffer: object,
        batch_size: int,
    ) -> None:
        super().on_task_end(
            task_index=task_index,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
        )
        self._task_guide_actors[task_index] = self._frozen_actor_copy()

    def initialize_task_from_guide(
        self,
        *,
        task_index: int,
        guide_task_index: int,
    ) -> None:
        self._validate_task_index(task_index)
        self._validate_task_index(guide_task_index)
        if guide_task_index >= task_index:
            raise ValueError("A rollout guide must come from an earlier task.")
        if guide_task_index not in self._task_guide_actors:
            raise RuntimeError(
                f"Missing frozen guide actor for task {guide_task_index}."
            )
        self._active_task_index = task_index
        self._selected_old_guide_index = guide_task_index

    @torch.inference_mode()
    def select_guide_action(
        self,
        observation: np.ndarray,
        *,
        guide_task_index: int,
        deterministic: bool = False,
    ) -> np.ndarray:
        if self._self_evolving_guide is not None:
            actor = self._self_evolving_guide
            guide_observation = np.asarray(observation, dtype=np.float32)
        else:
            if guide_task_index not in self._task_guide_actors:
                raise RuntimeError(
                    f"Missing frozen guide actor for task {guide_task_index}."
                )
            actor = self._task_guide_actors[guide_task_index]
            guide_observation = self._observation_for_actor_head(
                observation=observation,
                head_index=guide_task_index,
            )
        return actor.act(
            guide_observation,
            deterministic=deterministic,
            device=next(actor.parameters()).device,
        )

    def promote_current_policy_to_guide(
        self,
        *,
        task_index: int,
        success_rate: float,
        mean_return: float,
    ) -> bool:
        """Freeze a better current-task policy for future guide prefixes."""
        if self._active_task_index != task_index:
            raise RuntimeError("Cannot promote a guide for an inactive task.")
        if self._self_evolving_guide is not None:
            return False
        score = (float(success_rate), float(mean_return))
        self._self_evolving_guide = self._frozen_actor_copy()
        self._self_evolving_score = score
        return True

    def set_task_guide_actor_state(
        self,
        *,
        task_index: int,
        actor_state: dict[str, torch.Tensor],
    ) -> None:
        """Commit one externally selected best actor as a future-task guide."""
        self._validate_task_index(task_index)
        guide_actor = copy.deepcopy(self.actor)
        guide_actor.load_state_dict(actor_state, strict=True)
        guide_actor.eval()
        guide_actor.requires_grad_(False)
        self._task_guide_actors[task_index] = guide_actor

    @property
    def active_guide_source(self) -> str:
        if self._self_evolving_guide is not None:
            return "current_task_snapshot"
        if self._selected_old_guide_index is not None:
            return "old_task_snapshot"
        return "none"

    def _frozen_actor_copy(self) -> nn.Module:
        actor = copy.deepcopy(self.actor).eval()
        actor.requires_grad_(False)
        return actor
