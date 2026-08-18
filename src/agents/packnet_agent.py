from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from agents.replay_buffer import ReplayBuffer
from agents.sac_agent import SACAgent


@dataclass(frozen=True)
class ManagedParameter:
    name: str
    parameter: torch.nn.Parameter
    owner: torch.Tensor


class PackNetSACAgent(SACAgent):
    """Multi-head SAC with PackNet isolation on the actor backbone."""

    def __init__(
        self,
        *args,
        total_tasks: int,
        retrain_steps: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if total_tasks <= 0:
            raise ValueError("total_tasks must be positive.")
        if total_tasks != self.num_tasks:
            raise ValueError("PackNet requires one actor head per task.")
        if retrain_steps < 0:
            raise ValueError("retrain_steps must be non-negative.")

        self.total_tasks = int(total_tasks)
        self.retrain_steps = int(retrain_steps)
        self._managed_actor_parameters = self._build_managed_actor_parameters()
        self._managed_parameter_ids = {
            id(managed.parameter) for managed in self._managed_actor_parameters
        }
        self._backbone_parameter_ids = {
            id(parameter) for parameter in self.actor.backbone.parameters()
        }
        self._freeze_unmanaged_backbone = False
        self._retraining_task_index: int | None = None
        self._active_task_index = 0
        self._evaluation_snapshot: dict[int, torch.Tensor] | None = None

    def on_task_start(
        self,
        *,
        task_index: int,
        replay_buffer: object,
    ) -> None:
        del replay_buffer
        self._active_task_index = int(task_index)

    def adjust_actor_gradients(
        self,
        *,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[torch.nn.Parameter, ...],
        task_index: int,
    ) -> tuple[torch.Tensor, ...]:
        adjusted: list[torch.Tensor] = []
        for gradient, parameter in zip(gradients, parameters, strict=True):
            managed = self._find_managed_parameter(parameter)
            if managed is not None:
                if self._retraining_task_index is None:
                    trainable = (managed.owner == -1) | (
                        managed.owner == task_index
                    )
                else:
                    trainable = managed.owner == self._retraining_task_index
                adjusted.append(
                    gradient
                    * trainable.to(
                        device=gradient.device,
                        dtype=gradient.dtype,
                    )
                )
                continue

            if (
                self._freeze_unmanaged_backbone
                and id(parameter) in self._backbone_parameter_ids
            ):
                adjusted.append(torch.zeros_like(gradient))
            else:
                adjusted.append(gradient)
        return tuple(adjusted)

    def on_task_end(
        self,
        *,
        task_index: int,
        replay_buffer: ReplayBuffer,
        batch_size: int,
    ) -> None:
        if task_index == 0:
            # Continual World's PackNet freezes actor-core biases and
            # normalization parameters after the first task.
            self._freeze_unmanaged_backbone = True

        if task_index >= self.total_tasks - 1:
            self._assign_all_free_weights(task_index)
            self.rebuild_optimizer()
            return

        tasks_left = self.total_tasks - task_index - 1
        allocation_fraction = 1.0 / (tasks_left + 1)
        self._allocate_actor_weights(
            task_index=task_index,
            allocation_fraction=allocation_fraction,
        )

        # Reset Adam so masked, previously active parameters cannot drift from
        # stale momentum. During optional retraining only the retained current
        # task subnetwork is trainable; newly freed weights stay at zero.
        self.rebuild_optimizer()
        self._retraining_task_index = task_index
        try:
            for _ in range(self.retrain_steps):
                sample_size = min(batch_size, len(replay_buffer))
                batch = replay_buffer.sample(sample_size, self.device)
                self.update_batch(
                    batch.observations,
                    batch.actions,
                    batch.rewards,
                    batch.next_observations,
                    batch.terminated,
                    collect_metrics=False,
                )
        finally:
            self._retraining_task_index = None
            self.rebuild_optimizer()

    @torch.no_grad()
    def on_evaluation_start(self, task_index: int) -> None:
        if self._evaluation_snapshot is not None:
            raise RuntimeError("PackNet evaluation hooks cannot be nested.")
        self._evaluation_snapshot = {
            id(managed.parameter): managed.parameter.data.clone()
            for managed in self._managed_actor_parameters
        }
        for managed in self._managed_actor_parameters:
            visible = (managed.owner >= 0) & (managed.owner <= task_index)
            if task_index == self._active_task_index:
                visible = visible | (managed.owner == -1)
            managed.parameter.data.mul_(
                visible.to(
                    device=managed.parameter.device,
                    dtype=managed.parameter.dtype,
                )
            )

    @torch.no_grad()
    def on_evaluation_end(self, task_index: int) -> None:
        del task_index
        if self._evaluation_snapshot is None:
            raise RuntimeError("PackNet evaluation ended without a snapshot.")
        for managed in self._managed_actor_parameters:
            managed.parameter.data.copy_(
                self._evaluation_snapshot[id(managed.parameter)]
            )
        self._evaluation_snapshot = None

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "owners": {
                managed.name: managed.owner.detach().cpu()
                for managed in self._managed_actor_parameters
            },
            "freeze_unmanaged_backbone": self._freeze_unmanaged_backbone,
            "active_task_index": self._active_task_index,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("PackNet extra state must be a dictionary.")
        owners = state.get("owners", {})
        for managed in self._managed_actor_parameters:
            owner = owners.get(managed.name)
            if owner is None or owner.shape != managed.owner.shape:
                raise ValueError(
                    f"Missing or invalid PackNet owner mask for {managed.name}."
                )
            managed.owner.copy_(owner.to(managed.owner.device))
        self._freeze_unmanaged_backbone = bool(
            state.get("freeze_unmanaged_backbone", False)
        )
        self._active_task_index = int(state.get("active_task_index", 0))

    def _build_managed_actor_parameters(self) -> tuple[ManagedParameter, ...]:
        managed: list[ManagedParameter] = []
        for name, parameter in self.actor.backbone.named_parameters():
            if parameter.ndim < 2:
                continue
            managed.append(
                ManagedParameter(
                    name=name,
                    parameter=parameter,
                    owner=torch.full_like(
                        parameter,
                        fill_value=-1,
                        dtype=torch.int64,
                        device=parameter.device,
                    ),
                )
            )
        return tuple(managed)

    def _find_managed_parameter(
        self,
        parameter: torch.nn.Parameter,
    ) -> ManagedParameter | None:
        if id(parameter) not in self._managed_parameter_ids:
            return None
        for managed in self._managed_actor_parameters:
            if managed.parameter is parameter:
                return managed
        raise RuntimeError("Managed PackNet parameter lookup is inconsistent.")

    @torch.no_grad()
    def _allocate_actor_weights(
        self,
        *,
        task_index: int,
        allocation_fraction: float,
    ) -> None:
        for managed in self._managed_actor_parameters:
            free_mask = managed.owner == -1
            free_values = managed.parameter.data[free_mask].abs()
            if free_values.numel() == 0:
                continue

            num_allocate = max(
                1,
                int(round(allocation_fraction * free_values.numel())),
            )
            num_allocate = min(num_allocate, free_values.numel())
            top_indices = torch.topk(
                free_values,
                k=num_allocate,
                largest=True,
                sorted=False,
            ).indices
            free_flat_indices = free_mask.flatten().nonzero(as_tuple=False).flatten()
            assigned_flat_indices = free_flat_indices[top_indices]
            managed.owner.flatten()[assigned_flat_indices] = task_index

            # Parameters not retained for this task are the capacity released
            # for future tasks and must start from zero, as in PackNet pruning.
            remaining_free = managed.owner == -1
            managed.parameter.data[remaining_free] = 0.0

    @torch.no_grad()
    def _assign_all_free_weights(self, task_index: int) -> None:
        for managed in self._managed_actor_parameters:
            managed.owner[managed.owner == -1] = task_index
