from __future__ import annotations

from dataclasses import dataclass

import torch

from agents.replay_buffer import ReplayBuffer
from agents.sac_agent import SACAgent


@dataclass(frozen=True)
class ManagedParameter:
    parameter: torch.nn.Parameter
    owner: torch.Tensor
    saved: torch.Tensor


class PackNetSACAgent(SACAgent):
    """Single-head SAC with PackNet-style actor pruning."""

    def __init__(
        self,
        *args,
        total_tasks: int,
        retrain_steps: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.num_tasks != 1:
            raise ValueError("PackNetSACAgent currently supports single-head SAC only.")
        if total_tasks <= 0:
            raise ValueError("total_tasks must be positive.")
        if retrain_steps < 0:
            raise ValueError("retrain_steps must be non-negative.")

        self.total_tasks = int(total_tasks)
        self.retrain_steps = int(retrain_steps)
        self._managed_actor_parameters = self._build_managed_actor_parameters()
        self._freeze_non_kernel_actor_parameters = False

    def adjust_actor_gradients(
        self,
        *,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[torch.nn.Parameter, ...],
        task_index: int,
    ) -> tuple[torch.Tensor, ...]:
        adjusted = []
        for gradient, parameter in zip(gradients, parameters, strict=True):
            managed = self._find_managed_parameter(parameter)
            if managed is None:
                if self._freeze_non_kernel_actor_parameters and parameter.ndim < 2:
                    adjusted.append(torch.zeros_like(gradient))
                else:
                    adjusted.append(gradient)
                continue

            mask = (managed.owner == task_index).to(
                device=gradient.device,
                dtype=gradient.dtype,
            )
            adjusted.append(gradient * mask)
        return tuple(adjusted)

    def on_task_end(
        self,
        *,
        task_index: int,
        replay_buffer: ReplayBuffer,
        batch_size: int,
    ) -> None:
        if task_index >= self.total_tasks - 1:
            return

        if task_index == 0:
            self._freeze_non_kernel_actor_parameters = True

        num_tasks_left = self.total_tasks - task_index - 1
        prune_fraction = num_tasks_left / (num_tasks_left + 1)
        self._prune_actor(task_index=task_index, prune_fraction=prune_fraction)

        self.rebuild_optimizer()
        for _ in range(self.retrain_steps):
            batch = replay_buffer.sample(batch_size, self.device)
            self.update_batch(
                batch.observations,
                batch.actions,
                batch.rewards,
                batch.next_observations,
                batch.terminated,
                collect_metrics=False,
            )
        self.rebuild_optimizer()

    @torch.no_grad()
    def on_evaluation_start(self, task_index: int) -> None:
        for managed in self._managed_actor_parameters:
            managed.saved.copy_(managed.parameter.data)
            visible_mask = (managed.owner <= task_index).to(
                device=managed.parameter.device,
                dtype=managed.parameter.dtype,
            )
            managed.parameter.data.mul_(visible_mask)

    @torch.no_grad()
    def on_evaluation_end(self, task_index: int) -> None:
        del task_index
        for managed in self._managed_actor_parameters:
            managed.parameter.data.copy_(managed.saved)

    def _build_managed_actor_parameters(self) -> tuple[ManagedParameter, ...]:
        managed = []
        for parameter in self.actor.parameters():
            if parameter.ndim < 2:
                continue
            owner = torch.zeros_like(parameter, dtype=torch.int64, device=parameter.device)
            saved = torch.zeros_like(parameter.data)
            managed.append(
                ManagedParameter(
                    parameter=parameter,
                    owner=owner,
                    saved=saved,
                )
            )
        return tuple(managed)

    def _find_managed_parameter(
        self,
        parameter: torch.nn.Parameter,
    ) -> ManagedParameter | None:
        for managed in self._managed_actor_parameters:
            if managed.parameter is parameter:
                return managed
        return None

    @torch.no_grad()
    def _prune_actor(
        self,
        *,
        task_index: int,
        prune_fraction: float,
    ) -> None:
        for managed in self._managed_actor_parameters:
            task_mask = managed.owner == task_index
            if not torch.any(task_mask):
                continue

            values = managed.parameter.data[task_mask].abs()
            if values.numel() == 0:
                continue

            threshold_index = min(
                int(values.numel() * prune_fraction),
                values.numel() - 1,
            )
            threshold = torch.sort(values).values[threshold_index]
            keep_mask = (managed.parameter.data.abs() > threshold) | (~task_mask)
            managed.parameter.data.mul_(keep_mask.to(managed.parameter.dtype))
            managed.owner.copy_(
                torch.where(
                    keep_mask,
                    managed.owner,
                    torch.full_like(managed.owner, task_index + 1),
                )
            )
