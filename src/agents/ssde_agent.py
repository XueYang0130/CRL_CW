from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from agents.replay_buffer import ReplayBuffer
from agents.sac_agent import SACAgent


@dataclass(frozen=True)
class ManagedBackboneWeight:
    name: str
    parameter: torch.nn.Parameter
    owner: torch.Tensor


class _ActivationTracker:
    """Track mean absolute pre-activation magnitude per hidden neuron."""

    def __init__(self) -> None:
        self._sum: dict[str, torch.Tensor] = {}
        self._count: dict[str, int] = {}
        self.enabled = True

    def hook(self, name: str):
        def _fn(module, inputs, output):
            del module, inputs
            # Evaluation and action-selection paths may execute under
            # torch.inference_mode(). Tensors created there cannot later be
            # mutated during training, and those activations should not enter
            # dormancy statistics in the first place.
            if not self.enabled or torch.is_inference_mode_enabled():
                return
            activations = output.detach().abs().mean(dim=0).cpu()
            if name not in self._sum:
                self._sum[name] = torch.zeros_like(activations)
                self._count[name] = 0
            self._sum[name] += activations
            self._count[name] += 1

        return _fn

    def sensitivity(self, name: str) -> torch.Tensor | None:
        if name not in self._sum or self._count[name] == 0:
            return None
        return self._sum[name] / self._count[name]

    def reset(self) -> None:
        self._sum.clear()
        self._count.clear()


class SSDESACAgent(SACAgent):
    """Structured-sparsity and dormant-exploration SAC approximation.

    This implementation applies both mechanisms to the shared actor backbone:
    high-importance weights are assigned to completed tasks and frozen, while
    dormant entries that remain unassigned may be reinitialized.
    """

    def __init__(
        self,
        *args,
        dormancy_threshold: float = 0.01,
        dormancy_check_every: int = 10_000,
        retrain_steps: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not math.isfinite(dormancy_threshold) or dormancy_threshold < 0.0:
            raise ValueError(
                "dormancy_threshold must be finite and non-negative."
            )
        if dormancy_check_every <= 0:
            raise ValueError("dormancy_check_every must be positive.")
        if retrain_steps < 0:
            raise ValueError("retrain_steps must be non-negative.")

        self.dormancy_threshold = float(dormancy_threshold)
        self.dormancy_check_every = int(dormancy_check_every)
        self.retrain_steps = int(retrain_steps)
        self._step_count = 0
        self._active_task_index = 0
        self._retraining_task_index: int | None = None

        self._managed_weights = self._build_managed_weights()
        self._managed_parameter_ids = {
            id(managed.parameter) for managed in self._managed_weights
        }
        self._evaluation_snapshot: dict[int, torch.Tensor] | None = None

        self._activation_tracker = _ActivationTracker()
        self._activation_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_activation_hooks()

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
            managed = self._find_managed_weight(parameter)
            if managed is None:
                adjusted.append(gradient)
                continue
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
        return tuple(adjusted)

    def update_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        *,
        collect_metrics: bool = True,
    ) -> dict[str, float] | None:
        result = super().update_batch(
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            collect_metrics=collect_metrics,
        )
        self._step_count += 1
        if self._step_count % self.dormancy_check_every == 0:
            task_index = self._task_index_from_observations(observations)
            reactivated = self._reactivate_dormant(task_index)
            self._activation_tracker.reset()
            if result is not None:
                result["ssde_reactivated"] = float(reactivated)
        return result

    def on_task_end(
        self,
        *,
        task_index: int,
        replay_buffer: ReplayBuffer,
        batch_size: int,
    ) -> None:
        if len(replay_buffer) == 0:
            raise ValueError("SSDE cannot consolidate an empty replay buffer.")

        if task_index >= self.num_tasks - 1:
            self._assign_all_free_weights(task_index)
            self._activation_tracker.reset()
            self.rebuild_optimizer()
            return

        importance = self._compute_gradient_importance(
            replay_buffer=replay_buffer,
            task_index=task_index,
            batch_size=batch_size,
        )
        self._allocate_parameters(
            importance=importance,
            task_index=task_index,
        )
        self._activation_tracker.reset()

        # A task-boundary optimizer reset is required: with Adam, assigning a
        # zero gradient is insufficient to freeze a parameter that still has
        # nonzero first/second moments from the previous task.
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
            raise RuntimeError("SSDE evaluation hooks cannot be nested.")
        self._evaluation_snapshot = {
            id(managed.parameter): managed.parameter.data.clone()
            for managed in self._managed_weights
        }
        self._activation_tracker.enabled = False
        for managed in self._managed_weights:
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
            raise RuntimeError("SSDE evaluation ended without a snapshot.")
        for managed in self._managed_weights:
            managed.parameter.data.copy_(
                self._evaluation_snapshot[id(managed.parameter)]
            )
        self._evaluation_snapshot = None
        self._activation_tracker.enabled = True

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "owners": {
                managed.name: managed.owner.detach().cpu()
                for managed in self._managed_weights
            },
            "step_count": self._step_count,
            "active_task_index": self._active_task_index,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("SSDE extra state must be a dictionary.")
        owners = state.get("owners", {})
        for managed in self._managed_weights:
            owner = owners.get(managed.name)
            if owner is None or owner.shape != managed.owner.shape:
                raise ValueError(
                    f"Missing or invalid SSDE owner mask for {managed.name}."
                )
            managed.owner.copy_(owner.to(managed.owner.device))
        self._step_count = int(state.get("step_count", 0))
        self._active_task_index = int(state.get("active_task_index", 0))

    def _build_managed_weights(self) -> tuple[ManagedBackboneWeight, ...]:
        managed: list[ManagedBackboneWeight] = []
        for name, parameter in self.actor.backbone.named_parameters():
            if parameter.ndim < 2:
                continue
            managed.append(
                ManagedBackboneWeight(
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

    def _find_managed_weight(
        self,
        parameter: torch.nn.Parameter,
    ) -> ManagedBackboneWeight | None:
        if id(parameter) not in self._managed_parameter_ids:
            return None
        for managed in self._managed_weights:
            if managed.parameter is parameter:
                return managed
        raise RuntimeError("Managed SSDE parameter lookup is inconsistent.")

    def _register_activation_hooks(self) -> None:
        for layer_index, layer in enumerate(self.actor.backbone.hidden_layers):
            self._activation_hooks.append(
                layer.register_forward_hook(
                    self._activation_tracker.hook(f"hidden_{layer_index}")
                )
            )

    def _compute_gradient_importance(
        self,
        *,
        replay_buffer: ReplayBuffer,
        task_index: int,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        del task_index
        importance = {
            managed.name: torch.zeros_like(managed.parameter.detach())
            for managed in self._managed_weights
        }
        num_batches = min(
            10,
            max(1, (len(replay_buffer) + batch_size - 1) // batch_size),
        )

        was_training = self.actor.training
        tracking_was_enabled = self._activation_tracker.enabled
        self._activation_tracker.enabled = False
        self.actor.eval()
        try:
            for _ in range(num_batches):
                sample_size = min(batch_size, len(replay_buffer))
                batch = replay_buffer.sample(sample_size, self.device)
                actor_output = self.actor.sample(batch.observations)
                loss = -actor_output.log_probability.mean()
                gradients = torch.autograd.grad(
                    loss,
                    tuple(managed.parameter for managed in self._managed_weights),
                )
                for managed, gradient in zip(
                    self._managed_weights,
                    gradients,
                    strict=True,
                ):
                    importance[managed.name].add_(
                        gradient.detach().abs() / float(num_batches)
                    )
        finally:
            self._activation_tracker.enabled = tracking_was_enabled
            if was_training:
                self.actor.train()
        return importance

    @torch.no_grad()
    def _allocate_parameters(
        self,
        *,
        importance: dict[str, torch.Tensor],
        task_index: int,
    ) -> None:
        tasks_left = self.num_tasks - task_index - 1
        allocation_fraction = 1.0 / (tasks_left + 1)
        for managed in self._managed_weights:
            free_mask = managed.owner == -1
            free_importance = importance[managed.name][free_mask]
            if free_importance.numel() == 0:
                continue

            num_allocate = max(
                1,
                int(round(allocation_fraction * free_importance.numel())),
            )
            num_allocate = min(num_allocate, free_importance.numel())
            top_indices = torch.topk(
                free_importance,
                k=num_allocate,
                largest=True,
                sorted=False,
            ).indices
            free_flat_indices = free_mask.flatten().nonzero(as_tuple=False).flatten()
            allocated_flat_indices = free_flat_indices[top_indices]
            managed.owner.flatten()[allocated_flat_indices] = task_index

            # The unallocated capacity is released for later tasks.
            managed.parameter.data[managed.owner == -1] = 0.0

    @torch.no_grad()
    def _assign_all_free_weights(self, task_index: int) -> None:
        for managed in self._managed_weights:
            managed.owner[managed.owner == -1] = task_index

    @torch.no_grad()
    def _reactivate_dormant(self, task_index: int) -> int:
        del task_index
        reactivated = 0
        for layer_index, layer in enumerate(self.actor.backbone.hidden_layers):
            sensitivity = self._activation_tracker.sensitivity(
                f"hidden_{layer_index}"
            )
            if sensitivity is None:
                continue
            sensitivity = sensitivity.to(device=layer.weight.device)
            managed = self._find_managed_weight(layer.weight)
            if managed is None:
                continue

            dormant_rows = sensitivity < self.dormancy_threshold
            reset_mask = dormant_rows[:, None] & (managed.owner == -1)
            if not torch.any(reset_mask):
                continue

            replacement = torch.randn_like(layer.weight) * 0.01
            layer.weight.data[reset_mask] = replacement[reset_mask]
            self._clear_optimizer_moments(layer.weight, reset_mask)
            reactivated += int(torch.any(reset_mask, dim=1).sum().item())
        return reactivated

    def _clear_optimizer_moments(
        self,
        parameter: torch.nn.Parameter,
        mask: torch.Tensor,
    ) -> None:
        state = self.optimizer.state.get(parameter)
        if not state:
            return
        for state_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(state_name)
            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                value[mask] = 0.0
