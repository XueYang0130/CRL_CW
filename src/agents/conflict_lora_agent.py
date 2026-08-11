from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.optim import Adam

from agents.full_bc_agent import FullBehaviorCloningSACAgent
from agents.networks import LOG_STD_MAX, LOG_STD_MIN


class LowRankPolicyResidual(nn.Module):
    """Small residual adapter for old-task policy distribution parameters."""

    def __init__(self, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(observations))


class ConflictLoRAFullBCAgent(FullBehaviorCloningSACAgent):
    """PCGrad success replay with conflict-triggered LoRA residual memory.

    The main SAC/BC path remains unchanged. When the shared actor BC gradient
    conflicts with the SAC gradient and PCGrad projects it away, a separate
    low-rank adapter is trained on the same reference batch. This lets us test
    whether the discarded preservation signal can live in a small side module
    instead of fighting the current-task backbone update.
    """

    def __init__(
        self,
        *args,
        conflict_lora_rank: int = 4,
        conflict_lora_coefficient: float = 1.0,
        conflict_lora_max_norm_ratio: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if conflict_lora_rank <= 0:
            raise ValueError("conflict_lora_rank must be positive.")
        if (
            not math.isfinite(conflict_lora_coefficient)
            or conflict_lora_coefficient < 0.0
        ):
            raise ValueError(
                "conflict_lora_coefficient must be finite and non-negative."
            )
        if (
            not math.isfinite(conflict_lora_max_norm_ratio)
            or conflict_lora_max_norm_ratio <= 0.0
        ):
            raise ValueError(
                "conflict_lora_max_norm_ratio must be finite and positive."
            )

        self.conflict_lora_rank = int(conflict_lora_rank)
        self.conflict_lora_coefficient = float(conflict_lora_coefficient)
        self.conflict_lora_max_norm_ratio = float(conflict_lora_max_norm_ratio)
        self.conflict_lora_input_dim = (
            self.observation_dim - self.task_id_dim
            if self.hide_task_id
            else self.observation_dim
        )
        self.conflict_lora_output_dim = 2 * self.action_dim
        self.conflict_lora_adapters = nn.ModuleList(
            LowRankPolicyResidual(
                self.conflict_lora_input_dim,
                self.conflict_lora_output_dim,
                self.conflict_lora_rank,
            )
            for _ in range(self.num_tasks)
        ).to(self.device)
        self.conflict_lora_optimizer = Adam(
            self.conflict_lora_adapters.parameters(),
            lr=self.learning_rate,
            eps=1e-7,
        )
        self.conflict_lora_updates = 0
        self.last_conflict_lora_loss = float("nan")
        self.last_conflict_lora_grad_norm = float("nan")
        self.last_conflict_lora_grad_scale = float("nan")

    def reset_optimizer_state(self) -> None:
        super().reset_optimizer_state()
        self.conflict_lora_optimizer.state.clear()

    def rebuild_optimizer(self) -> None:
        super().rebuild_optimizer()
        self.conflict_lora_optimizer = Adam(
            self.conflict_lora_adapters.parameters(),
            lr=self.learning_rate,
            eps=1e-7,
        )

    @torch.inference_mode()
    def select_action(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        return self._act_with_lora(
            observation,
            deterministic=deterministic,
        )

    @torch.inference_mode()
    def select_action_with_head(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        head_index: int,
        deterministic: bool = False,
    ) -> np.ndarray:
        overridden_observation = self._observation_for_actor_head(
            observation=observation,
            head_index=head_index,
        )
        return self._act_with_lora(
            overridden_observation,
            deterministic=deterministic,
        )

    @torch.inference_mode()
    def select_guide_action(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        guide_task_index: int,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select a clean prior-task guide action without LoRA residuals.

        LoRA adapters are retention modules. They should affect normal
        old-task evaluation, but not the best-return guide selection or
        start-step exploration policy used to bootstrap a new task.
        """
        overridden_observation = self._observation_for_actor_head(
            observation=observation,
            head_index=guide_task_index,
        )
        return self.actor.act(
            overridden_observation,
            deterministic=deterministic,
            device=self.device,
        )

    def adjust_actor_gradients(
        self,
        *,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[torch.nn.Parameter, ...],
        task_index: int,
    ) -> tuple[torch.Tensor, ...]:
        if task_index == 0 or self.reference_state_count == 0:
            return gradients
        if self._max_reference_source_task_index >= task_index:
            invalid_sources = torch.unique(
                self._episodic_source_task_indices[
                    self._episodic_source_task_indices >= task_index
                ]
            ).tolist()
            raise RuntimeError(
                "Behavior-cloning memory contains current or future task data: "
                f"current_task_index={task_index}, invalid_sources={invalid_sources}."
            )
        observations, target_means, target_log_stds = self._sample_episodic_batch()
        current_means, current_log_stds = self._cloning_distribution_parameters(
            observations
        )
        raw_cloning_loss = self._gaussian_kl(
            target_means,
            target_log_stds,
            current_means,
            current_log_stds,
        ).mean()
        weighted_cloning_loss = (
            self.actor_cloning_coefficient * raw_cloning_loss
        )
        cloning_gradients = torch.autograd.grad(
            weighted_cloning_loss,
            parameters,
        )
        applied_cloning_gradients, gradient_adjustment = (
            self._apply_bc_gradient_strategy(
                sac_gradients=gradients,
                cloning_gradients=cloning_gradients,
            )
        )
        if bool(gradient_adjustment["projection_applied"]):
            self._update_conflict_lora(
                observations=observations,
                target_means=target_means,
                target_log_stds=target_log_stds,
                sac_gradients=gradients,
            )
        final_actor_gradients, combination_scale = self._combine_actor_gradients(
            sac_gradients=gradients,
            applied_cloning_gradients=applied_cloning_gradients,
            conflict=bool(gradient_adjustment["conflict"]),
        )
        collect_global = self.gradient_diagnostics.begin_bc_update()
        if collect_global:
            self.gradient_diagnostics.record(
                sac_gradients=gradients,
                bc_gradients=cloning_gradients,
                current_task_index=task_index,
                raw_bc_loss=float(raw_cloning_loss.detach().item()),
                bc_coefficient=self.actor_cloning_coefficient,
                sac_actor_loss=float(self._current_actor_loss.item()),
                reference_memory_states=self.reference_state_count,
                applied_bc_gradients=applied_cloning_gradients,
                final_actor_gradients=final_actor_gradients,
                gradient_strategy=self.bc_gradient_strategy,
                projection_applied=gradient_adjustment["projection_applied"],
                bc_norm_scale=gradient_adjustment["bc_norm_scale"],
                bc_combination_strategy=self.bc_combination_strategy,
                bc_combination_scale=combination_scale,
            )
        if collect_global:
            self._record_source_task_diagnostics(
                sac_gradients=gradients,
                parameters=parameters,
                current_task_index=task_index,
            )
        return final_actor_gradients

    def _act_with_lora(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        deterministic: bool,
    ) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32)
        expected_shape = (self.observation_dim,)
        if observation_array.shape != expected_shape:
            raise ValueError(
                f"Expected observation shape {expected_shape}, "
                f"received {observation_array.shape}."
            )
        observation_tensor = torch.as_tensor(
            observation_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mean, log_std = self._lora_distribution_parameters(observation_tensor)
        if deterministic:
            raw_action = mean
        else:
            raw_action = Normal(mean, log_std.exp()).sample()
        action = torch.tanh(raw_action) * self.actor.action_scale + self.actor.action_bias
        return action.squeeze(0).cpu().numpy()

    def _lora_distribution_parameters(
        self,
        observations: torch.Tensor,
        *,
        detach_base: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_mean, base_log_std = self.actor.distribution_parameters(observations)
        if detach_base:
            base_mean = base_mean.detach()
            base_log_std = base_log_std.detach()
        residual = self._lora_residual(observations)
        mean_delta, log_std_delta = residual.chunk(2, dim=-1)
        return (
            base_mean + mean_delta,
            torch.clamp(
                base_log_std + log_std_delta,
                min=LOG_STD_MIN,
                max=LOG_STD_MAX,
            ),
        )

    def _lora_residual(self, observations: torch.Tensor) -> torch.Tensor:
        features = (
            observations[:, :-self.task_id_dim]
            if self.hide_task_id and self.task_id_dim > 0
            else observations
        )
        task_indices = self._infer_source_task_indices(observations.detach().cpu())
        task_indices = task_indices.to(observations.device)
        residual = torch.zeros(
            (observations.shape[0], self.conflict_lora_output_dim),
            dtype=observations.dtype,
            device=observations.device,
        )
        for task_index in torch.unique(task_indices).tolist():
            task_int = int(task_index)
            mask = task_indices == task_int
            residual[mask] = self.conflict_lora_adapters[task_int](features[mask])
        return residual

    def _update_conflict_lora(
        self,
        *,
        observations: torch.Tensor,
        target_means: torch.Tensor,
        target_log_stds: torch.Tensor,
        sac_gradients: tuple[torch.Tensor, ...],
    ) -> None:
        if (
            self.conflict_lora_coefficient == 0.0
            or self.bc_progress_multiplier == 0.0
        ):
            return
        self.conflict_lora_optimizer.zero_grad(set_to_none=True)
        current_means, current_log_stds = self._lora_distribution_parameters(
            observations,
            detach_base=True,
        )
        raw_loss = self._gaussian_kl(
            target_means.detach(),
            target_log_stds.detach(),
            current_means,
            current_log_stds,
        ).mean()
        lora_loss = (
            self.actor_cloning_coefficient
            * self.conflict_lora_coefficient
            * self.bc_progress_multiplier
            * raw_loss
        )
        lora_loss.backward()
        grad_norm, grad_scale = self._cap_conflict_lora_gradients(
            sac_gradients=sac_gradients,
        )
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.conflict_lora_adapters.parameters(),
                self.gradient_clip_norm,
            )
        self.conflict_lora_optimizer.step()
        self.conflict_lora_updates += 1
        self.last_conflict_lora_loss = float(raw_loss.detach().item())
        self.last_conflict_lora_grad_norm = grad_norm
        self.last_conflict_lora_grad_scale = grad_scale

    def _cap_conflict_lora_gradients(
        self,
        *,
        sac_gradients: tuple[torch.Tensor, ...],
    ) -> tuple[float, float]:
        epsilon = 1e-12
        shared_indices = self.gradient_diagnostics.shared_indices
        sac_norm = sum(
            sac_gradients[index].detach().square().sum()
            for index in shared_indices
        ).sqrt()
        lora_parameters = tuple(self.conflict_lora_adapters.parameters())
        lora_norm_sq = torch.zeros((), device=self.device)
        for parameter in lora_parameters:
            if parameter.grad is not None:
                lora_norm_sq = lora_norm_sq + parameter.grad.detach().square().sum()
        lora_norm = lora_norm_sq.sqrt()
        scale = torch.clamp(
            self.conflict_lora_max_norm_ratio
            * sac_norm
            / (lora_norm + epsilon),
            max=1.0,
        )
        scale_value = float(scale.detach().item())
        for parameter in lora_parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        return float(lora_norm.detach().item()), scale_value
