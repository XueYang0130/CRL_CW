from __future__ import annotations

import numpy as np
import torch

from agents.sac_agent import SACAgent


class FullBehaviorCloningSACAgent(SACAgent):
    """CW10 full-trajectory behavioral cloning with ClonEx-style policy distillation."""

    def __init__(
        self,
        *args,
        episodic_batch_size: int = 128,
        actor_cloning_coefficient: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.num_tasks <= 1 or self.task_id_dim != self.num_tasks:
            raise ValueError("full_bc requires task-specific actor and critic heads.")
        if episodic_batch_size <= 0:
            raise ValueError("episodic_batch_size must be positive.")
        if actor_cloning_coefficient < 0.0:
            raise ValueError("actor_cloning_coefficient must be non-negative.")

        self.episodic_batch_size = int(episodic_batch_size)
        self.actor_cloning_coefficient = float(actor_cloning_coefficient)
        self._episodic_observations = torch.empty((0, self.observation_dim))
        self._episodic_target_means = torch.empty((0, self.action_dim))
        self._episodic_target_log_stds = torch.empty((0, self.action_dim))

    @property
    def reference_state_count(self) -> int:
        return int(self._episodic_observations.shape[0])

    def clear_reference_memory(self) -> None:
        self._episodic_observations = torch.empty((0, self.observation_dim))
        self._episodic_target_means = torch.empty((0, self.action_dim))
        self._episodic_target_log_stds = torch.empty((0, self.action_dim))

    @torch.no_grad()
    def add_reference_memory(
        self,
        *,
        observations: np.ndarray | torch.Tensor,
    ) -> None:
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32)
        if observations_tensor.ndim != 2 or observations_tensor.shape[1] != self.observation_dim:
            raise ValueError("reference observations have the wrong shape.")
        if observations_tensor.shape[0] == 0:
            return
        means, log_stds = self.compute_reference_targets(observations_tensor)
        self._episodic_observations = torch.cat(
            (self._episodic_observations, observations_tensor.detach().cpu())
        )
        self._episodic_target_means = torch.cat(
            (self._episodic_target_means, means.detach().cpu())
        )
        self._episodic_target_log_stds = torch.cat(
            (self._episodic_target_log_stds, log_stds.detach().cpu())
        )

    def add_reference_memory_with_targets(
        self,
        *,
        observations: np.ndarray | torch.Tensor,
        target_means: np.ndarray | torch.Tensor,
        target_log_stds: np.ndarray | torch.Tensor,
    ) -> None:
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32)
        target_means_tensor = torch.as_tensor(target_means, dtype=torch.float32)
        target_log_stds_tensor = torch.as_tensor(target_log_stds, dtype=torch.float32)
        if observations_tensor.ndim != 2 or observations_tensor.shape[1] != self.observation_dim:
            raise ValueError("reference observations have the wrong shape.")
        if target_means_tensor.ndim != 2 or target_means_tensor.shape[1] != self.action_dim:
            raise ValueError("reference target means have the wrong shape.")
        if target_log_stds_tensor.ndim != 2 or target_log_stds_tensor.shape[1] != self.action_dim:
            raise ValueError("reference target log stds have the wrong shape.")
        if observations_tensor.shape[0] != target_means_tensor.shape[0]:
            raise ValueError("reference observations and target means must have the same length.")
        if observations_tensor.shape[0] != target_log_stds_tensor.shape[0]:
            raise ValueError("reference observations and target log stds must have the same length.")
        if observations_tensor.shape[0] == 0:
            return
        self._episodic_observations = torch.cat(
            (self._episodic_observations, observations_tensor.detach().cpu())
        )
        self._episodic_target_means = torch.cat(
            (self._episodic_target_means, target_means_tensor.detach().cpu())
        )
        self._episodic_target_log_stds = torch.cat(
            (self._episodic_target_log_stds, target_log_stds_tensor.detach().cpu())
        )

    def set_reference_memory_with_targets(
        self,
        *,
        observations: np.ndarray | torch.Tensor,
        target_means: np.ndarray | torch.Tensor,
        target_log_stds: np.ndarray | torch.Tensor,
    ) -> None:
        self.clear_reference_memory()
        self.add_reference_memory_with_targets(
            observations=observations,
            target_means=target_means,
            target_log_stds=target_log_stds,
        )

    @torch.no_grad()
    def compute_reference_targets(
        self,
        observations: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32)
        if observations_tensor.ndim != 2 or observations_tensor.shape[1] != self.observation_dim:
            raise ValueError("reference observations have the wrong shape.")
        return self._cloning_distribution_parameters(
            observations_tensor.to(self.device)
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
        observations, target_means, target_log_stds = self._sample_episodic_batch()
        current_means, current_log_stds = self._cloning_distribution_parameters(observations)
        cloning_loss = self.actor_cloning_coefficient * self._gaussian_kl(
            target_means,
            target_log_stds,
            current_means,
            current_log_stds,
        ).mean()
        cloning_gradients = torch.autograd.grad(cloning_loss, parameters)
        return tuple(
            (sac_gradient + cloning_gradient) / 2.0
            for sac_gradient, cloning_gradient in zip(
                gradients,
                cloning_gradients,
                strict=True,
            )
        )

    def _sample_episodic_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.randint(
            0,
            self._episodic_observations.shape[0],
            size=self.episodic_batch_size,
        )
        return (
            self._episodic_observations[indices].to(self.device),
            self._episodic_target_means[indices].to(self.device),
            self._episodic_target_log_stds[indices].to(self.device),
        )

    @staticmethod
    def _gaussian_kl(
        first_mean: torch.Tensor,
        first_log_std: torch.Tensor,
        second_mean: torch.Tensor,
        second_log_std: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = 1e-6
        first_variance = (first_log_std.exp() + epsilon).square()
        second_variance = (second_log_std.exp() + epsilon).square()
        return (
            second_log_std
            - first_log_std
            + (first_variance + (first_mean - second_mean).square())
            / (2.0 * second_variance)
            - 0.5
        ).sum(dim=-1)

    def _cloning_distribution_parameters(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_means, log_stds = self.actor.distribution_parameters(observations)
        means = (
            torch.tanh(raw_means) * self.actor.action_scale
            + self.actor.action_bias
        )
        return means, log_stds
