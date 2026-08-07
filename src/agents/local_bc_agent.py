from __future__ import annotations

import numpy as np
import torch

from agents.sac_agent import SACAgent


class LocalBehaviorCloningSACAgent(SACAgent):
    """SAC with an auxiliary ClonEx-style KL loss on selected old-task states."""

    def __init__(
        self,
        *args,
        local_bc_coefficient: float = 0.0,
        local_bc_batch_size: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if local_bc_coefficient < 0.0:
            raise ValueError("local_bc_coefficient must be non-negative.")
        if local_bc_batch_size <= 0:
            raise ValueError("local_bc_batch_size must be positive.")
        self.local_bc_coefficient = float(local_bc_coefficient)
        self.local_bc_batch_size = int(local_bc_batch_size)
        self._reference_observations = torch.empty((0, self.observation_dim))
        self._reference_target_means = torch.empty((0, self.action_dim))
        self._reference_target_log_stds = torch.empty((0, self.action_dim))

    def set_reference_memory(
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
        self._reference_observations = observations_tensor.detach().cpu()
        self._reference_target_means = target_means_tensor.detach().cpu()
        self._reference_target_log_stds = target_log_stds_tensor.detach().cpu()

    @property
    def has_reference_memory(self) -> bool:
        return self._reference_observations.shape[0] > 0

    def adjust_actor_gradients(
        self,
        *,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[torch.nn.Parameter, ...],
        task_index: int,
    ) -> tuple[torch.Tensor, ...]:
        del task_index
        if self.local_bc_coefficient == 0.0 or not self.has_reference_memory:
            return gradients
        observations, target_means, target_log_stds = self._sample_reference_batch()
        current_means, current_log_stds = self._cloning_distribution_parameters(observations)
        bc_loss = self.local_bc_coefficient * torch.mean(
            self._gaussian_kl(
                target_means,
                target_log_stds,
                current_means,
                current_log_stds,
            )
        )
        raw_bc_gradients = torch.autograd.grad(
            bc_loss,
            parameters,
            allow_unused=True,
        )
        bc_gradients = tuple(
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(
                parameters,
                raw_bc_gradients,
                strict=True,
            )
        )
        return tuple(
            sac_gradient + bc_gradient
            for sac_gradient, bc_gradient in zip(
                gradients,
                bc_gradients,
                strict=True,
            )
        )

    def _sample_reference_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_size = min(self.local_bc_batch_size, self._reference_observations.shape[0])
        indices = np.random.randint(0, self._reference_observations.shape[0], size=sample_size)
        return (
            self._reference_observations[indices].to(self.device),
            self._reference_target_means[indices].to(self.device),
            self._reference_target_log_stds[indices].to(self.device),
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
        # Return pre-tanh raw means so KL is computed in the same space as log_stds.
        return self.actor.distribution_parameters(observations)
