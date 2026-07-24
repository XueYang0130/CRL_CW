from __future__ import annotations

import numpy as np
import torch

from agents.replay_buffer import ReplayBuffer
from agents.sac_agent import SACAgent


class ClonExSACAgent(SACAgent):
    """Final ClonEx-SAC with actor-only behavioral cloning."""

    def __init__(
        self,
        *args,
        episodic_memory_per_task: int = 10_000,
        episodic_batch_size: int = 128,
        actor_cloning_coefficient: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.num_tasks <= 1 or self.task_id_dim != self.num_tasks:
            raise ValueError("ClonEx-SAC requires task-specific actor and critic heads.")
        if episodic_memory_per_task <= 0 or episodic_batch_size <= 0:
            raise ValueError("ClonEx-SAC episodic memory sizes must be positive.")
        if actor_cloning_coefficient < 0.0:
            raise ValueError("actor_cloning_coefficient must be non-negative.")

        self.episodic_memory_per_task = int(episodic_memory_per_task)
        self.episodic_batch_size = int(episodic_batch_size)
        self.actor_cloning_coefficient = float(actor_cloning_coefficient)
        self._episodic_observations = torch.empty((0, self.observation_dim))
        self._episodic_target_means = torch.empty((0, self.action_dim))
        self._episodic_target_log_stds = torch.empty((0, self.action_dim))

    @torch.no_grad()
    def on_task_start(
        self,
        *,
        task_index: int,
        replay_buffer: ReplayBuffer,
    ) -> None:
        if task_index == 0:
            return
        batch = replay_buffer.sample(self.episodic_memory_per_task, self.device)
        means, log_stds = self._cloning_distribution_parameters(batch.observations)
        self._episodic_observations = torch.cat(
            (self._episodic_observations, batch.observations.detach().cpu())
        )
        self._episodic_target_means = torch.cat(
            (self._episodic_target_means, means.detach().cpu())
        )
        self._episodic_target_log_stds = torch.cat(
            (self._episodic_target_log_stds, log_stds.detach().cpu())
        )

    def adjust_actor_gradients(
        self,
        *,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[torch.nn.Parameter, ...],
        task_index: int,
    ) -> tuple[torch.Tensor, ...]:
        if task_index == 0:
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
