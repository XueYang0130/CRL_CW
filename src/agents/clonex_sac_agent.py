from __future__ import annotations

import torch

from agents.full_bc_agent import FullBehaviorCloningSACAgent
from agents.replay_buffer import ReplayBuffer


class ClonExSACAgent(FullBehaviorCloningSACAgent):
    """Final ClonEx-SAC with replay-sampled actor policy distillation."""

    def __init__(
        self,
        *args,
        episodic_memory_per_task: int = 10_000,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if episodic_memory_per_task <= 0:
            raise ValueError("ClonEx-SAC episodic memory size must be positive.")
        self.episodic_memory_per_task = int(episodic_memory_per_task)

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
        self.add_reference_memory_with_targets(
            observations=batch.observations,
            target_means=means,
            target_log_stds=log_stds,
        )
