"""Replay buffer for online off-policy SAC training.

SAC is an off-policy algorithm: transitions collected during online
environment interaction are stored and reused for gradient updates.

This replay buffer is different from any future policy bank,
demonstration memory, or task-specific trained-policy storage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReplayBatch:
    """A sampled transition batch represented as PyTorch tensors."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity circular replay buffer used by SAC.

    Args:
        observation_dim: Flat observation-vector dimension.
        action_dim: Flat continuous-action dimension.
        capacity: Maximum number of transitions. The default value of
            1,000,000 follows the ClonEx-SAC experimental configuration.
        seed: Seed used only for replay-buffer sampling.

    Notes:
        ``terminated`` records genuine MDP termination. A Gymnasium
        ``truncated`` signal caused only by the episode time limit should
        not normally be stored as terminal, because SAC should continue
        bootstrapping through a time-limit truncation.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        seed: int = 0,
    ) -> None:
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.capacity = capacity

        self._observations = np.empty(
            (capacity, observation_dim),
            dtype=np.float32,
        )
        self._actions = np.empty(
            (capacity, action_dim),
            dtype=np.float32,
        )
        self._rewards = np.empty(
            (capacity, 1),
            dtype=np.float32,
        )
        self._next_observations = np.empty(
            (capacity, observation_dim),
            dtype=np.float32,
        )
        self._terminated = np.empty(
            (capacity, 1),
            dtype=np.float32,
        )

        self._position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Return whether the buffer has reached its capacity."""
        return self._size == self.capacity

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one transition without per-step diagnostic validation."""
        index = self._position

        self._observations[index] = observation
        self._actions[index] = action
        self._rewards[index, 0] = reward
        self._next_observations[index] = next_observation
        self._terminated[index, 0] = terminated

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> ReplayBatch:
        """Uniformly sample a transition batch.

        Sampling is performed with replacement, as is standard in SAC
        replay-buffer implementations.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        indices = self._rng.integers(
            low=0,
            high=self._size,
            size=batch_size,
        )

        return ReplayBatch(
            observations=torch.as_tensor(
                self._observations[indices],
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(
                self._actions[indices],
                dtype=torch.float32,
                device=device,
            ),
            rewards=torch.as_tensor(
                self._rewards[indices],
                dtype=torch.float32,
                device=device,
            ),
            next_observations=torch.as_tensor(
                self._next_observations[indices],
                dtype=torch.float32,
                device=device,
            ),
            terminated=torch.as_tensor(
                self._terminated[indices],
                dtype=torch.float32,
                device=device,
            ),
        )

    def clear(self) -> None:
        """Remove all stored transitions without reallocating arrays.

        ClonEx-SAC clears the current SAC replay buffer when switching to
        the next task. This behavior will be invoked by the future CW10
        training script, not automatically inside this class.
        """
        self._position = 0
        self._size = 0

