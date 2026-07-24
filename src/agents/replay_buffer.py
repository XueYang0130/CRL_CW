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
            1,000,000 follows the default SAC experiment configuration.
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
        """Store one finite transition with the configured dimensions."""
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        if observation.shape != (self.observation_dim,):
            raise ValueError("observation has the wrong shape.")
        if action.shape != (self.action_dim,):
            raise ValueError("action has the wrong shape.")
        if next_observation.shape != (self.observation_dim,):
            raise ValueError("next_observation has the wrong shape.")
        if not (
            np.all(np.isfinite(observation))
            and np.all(np.isfinite(action))
            and np.all(np.isfinite(next_observation))
            and np.isfinite(reward)
        ):
            raise ValueError("Replay transitions must contain only finite values.")
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

        indices = np.random.randint(
            0,
            self._size,
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

        Continual training may clear the replay buffer at task
        boundaries. That behavior is controlled by the outer runner.
        """
        self._position = 0
        self._size = 0
