"""Prioritized n-step replay used by the discovery-focused SAC variant.

The standard :mod:`agents.replay_buffer` remains unchanged.  This buffer is
opted into by one method and adds three pieces of information needed by that
method's critic update:

* an n-step reward and its exact bootstrap discount;
* replay indices for priority updates;
* importance weights for the prioritized part of the minibatch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class PrioritizedReplayBatch:
    """One sampled batch with PER and n-step metadata."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor
    discounts: torch.Tensor
    indices: np.ndarray
    importance_weights: torch.Tensor
    sampling_probabilities: torch.Tensor
    priority_beta: float


@dataclass(frozen=True)
class _PendingTransition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool


class PrioritizedNStepReplayBuffer:
    """Circular n-step replay with a fixed uniform/prioritized mixture.

    A fraction of every minibatch is drawn uniformly, so newly discovered or
    temporarily low-error transitions never become unreachable.  The rest is
    drawn according to absolute TD-error priorities.  Importance weights use
    the probability of the complete mixture rather than only the prioritized
    component.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        seed: int = 0,
        *,
        gamma: float = 0.99,
        n_step: int = 3,
        prioritized_fraction: float = 0.2,
        priority_alpha: float = 0.6,
        priority_beta_start: float = 0.4,
        priority_beta_end: float = 1.0,
        priority_beta_steps: int = 500_000,
        priority_epsilon: float = 1e-6,
    ) -> None:
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1].")
        if n_step <= 0:
            raise ValueError("n_step must be positive.")
        if not math.isfinite(prioritized_fraction) or not 0.0 <= prioritized_fraction <= 1.0:
            raise ValueError("prioritized_fraction must be finite and in [0, 1].")
        if not math.isfinite(priority_alpha) or priority_alpha < 0.0:
            raise ValueError("priority_alpha must be finite and non-negative.")
        if (
            not math.isfinite(priority_beta_start)
            or not math.isfinite(priority_beta_end)
            or not 0.0 <= priority_beta_start <= priority_beta_end <= 1.0
        ):
            raise ValueError("priority betas must satisfy 0 <= start <= end <= 1.")
        if priority_beta_steps <= 0:
            raise ValueError("priority_beta_steps must be positive.")
        if not math.isfinite(priority_epsilon) or priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be finite and positive.")

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.prioritized_fraction = float(prioritized_fraction)
        self.priority_alpha = float(priority_alpha)
        self.priority_beta_start = float(priority_beta_start)
        self.priority_beta_end = float(priority_beta_end)
        self.priority_beta_steps = int(priority_beta_steps)
        self.priority_epsilon = float(priority_epsilon)
        self._rng = np.random.default_rng(seed)

        self._observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self._actions = np.empty((capacity, action_dim), dtype=np.float32)
        self._rewards = np.empty((capacity, 1), dtype=np.float32)
        self._next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self._terminated = np.empty((capacity, 1), dtype=np.float32)
        self._discounts = np.empty((capacity, 1), dtype=np.float32)
        self._priorities = np.zeros((capacity,), dtype=np.float64)

        self._position = 0
        self._size = 0
        self._sample_calls = 0
        self._pending: deque[_PendingTransition] = deque()

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def current_priority_beta(self) -> float:
        progress = min(1.0, self._sample_calls / self.priority_beta_steps)
        return self.priority_beta_start + progress * (
            self.priority_beta_end - self.priority_beta_start
        )

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        observation = self._finite_vector(
            observation, self.observation_dim, "observation"
        )
        action = self._finite_vector(action, self.action_dim, "action")
        next_observation = self._finite_vector(
            next_observation, self.observation_dim, "next_observation"
        )
        if not math.isfinite(reward):
            raise ValueError("reward must be finite.")

        self._pending.append(
            _PendingTransition(
                observation=observation.copy(),
                action=action.copy(),
                reward=float(reward),
                next_observation=next_observation.copy(),
                terminated=bool(terminated),
            )
        )
        if terminated:
            self.end_episode()
        elif len(self._pending) >= self.n_step:
            self._store_oldest_pending()
            self._pending.popleft()

    def end_episode(self) -> None:
        """Flush pending transitions without crossing an environment reset."""
        while self._pending:
            self._store_oldest_pending()
            self._pending.popleft()

    def flush_pending(self) -> None:
        """Flush the final partial trajectory at the end of a training period."""
        self.end_episode()

    def clear(self) -> None:
        self._position = 0
        self._size = 0
        self._sample_calls = 0
        self._pending.clear()
        self._priorities.fill(0.0)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> PrioritizedReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        priority_probabilities = self._priority_probabilities()
        prioritized_count = int(round(batch_size * self.prioritized_fraction))
        uniform_count = batch_size - prioritized_count
        parts: list[np.ndarray] = []
        if uniform_count:
            parts.append(self._rng.integers(0, self._size, size=uniform_count))
        if prioritized_count:
            parts.append(
                self._rng.choice(
                    self._size,
                    size=prioritized_count,
                    replace=True,
                    p=priority_probabilities,
                )
            )
        indices = np.concatenate(parts).astype(np.int64, copy=False)
        # Preserve the exact standard-ReplayBuffer RNG path when PER is
        # disabled. Shuffling is only needed to interleave two real strata.
        if len(parts) > 1:
            self._rng.shuffle(indices)

        mixture_probabilities = (
            (1.0 - self.prioritized_fraction) / self._size
            + self.prioritized_fraction * priority_probabilities
        )
        selected_probabilities = mixture_probabilities[indices]
        beta = self.current_priority_beta
        all_weights = np.power(
            self._size * mixture_probabilities,
            -beta,
            dtype=np.float64,
        )
        normalization = float(np.max(all_weights))
        weights = np.power(
            self._size * selected_probabilities,
            -beta,
            dtype=np.float64,
        ) / normalization
        self._sample_calls += 1

        tensor = lambda values: torch.as_tensor(  # noqa: E731
            values, dtype=torch.float32, device=device
        )
        return PrioritizedReplayBatch(
            observations=tensor(self._observations[indices]),
            actions=tensor(self._actions[indices]),
            rewards=tensor(self._rewards[indices]),
            next_observations=tensor(self._next_observations[indices]),
            terminated=tensor(self._terminated[indices]),
            discounts=tensor(self._discounts[indices]),
            indices=indices.copy(),
            importance_weights=tensor(weights.reshape(-1, 1)),
            sampling_probabilities=tensor(selected_probabilities.reshape(-1, 1)),
            priority_beta=float(beta),
        )

    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        indices_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        priorities_array = np.asarray(priorities, dtype=np.float64).reshape(-1)
        if indices_array.shape != priorities_array.shape:
            raise ValueError("indices and priorities must have identical shapes.")
        if np.any(indices_array < 0) or np.any(indices_array >= self._size):
            raise IndexError("Priority index is outside the active replay range.")
        if not np.all(np.isfinite(priorities_array)) or np.any(priorities_array < 0.0):
            raise ValueError("priorities must be finite and non-negative.")

        # Sampling with replacement can repeat an index.  Taking the largest
        # observed TD error prevents update order from changing the result.
        for index in np.unique(indices_array):
            value = float(np.max(priorities_array[indices_array == index]))
            self._priorities[index] = value + self.priority_epsilon

    def diagnostics(self) -> dict[str, float | int]:
        active = self._priorities[: self._size]
        return {
            "replay_states": self._size,
            "pending_nstep_states": len(self._pending),
            "n_step": self.n_step,
            "prioritized_fraction": self.prioritized_fraction,
            "priority_alpha": self.priority_alpha,
            "priority_beta": self.current_priority_beta,
            "mean_priority": float(active.mean()) if self._size else 0.0,
            "max_priority": float(active.max()) if self._size else 0.0,
        }

    def _store_oldest_pending(self) -> None:
        if not self._pending:
            raise RuntimeError("Cannot aggregate an empty n-step queue.")
        reward = 0.0
        steps = 0
        final = self._pending[0]
        for transition in self._pending:
            reward += (self.gamma**steps) * transition.reward
            steps += 1
            final = transition
            if transition.terminated or steps >= self.n_step:
                break

        first = self._pending[0]
        index = self._position
        self._observations[index] = first.observation
        self._actions[index] = first.action
        self._rewards[index, 0] = reward
        self._next_observations[index] = final.next_observation
        self._terminated[index, 0] = final.terminated
        self._discounts[index, 0] = self.gamma**steps

        active_priorities = self._priorities[: self._size]
        initial_priority = (
            float(active_priorities.max())
            if self._size and bool(np.any(active_priorities > 0.0))
            else 1.0
        )
        self._priorities[index] = initial_priority
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _priority_probabilities(self) -> np.ndarray:
        active = np.maximum(
            self._priorities[: self._size],
            self.priority_epsilon,
        )
        scaled = np.power(active, self.priority_alpha, dtype=np.float64)
        total = float(scaled.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise FloatingPointError("Replay priorities produced an invalid distribution.")
        return scaled / total

    @staticmethod
    def _finite_vector(
        value: np.ndarray,
        expected_dim: int,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (expected_dim,):
            raise ValueError(f"{name} has shape {array.shape}, expected {(expected_dim,)}.")
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"{name} must contain only finite values.")
        return array
