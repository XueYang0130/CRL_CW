from __future__ import annotations

import random

import numpy as np


class SuccessfulStateReservoir:
    """Uniform fixed-capacity sample of states from successful train episodes."""

    def __init__(self, *, capacity: int, observation_dim: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self._rng = random.Random(seed)
        self._states: list[np.ndarray] = []
        self.seen_successful_states = 0
        self.successful_episodes = 0

    def add_episode(
        self,
        observations: tuple[np.ndarray, ...],
        succeeded: bool,
    ) -> None:
        if not succeeded:
            return
        self.successful_episodes += 1
        for observation in observations:
            state = np.asarray(observation, dtype=np.float32)
            if state.shape != (self.observation_dim,):
                raise ValueError(
                    "Successful replay observation has shape "
                    f"{state.shape}, expected {(self.observation_dim,)}."
                )
            if not bool(np.isfinite(state).all()):
                raise ValueError("Successful replay observations must be finite.")
            self.seen_successful_states += 1
            if len(self._states) < self.capacity:
                self._states.append(state.copy())
                continue
            replacement = self._rng.randrange(self.seen_successful_states)
            if replacement < self.capacity:
                self._states[replacement] = state.copy()

    def observations(self) -> np.ndarray:
        if not self._states:
            return np.empty((0, self.observation_dim), dtype=np.float32)
        return np.stack(self._states).astype(np.float32, copy=False)
