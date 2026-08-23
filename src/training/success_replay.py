from __future__ import annotations

import random

import numpy as np


BROADER_TEMPORAL_BINS: tuple[str, ...] = ("early", "middle", "late")


def select_success_dominant_memory(
    *,
    successful: "SuccessfulStateReservoir",
    broader: "BroaderStateReservoir",
    total_capacity: int,
    nominal_success_capacity: int,
    selected_bins: tuple[str, ...] = BROADER_TEMPORAL_BINS,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a fixed-budget success-preferred memory with broad fallback.

    The nominal successful quota is used whenever enough successful states and
    broad candidates exist.  Missing successful states are replaced by broad
    states.  If the selected broad bins are themselves underfilled, additional
    successful states fill the remaining budget.  The two reservoirs are
    disjoint by construction because broader replay only records failed
    episodes.
    """
    if total_capacity <= 0:
        raise ValueError("total_capacity must be positive.")
    if not 0 <= nominal_success_capacity <= total_capacity:
        raise ValueError(
            "nominal_success_capacity must be within the total capacity."
        )

    available_success = successful.observations()
    initial_success_count = min(
        nominal_success_capacity,
        int(available_success.shape[0]),
    )
    initial_broad_target = total_capacity - initial_success_count
    broad_observations = broader.observations(
        selected_bins=selected_bins,
        capacity=initial_broad_target,
        seed=seed + 1,
    )
    final_success_target = min(
        total_capacity - int(broad_observations.shape[0]),
        int(available_success.shape[0]),
    )
    success_observations = successful.observations(
        capacity=final_success_target,
        seed=seed,
    )
    if success_observations.shape[0] + broad_observations.shape[0] > total_capacity:
        raise RuntimeError("Dynamic replay selection exceeded its fixed budget.")
    return success_observations, broad_observations


class BroaderStateReservoir:
    """Fixed-capacity temporal reservoirs from non-successful episodes."""

    def __init__(self, *, capacity_per_bin: int, observation_dim: int, seed: int) -> None:
        if capacity_per_bin <= 0:
            raise ValueError("capacity_per_bin must be positive.")
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        self.capacity_per_bin = int(capacity_per_bin)
        self.observation_dim = int(observation_dim)
        self._rng = {
            label: random.Random(seed + index)
            for index, label in enumerate(BROADER_TEMPORAL_BINS)
        }
        self._states: dict[str, list[np.ndarray]] = {
            label: [] for label in BROADER_TEMPORAL_BINS
        }
        self._seen: dict[str, int] = {label: 0 for label in BROADER_TEMPORAL_BINS}
        self.non_successful_episodes = 0

    def add_episode(
        self,
        observations: tuple[np.ndarray, ...],
        succeeded: bool,
    ) -> None:
        # The broader component is deliberately disjoint from the successful 80%.
        if succeeded or not observations:
            return
        self.non_successful_episodes += 1
        episode_length = len(observations)
        for step, observation in enumerate(observations):
            state = np.asarray(observation, dtype=np.float32)
            if state.shape != (self.observation_dim,):
                raise ValueError(
                    f"Broader replay observation has shape {state.shape}, "
                    f"expected {(self.observation_dim,)}."
                )
            if not bool(np.isfinite(state).all()):
                raise ValueError("Broader replay observations must be finite.")
            progress = (step + 0.5) / episode_length
            label = BROADER_TEMPORAL_BINS[min(int(progress * 3), 2)]
            self._seen[label] += 1
            states = self._states[label]
            if len(states) < self.capacity_per_bin:
                states.append(state.copy())
                continue
            replacement = self._rng[label].randrange(self._seen[label])
            if replacement < self.capacity_per_bin:
                states[replacement] = state.copy()

    def counts(self) -> dict[str, int]:
        return {label: len(self._states[label]) for label in BROADER_TEMPORAL_BINS}

    def observations(
        self,
        *,
        selected_bins: tuple[str, ...] = BROADER_TEMPORAL_BINS,
        capacity: int,
        seed: int,
    ) -> np.ndarray:
        if capacity < 0:
            raise ValueError("capacity must be non-negative.")
        invalid = sorted(set(selected_bins).difference(BROADER_TEMPORAL_BINS))
        if invalid:
            raise ValueError(f"Unknown broader temporal bins: {invalid}.")
        if capacity == 0 or not selected_bins:
            return np.empty((0, self.observation_dim), dtype=np.float32)
        candidates = [
            state
            for label in selected_bins
            for state in self._states[label]
        ]
        if not candidates:
            return np.empty((0, self.observation_dim), dtype=np.float32)
        rng = random.Random(seed)
        if len(candidates) > capacity:
            candidates = rng.sample(candidates, capacity)
        return np.stack(candidates).astype(np.float32, copy=False)


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

    def observations(
        self,
        *,
        capacity: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        if capacity is not None and capacity < 0:
            raise ValueError("capacity must be non-negative.")
        if not self._states:
            return np.empty((0, self.observation_dim), dtype=np.float32)
        states = self._states
        if capacity is not None:
            if capacity == 0:
                return np.empty((0, self.observation_dim), dtype=np.float32)
            if len(states) > capacity:
                if seed is None:
                    raise ValueError("seed is required when subsampling observations.")
                states = random.Random(seed).sample(states, capacity)
        return np.stack(states).astype(np.float32, copy=False)
