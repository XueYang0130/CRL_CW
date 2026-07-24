"""Unit tests for the SAC replay buffer."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import ReplayBuffer
from envs import make_cw_env


class TestReplayBuffer(unittest.TestCase):
    """Test replay-buffer behavior with small synthetic dimensions."""

    def setUp(self) -> None:
        # Small dimensions make unit tests fast and easy to inspect.
        # These are test fixtures, not the MetaWorld dimensions.
        self.buffer = ReplayBuffer(
            observation_dim=3,
            action_dim=2,
            capacity=5,
            seed=0,
        )

    def add_transition(
        self,
        value: float,
        terminated: bool = False,
    ) -> None:
        observation = np.full(3, value, dtype=np.float32)
        action = np.full(2, value, dtype=np.float32)
        next_observation = observation + 1.0

        self.buffer.add(
            observation=observation,
            action=action,
            reward=value,
            next_observation=next_observation,
            terminated=terminated,
        )

    def test_new_buffer_is_empty(self) -> None:
        self.assertEqual(len(self.buffer), 0)
        self.assertFalse(self.buffer.is_full)

    def test_add_transition(self) -> None:
        self.add_transition(value=1.0)

        self.assertEqual(len(self.buffer), 1)

    def test_capacity_wraparound(self) -> None:
        for value in range(8):
            self.add_transition(float(value))

        self.assertEqual(len(self.buffer), 5)
        self.assertTrue(self.buffer.is_full)

    def test_sample_shapes_and_device(self) -> None:
        for value in range(5):
            self.add_transition(float(value))

        batch = self.buffer.sample(
            batch_size=8,
            device="cpu",
        )

        # Sampling with replacement permits batch_size > current size.
        self.assertEqual(batch.observations.shape, (8, 3))
        self.assertEqual(batch.actions.shape, (8, 2))
        self.assertEqual(batch.rewards.shape, (8, 1))
        self.assertEqual(batch.next_observations.shape, (8, 3))
        self.assertEqual(batch.terminated.shape, (8, 1))

        self.assertEqual(batch.observations.dtype, torch.float32)
        self.assertEqual(batch.observations.device.type, "cpu")

    def test_sampling_empty_buffer_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.buffer.sample(batch_size=1, device="cpu")

    def test_clear(self) -> None:
        self.add_transition(value=1.0)
        self.buffer.clear()

        self.assertEqual(len(self.buffer), 0)
        self.assertFalse(self.buffer.is_full)

    def test_invalid_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.buffer.add(
                observation=np.zeros(4, dtype=np.float32),
                action=np.zeros(2, dtype=np.float32),
                reward=0.0,
                next_observation=np.zeros(3, dtype=np.float32),
                terminated=False,
            )

    def test_non_finite_values_are_rejected(self) -> None:
        observation = np.zeros(3, dtype=np.float32)
        observation[0] = np.nan

        with self.assertRaises(ValueError):
            self.buffer.add(
                observation=observation,
                action=np.zeros(2, dtype=np.float32),
                reward=0.0,
                next_observation=np.zeros(3, dtype=np.float32),
                terminated=False,
            )

    def test_real_environment_dimensions_are_supported(self) -> None:
        """Verify compatibility with the actual MetaWorld observation/action size."""
        env = make_cw_env("hammer-v3", seed=0)

        try:
            observation, _ = env.reset(seed=0)

            buffer = ReplayBuffer(
                observation_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                capacity=10,
                seed=0,
            )

            action = env.action_space.sample()

            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            # Store only genuine termination. Time-limit truncation is
            # deliberately not converted into terminal=True.
            buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
            )

            self.assertEqual(len(buffer), 1)

            batch = buffer.sample(
                batch_size=1,
                device="cpu",
            )

            self.assertEqual(
                batch.observations.shape,
                (1, env.observation_space.shape[0]),
            )
            self.assertEqual(
                batch.actions.shape,
                (1, env.action_space.shape[0]),
            )

        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
