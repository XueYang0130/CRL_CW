"""Smoke tests for the modern MetaWorld CW10 compatibility layer."""

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crl_cw.envs import (
    CLONEX_EPISODE_LENGTH,
    CW10_TASKS,
    extract_success,
    make_cw_env,
    modern_task_name,
)


class TestCWEnvironment(unittest.TestCase):
    def test_all_cw10_names_have_modern_mapping(self) -> None:
        for task_name in CW10_TASKS:
            self.assertTrue(modern_task_name(task_name).endswith("-v3"))

    def test_unknown_task_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            modern_task_name("unknown-task-v1")

    def test_success_extraction(self) -> None:
        self.assertEqual(extract_success({}), 0.0)
        self.assertEqual(extract_success({"success": 0.0}), 0.0)
        self.assertEqual(extract_success({"success": 1.0}), 1.0)
        self.assertEqual(extract_success({"success": np.array([1.0])}), 1.0)

    def test_hammer_reset_and_step(self) -> None:
        env = make_cw_env("hammer-v1", seed=0)

        try:
            observation, info = env.reset(seed=0)

            self.assertIsInstance(observation, np.ndarray)
            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertIsInstance(info, dict)
            self.assertEqual(
                env.spec.max_episode_steps,
                CLONEX_EPISODE_LENGTH,
            )

            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, step_info = env.step(
                action
            )

            self.assertEqual(
                next_observation.shape,
                env.observation_space.shape,
            )
            self.assertTrue(np.isfinite(float(reward)))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(step_info, dict)
            self.assertIn(extract_success(step_info), (0.0, 1.0))

        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()