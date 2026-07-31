import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from envs import CW10_NUM_TASKS, DEFAULT_EPISODE_LENGTH, CW10_TASKS, extract_success, make_cw_env


class TestCWEnvironment(unittest.TestCase):
    def test_success_extraction(self) -> None:
        self.assertEqual(extract_success({}), 0.0)
        self.assertEqual(extract_success({"success": 0.0}), 0.0)
        self.assertEqual(extract_success({"success": 1.0}), 1.0)
        self.assertEqual(extract_success({"success": np.array([1.0])}), 1.0)

    def test_hammer_reset_and_step(self) -> None:
        env = make_cw_env("hammer-v3", seed=0)
        try:
            observation, info = env.reset(seed=0)
            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertEqual(env.spec.max_episode_steps, DEFAULT_EPISODE_LENGTH)
            self.assertIsInstance(info, dict)

            next_observation, reward, terminated, truncated, step_info = env.step(
                env.action_space.sample()
            )
            self.assertEqual(next_observation.shape, env.observation_space.shape)
            self.assertTrue(np.isfinite(float(reward)))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(step_info, dict)
        finally:
            env.close()

    def test_task_id_appending(self) -> None:
        base_env = make_cw_env("hammer-v3", seed=0, append_task_id=False)
        task_env = make_cw_env("hammer-v3", seed=0, append_task_id=True)
        try:
            base_obs, _ = base_env.reset(seed=0)
            task_obs, _ = task_env.reset(seed=0)
            self.assertEqual(task_obs.shape[0], base_obs.shape[0] + CW10_NUM_TASKS)
        finally:
            base_env.close()
            task_env.close()

    def test_task_id_dimension_can_match_a_cw5_prefix(self) -> None:
        env = make_cw_env(
            "stick-pull-v3",
            seed=0,
            append_task_id=True,
            num_task_ids=5,
        )
        try:
            observation, _ = env.reset(seed=0)
            self.assertEqual(observation[-5:].tolist(), [0.0, 0.0, 0.0, 0.0, 1.0])
        finally:
            env.close()

    def test_stick_pull_v1_compatible_reward_env(self) -> None:
        env = make_cw_env("stick-pull-v3", seed=0, reward_function_version="v1_compatible")
        try:
            observation, info = env.reset(seed=0)
            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertIsInstance(info, dict)

            next_observation, reward, terminated, truncated, step_info = env.step(
                env.action_space.sample()
            )
            self.assertEqual(next_observation.shape, env.observation_space.shape)
            self.assertTrue(np.isfinite(float(reward)))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(step_info, dict)
        finally:
            env.close()

    def test_known_task_names(self) -> None:
        self.assertEqual(len(CW10_TASKS), 10)


if __name__ == "__main__":
    unittest.main()
