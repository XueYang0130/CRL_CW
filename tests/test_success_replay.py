import unittest

import numpy as np
import torch

from agents import FullBehaviorCloningSACAgent
from methods import bc_gradient_strategy_for_method, get_method
from training.continual_experiment import (
    add_relabelled_reference_memory,
    clone_module_state,
)
from training.success_replay import SuccessfulStateReservoir


class SuccessfulStateReservoirTests(unittest.TestCase):
    def test_ignores_failed_episodes(self) -> None:
        memory = SuccessfulStateReservoir(capacity=3, observation_dim=2, seed=7)
        memory.add_episode((np.array([1.0, 2.0], dtype=np.float32),), False)
        self.assertEqual(memory.successful_episodes, 0)
        self.assertEqual(memory.seen_successful_states, 0)
        self.assertEqual(memory.observations().shape, (0, 2))

    def test_capacity_and_seed_are_deterministic(self) -> None:
        episodes = tuple(
            np.array([float(index), float(index + 1)], dtype=np.float32)
            for index in range(20)
        )
        first = SuccessfulStateReservoir(capacity=5, observation_dim=2, seed=11)
        second = SuccessfulStateReservoir(capacity=5, observation_dim=2, seed=11)
        first.add_episode(episodes, True)
        second.add_episode(episodes, True)
        self.assertEqual(first.successful_episodes, 1)
        self.assertEqual(first.seen_successful_states, 20)
        self.assertEqual(first.observations().shape, (5, 2))
        np.testing.assert_array_equal(first.observations(), second.observations())

    def test_rejects_invalid_observation_shape(self) -> None:
        memory = SuccessfulStateReservoir(capacity=2, observation_dim=3, seed=0)
        with self.assertRaisesRegex(ValueError, "expected"):
            memory.add_episode((np.zeros(2, dtype=np.float32),), True)


class SuccessfulReplayMethodTests(unittest.TestCase):
    def test_teacher_and_gradient_strategy_presets(self) -> None:
        expected = {
            "success_replay_final": ("final", "standard"),
            "success_replay_best": ("best", "standard"),
            "success_replay_best_pcgrad": ("best", "pcgrad_sac_priority"),
            "success_replay_best_adaptive_pcgrad": (
                "best",
                "pcgrad_sac_priority",
            ),
        }
        for method_id, (teacher, strategy) in expected.items():
            with self.subTest(method=method_id):
                method = get_method(method_id)
                self.assertEqual(method.success_replay_teacher, teacher)
                self.assertEqual(
                    bc_gradient_strategy_for_method(method_id),
                    strategy,
                )
                self.assertEqual(method.defaults["episodic_memory_per_task"], 10_000)
                if method_id == "success_replay_best_adaptive_pcgrad":
                    self.assertEqual(
                        method.defaults["bc_combination_strategy"],
                        "adaptive_additive",
                    )

    def test_teacher_relabel_restores_live_actor(self) -> None:
        torch.manual_seed(5)
        agent = FullBehaviorCloningSACAgent(
            observation_dim=5,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=2,
            task_id_dim=2,
            hide_task_id=True,
            device="cpu",
            episodic_batch_size=2,
        )
        observations = np.zeros((3, 5), dtype=np.float32)
        observations[:, -2] = 1.0
        teacher_state = clone_module_state(agent.actor)
        teacher_means, teacher_log_stds = agent.compute_reference_targets(observations)
        with torch.no_grad():
            for parameter in agent.actor.parameters():
                parameter.add_(0.05)
        live_state = clone_module_state(agent.actor)

        add_relabelled_reference_memory(
            agent=agent,
            observations=observations,
            teacher_actor_state=teacher_state,
        )

        for name, value in agent.actor.state_dict().items():
            torch.testing.assert_close(value.cpu(), live_state[name])
        torch.testing.assert_close(
            agent._episodic_target_means,
            teacher_means.cpu(),
        )
        torch.testing.assert_close(
            agent._episodic_target_log_stds,
            teacher_log_stds.cpu(),
        )


if __name__ == "__main__":
    unittest.main()
