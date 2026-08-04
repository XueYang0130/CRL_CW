import unittest

import numpy as np
import torch

from agents import ClonExSACAgent, ReplayBuffer


class TestClonExAgent(unittest.TestCase):
    def test_task_start_captures_old_replay_before_buffer_clear(self) -> None:
        torch.manual_seed(3)
        agent = ClonExSACAgent(
            observation_dim=6,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=2,
            task_id_dim=2,
            hide_task_id=True,
            device="cpu",
            episodic_memory_per_task=6,
            episodic_batch_size=4,
            actor_cloning_coefficient=100.0,
        )
        replay = ReplayBuffer(
            observation_dim=6,
            action_dim=2,
            capacity=16,
            seed=7,
        )
        for index in range(4):
            observation = np.array(
                [index, 0.0, 0.0, 0.0, 1.0, 0.0],
                dtype=np.float32,
            )
            replay.add(
                observation=observation,
                action=np.zeros(2, dtype=np.float32),
                reward=0.0,
                next_observation=observation,
                terminated=False,
            )

        agent.on_task_start(task_index=1, replay_buffer=replay)
        replay.clear()

        self.assertEqual(agent.reference_state_count, 6)
        self.assertEqual(agent._max_reference_source_task_index, 0)
        current_observations = torch.randn(8, 6)
        current_observations[:, -2:] = torch.tensor([0.0, 1.0])
        agent.update_batch(
            observations=current_observations,
            actions=torch.rand(8, 2) * 2.0 - 1.0,
            rewards=torch.randn(8, 1),
            next_observations=current_observations.clone(),
            dones=torch.zeros(8, 1),
            collect_metrics=False,
        )
        self.assertTrue(
            all(torch.isfinite(parameter).all() for parameter in agent.parameters())
        )


if __name__ == "__main__":
    unittest.main()
