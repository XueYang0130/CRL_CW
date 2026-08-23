import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from agents import (
    FullBehaviorCloningSACAgent,
    OptimisticEnsembleFullBCAgent,
    PrioritizedNStepReplayBuffer,
    ReplayBuffer,
)
from methods import get_method
from scripts.run import parse_args
from utils import load_sac_checkpoint, save_sac_checkpoint


def make_observation(value: float, task_index: int = 1) -> np.ndarray:
    task_id = [0.0, 0.0]
    task_id[task_index] = 1.0
    return np.asarray([value, -value, 0.25, *task_id], dtype=np.float32)


def make_agent(seed: int = 0) -> OptimisticEnsembleFullBCAgent:
    torch.manual_seed(seed)
    return OptimisticEnsembleFullBCAgent(
        observation_dim=5,
        action_dim=2,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        num_tasks=2,
        task_id_dim=2,
        hide_task_id=True,
        device="cpu",
        gradient_clip_norm=0.1,
        episodic_batch_size=8,
        bc_gradient_strategy="pcgrad_sac_priority",
        bc_combination_strategy="adaptive_additive",
        critic_ensemble_size=4,
        optimistic_ucb_beta=0.5,
        optimistic_action_candidates=4,
        ensemble_seed=seed + 100,
    )


def make_base_agent(seed: int = 0) -> FullBehaviorCloningSACAgent:
    torch.manual_seed(seed)
    return FullBehaviorCloningSACAgent(
        observation_dim=5,
        action_dim=2,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        num_tasks=2,
        task_id_dim=2,
        hide_task_id=True,
        device="cpu",
        gradient_clip_norm=0.1,
        episodic_batch_size=8,
        bc_gradient_strategy="pcgrad_sac_priority",
        bc_combination_strategy="adaptive_additive",
    )


class PrioritizedNStepReplayTests(unittest.TestCase):
    def test_one_step_uniform_sampling_matches_standard_replay_exactly(self) -> None:
        standard = ReplayBuffer(
            observation_dim=2,
            action_dim=1,
            capacity=32,
            seed=17,
        )
        candidate = PrioritizedNStepReplayBuffer(
            observation_dim=2,
            action_dim=1,
            capacity=32,
            seed=17,
            gamma=0.99,
            n_step=1,
            prioritized_fraction=0.0,
        )
        for index in range(20):
            observation = np.asarray([index, -index], dtype=np.float32)
            action = np.asarray([index / 10.0], dtype=np.float32)
            next_observation = observation + 0.5
            transition = (
                observation,
                action,
                float(index),
                next_observation,
                bool(index % 7 == 0),
            )
            standard.add(*transition)
            candidate.add(*transition)

        standard_batch = standard.sample(batch_size=64, device="cpu")
        candidate_batch = candidate.sample(batch_size=64, device="cpu")
        for name in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminated",
        ):
            torch.testing.assert_close(
                getattr(candidate_batch, name),
                getattr(standard_batch, name),
                rtol=0.0,
                atol=0.0,
            )
        torch.testing.assert_close(
            candidate_batch.discounts,
            torch.full((64, 1), 0.99),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            candidate_batch.importance_weights,
            torch.ones((64, 1)),
            rtol=0.0,
            atol=0.0,
        )

    def test_exact_three_step_return_and_discount(self) -> None:
        buffer = PrioritizedNStepReplayBuffer(
            observation_dim=1,
            action_dim=1,
            capacity=10,
            seed=0,
            gamma=0.5,
            n_step=3,
            prioritized_fraction=0.0,
        )
        for index, reward in enumerate((1.0, 2.0, 4.0)):
            buffer.add(
                np.asarray([index], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                reward,
                np.asarray([index + 1], dtype=np.float32),
                False,
            )

        self.assertEqual(len(buffer), 1)
        batch = buffer.sample(1, "cpu")
        self.assertAlmostEqual(float(batch.rewards.item()), 3.0)
        self.assertAlmostEqual(float(batch.discounts.item()), 0.125)
        self.assertEqual(float(batch.next_observations.item()), 3.0)
        self.assertEqual(float(batch.terminated.item()), 0.0)

    def test_episode_flush_never_crosses_reset_boundary(self) -> None:
        buffer = PrioritizedNStepReplayBuffer(
            observation_dim=1,
            action_dim=1,
            capacity=20,
            seed=0,
            gamma=1.0,
            n_step=3,
        )
        for value in (1.0, 2.0):
            buffer.add(
                np.asarray([value], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                value,
                np.asarray([value + 0.5], dtype=np.float32),
                False,
            )
        buffer.end_episode()
        for value in (10.0, 20.0, 30.0):
            buffer.add(
                np.asarray([value], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                value,
                np.asarray([value + 0.5], dtype=np.float32),
                False,
            )

        stored = {
            float(buffer._observations[index, 0]): float(buffer._rewards[index, 0])
            for index in range(len(buffer))
        }
        self.assertEqual(stored[1.0], 3.0)
        self.assertEqual(stored[2.0], 2.0)
        self.assertEqual(stored[10.0], 60.0)

    def test_terminated_tail_has_no_bootstrap(self) -> None:
        buffer = PrioritizedNStepReplayBuffer(
            observation_dim=1,
            action_dim=1,
            capacity=10,
            gamma=0.9,
            n_step=3,
        )
        buffer.add(
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            1.0,
            np.asarray([1.0], dtype=np.float32),
            False,
        )
        buffer.add(
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            2.0,
            np.asarray([2.0], dtype=np.float32),
            True,
        )
        self.assertEqual(len(buffer), 2)
        np.testing.assert_array_equal(buffer._terminated[:2, 0], [1.0, 1.0])
        self.assertAlmostEqual(float(buffer._rewards[0, 0]), 2.8, places=6)

    def test_priority_updates_and_importance_weights_are_valid(self) -> None:
        buffer = PrioritizedNStepReplayBuffer(
            observation_dim=1,
            action_dim=1,
            capacity=20,
            seed=4,
            n_step=1,
            prioritized_fraction=1.0,
            priority_alpha=1.0,
        )
        for value in range(4):
            buffer.add(
                np.asarray([value], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
                0.0,
                np.asarray([value + 1], dtype=np.float32),
                False,
            )
        buffer.update_priorities(
            np.arange(4),
            np.asarray([1.0, 1.0, 1.0, 100.0]),
        )
        batch = buffer.sample(2_000, "cpu")
        high_fraction = float(np.mean(batch.indices == 3))
        self.assertGreater(high_fraction, 0.9)
        self.assertTrue(bool(torch.isfinite(batch.importance_weights).all()))
        self.assertGreater(float(batch.importance_weights.min()), 0.0)
        self.assertLessEqual(float(batch.importance_weights.max()), 1.0)


class OptimisticEnsembleAgentTests(unittest.TestCase):
    def test_live_cli_applies_candidate_defaults_and_explicit_override(self) -> None:
        argv = [
            "run.py",
            "--mode",
            "continual",
            "--method",
            "success_replay_mixed80_optimistic_nstep_pcgrad",
        ]
        with patch.object(sys, "argv", argv):
            defaults = parse_args()
        self.assertEqual(defaults.critic_ensemble_size, 4)
        self.assertEqual(defaults.critic_bootstrap_probability, 0.8)
        self.assertEqual(defaults.optimistic_ucb_beta, 0.5)
        self.assertEqual(defaults.optimistic_action_candidates, 8)
        self.assertEqual(defaults.n_step_return, 3)
        self.assertEqual(defaults.prioritized_replay_fraction, 0.2)

        with patch.object(
            sys,
            "argv",
            [*argv, "--optimistic-ucb-beta", "0.25"],
        ):
            overridden = parse_args()
        self.assertEqual(overridden.optimistic_ucb_beta, 0.25)
        self.assertEqual(overridden.n_step_return, 3)

    def test_base_twin_update_matches_baseline_when_replay_is_uniform_one_step(
        self,
    ) -> None:
        baseline = make_base_agent(seed=12)
        candidate = make_agent(seed=13)
        candidate.initialize_from_base_agent(baseline)
        observations = torch.as_tensor(
            np.stack([make_observation(index / 20.0) for index in range(16)]),
            dtype=torch.float32,
        )
        next_observations = observations.clone()
        next_observations[:, 0] += 0.01
        actions = torch.linspace(-0.5, 0.5, 32).reshape(16, 2)
        rewards = torch.linspace(-1.0, 1.0, 16).reshape(-1, 1)
        dones = torch.zeros((16, 1), dtype=torch.float32)

        torch.manual_seed(999)
        baseline.update_batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            collect_metrics=False,
        )
        torch.manual_seed(999)
        candidate._update_prioritized_batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            discounts=torch.full((16, 1), candidate.gamma),
            importance_weights=torch.ones((16, 1)),
            priority_beta=0.0,
            collect_metrics=False,
        )

        for baseline_module, candidate_module in (
            (baseline.actor, candidate.actor),
            (baseline.critic1, candidate.critic1),
            (baseline.critic2, candidate.critic2),
            (baseline.target_critic1, candidate.target_critic1),
            (baseline.target_critic2, candidate.target_critic2),
        ):
            for name, expected in baseline_module.state_dict().items():
                torch.testing.assert_close(
                    candidate_module.state_dict()[name],
                    expected,
                    rtol=0.0,
                    atol=1e-7,
                )
        torch.testing.assert_close(
            candidate.log_alpha,
            baseline.log_alpha,
            rtol=0.0,
            atol=1e-7,
        )

    def test_update_changes_auxiliary_critic_and_priorities(self) -> None:
        agent = make_agent()
        buffer = PrioritizedNStepReplayBuffer(
            observation_dim=5,
            action_dim=2,
            capacity=500,
            seed=3,
            gamma=agent.gamma,
            n_step=3,
        )
        for index in range(100):
            buffer.add(
                make_observation(index / 100.0),
                np.asarray([0.1, -0.2], dtype=np.float32),
                float(index % 7),
                make_observation((index + 1) / 100.0),
                False,
            )
        before = next(agent.extra_critics[0].parameters()).detach().clone()
        metrics = agent.update_from_replay_buffer(
            replay_buffer=buffer,
            batch_size=32,
            collect_metrics=True,
        )
        after = next(agent.extra_critics[0].parameters()).detach()

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(np.isfinite(metrics["ensemble_critic_loss"]))
        self.assertGreater(metrics["q_ensemble_std_mean"], 0.0)
        self.assertGreater(float(buffer._priorities[: len(buffer)].max()), 1.0)

    def test_ucb_is_training_only(self) -> None:
        agent = make_agent()
        observation = make_observation(0.0)
        agent.eval()
        action = agent.select_action(observation, deterministic=False)
        self.assertEqual(action.shape, (2,))
        self.assertEqual(agent._ucb_action_calls, 0)

        agent.train()
        action = agent.select_action(observation, deterministic=False)
        self.assertEqual(action.shape, (2,))
        self.assertEqual(agent._ucb_action_calls, 1)

        agent.on_task_start(task_index=1, replay_buffer=object())
        self.assertEqual(agent._ucb_action_calls, 0)
        self.assertEqual(agent._ucb_uncertainty_sum, 0.0)

    def test_checkpoint_round_trip_includes_auxiliary_critics(self) -> None:
        first = make_agent(seed=10)
        with torch.no_grad():
            next(first.extra_critics[0].parameters()).fill_(0.123)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pt"
            save_sac_checkpoint(agent=first, path=path, environment_step=7)
            second = make_agent(seed=20)
            info = load_sac_checkpoint(agent=second, path=path)
        self.assertTrue(info.optimizer_loaded)
        first_value = next(first.extra_critics[0].parameters()).detach()
        second_value = next(second.extra_critics[0].parameters()).detach()
        self.assertTrue(torch.equal(first_value, second_value))

    def test_method_specific_replay_does_not_change_existing_method(self) -> None:
        args = SimpleNamespace(
            replay_size=100,
            seed=2,
            gamma=0.99,
            steps_per_task=1000,
            priority_beta_steps=0,
            n_step_return=3,
            prioritized_replay_fraction=0.2,
            priority_alpha=0.6,
            priority_beta_start=0.4,
            priority_beta_end=1.0,
            priority_epsilon=1e-6,
        )
        baseline = get_method("success_replay_best_adaptive_pcgrad")
        candidate = get_method("success_replay_mixed80_optimistic_nstep_pcgrad")
        self.assertTrue(candidate.dynamic_broader_replay_fill)
        self.assertIsInstance(
            baseline.build_replay_buffer(
                args=args, observation_dim=5, action_dim=2
            ),
            ReplayBuffer,
        )
        self.assertIsInstance(
            candidate.build_replay_buffer(
                args=args, observation_dim=5, action_dim=2
            ),
            PrioritizedNStepReplayBuffer,
        )


if __name__ == "__main__":
    unittest.main()
