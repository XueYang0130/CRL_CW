import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agents import (
    FullBehaviorCloningSACAgent,
    KLBudgetFullBehaviorCloningSACAgent,
)
from methods import get_method
from scripts.run import parse_args


def make_agent(**overrides: float) -> KLBudgetFullBehaviorCloningSACAgent:
    kwargs = {
        "observation_dim": 6,
        "action_dim": 2,
        "action_low": np.full(2, -1.0, dtype=np.float32),
        "action_high": np.full(2, 1.0, dtype=np.float32),
        "num_tasks": 3,
        "task_id_dim": 3,
        "hide_task_id": True,
        "device": "cpu",
        "episodic_batch_size": 4,
        "actor_cloning_coefficient": 100.0,
        "bc_gradient_strategy": "pcgrad_sac_priority",
        "bc_max_norm_ratio": 1.0,
        "bc_combination_strategy": "adaptive_additive",
        "bc_adaptive_target_ratio": 0.2,
        "bc_adaptive_conflict_ratio": 0.05,
        "gradient_diagnostics": False,
        "kl_budget_low": 0.05,
        "kl_budget_high": 0.5,
        "kl_budget_ema_beta": 0.0,
    }
    kwargs.update(overrides)
    return KLBudgetFullBehaviorCloningSACAgent(**kwargs)


def observations_for_task(task_index: int, count: int) -> torch.Tensor:
    observations = torch.randn(count, 6)
    observations[:, -3:] = 0.0
    observations[:, 3 + task_index] = 1.0
    return observations


class KLBudgetAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        np.random.seed(11)

    def test_piecewise_budget_mapping_and_task_reset(self) -> None:
        agent = make_agent()
        agent._start_kl_budget_task(task_index=1, baseline_kl=0.25)
        self.assertEqual(agent._update_kl_budget(raw_kl=0.25, task_index=1), 0.0)
        self.assertAlmostEqual(
            agent._update_kl_budget(raw_kl=0.525, task_index=1), 0.5
        )
        self.assertEqual(agent._update_kl_budget(raw_kl=0.75, task_index=1), 1.0)
        self.assertEqual(agent._update_kl_budget(raw_kl=0.1, task_index=2), 0.0)
        self.assertAlmostEqual(agent.kl_budget_ema, 0.1)
        self.assertAlmostEqual(agent.kl_budget_baseline, 0.1)

    def test_zero_drift_leaves_sac_gradient_unchanged(self) -> None:
        agent = make_agent()
        memory = observations_for_task(0, 8)
        agent.add_reference_memory(observations=memory)
        parameters = tuple(agent.actor.parameters())
        sac_gradients = tuple(torch.randn_like(parameter) for parameter in parameters)

        np.random.seed(7)
        adjusted = agent.adjust_actor_gradients(
            gradients=sac_gradients,
            parameters=parameters,
            task_index=1,
        )

        self.assertEqual(agent.kl_budget_multiplier, 0.0)
        for expected, actual in zip(sac_gradients, adjusted, strict=True):
            torch.testing.assert_close(expected, actual)

    def test_checkpoint_parameters_match_original_agent(self) -> None:
        adaptive = FullBehaviorCloningSACAgent(
            observation_dim=6,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=3,
            task_id_dim=3,
            hide_task_id=True,
            device="cpu",
        )
        budget = make_agent()
        self.assertEqual(set(adaptive.state_dict()), set(budget.state_dict()))
        budget.load_state_dict(adaptive.state_dict(), strict=True)

    def test_rejects_invalid_budget_parameters(self) -> None:
        for overrides in (
            {"kl_budget_low": -0.1},
            {"kl_budget_high": 0.05},
            {"kl_budget_ema_beta": 1.0},
            {"kl_budget_ema_beta": float("nan")},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                make_agent(**overrides)

    def test_method_registry_builds_independent_agent(self) -> None:
        method = get_method("success_replay_best_kl_budget_pcgrad")
        self.assertEqual(method.success_replay_teacher, "best")
        self.assertEqual(
            method.agent_factory.__module__,
            "methods.success_replay_best_kl_budget_pcgrad",
        )

    def test_cli_applies_method_defaults_and_rejects_invalid_range(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--mode",
                "continual",
                "--method",
                "success_replay_best_kl_budget_pcgrad",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.kl_budget_low, 0.05)
        self.assertEqual(args.kl_budget_high, 0.5)
        self.assertEqual(args.kl_budget_ema_beta, 0.99)

        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--mode",
                "continual",
                "--method",
                "success_replay_best_kl_budget_pcgrad",
                "--kl-budget-low",
                "2.0",
                "--kl-budget-high",
                "1.0",
            ],
        ), self.assertRaises(SystemExit):
            parse_args()


if __name__ == "__main__":
    unittest.main()
