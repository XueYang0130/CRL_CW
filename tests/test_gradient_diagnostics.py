import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from agents import FullBehaviorCloningSACAgent
from utils import load_sac_checkpoint, save_sac_checkpoint


def make_agent(
    *,
    diagnostics: bool,
    gradient_strategy: str = "standard",
    diagnostics_interval: int = 1,
    combination_strategy: str = "average",
    adaptive_target_ratio: float = 0.2,
    adaptive_conflict_ratio: float = 0.05,
    cagrad_alpha: float = 0.5,
    bc_update_interval: int = 1,
) -> FullBehaviorCloningSACAgent:
    return FullBehaviorCloningSACAgent(
        observation_dim=6,
        action_dim=2,
        action_low=np.full(2, -1.0, dtype=np.float32),
        action_high=np.full(2, 1.0, dtype=np.float32),
        num_tasks=3,
        task_id_dim=3,
        hide_task_id=True,
        device="cpu",
        episodic_batch_size=4,
        actor_cloning_coefficient=10.0,
        bc_gradient_strategy=gradient_strategy,
        bc_max_norm_ratio=1.0,
        bc_combination_strategy=combination_strategy,
        bc_adaptive_target_ratio=adaptive_target_ratio,
        bc_adaptive_conflict_ratio=adaptive_conflict_ratio,
        bc_cagrad_alpha=cagrad_alpha,
        bc_update_interval=bc_update_interval,
        gradient_diagnostics=diagnostics,
        gradient_diagnostics_interval=diagnostics_interval,
        gradient_diagnostics_source_batch_size=4,
        gradient_diagnostics_seed=123,
    )


def observations_for_task(task_index: int, count: int) -> torch.Tensor:
    observations = torch.randn(count, 6)
    observations[:, -3:] = 0.0
    observations[:, 3 + task_index] = 1.0
    return observations


class TestGradientDiagnostics(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        np.random.seed(7)

    def add_task_zero_memory(self, agent: FullBehaviorCloningSACAgent) -> None:
        observations = observations_for_task(0, 8)
        agent.add_reference_memory(observations=observations)

    def test_rejects_nonfinite_bc_hyperparameters(self) -> None:
        for field, value in (
            ("actor_cloning_coefficient", float("nan")),
            ("actor_cloning_coefficient", float("inf")),
            ("bc_max_norm_ratio", float("nan")),
            ("bc_max_norm_ratio", float("inf")),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                kwargs = {field: value}
                FullBehaviorCloningSACAgent(
                    observation_dim=6,
                    action_dim=2,
                    action_low=np.full(2, -1.0, dtype=np.float32),
                    action_high=np.full(2, 1.0, dtype=np.float32),
                    num_tasks=3,
                    task_id_dim=3,
                    hide_task_id=True,
                    device="cpu",
                    **kwargs,
                )

    def update_task_one(
        self,
        agent: FullBehaviorCloningSACAgent,
        *,
        collect_metrics: bool = True,
    ) -> None:
        observations = observations_for_task(1, 8)
        agent.update_batch(
            observations=observations,
            actions=torch.rand(8, 2) * 2.0 - 1.0,
            rewards=torch.randn(8, 1),
            next_observations=observations_for_task(1, 8),
            dones=torch.zeros(8, 1),
            collect_metrics=collect_metrics,
        )

    def test_records_global_layer_and_source_task_rows(self) -> None:
        agent = make_agent(diagnostics=True)
        self.add_task_zero_memory(agent)
        self.update_task_one(agent)

        rows = agent.drain_gradient_diagnostics()
        self.assertEqual(len(rows["windows"]), 2)
        self.assertEqual(
            {row["scope"] for row in rows["windows"]},
            {"shared_actor_backbone", "full_actor"},
        )
        self.assertGreaterEqual(len(rows["layers"]), 4)
        self.assertEqual(len(rows["task_pairs"]), 1)
        self.assertEqual(rows["task_pairs"][0]["source_task_index"], 0)
        self.assertIn("cosine_similarity", rows["windows"][0])
        self.assertIn("conflict_mass", rows["windows"][0])

    def test_source_task_rows_follow_sparse_diagnostic_interval(self) -> None:
        agent = make_agent(diagnostics=True, diagnostics_interval=2)
        self.add_task_zero_memory(agent)

        self.update_task_one(agent, collect_metrics=True)
        first_rows = agent.drain_gradient_diagnostics()
        self.assertEqual(first_rows["windows"], [])
        self.assertEqual(first_rows["layers"], [])
        self.assertEqual(first_rows["task_pairs"], [])

        self.update_task_one(agent, collect_metrics=False)
        second_rows = agent.drain_gradient_diagnostics()
        self.assertEqual(len(second_rows["windows"]), 2)
        self.assertGreaterEqual(len(second_rows["layers"]), 4)
        self.assertEqual(len(second_rows["task_pairs"]), 1)

    def test_diagnostics_do_not_change_parameter_update(self) -> None:
        control = make_agent(diagnostics=False)
        diagnosed = make_agent(diagnostics=True)
        diagnosed.load_state_dict(copy.deepcopy(control.state_dict()))

        memory = observations_for_task(0, 8)
        control.add_reference_memory(observations=memory)
        diagnosed.add_reference_memory(observations=memory)

        observations = observations_for_task(1, 8)
        next_observations = observations_for_task(1, 8)
        actions = torch.rand(8, 2) * 2.0 - 1.0
        rewards = torch.randn(8, 1)
        dones = torch.zeros(8, 1)

        np.random.seed(11)
        torch.manual_seed(11)
        control.update_batch(
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            collect_metrics=True,
        )
        np.random.seed(11)
        torch.manual_seed(11)
        diagnosed.update_batch(
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            collect_metrics=True,
        )

        for control_parameter, diagnosed_parameter in zip(
            control.parameters(), diagnosed.parameters(), strict=True
        ):
            torch.testing.assert_close(control_parameter, diagnosed_parameter)

    def test_rejects_current_or_future_task_memory(self) -> None:
        agent = make_agent(diagnostics=True)
        agent.add_reference_memory(observations=observations_for_task(1, 8))
        with self.assertRaisesRegex(RuntimeError, "current or future task data"):
            self.update_task_one(agent)

    def test_rejects_malformed_task_vector_that_only_sums_to_one(self) -> None:
        agent = make_agent(diagnostics=False)
        observations = observations_for_task(0, 8)
        observations[0, -3:] = torch.tensor([1.1, -0.1, 0.0])
        with self.assertRaisesRegex(ValueError, "valid one-hot task IDs"):
            agent.add_reference_memory(observations=observations)

    def test_rejects_nonfinite_reference_memory(self) -> None:
        agent = make_agent(diagnostics=False)
        observations = observations_for_task(0, 8)
        observations[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "observations must be finite"):
            agent.add_reference_memory(observations=observations)

    def test_reference_source_cache_tracks_add_and_clear(self) -> None:
        agent = make_agent(diagnostics=False)
        agent.add_reference_memory(observations=observations_for_task(0, 8))
        self.assertEqual(agent._max_reference_source_task_index, 0)
        agent.add_reference_memory(observations=observations_for_task(1, 8))
        self.assertEqual(agent._max_reference_source_task_index, 1)
        agent.clear_reference_memory()
        self.assertEqual(agent._max_reference_source_task_index, -1)

    def test_valid_bc_update_does_not_rescan_all_reference_sources(self) -> None:
        agent = make_agent(diagnostics=False)
        self.add_task_zero_memory(agent)
        with patch(
            "agents.full_bc_agent.torch.unique",
            side_effect=AssertionError("unexpected full-memory source scan"),
        ):
            self.update_task_one(agent, collect_metrics=False)

    def test_episodic_batch_matches_reference_sampling_without_replacement(self) -> None:
        agent = make_agent(diagnostics=False)
        observations = observations_for_task(0, 3)
        agent.add_reference_memory(observations=observations)

        with patch(
            "agents.full_bc_agent.np.random.choice",
            wraps=np.random.choice,
        ) as choice:
            sampled_observations, _, _ = agent._sample_episodic_batch()

        self.assertEqual(sampled_observations.shape[0], 3)
        self.assertEqual(choice.call_args.kwargs["size"], 3)
        self.assertFalse(choice.call_args.kwargs["replace"])

    def test_reference_memory_accumulates_across_task_lifecycle(self) -> None:
        agent = make_agent(diagnostics=False)
        task_zero_memory = observations_for_task(0, 8)
        task_one_memory = observations_for_task(1, 6)

        agent.add_reference_memory(observations=task_zero_memory)
        self.update_task_one(agent, collect_metrics=False)
        agent.add_reference_memory(observations=task_one_memory)

        self.assertEqual(agent.reference_state_count, 14)
        self.assertEqual(agent._max_reference_source_task_index, 1)

        task_two_observations = observations_for_task(2, 8)
        agent.update_batch(
            observations=task_two_observations,
            actions=torch.rand(8, 2) * 2.0 - 1.0,
            rewards=torch.randn(8, 1),
            next_observations=observations_for_task(2, 8),
            dones=torch.zeros(8, 1),
            collect_metrics=False,
        )

        with self.assertRaisesRegex(RuntimeError, "current or future task data"):
            self.update_task_one(agent, collect_metrics=False)

    def test_full_bc_checkpoint_restores_policy_exactly(self) -> None:
        source = make_agent(
            diagnostics=True,
            gradient_strategy="pcgrad_sac_priority",
        )
        self.add_task_zero_memory(source)
        self.update_task_one(source, collect_metrics=False)
        probe = observations_for_task(2, 5)
        with torch.no_grad():
            expected = source.actor.distribution_parameters(probe)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full_bc.pt"
            save_sac_checkpoint(
                agent=source,
                path=path,
                environment_step=123,
                metadata={
                    "method": "full_bc_pcgrad",
                    "bc_gradient_strategy": "pcgrad_sac_priority",
                },
            )
            restored = make_agent(
                diagnostics=True,
                gradient_strategy="pcgrad_sac_priority",
            )
            info = load_sac_checkpoint(
                agent=restored,
                path=path,
                map_location="cpu",
                load_optimizer=True,
            )
            with torch.no_grad():
                actual = restored.actor.distribution_parameters(probe)

        self.assertEqual(info.environment_step, 123)
        self.assertEqual(info.metadata["method"], "full_bc_pcgrad")
        self.assertEqual(
            info.metadata["bc_gradient_strategy"],
            "pcgrad_sac_priority",
        )
        for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
            torch.testing.assert_close(
                expected_tensor,
                actual_tensor,
                rtol=0.0,
                atol=0.0,
            )

    def conflicting_gradients(
        self,
        agent: FullBehaviorCloningSACAgent,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        sac_gradients = tuple(
            torch.zeros_like(parameter) for parameter in agent.actor.parameters()
        )
        bc_gradients = tuple(
            torch.zeros_like(parameter) for parameter in agent.actor.parameters()
        )
        sac_gradients = list(sac_gradients)
        bc_gradients = list(bc_gradients)
        for index in agent.gradient_diagnostics.shared_indices:
            sac_gradients[index].fill_(1.0)
            bc_gradients[index].fill_(-2.0)
        for index in set(range(len(bc_gradients))).difference(
            agent.gradient_diagnostics.shared_indices
        ):
            bc_gradients[index].fill_(3.0)
        return tuple(sac_gradients), tuple(bc_gradients)

    def test_norm_balancing_caps_shared_bc_norm_without_projection(self) -> None:
        agent = make_agent(
            diagnostics=True,
            gradient_strategy="norm_balanced",
        )
        sac_gradients, bc_gradients = self.conflicting_gradients(agent)
        adjusted, metadata = agent._apply_bc_gradient_strategy(
            sac_gradients=sac_gradients,
            cloning_gradients=bc_gradients,
        )
        shared = agent.gradient_diagnostics.shared_indices
        sac_norm = sum(sac_gradients[i].square().sum() for i in shared).sqrt()
        adjusted_norm = sum(adjusted[i].square().sum() for i in shared).sqrt()
        dot = sum((sac_gradients[i] * adjusted[i]).sum() for i in shared)
        self.assertLessEqual(float(adjusted_norm), float(sac_norm) + 1e-5)
        self.assertLess(float(dot), 0.0)
        self.assertFalse(metadata["projection_applied"])
        self.assertAlmostEqual(float(metadata["bc_norm_scale"]), 0.5, places=5)

    def test_pcgrad_projects_only_shared_conflict_then_applies_common_scale(self) -> None:
        agent = make_agent(
            diagnostics=True,
            gradient_strategy="pcgrad_sac_priority",
        )
        sac_gradients, bc_gradients = self.conflicting_gradients(agent)
        adjusted, metadata = agent._apply_bc_gradient_strategy(
            sac_gradients=sac_gradients,
            cloning_gradients=bc_gradients,
        )
        shared = agent.gradient_diagnostics.shared_indices
        dot = sum((sac_gradients[i] * adjusted[i]).sum() for i in shared)
        self.assertGreaterEqual(float(dot), -1e-5)
        self.assertTrue(metadata["projection_applied"])
        self.assertAlmostEqual(float(metadata["bc_norm_scale"]), 1.0, places=5)

        nonshared = next(
            index for index in range(len(adjusted)) if index not in shared
        )
        torch.testing.assert_close(
            adjusted[nonshared],
            bc_gradients[nonshared],
        )

    def test_bc_priority_pcgrad_projects_only_shared_sac_gradient(self) -> None:
        agent = make_agent(
            diagnostics=True,
            gradient_strategy="pcgrad_bc_priority",
        )
        sac_gradients, bc_gradients = self.conflicting_gradients(agent)
        adjusted, metadata = agent._apply_sac_gradient_strategy(
            sac_gradients=sac_gradients,
            cloning_gradients=bc_gradients,
        )
        shared = agent.gradient_diagnostics.shared_indices
        dot = sum((adjusted[i] * bc_gradients[i]).sum() for i in shared)
        self.assertGreaterEqual(float(dot), -1e-5)
        self.assertTrue(metadata["projection_applied"])
        self.assertEqual(metadata["bc_norm_scale"], 1.0)

        nonshared = next(
            index for index in range(len(adjusted)) if index not in shared
        )
        torch.testing.assert_close(adjusted[nonshared], sac_gradients[nonshared])

        combined, scale = agent._combine_actor_gradients(
            sac_gradients=adjusted,
            applied_cloning_gradients=bc_gradients,
            conflict=True,
        )
        self.assertEqual(scale, 1.0)
        for sac, bc, final in zip(adjusted, bc_gradients, combined, strict=True):
            torch.testing.assert_close(final, (sac + bc) / 2.0)

        agent.gradient_diagnostics.record(
            sac_gradients=sac_gradients,
            bc_gradients=bc_gradients,
            current_task_index=1,
            raw_bc_loss=1.0,
            bc_coefficient=100.0,
            sac_actor_loss=1.0,
            reference_memory_states=10,
            applied_bc_gradients=bc_gradients,
            final_actor_gradients=combined,
            gradient_strategy="pcgrad_bc_priority",
            projection_applied=True,
            bc_norm_scale=1.0,
            bc_combination_strategy="average",
            bc_combination_scale=1.0,
            applied_sac_gradients=adjusted,
        )
        shared_row = agent.gradient_diagnostics.drain()["windows"][0]
        self.assertEqual(shared_row["projection_target"], "sac")
        self.assertGreaterEqual(shared_row["applied_cosine_similarity"], -1e-5)
        self.assertLess(
            shared_row["applied_sac_to_raw_sac_norm_ratio"],
            1.0,
        )

    def test_parallel_norm_is_nonnegative_for_conflicting_gradients(self) -> None:
        agent = make_agent(diagnostics=True)
        sac_gradients = [
            torch.zeros_like(parameter) for parameter in agent.actor.parameters()
        ]
        bc_gradients = [
            torch.zeros_like(parameter) for parameter in agent.actor.parameters()
        ]
        shared_index = agent.gradient_diagnostics.shared_indices[0]
        sac_gradients[shared_index].reshape(-1)[0] = 1.0
        bc_flat = bc_gradients[shared_index].reshape(-1)
        bc_flat[0] = -1.0
        bc_flat[1] = 2.0
        sac_gradients = tuple(sac_gradients)
        bc_gradients = tuple(bc_gradients)
        agent.gradient_diagnostics.record(
            sac_gradients=sac_gradients,
            bc_gradients=bc_gradients,
            current_task_index=1,
            raw_bc_loss=1.0,
            bc_coefficient=10.0,
            sac_actor_loss=1.0,
            reference_memory_states=8,
            applied_bc_gradients=bc_gradients,
            final_actor_gradients=tuple(
                (sac + bc) / 2.0
                for sac, bc in zip(sac_gradients, bc_gradients, strict=True)
            ),
            gradient_strategy="standard",
            projection_applied=False,
            bc_norm_scale=1.0,
            bc_combination_strategy="average",
            bc_combination_scale=1.0,
        )
        rows = agent.drain_gradient_diagnostics()["windows"]
        self.assertTrue(rows)
        self.assertTrue(all(float(row["bc_parallel_norm"]) >= 0.0 for row in rows))

    def test_pcgrad_caps_the_projected_shared_gradient(self) -> None:
        agent = make_agent(
            diagnostics=True,
            gradient_strategy="pcgrad_sac_priority",
        )
        sac_gradients = [
            torch.zeros_like(parameter) for parameter in agent.actor.parameters()
        ]
        bc_gradients = [
            torch.zeros_like(parameter) for parameter in agent.actor.parameters()
        ]
        shared_index = agent.gradient_diagnostics.shared_indices[0]
        sac_flat = sac_gradients[shared_index].reshape(-1)
        bc_flat = bc_gradients[shared_index].reshape(-1)
        sac_flat[0] = 1.0
        bc_flat[0] = -1.0
        bc_flat[1] = 2.0

        adjusted, metadata = agent._apply_bc_gradient_strategy(
            sac_gradients=tuple(sac_gradients),
            cloning_gradients=tuple(bc_gradients),
        )
        shared = agent.gradient_diagnostics.shared_indices
        sac_norm = sum(sac_gradients[i].square().sum() for i in shared).sqrt()
        applied_norm = sum(adjusted[i].square().sum() for i in shared).sqrt()

        self.assertTrue(metadata["projection_applied"])
        self.assertAlmostEqual(float(metadata["bc_norm_scale"]), 0.5, places=6)
        self.assertLessEqual(float(applied_norm), float(sac_norm) + 1e-6)
        self.assertAlmostEqual(float(applied_norm), 1.0, places=6)

    def test_balancing_suppresses_bc_when_shared_sac_gradient_is_zero(self) -> None:
        for strategy in ("norm_balanced", "pcgrad_sac_priority"):
            with self.subTest(strategy=strategy):
                agent = make_agent(
                    diagnostics=True,
                    gradient_strategy=strategy,
                )
                sac_gradients = tuple(
                    torch.zeros_like(parameter)
                    for parameter in agent.actor.parameters()
                )
                bc_gradients = tuple(
                    torch.ones_like(parameter)
                    for parameter in agent.actor.parameters()
                )
                adjusted, metadata = agent._apply_bc_gradient_strategy(
                    sac_gradients=sac_gradients,
                    cloning_gradients=bc_gradients,
                )
                for gradient in adjusted:
                    torch.testing.assert_close(
                        gradient,
                        torch.zeros_like(gradient),
                    )
                self.assertFalse(metadata["projection_applied"])
                self.assertEqual(metadata["bc_norm_scale"], 0.0)

    def test_pcgrad_update_records_raw_and_applied_gradient_geometry(self) -> None:
        agent = make_agent(
            diagnostics=True,
            gradient_strategy="pcgrad_sac_priority",
        )
        self.add_task_zero_memory(agent)
        self.update_task_one(agent)
        shared_row = next(
            row
            for row in agent.drain_gradient_diagnostics()["windows"]
            if row["scope"] == "shared_actor_backbone"
        )
        self.assertEqual(
            shared_row["gradient_strategy"],
            "pcgrad_sac_priority",
        )
        self.assertLessEqual(float(shared_row["bc_norm_scale"]), 1.0)
        self.assertLessEqual(
            float(shared_row["applied_bc_to_sac_norm_ratio"]),
            1.0 + 1e-5,
        )

    def test_standard_strategy_is_exactly_the_original_gradient_average(self) -> None:
        agent = make_agent(diagnostics=False, gradient_strategy="standard")
        sac_gradients, bc_gradients = self.conflicting_gradients(agent)
        adjusted, metadata = agent._apply_bc_gradient_strategy(
            sac_gradients=sac_gradients,
            cloning_gradients=bc_gradients,
        )
        for original, returned in zip(bc_gradients, adjusted, strict=True):
            self.assertIs(original, returned)
        expected = tuple(
            (sac + bc) / 2.0
            for sac, bc in zip(sac_gradients, bc_gradients, strict=True)
        )
        actual = tuple(
            (sac + bc) / 2.0
            for sac, bc in zip(sac_gradients, adjusted, strict=True)
        )
        for expected_gradient, actual_gradient in zip(
            expected, actual, strict=True
        ):
            torch.testing.assert_close(expected_gradient, actual_gradient)
        self.assertEqual(metadata["bc_norm_scale"], 1.0)

    def test_adaptive_additive_keeps_sac_dominant_under_conflict(self) -> None:
        agent = make_agent(
            diagnostics=False,
            gradient_strategy="pcgrad_sac_priority",
            combination_strategy="adaptive_additive",
            adaptive_target_ratio=0.2,
            adaptive_conflict_ratio=0.05,
        )
        sac_gradients = [
            torch.zeros_like(parameter) for parameter in agent.actor.parameters()
        ]
        bc_gradients = [
            torch.zeros_like(parameter) for parameter in agent.actor.parameters()
        ]
        shared_index = agent.gradient_diagnostics.shared_indices[0]
        sac_gradients[shared_index].reshape(-1)[0] = 1.0
        bc_flat = bc_gradients[shared_index].reshape(-1)
        bc_flat[0] = -1.0
        bc_flat[1] = 2.0
        sac_gradients = tuple(sac_gradients)
        bc_gradients = tuple(bc_gradients)
        adjusted, metadata = agent._apply_bc_gradient_strategy(
            sac_gradients=sac_gradients,
            cloning_gradients=bc_gradients,
        )
        actual, scale = agent._combine_actor_gradients(
            sac_gradients=sac_gradients,
            applied_cloning_gradients=adjusted,
            conflict=bool(metadata["conflict"]),
        )
        shared = agent.gradient_diagnostics.shared_indices
        sac_norm = sum(sac_gradients[i].square().sum() for i in shared).sqrt()
        applied_norm = sum(adjusted[i].square().sum() for i in shared).sqrt()
        expected_scale = 0.05 * sac_norm / applied_norm
        self.assertTrue(metadata["projection_applied"])
        self.assertAlmostEqual(float(scale), float(expected_scale), places=5)
        for sac, bc, final in zip(sac_gradients, adjusted, actual, strict=True):
            torch.testing.assert_close(final, sac + scale * bc)

    def test_adaptive_scaling_caps_conflict_without_projection(self) -> None:
        agent = make_agent(
            diagnostics=False,
            gradient_strategy="standard",
            combination_strategy="adaptive_additive",
            adaptive_target_ratio=0.2,
            adaptive_conflict_ratio=0.05,
        )
        sac_gradients, bc_gradients = self.conflicting_gradients(agent)
        adjusted, metadata = agent._apply_bc_gradient_strategy(
            sac_gradients=sac_gradients,
            cloning_gradients=bc_gradients,
        )
        actual, scale = agent._combine_actor_gradients(
            sac_gradients=sac_gradients,
            applied_cloning_gradients=adjusted,
            conflict=bool(metadata["conflict"]),
        )
        shared = agent.gradient_diagnostics.shared_indices
        sac_norm = sum(sac_gradients[i].square().sum() for i in shared).sqrt()
        bc_norm = sum(bc_gradients[i].square().sum() for i in shared).sqrt()
        expected_scale = 0.05 * sac_norm / bc_norm

        self.assertTrue(metadata["conflict"])
        self.assertFalse(metadata["projection_applied"])
        self.assertEqual(metadata["bc_norm_scale"], 1.0)
        self.assertAlmostEqual(float(scale), float(expected_scale), places=5)
        for original, returned in zip(bc_gradients, adjusted, strict=True):
            self.assertIs(original, returned)
        for sac, bc, final in zip(sac_gradients, bc_gradients, actual, strict=True):
            torch.testing.assert_close(final, sac + scale * bc)

    def test_standard_update_uses_original_weighted_loss_gradient_order(self) -> None:
        agent = make_agent(diagnostics=False, gradient_strategy="standard")
        self.add_task_zero_memory(agent)
        parameters = tuple(agent.actor.parameters())
        sac_gradients = tuple(torch.randn_like(parameter) for parameter in parameters)

        np.random.seed(91)
        observations, target_means, target_log_stds = agent._sample_episodic_batch()
        current_means, current_log_stds = agent._cloning_distribution_parameters(
            observations
        )
        weighted_loss = agent.actor_cloning_coefficient * agent._gaussian_kl(
            target_means,
            target_log_stds,
            current_means,
            current_log_stds,
        ).mean()
        original_bc_gradients = torch.autograd.grad(weighted_loss, parameters)
        expected = tuple(
            (sac + bc) / 2.0
            for sac, bc in zip(sac_gradients, original_bc_gradients, strict=True)
        )

        np.random.seed(91)
        actual = agent.adjust_actor_gradients(
            gradients=sac_gradients,
            parameters=parameters,
            task_index=1,
        )
        for expected_gradient, actual_gradient in zip(
            expected,
            actual,
            strict=True,
        ):
            torch.testing.assert_close(
                expected_gradient,
                actual_gradient,
                rtol=0.0,
                atol=0.0,
            )

    def test_sparse_bc_applies_only_on_configured_actor_updates(self) -> None:
        agent = make_agent(
            diagnostics=False,
            gradient_strategy="standard",
            bc_update_interval=4,
        )
        self.add_task_zero_memory(agent)
        parameters = tuple(agent.actor.parameters())
        sac_gradients = tuple(torch.randn_like(parameter) for parameter in parameters)

        for _ in range(3):
            actual = agent.adjust_actor_gradients(
                gradients=sac_gradients,
                parameters=parameters,
                task_index=1,
            )
            for expected_gradient, actual_gradient in zip(
                sac_gradients, actual, strict=True
            ):
                torch.testing.assert_close(actual_gradient, expected_gradient)

        fourth = agent.adjust_actor_gradients(
            gradients=sac_gradients,
            parameters=parameters,
            task_index=1,
        )
        self.assertEqual(agent.bc_update_opportunities, 4)
        self.assertEqual(agent.bc_updates_applied, 1)
        self.assertTrue(
            any(
                not torch.equal(actual_gradient, expected_gradient)
                for actual_gradient, expected_gradient in zip(
                    fourth, sac_gradients, strict=True
                )
            )
        )

        agent.on_task_start(task_index=2, replay_buffer=object())
        self.assertEqual(agent.bc_update_opportunities, 0)
        self.assertEqual(agent.bc_updates_applied, 0)

    def test_bc_update_interval_rejects_invalid_values(self) -> None:
        agent = make_agent(diagnostics=False)
        for interval in (0, -1, True, 1.5):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                agent.configure_bc_update_interval(interval)  # type: ignore[arg-type]

    def test_zero_progress_multiplier_preserves_full_sac_gradient(self) -> None:
        agent = make_agent(diagnostics=True)
        agent.set_bc_progress_multiplier(0.0)
        sac_gradients, bc_gradients = self.conflicting_gradients(agent)
        combined, scale = agent._combine_actor_gradients(
            sac_gradients=sac_gradients,
            applied_cloning_gradients=bc_gradients,
            conflict=True,
        )
        self.assertEqual(scale, 0.0)
        for actual, expected in zip(combined, sac_gradients, strict=True):
            torch.testing.assert_close(actual, expected)

    def test_cagrad_produces_finite_common_descent_direction(self) -> None:
        agent = make_agent(
            diagnostics=False,
            combination_strategy="cagrad",
            cagrad_alpha=0.5,
        )
        sac_gradients = [
            torch.zeros_like(parameter) for parameter in agent.actor.parameters()
        ]
        bc_gradients = [
            torch.zeros_like(parameter) for parameter in agent.actor.parameters()
        ]
        shared_index = agent.gradient_diagnostics.shared_indices[0]
        sac_gradients[shared_index].reshape(-1)[:2] = torch.tensor([1.0, 0.0])
        bc_gradients[shared_index].reshape(-1)[:2] = torch.tensor([-0.2, 1.0])
        sac_gradients = tuple(sac_gradients)
        bc_gradients = tuple(bc_gradients)

        combined, sac_weight = agent._combine_actor_gradients(
            sac_gradients=sac_gradients,
            applied_cloning_gradients=bc_gradients,
            conflict=True,
        )

        self.assertGreaterEqual(sac_weight, 0.0)
        self.assertLessEqual(sac_weight, 1.0)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in combined))
        sac_dot = sum(
            (combined[index] * sac_gradients[index]).sum()
            for index in agent.gradient_diagnostics.shared_indices
        )
        bc_dot = sum(
            (combined[index] * bc_gradients[index]).sum()
            for index in agent.gradient_diagnostics.shared_indices
        )
        self.assertGreater(float(sac_dot), 0.0)
        self.assertGreater(float(bc_dot), 0.0)

    def test_cagrad_zero_progress_is_exact_sac_update(self) -> None:
        agent = make_agent(diagnostics=False, combination_strategy="cagrad")
        agent.set_bc_progress_multiplier(0.0)
        sac_gradients, bc_gradients = self.conflicting_gradients(agent)
        combined, scale = agent._combine_actor_gradients(
            sac_gradients=sac_gradients,
            applied_cloning_gradients=bc_gradients,
            conflict=True,
        )
        self.assertEqual(scale, 0.0)
        for actual, expected in zip(combined, sac_gradients, strict=True):
            torch.testing.assert_close(actual, expected)

    def test_cagrad_runs_through_actor_update_and_diagnostics(self) -> None:
        agent = make_agent(
            diagnostics=True,
            combination_strategy="cagrad",
            cagrad_alpha=0.5,
        )
        self.add_task_zero_memory(agent)
        self.update_task_one(agent)
        rows = agent.drain_gradient_diagnostics()["windows"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["bc_combination_strategy"] == "cagrad" for row in rows))
        self.assertTrue(
            all(0.0 <= float(row["bc_combination_scale"]) <= 1.0 for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
