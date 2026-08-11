import unittest

import numpy as np
import torch

from agents import ConflictLoRAFullBCAgent, FullBehaviorCloningSACAgent
from methods import bc_gradient_strategy_for_method, get_method
from training.continual_experiment import (
    add_relabelled_reference_memory,
    clone_module_state,
)
from training.bc_progress_gate import BCProgressGate
from training.success_replay import BroaderStateReservoir, SuccessfulStateReservoir


class SuccessfulStateReservoirTests(unittest.TestCase):
    def test_broader_reservoir_is_disjoint_from_success_memory(self) -> None:
        memory = BroaderStateReservoir(
            capacity_per_bin=2, observation_dim=2, seed=5
        )
        failed = tuple(
            np.asarray([float(index), -float(index)], dtype=np.float32)
            for index in range(9)
        )
        successful = tuple(
            np.asarray([100.0 + index, 100.0], dtype=np.float32)
            for index in range(9)
        )
        memory.add_episode(failed, False)
        memory.add_episode(successful, True)

        self.assertEqual(memory.non_successful_episodes, 1)
        self.assertEqual(memory.counts(), {"early": 2, "middle": 2, "late": 2})
        selected = memory.observations(
            selected_bins=("middle",), capacity=2, seed=9
        )
        self.assertEqual(selected.shape, (2, 2))
        self.assertTrue(bool((selected[:, 0] < 100.0).all()))

    def test_broader_selection_is_reproducible(self) -> None:
        memory = BroaderStateReservoir(
            capacity_per_bin=10, observation_dim=1, seed=2
        )
        episode = tuple(
            np.asarray([float(index)], dtype=np.float32) for index in range(30)
        )
        memory.add_episode(episode, False)
        first = memory.observations(capacity=7, seed=17)
        second = memory.observations(capacity=7, seed=17)
        np.testing.assert_array_equal(first, second)

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


class BCProgressGateTests(unittest.TestCase):
    def test_uses_recent_success_window_and_piecewise_multipliers(self) -> None:
        gate = BCProgressGate(window=3)
        self.assertEqual(gate.multiplier, 0.0)
        self.assertEqual(gate.update(0.0), 0.0)
        self.assertEqual(gate.update(0.6), 0.1)
        self.assertEqual(gate.update(0.9), 0.4)
        self.assertAlmostEqual(gate.score, 0.5)
        self.assertEqual(gate.update(1.0), 1.0)

    def test_reset_restores_early_task_gate(self) -> None:
        gate = BCProgressGate(window=1)
        self.assertEqual(gate.update(1.0), 1.0)
        gate.reset()
        self.assertEqual(gate.score, 0.0)
        self.assertEqual(gate.multiplier, 0.0)


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
            "success_replay_best_random_broader_adaptive_pcgrad": (
                "best",
                "pcgrad_sac_priority",
            ),
            "success_replay_best_llm_broader_adaptive_pcgrad": (
                "best",
                "pcgrad_sac_priority",
            ),
            "success_replay_best_cagrad": ("best", "standard"),
            "success_replay_best_conflict_lora": (
                "best",
                "pcgrad_sac_priority",
            ),
            "success_replay_best_progress_pcgrad": (
                "best",
                "pcgrad_sac_priority",
            ),
            "success_replay_best_llm_schedule_pcgrad": (
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
                if method_id in {
                    "success_replay_best_adaptive_pcgrad",
                    "success_replay_best_random_broader_adaptive_pcgrad",
                    "success_replay_best_llm_broader_adaptive_pcgrad",
                }:
                    self.assertEqual(
                        method.defaults["bc_combination_strategy"],
                        "adaptive_additive",
                    )
                if "broader" in method_id:
                    self.assertEqual(method.broader_replay_ratio, 0.2)
                    self.assertEqual(
                        method.broader_replay_selector,
                        "llm" if "llm_broader" in method_id else "random",
                    )
                if method_id == "success_replay_best_cagrad":
                    self.assertEqual(
                        method.defaults["bc_combination_strategy"],
                        "cagrad",
                    )
                    self.assertEqual(method.defaults["bc_cagrad_alpha"], 0.5)
                if method_id == "success_replay_best_conflict_lora":
                    self.assertEqual(
                        method.defaults["bc_combination_strategy"],
                        "adaptive_additive",
                    )
                    self.assertEqual(method.defaults["conflict_lora_rank"], 4)
                if method_id == "success_replay_best_progress_pcgrad":
                    self.assertTrue(method.defaults["bc_progress_gate"])
                if method_id == "success_replay_best_llm_schedule_pcgrad":
                    self.assertTrue(method.defaults["llm_task_bc_schedule"])

    def test_conflict_lora_uses_separate_optimizer(self) -> None:
        agent = ConflictLoRAFullBCAgent(
            observation_dim=5,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=2,
            task_id_dim=2,
            hide_task_id=True,
            device="cpu",
            episodic_batch_size=2,
            bc_gradient_strategy="pcgrad_sac_priority",
            bc_combination_strategy="adaptive_additive",
            conflict_lora_rank=2,
        )
        lora_parameter_ids = {
            id(parameter)
            for parameter in agent.conflict_lora_adapters.parameters()
        }
        main_optimizer_parameter_ids = {
            id(parameter)
            for group in agent.optimizer.param_groups
            for parameter in group["params"]
        }
        lora_optimizer_parameter_ids = {
            id(parameter)
            for group in agent.conflict_lora_optimizer.param_groups
            for parameter in group["params"]
        }

        self.assertFalse(lora_parameter_ids & main_optimizer_parameter_ids)
        self.assertEqual(lora_parameter_ids, lora_optimizer_parameter_ids)

    def test_conflict_lora_does_not_enter_guide_actions(self) -> None:
        agent = ConflictLoRAFullBCAgent(
            observation_dim=5,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=2,
            task_id_dim=2,
            hide_task_id=True,
            device="cpu",
            episodic_batch_size=2,
            bc_gradient_strategy="pcgrad_sac_priority",
            bc_combination_strategy="adaptive_additive",
            conflict_lora_rank=2,
        )
        observation = np.asarray([0.2, -0.1, 0.3, 0.0, 1.0], dtype=np.float32)
        base_observation = agent._observation_for_actor_head(
            observation=observation,
            head_index=0,
        )
        with torch.no_grad():
            for parameter in agent.conflict_lora_adapters[0].parameters():
                parameter.fill_(0.25)

        base_action = agent.actor.act(
            base_observation,
            deterministic=True,
            device=agent.device,
        )
        lora_action = agent.select_action_with_head(
            observation,
            head_index=0,
            deterministic=True,
        )
        guide_action = agent.select_guide_action(
            observation,
            guide_task_index=0,
            deterministic=True,
        )

        self.assertGreater(
            float(np.linalg.norm(lora_action - base_action)),
            1e-5,
        )
        np.testing.assert_allclose(guide_action, base_action, atol=1e-7)

    def test_conflict_lora_updates_only_on_projected_conflict(self) -> None:
        torch.manual_seed(11)
        agent = ConflictLoRAFullBCAgent(
            observation_dim=5,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=2,
            task_id_dim=2,
            hide_task_id=True,
            device="cpu",
            episodic_batch_size=4,
            bc_gradient_strategy="pcgrad_sac_priority",
            bc_combination_strategy="adaptive_additive",
            conflict_lora_rank=2,
            conflict_lora_coefficient=1.0,
        )
        observations = torch.zeros((4, 5), dtype=torch.float32)
        observations[:, 0] = torch.linspace(0.0, 0.3, 4)
        observations[:, -2] = 1.0
        with torch.no_grad():
            means, log_stds = agent.compute_reference_targets(observations)
        target_means = means + 0.5
        agent.add_reference_memory_with_targets(
            observations=observations,
            target_means=target_means,
            target_log_stds=log_stds,
        )

        parameters = tuple(agent.actor.parameters())
        batch = tuple(tensor.to(agent.device) for tensor in agent._sample_episodic_batch())
        sampled_observations, sampled_target_means, sampled_target_log_stds = batch
        current_means, current_log_stds = agent._cloning_distribution_parameters(
            sampled_observations
        )
        raw_loss = agent._gaussian_kl(
            sampled_target_means,
            sampled_target_log_stds,
            current_means,
            current_log_stds,
        ).mean()
        weighted_loss = agent.actor_cloning_coefficient * raw_loss
        cloning_gradients = torch.autograd.grad(weighted_loss, parameters)
        sac_gradients = tuple(-gradient.detach().clone() for gradient in cloning_gradients)
        agent._sample_episodic_batch = lambda: (
            sampled_observations,
            sampled_target_means,
            sampled_target_log_stds,
        )

        for parameter in parameters:
            parameter.grad = None
        agent.adjust_actor_gradients(
            gradients=sac_gradients,
            parameters=parameters,
            task_index=1,
        )

        self.assertGreater(agent.conflict_lora_updates, 0)
        self.assertTrue(agent.last_bc_conflict)
        self.assertTrue(
            all(parameter.grad is None for parameter in parameters)
        )

    def test_conflict_lora_routes_mixed_batch_by_source_task(self) -> None:
        torch.manual_seed(13)
        agent = ConflictLoRAFullBCAgent(
            observation_dim=6,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=3,
            task_id_dim=3,
            hide_task_id=True,
            device="cpu",
            episodic_batch_size=4,
            bc_gradient_strategy="pcgrad_sac_priority",
            bc_combination_strategy="adaptive_additive",
            conflict_lora_rank=2,
            conflict_lora_coefficient=1.0,
            gradient_clip_norm=None,
        )
        observations = torch.zeros((4, 6), dtype=torch.float32)
        observations[:, 0] = torch.tensor([0.0, 0.1, 0.2, 0.3])
        observations[:2, -3] = 1.0
        observations[2:, -2] = 1.0
        target_means, target_log_stds = agent.compute_reference_targets(
            observations
        )
        sac_gradients = tuple(
            torch.ones_like(parameter) for parameter in agent.actor.parameters()
        )

        agent._update_conflict_lora(
            observations=observations.to(agent.device),
            target_means=(target_means + 0.5).to(agent.device),
            target_log_stds=target_log_stds.to(agent.device),
            sac_gradients=sac_gradients,
        )

        grad_norms = []
        for adapter in agent.conflict_lora_adapters:
            norm_sq = sum(
                (
                    parameter.grad.detach().square().sum()
                    if parameter.grad is not None
                    else torch.tensor(0.0)
                )
                for parameter in adapter.parameters()
            )
            grad_norms.append(float(norm_sq.sqrt().item()))
        self.assertGreater(grad_norms[0], 0.0)
        self.assertGreater(grad_norms[1], 0.0)
        self.assertEqual(grad_norms[2], 0.0)

    def test_conflict_lora_mixed_loss_keeps_task_gradients_separate(self) -> None:
        torch.manual_seed(17)
        agent = ConflictLoRAFullBCAgent(
            observation_dim=6,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=3,
            task_id_dim=3,
            hide_task_id=True,
            device="cpu",
            episodic_batch_size=4,
            bc_gradient_strategy="pcgrad_sac_priority",
            bc_combination_strategy="adaptive_additive",
            conflict_lora_rank=2,
            conflict_lora_coefficient=1.0,
            gradient_clip_norm=None,
        )
        observations = torch.zeros((4, 6), dtype=torch.float32)
        observations[:, 0] = torch.tensor([0.0, 0.1, 0.8, 0.9])
        observations[:2, -3] = 1.0
        observations[2:, -2] = 1.0
        with torch.no_grad():
            means, log_stds = agent.compute_reference_targets(observations)
        target_means = means + torch.tensor([[0.4, -0.2]], dtype=torch.float32)

        def adapter_gradients_for_loss(
            selected_observations: torch.Tensor,
            selected_means: torch.Tensor,
            selected_log_stds: torch.Tensor,
        ) -> list[list[torch.Tensor | None]]:
            agent.conflict_lora_optimizer.zero_grad(set_to_none=True)
            current_means, current_log_stds = agent._lora_distribution_parameters(
                selected_observations.to(agent.device),
                detach_base=True,
            )
            loss = agent._gaussian_kl(
                selected_means.to(agent.device),
                selected_log_stds.to(agent.device),
                current_means,
                current_log_stds,
            ).mean()
            loss.backward()
            return [
                [
                    (
                        parameter.grad.detach().clone()
                        if parameter.grad is not None
                        else None
                    )
                    for parameter in adapter.parameters()
                ]
                for adapter in agent.conflict_lora_adapters
            ]

        mixed_grads = adapter_gradients_for_loss(
            observations,
            target_means,
            log_stds,
        )
        task0_grads = adapter_gradients_for_loss(
            observations[:2],
            target_means[:2],
            log_stds[:2],
        )
        task1_grads = adapter_gradients_for_loss(
            observations[2:],
            target_means[2:],
            log_stds[2:],
        )

        for mixed, task0 in zip(mixed_grads[0], task0_grads[0], strict=True):
            self.assertIsNotNone(mixed)
            self.assertIsNotNone(task0)
            torch.testing.assert_close(mixed, 0.5 * task0)
        for mixed, task1 in zip(mixed_grads[1], task1_grads[1], strict=True):
            self.assertIsNotNone(mixed)
            self.assertIsNotNone(task1)
            torch.testing.assert_close(mixed, 0.5 * task1)
        for mixed in mixed_grads[2]:
            self.assertIsNone(mixed)

    def test_conflict_lora_norm_cap_is_applied(self) -> None:
        agent = ConflictLoRAFullBCAgent(
            observation_dim=5,
            action_dim=2,
            action_low=np.full(2, -1.0, dtype=np.float32),
            action_high=np.full(2, 1.0, dtype=np.float32),
            num_tasks=2,
            task_id_dim=2,
            hide_task_id=True,
            device="cpu",
            episodic_batch_size=2,
            bc_gradient_strategy="pcgrad_sac_priority",
            bc_combination_strategy="adaptive_additive",
            conflict_lora_rank=2,
            conflict_lora_max_norm_ratio=0.1,
        )
        sac_gradients = tuple(
            torch.full_like(parameter, 0.01)
            for parameter in agent.actor.parameters()
        )
        for parameter in agent.conflict_lora_adapters.parameters():
            parameter.grad = torch.full_like(parameter, 10.0)

        pre_norm, scale = agent._cap_conflict_lora_gradients(
            sac_gradients=sac_gradients
        )
        capped_norm_sq = sum(
            parameter.grad.detach().square().sum()
            for parameter in agent.conflict_lora_adapters.parameters()
            if parameter.grad is not None
        )
        capped_norm = float(capped_norm_sq.sqrt().item())
        shared_indices = agent.gradient_diagnostics.shared_indices
        sac_norm = sum(
            sac_gradients[index].square().sum()
            for index in shared_indices
        ).sqrt()
        max_allowed = 0.1 * float(sac_norm.item())

        self.assertGreater(pre_norm, max_allowed)
        self.assertLess(scale, 1.0)
        self.assertLessEqual(capped_norm, max_allowed + 1e-6)

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
