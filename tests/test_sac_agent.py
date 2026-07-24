import math
import unittest

import numpy as np
import torch

from agents import SACAgent


OBSERVATION_DIM = 39
ACTION_DIM = 4

ACTION_LOW = np.full(
    ACTION_DIM,
    -1.0,
    dtype=np.float32,
)

ACTION_HIGH = np.full(
    ACTION_DIM,
    1.0,
    dtype=np.float32,
)


class TestSACAgent(unittest.TestCase):
    def setUp(self) -> None:
        """Create a fresh SAC agent before every test."""
        torch.manual_seed(0)
        np.random.seed(0)

        self.agent = SACAgent(
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            action_low=ACTION_LOW,
            action_high=ACTION_HIGH,
            device="cpu",
        )

    def test_online_critics_are_independent_and_targets_match(
        self,
    ) -> None:
        """The two critics must be independent networks."""
        critics_differ = any(
            not torch.equal(
                parameter1,
                parameter2,
            )
            for parameter1, parameter2 in zip(
                self.agent.critic1.parameters(),
                self.agent.critic2.parameters(),
                strict=True,
            )
        )

        self.assertTrue(critics_differ)

        # Each target critic must initially be an exact copy
        # of its corresponding online critic.
        for online, target in (
            (
                self.agent.critic1,
                self.agent.target_critic1,
            ),
            (
                self.agent.critic2,
                self.agent.target_critic2,
            ),
        ):
            for online_parameter, target_parameter in zip(
                online.parameters(),
                target.parameters(),
                strict=True,
            ):
                self.assertTrue(
                    torch.equal(
                        online_parameter,
                        target_parameter,
                    )
                )

                self.assertFalse(
                    target_parameter.requires_grad
                )

    def test_default_entropy_configuration(
        self,
    ) -> None:
        """Check automatic entropy defaults."""
        self.assertEqual(
            self.agent.target_entropy,
            -float(ACTION_DIM),
        )

        self.assertAlmostEqual(
            self.agent.log_alpha.item(),
            1.0,
            places=6,
        )

        # alpha = exp(log_alpha) = exp(1)
        self.assertAlmostEqual(
            self.agent.alpha_value,
            math.e,
            places=5,
        )

    def test_optimizer_contains_only_trainable_online_state(
        self,
    ) -> None:
        """Target critics must not be managed by Adam."""
        expected_parameter_ids = {
            id(parameter)
            for module in (
                self.agent.actor,
                self.agent.critic1,
                self.agent.critic2,
            )
            for parameter in module.parameters()
        }

        expected_parameter_ids.add(
            id(self.agent.log_alpha)
        )

        optimizer_parameter_ids = {
            id(parameter)
            for parameter_group
            in self.agent.optimizer.param_groups
            for parameter
            in parameter_group["params"]
        }

        self.assertEqual(
            expected_parameter_ids,
            optimizer_parameter_ids,
        )

        target_parameter_ids = {
            id(parameter)
            for module in (
                self.agent.target_critic1,
                self.agent.target_critic2,
            )
            for parameter in module.parameters()
        }

        self.assertTrue(
            optimizer_parameter_ids.isdisjoint(
                target_parameter_ids
            )
        )

    def test_select_action_uses_metaworld_dimensions(
        self,
    ) -> None:
        """Environment actions must have shape 4 and remain bounded."""
        observation = np.zeros(
            OBSERVATION_DIM,
            dtype=np.float32,
        )

        deterministic_action = self.agent.select_action(
            observation,
            deterministic=True,
        )

        stochastic_action = self.agent.select_action(
            observation,
            deterministic=False,
        )

        for action in (
            deterministic_action,
            stochastic_action,
        ):
            self.assertEqual(
                action.shape,
                (ACTION_DIM,),
            )

            self.assertTrue(
                np.all(action >= ACTION_LOW)
            )

            self.assertTrue(
                np.all(action <= ACTION_HIGH)
            )

    def test_hard_update_targets_copies_online_critics(
        self,
    ) -> None:
        """Hard update must exactly copy both online critics."""
        with torch.no_grad():
            for parameter in (
                self.agent.critic1.parameters()
            ):
                parameter.fill_(2.0)

            for parameter in (
                self.agent.critic2.parameters()
            ):
                parameter.fill_(4.0)

        self.agent.hard_update_targets()

        for online, target in (
            (
                self.agent.critic1,
                self.agent.target_critic1,
            ),
            (
                self.agent.critic2,
                self.agent.target_critic2,
            ),
        ):
            for online_parameter, target_parameter in zip(
                online.parameters(),
                target.parameters(),
                strict=True,
            ):
                self.assertTrue(
                    torch.equal(
                        online_parameter,
                        target_parameter,
                    )
                )

    def test_soft_update_targets_uses_polyak_formula(
        self,
    ) -> None:
        """Check target = 0.5 target + 0.5 online."""
        agent = SACAgent(
            observation_dim=OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            action_low=ACTION_LOW,
            action_high=ACTION_HIGH,
            polyak=0.5,
            device="cpu",
        )

        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                agent.critic1.parameters(),
                agent.target_critic1.parameters(),
                strict=True,
            ):
                online_parameter.fill_(2.0)
                target_parameter.zero_()

            for online_parameter, target_parameter in zip(
                agent.critic2.parameters(),
                agent.target_critic2.parameters(),
                strict=True,
            ):
                online_parameter.fill_(4.0)
                target_parameter.zero_()

        agent.soft_update_targets()

        # 0.5 * 0 + 0.5 * 2 = 1
        for target_parameter in (
            agent.target_critic1.parameters()
        ):
            torch.testing.assert_close(
                target_parameter,
                torch.ones_like(
                    target_parameter
                ),
            )

        # 0.5 * 0 + 0.5 * 4 = 2
        for target_parameter in (
            agent.target_critic2.parameters()
        ):
            torch.testing.assert_close(
                target_parameter,
                torch.full_like(
                    target_parameter,
                    2.0,
                ),
            )

    def test_invalid_action_bound_shape_is_rejected(
        self,
    ) -> None:
        """Action bounds must match action dimension 4."""
        with self.assertRaises(ValueError):
            SACAgent(
                observation_dim=OBSERVATION_DIM,
                action_dim=ACTION_DIM,
                action_low=np.full(
                    3,
                    -1.0,
                    dtype=np.float32,
                ),
                action_high=ACTION_HIGH,
            )

    def test_update_batch_updates_online_and_target_state(
        self,
    ) -> None:
        """One minibatch must update the complete SAC state."""
        batch_size = 16

        observations = torch.randn(
            batch_size,
            OBSERVATION_DIM,
        )

        actions = torch.empty(
            batch_size,
            ACTION_DIM,
        ).uniform_(-1.0, 1.0)

        rewards = torch.randn(
            batch_size,
            1,
        )

        next_observations = torch.randn(
            batch_size,
            OBSERVATION_DIM,
        )

        dones = torch.zeros(
            batch_size,
            1,
        )

        old_actor_parameters = [
            parameter.detach().clone()
            for parameter
            in self.agent.actor.parameters()
        ]

        old_critic1_parameters = [
            parameter.detach().clone()
            for parameter
            in self.agent.critic1.parameters()
        ]

        old_critic2_parameters = [
            parameter.detach().clone()
            for parameter
            in self.agent.critic2.parameters()
        ]

        old_target1_parameters = [
            parameter.detach().clone()
            for parameter
            in self.agent.target_critic1.parameters()
        ]

        old_target2_parameters = [
            parameter.detach().clone()
            for parameter
            in self.agent.target_critic2.parameters()
        ]

        old_log_alpha = (
            self.agent.log_alpha
            .detach()
            .clone()
        )

        metrics = self.agent.update_batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
        )

        expected_metric_names = {
            "actor_loss",
            "q1_loss",
            "q2_loss",
            "alpha_loss",
            "alpha",
            "q1_mean",
            "q2_mean",
            "q_target_mean",
            "log_prob_mean",
        }

        self.assertEqual(
            set(metrics.keys()),
            expected_metric_names,
        )

        for metric_value in metrics.values():
            self.assertTrue(
                math.isfinite(metric_value)
            )

        actor_changed = any(
            not torch.equal(
                old_parameter,
                new_parameter,
            )
            for old_parameter, new_parameter in zip(
                old_actor_parameters,
                self.agent.actor.parameters(),
                strict=True,
            )
        )

        critic1_changed = any(
            not torch.equal(
                old_parameter,
                new_parameter,
            )
            for old_parameter, new_parameter in zip(
                old_critic1_parameters,
                self.agent.critic1.parameters(),
                strict=True,
            )
        )

        critic2_changed = any(
            not torch.equal(
                old_parameter,
                new_parameter,
            )
            for old_parameter, new_parameter in zip(
                old_critic2_parameters,
                self.agent.critic2.parameters(),
                strict=True,
            )
        )

        self.assertTrue(actor_changed)
        self.assertTrue(critic1_changed)
        self.assertTrue(critic2_changed)

        self.assertFalse(
            torch.equal(
                old_log_alpha,
                self.agent.log_alpha.detach(),
            )
        )

        # Check that soft target update happened after the
        # online critic optimizer update.
        for (
            old_target,
            updated_online,
            updated_target,
        ) in zip(
            old_target1_parameters,
            self.agent.critic1.parameters(),
            self.agent.target_critic1.parameters(),
            strict=True,
        ):
            expected_target = (
                self.agent.polyak
                * old_target
                + (
                    1.0
                    - self.agent.polyak
                )
                * updated_online.detach()
            )

            torch.testing.assert_close(
                updated_target,
                expected_target,
            )

            self.assertFalse(
                updated_target.requires_grad
            )

        for (
            old_target,
            updated_online,
            updated_target,
        ) in zip(
            old_target2_parameters,
            self.agent.critic2.parameters(),
            self.agent.target_critic2.parameters(),
            strict=True,
        ):
            expected_target = (
                self.agent.polyak
                * old_target
                + (
                    1.0
                    - self.agent.polyak
                )
                * updated_online.detach()
            )

            torch.testing.assert_close(
                updated_target,
                expected_target,
            )

            self.assertFalse(
                updated_target.requires_grad
            )

    def test_update_batch_supports_flat_reward_and_done(
        self,
    ) -> None:
        """Rewards and dones may use shape `(batch_size,)`."""
        batch_size = 8

        metrics = self.agent.update_batch(
            observations=torch.randn(
                batch_size,
                OBSERVATION_DIM,
            ),
            actions=torch.empty(
                batch_size,
                ACTION_DIM,
            ).uniform_(-1.0, 1.0),
            rewards=torch.randn(
                batch_size
            ),
            next_observations=torch.randn(
                batch_size,
                OBSERVATION_DIM,
            ),
            dones=torch.zeros(
                batch_size
            ),
        )

        self.assertTrue(
            all(
                math.isfinite(value)
                for value in metrics.values()
            )
        )

    def test_update_batch_rejects_wrong_observation_dimension(
        self,
    ) -> None:
        """Observation dimension must remain 39."""
        batch_size = 4

        with self.assertRaises(ValueError):
            self.agent.update_batch(
                observations=torch.randn(
                    batch_size,
                    3,
                ),
                actions=torch.randn(
                    batch_size,
                    ACTION_DIM,
                ),
                rewards=torch.randn(
                    batch_size
                ),
                next_observations=torch.randn(
                    batch_size,
                    OBSERVATION_DIM,
                ),
                dones=torch.zeros(
                    batch_size
                ),
            )

    def test_update_batch_rejects_invalid_done_values(
        self,
    ) -> None:
        """Done indicators must remain between zero and one."""
        batch_size = 4

        with self.assertRaises(ValueError):
            self.agent.update_batch(
                observations=torch.randn(
                    batch_size,
                    OBSERVATION_DIM,
                ),
                actions=torch.randn(
                    batch_size,
                    ACTION_DIM,
                ),
                rewards=torch.randn(
                    batch_size
                ),
                next_observations=torch.randn(
                    batch_size,
                    OBSERVATION_DIM,
                ),
                dones=torch.tensor(
                    [0.0, 0.0, 1.0, 2.0]
                ),
            )


if __name__ == "__main__":
    unittest.main()
