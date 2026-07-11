"""Unit tests for the ClonEx-compatible SAC networks."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crl_cw.agents import (
    DEFAULT_HIDDEN_SIZES,
    LOG_STD_MAX,
    LOG_STD_MIN,
    GaussianActor,
    QCritic,
)
from crl_cw.envs import make_cw_env


class TestSACNetworks(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

        self.observation_dim = 39
        self.action_dim = 4

        self.action_low = -np.ones(
            self.action_dim,
            dtype=np.float32,
        )
        self.action_high = np.ones(
            self.action_dim,
            dtype=np.float32,
        )

        self.actor = GaussianActor(
            observation_dim=self.observation_dim,
            action_low=self.action_low,
            action_high=self.action_high,
        )

        self.critic = QCritic(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
        )

    def test_default_architecture_matches_clonex(self) -> None:
        self.assertEqual(
            DEFAULT_HIDDEN_SIZES,
            (256, 256, 256, 256),
        )

        self.assertEqual(
            len(self.actor.backbone.hidden_layers),
            4,
        )

        self.assertIsNotNone(
            self.actor.backbone.first_layer_norm,
        )

    def test_actor_output_shapes(self) -> None:
        observations = torch.zeros(
            (8, self.observation_dim),
            dtype=torch.float32,
        )

        output = self.actor.sample(observations)

        self.assertEqual(
            output.mean_action.shape,
            (8, self.action_dim),
        )
        self.assertEqual(
            output.sampled_action.shape,
            (8, self.action_dim),
        )
        self.assertEqual(
            output.log_probability.shape,
            (8, 1),
        )
        self.assertEqual(
            output.log_std.shape,
            (8, self.action_dim),
        )

    def test_actor_actions_respect_environment_bounds(self) -> None:
        observations = torch.randn(
            (32, self.observation_dim),
            dtype=torch.float32,
        )

        output = self.actor.sample(observations)

        self.assertTrue(
            torch.all(output.sampled_action <= 1.0),
        )
        self.assertTrue(
            torch.all(output.sampled_action >= -1.0),
        )
        self.assertTrue(
            torch.all(output.mean_action <= 1.0),
        )
        self.assertTrue(
            torch.all(output.mean_action >= -1.0),
        )

    def test_log_std_is_clipped(self) -> None:
        observations = torch.randn(
            (32, self.observation_dim),
            dtype=torch.float32,
        )

        _, log_std = self.actor.distribution_parameters(
            observations,
        )

        self.assertTrue(
            torch.all(log_std <= LOG_STD_MAX),
        )
        self.assertTrue(
            torch.all(log_std >= LOG_STD_MIN),
        )

    def test_critic_output_shape(self) -> None:
        observations = torch.zeros(
            (8, self.observation_dim),
            dtype=torch.float32,
        )
        actions = torch.zeros(
            (8, self.action_dim),
            dtype=torch.float32,
        )

        q_values = self.critic(
            observations,
            actions,
        )

        self.assertEqual(q_values.shape, (8, 1))

    def test_actor_reparameterization_supports_gradients(self) -> None:
        observations = torch.randn(
            (4, self.observation_dim),
            dtype=torch.float32,
        )

        output = self.actor.sample(observations)
        loss = output.sampled_action.mean()

        loss.backward()

        gradients = [
            parameter.grad
            for parameter in self.actor.parameters()
            if parameter.requires_grad
        ]

        self.assertTrue(
            any(
                gradient is not None
                and torch.any(torch.isfinite(gradient))
                for gradient in gradients
            )
        )

    def test_real_metaworld_dimensions(self) -> None:
        env = make_cw_env("hammer-v1", seed=0)

        try:
            observation, _ = env.reset(seed=0)

            actor = GaussianActor(
                observation_dim=env.observation_space.shape[0],
                action_low=env.action_space.low,
                action_high=env.action_space.high,
            )

            critic = QCritic(
                observation_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
            )

            action = actor.act(
                observation=observation,
                deterministic=False,
                device="cpu",
            )

            self.assertEqual(
                action.shape,
                env.action_space.shape,
            )

            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
            ).unsqueeze(0)

            action_tensor = torch.as_tensor(
                action,
                dtype=torch.float32,
            ).unsqueeze(0)

            q_value = critic(
                observation_tensor,
                action_tensor,
            )

            self.assertEqual(q_value.shape, (1, 1))

        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()