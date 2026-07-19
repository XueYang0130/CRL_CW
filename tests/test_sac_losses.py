import unittest

import torch

from crl_cw.agents import (
    compute_actor_loss,
    compute_alpha_loss,
    compute_critic_losses,
    compute_q_target,
)


class TestSACLosses(unittest.TestCase):
    def test_q_target_uses_minimum_q_and_entropy(
        self,
    ) -> None:
        """Bellman target must use minimum target Q and entropy."""
        rewards = torch.tensor(
            [[1.0], [2.0]],
            dtype=torch.float32,
        )

        dones = torch.tensor(
            [[False], [True]],
        )

        next_q1 = torch.tensor(
            [[5.0], [4.0]],
            dtype=torch.float32,
        )

        next_q2 = torch.tensor(
            [[3.0], [6.0]],
            dtype=torch.float32,
        )

        next_log_prob = torch.tensor(
            [[-0.5], [-0.25]],
            dtype=torch.float32,
        )

        alpha = torch.tensor(
            2.0,
            dtype=torch.float32,
        )

        q_target = compute_q_target(
            rewards=rewards,
            dones=dones,
            next_q1=next_q1,
            next_q2=next_q2,
            next_log_prob=next_log_prob,
            alpha=alpha,
            gamma=0.9,
        )

        # First sample:
        #
        # min_q = min(5, 3) = 3
        # entropy-adjusted q = 3 - 2 * (-0.5) = 4
        # target = 1 + 0.9 * 4 = 4.6
        #
        # Second sample:
        #
        # done = 1, therefore no bootstrapping.
        # target = reward = 2
        expected = torch.tensor(
            [4.6, 2.0],
            dtype=torch.float32,
        )

        torch.testing.assert_close(
            q_target,
            expected,
        )

    def test_q_target_is_detached(
        self,
    ) -> None:
        """Bellman target must not propagate target-network gradients."""
        rewards = torch.ones(
            3,
            dtype=torch.float32,
        )

        dones = torch.zeros(
            3,
            dtype=torch.float32,
        )

        next_q1 = torch.ones(
            3,
            dtype=torch.float32,
            requires_grad=True,
        )

        next_q2 = torch.full(
            (3,),
            2.0,
            dtype=torch.float32,
            requires_grad=True,
        )

        next_log_prob = torch.full(
            (3,),
            -0.5,
            dtype=torch.float32,
            requires_grad=True,
        )

        alpha = torch.tensor(
            1.0,
            dtype=torch.float32,
            requires_grad=True,
        )

        q_target = compute_q_target(
            rewards=rewards,
            dones=dones,
            next_q1=next_q1,
            next_q2=next_q2,
            next_log_prob=next_log_prob,
            alpha=alpha,
            gamma=0.99,
        )

        self.assertFalse(
            q_target.requires_grad
        )

        self.assertIsNone(
            q_target.grad_fn
        )

    def test_critic_losses_match_reference_formula(
        self,
    ) -> None:
        """Each critic uses 0.5 times its mean squared error."""
        q1_predictions = torch.tensor(
            [[0.0], [5.0]],
            dtype=torch.float32,
        )

        q2_predictions = torch.tensor(
            [2.0, 1.0],
            dtype=torch.float32,
        )

        q_targets = torch.tensor(
            [1.0, 3.0],
            dtype=torch.float32,
        )

        q1_loss, q2_loss = compute_critic_losses(
            q1_predictions=q1_predictions,
            q2_predictions=q2_predictions,
            q_targets=q_targets,
        )

        # Critic 1 squared errors:
        #
        # (0 - 1)^2 = 1
        # (5 - 3)^2 = 4
        #
        # mean = 2.5
        # 0.5 * mean = 1.25
        self.assertAlmostEqual(
            q1_loss.item(),
            1.25,
            places=6,
        )

        # Critic 2 has the same squared errors:
        #
        # (2 - 1)^2 = 1
        # (1 - 3)^2 = 4
        self.assertAlmostEqual(
            q2_loss.item(),
            1.25,
            places=6,
        )

    def test_actor_loss_matches_reference_formula(
        self,
    ) -> None:
        """Actor loss must use minimum online Q and detached alpha."""
        log_prob = torch.tensor(
            [-1.0, -0.5],
            dtype=torch.float32,
            requires_grad=True,
        )

        q1_policy = torch.tensor(
            [2.0, 4.0],
            dtype=torch.float32,
            requires_grad=True,
        )

        q2_policy = torch.tensor(
            [1.0, 5.0],
            dtype=torch.float32,
            requires_grad=True,
        )

        alpha = torch.tensor(
            2.0,
            dtype=torch.float32,
            requires_grad=True,
        )

        actor_loss = compute_actor_loss(
            log_prob=log_prob,
            q1_policy=q1_policy,
            q2_policy=q2_policy,
            alpha=alpha,
        )

        # Sample 1:
        # alpha * log_prob - min_q
        # = 2 * (-1) - 1
        # = -3
        #
        # Sample 2:
        # = 2 * (-0.5) - 4
        # = -5
        #
        # mean = -4
        self.assertAlmostEqual(
            actor_loss.item(),
            -4.0,
            places=6,
        )

        actor_loss.backward()

        # Actor loss must not update alpha.
        self.assertIsNone(
            alpha.grad
        )

        # But gradients must still pass through policy outputs.
        self.assertIsNotNone(
            log_prob.grad
        )

    def test_alpha_loss_detaches_actor_log_probability(
        self,
    ) -> None:
        """Alpha update must not send gradients into the actor."""
        log_alpha = torch.tensor(
            1.0,
            dtype=torch.float32,
            requires_grad=True,
        )

        log_prob = torch.tensor(
            [-2.0, -1.0],
            dtype=torch.float32,
            requires_grad=True,
        )

        alpha_loss = compute_alpha_loss(
            log_alpha=log_alpha,
            log_prob=log_prob,
            target_entropy=-1.0,
        )

        # Entropy errors:
        #
        # -2 + (-1) = -3
        # -1 + (-1) = -2
        #
        # alpha loss:
        #
        # -mean(1 * [-3, -2]) = 2.5
        self.assertAlmostEqual(
            alpha_loss.item(),
            2.5,
            places=6,
        )

        alpha_loss.backward()

        self.assertIsNotNone(
            log_alpha.grad
        )

        self.assertAlmostEqual(
            log_alpha.grad.item(),
            2.5,
            places=6,
        )

        # The entropy target is fixed during alpha optimization.
        self.assertIsNone(
            log_prob.grad
        )

    def test_flat_and_column_shapes_are_supported(
        self,
    ) -> None:
        """Losses must safely support `(batch,)` and `(batch, 1)`."""
        actor_loss = compute_actor_loss(
            log_prob=torch.tensor(
                [[-1.0], [-2.0]],
                dtype=torch.float32,
            ),
            q1_policy=torch.tensor(
                [1.0, 2.0],
                dtype=torch.float32,
            ),
            q2_policy=torch.tensor(
                [[2.0], [1.0]],
                dtype=torch.float32,
            ),
            alpha=torch.tensor(
                1.0,
                dtype=torch.float32,
            ),
        )

        self.assertEqual(
            actor_loss.ndim,
            0,
        )

        self.assertTrue(
            torch.isfinite(actor_loss)
        )

    def test_mismatched_batch_sizes_are_rejected(
        self,
    ) -> None:
        """Silent broadcasting between different batches is forbidden."""
        with self.assertRaises(ValueError):
            compute_actor_loss(
                log_prob=torch.zeros(2),
                q1_policy=torch.zeros(3),
                q2_policy=torch.zeros(2),
                alpha=torch.tensor(1.0),
            )

    def test_invalid_gamma_is_rejected(
        self,
    ) -> None:
        """Discount factor must remain in the valid interval."""
        with self.assertRaises(ValueError):
            compute_q_target(
                rewards=torch.zeros(2),
                dones=torch.zeros(2),
                next_q1=torch.zeros(2),
                next_q2=torch.zeros(2),
                next_log_prob=torch.zeros(2),
                alpha=torch.tensor(1.0),
                gamma=1.1,
            )


if __name__ == "__main__":
    unittest.main()