import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from crl_cw.agents import SACAgent
from crl_cw.utils import (
    CHECKPOINT_VERSION,
    load_sac_checkpoint,
    save_sac_checkpoint,
)


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


def create_agent(
    *,
    observation_dim: int = OBSERVATION_DIM,
    action_dim: int = ACTION_DIM,
    gamma: float = 0.99,
) -> SACAgent:
    """Create one deterministic CPU test agent."""
    torch.manual_seed(0)

    return SACAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        action_low=np.full(
            action_dim,
            -1.0,
            dtype=np.float32,
        ),
        action_high=np.full(
            action_dim,
            1.0,
            dtype=np.float32,
        ),
        gamma=gamma,
        device="cpu",
    )


def run_one_update(
    agent: SACAgent,
) -> None:
    """Populate both model and optimizer state."""
    batch_size = 8

    agent.update_batch(
        observations=torch.randn(
            batch_size,
            agent.observation_dim,
        ),
        actions=torch.empty(
            batch_size,
            agent.action_dim,
        ).uniform_(-1.0, 1.0),
        rewards=torch.randn(
            batch_size,
            1,
        ),
        next_observations=torch.randn(
            batch_size,
            agent.observation_dim,
        ),
        dones=torch.zeros(
            batch_size,
            1,
        ),
    )


class TestSACCheckpoint(unittest.TestCase):
    def test_save_and_load_complete_training_state(
        self,
    ) -> None:
        """Checkpoint must restore model, alpha, and Adam state."""
        source_agent = create_agent()

        run_one_update(
            source_agent
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = (
                Path(directory)
                / "checkpoint.pt"
            )

            saved_path = save_sac_checkpoint(
                agent=source_agent,
                path=checkpoint_path,
                environment_step=12_345,
                metadata={
                    "task_name": "hammer-v1",
                    "seed": 7,
                },
            )

            self.assertEqual(
                saved_path,
                checkpoint_path,
            )

            self.assertTrue(
                checkpoint_path.is_file()
            )

            restored_agent = create_agent()

            info = load_sac_checkpoint(
                agent=restored_agent,
                path=checkpoint_path,
                map_location="cpu",
                load_optimizer=True,
            )

            self.assertEqual(
                info.environment_step,
                12_345,
            )

            self.assertEqual(
                info.metadata,
                {
                    "task_name": "hammer-v1",
                    "seed": 7,
                },
            )

            self.assertTrue(
                info.optimizer_loaded
            )

            self.assertEqual(
                info.checkpoint_version,
                CHECKPOINT_VERSION,
            )

            source_state = (
                source_agent.state_dict()
            )

            restored_state = (
                restored_agent.state_dict()
            )

            self.assertEqual(
                set(source_state),
                set(restored_state),
            )

            for name in source_state:
                torch.testing.assert_close(
                    source_state[name],
                    restored_state[name],
                )

            source_optimizer_state = (
                source_agent
                .optimizer
                .state_dict()
            )

            restored_optimizer_state = (
                restored_agent
                .optimizer
                .state_dict()
            )

            self.assertEqual(
                source_optimizer_state.keys(),
                restored_optimizer_state.keys(),
            )

            self.assertEqual(
                len(
                    source_optimizer_state[
                        "state"
                    ]
                ),
                len(
                    restored_optimizer_state[
                        "state"
                    ]
                ),
            )

            self.assertGreater(
                len(
                    restored_optimizer_state[
                        "state"
                    ]
                ),
                0,
            )

    def test_load_without_optimizer_restores_model_only(
        self,
    ) -> None:
        """Evaluation loading may skip the optimizer state."""
        source_agent = create_agent()

        run_one_update(
            source_agent
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = (
                Path(directory)
                / "model_only.pt"
            )

            save_sac_checkpoint(
                agent=source_agent,
                path=checkpoint_path,
                environment_step=100,
            )

            restored_agent = create_agent()

            self.assertEqual(
                len(
                    restored_agent
                    .optimizer
                    .state
                ),
                0,
            )

            info = load_sac_checkpoint(
                agent=restored_agent,
                path=checkpoint_path,
                load_optimizer=False,
            )

            self.assertFalse(
                info.optimizer_loaded
            )

            self.assertEqual(
                len(
                    restored_agent
                    .optimizer
                    .state
                ),
                0,
            )

            for (
                source_parameter,
                restored_parameter,
            ) in zip(
                source_agent.parameters(),
                restored_agent.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(
                    source_parameter,
                    restored_parameter,
                )

    def test_incompatible_observation_dimension_is_rejected(
        self,
    ) -> None:
        """A checkpoint cannot load into a different observation size."""
        source_agent = create_agent()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = (
                Path(directory)
                / "checkpoint.pt"
            )

            save_sac_checkpoint(
                agent=source_agent,
                path=checkpoint_path,
                environment_step=0,
            )

            incompatible_agent = create_agent(
                observation_dim=3,
            )

            with self.assertRaises(
                ValueError
            ):
                load_sac_checkpoint(
                    agent=incompatible_agent,
                    path=checkpoint_path,
                )

    def test_incompatible_gamma_is_rejected(
        self,
    ) -> None:
        """Resume training must not silently change gamma."""
        source_agent = create_agent(
            gamma=0.99
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = (
                Path(directory)
                / "checkpoint.pt"
            )

            save_sac_checkpoint(
                agent=source_agent,
                path=checkpoint_path,
                environment_step=0,
            )

            incompatible_agent = create_agent(
                gamma=0.95
            )

            with self.assertRaises(
                ValueError
            ):
                load_sac_checkpoint(
                    agent=incompatible_agent,
                    path=checkpoint_path,
                )

    def test_missing_checkpoint_is_rejected(
        self,
    ) -> None:
        """Loading a nonexistent file must fail clearly."""
        agent = create_agent()

        with self.assertRaises(
            FileNotFoundError
        ):
            load_sac_checkpoint(
                agent=agent,
                path=(
                    "does_not_exist.pt"
                ),
            )

    def test_negative_environment_step_is_rejected(
        self,
    ) -> None:
        """Training progress cannot be negative."""
        agent = create_agent()

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(
                ValueError
            ):
                save_sac_checkpoint(
                    agent=agent,
                    path=(
                        Path(directory)
                        / "checkpoint.pt"
                    ),
                    environment_step=-1,
                )


if __name__ == "__main__":
    unittest.main()