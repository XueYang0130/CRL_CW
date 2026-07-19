import unittest
from dataclasses import dataclass

import numpy as np
import torch

from crl_cw.training import (
    SACTrainer,
    SACTrainerConfig,
)


OBSERVATION_DIM = 39
ACTION_DIM = 4


class FakeObservationSpace:
    """Minimal observation-space object."""

    shape = (
        OBSERVATION_DIM,
    )


class CountingActionSpace:
    """Action space that records random exploration calls."""

    shape = (
        ACTION_DIM,
    )

    def __init__(
        self,
    ) -> None:
        self.sample_count = 0
        self.seed_value: int | None = None

    def sample(
        self,
    ) -> np.ndarray:
        """Return one fixed random-exploration action."""
        self.sample_count += 1

        return np.full(
            ACTION_DIM,
            -0.5,
            dtype=np.float32,
        )

    def seed(
        self,
        seed: int,
    ) -> None:
        """Record the action-space seed."""
        self.seed_value = seed


class FakeEnvironment:
    """Small Gymnasium-style environment for Trainer tests."""

    observation_space = (
        FakeObservationSpace()
    )

    def __init__(
        self,
        *,
        terminate_at: int | None = None,
    ) -> None:
        self.action_space = (
            CountingActionSpace()
        )

        self.terminate_at = (
            terminate_at
        )

        self.episode_step = 0
        self.reset_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[
        np.ndarray,
        dict,
    ]:
        """Reset the fake episode."""
        del seed

        self.episode_step = 0
        self.reset_count += 1

        observation = np.zeros(
            OBSERVATION_DIM,
            dtype=np.float32,
        )

        return observation, {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict,
    ]:
        """Execute one fake transition."""
        if action.shape != (
            ACTION_DIM,
        ):
            raise AssertionError(
                "Trainer supplied invalid action shape."
            )

        self.episode_step += 1

        observation = np.full(
            OBSERVATION_DIM,
            float(self.episode_step),
            dtype=np.float32,
        )

        terminated = (
            self.terminate_at is not None
            and self.episode_step
            >= self.terminate_at
        )

        return (
            observation,
            1.0,
            terminated,
            False,
            {},
        )


@dataclass(frozen=True)
class FakeReplayBatch:
    """Fake batch matching the real ReplayBatch interface."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor


class FakeReplayBuffer:
    """Fake buffer matching the real ReplayBuffer API."""

    def __init__(
        self,
    ) -> None:
        self.transitions: list[
            tuple[
                np.ndarray,
                np.ndarray,
                float,
                np.ndarray,
                bool,
            ]
        ] = []

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one fake transition."""
        self.transitions.append(
            (
                np.asarray(
                    observation,
                    dtype=np.float32,
                ).copy(),
                np.asarray(
                    action,
                    dtype=np.float32,
                ).copy(),
                float(reward),
                np.asarray(
                    next_observation,
                    dtype=np.float32,
                ).copy(),
                bool(terminated),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> FakeReplayBatch:
        """Return a fake batch matching ReplayBatch."""
        if not self.transitions:
            raise RuntimeError(
                "Cannot sample an empty fake buffer."
            )

        indexes = [
            index
            % len(self.transitions)
            for index in range(
                batch_size
            )
        ]

        selected = [
            self.transitions[index]
            for index in indexes
        ]

        return FakeReplayBatch(
            observations=torch.as_tensor(
                np.stack(
                    [
                        item[0]
                        for item in selected
                    ]
                ),
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(
                np.stack(
                    [
                        item[1]
                        for item in selected
                    ]
                ),
                dtype=torch.float32,
                device=device,
            ),
            rewards=torch.as_tensor(
                [
                    item[2]
                    for item in selected
                ],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1),
            next_observations=torch.as_tensor(
                np.stack(
                    [
                        item[3]
                        for item in selected
                    ]
                ),
                dtype=torch.float32,
                device=device,
            ),
            terminated=torch.as_tensor(
                [
                    item[4]
                    for item in selected
                ],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1),
        )


class FakeAgent:
    """Agent spy used to test Trainer scheduling."""

    observation_dim = (
        OBSERVATION_DIM
    )
    action_dim = ACTION_DIM
    device = torch.device(
        "cpu"
    )

    def __init__(
        self,
    ) -> None:
        self.policy_action_count = 0
        self.update_count = 0

    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Return one fixed learned-policy action."""
        if observation.shape != (
            OBSERVATION_DIM,
        ):
            raise AssertionError(
                "Trainer supplied invalid observation."
            )

        if deterministic:
            raise AssertionError(
                "Training must use stochastic actor actions."
            )

        self.policy_action_count += 1

        return np.full(
            ACTION_DIM,
            0.25,
            dtype=np.float32,
        )

    def update_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        """Record one fake SAC update."""
        batch_size = (
            observations.shape[0]
        )

        if observations.shape != (
            batch_size,
            OBSERVATION_DIM,
        ):
            raise AssertionError(
                "Unexpected observation batch shape."
            )

        if actions.shape != (
            batch_size,
            ACTION_DIM,
        ):
            raise AssertionError(
                "Unexpected action batch shape."
            )

        if next_observations.shape != (
            batch_size,
            OBSERVATION_DIM,
        ):
            raise AssertionError(
                "Unexpected next-observation shape."
            )

        if rewards.shape[0] != batch_size:
            raise AssertionError(
                "Unexpected reward batch size."
            )

        if dones.shape[0] != batch_size:
            raise AssertionError(
                "Unexpected done batch size."
            )

        self.update_count += 1

        return {
            "actor_loss": 1.0,
            "q1_loss": 2.0,
            "q2_loss": 3.0,
            "alpha_loss": 4.0,
            "alpha": 0.5,
            "q1_mean": 0.0,
            "q2_mean": 0.0,
            "q_target_mean": 1.0,
            "log_prob_mean": -1.0,
        }


class TestSACTrainer(unittest.TestCase):
    def test_action_and_update_schedule(
        self,
    ) -> None:
        """Trainer must follow random-action and update schedules."""
        env = FakeEnvironment()
        agent = FakeAgent()
        replay_buffer = (
            FakeReplayBuffer()
        )

        config = SACTrainerConfig(
            total_steps=7,
            batch_size=2,
            start_steps=2,
            update_after=2,
            update_every=2,
            max_episode_steps=100,
            seed=7,
        )

        trainer = SACTrainer(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            config=config,
        )

        summary = trainer.train()

        # Current reference-compatible condition:
        #
        # environment_step <= start_steps
        #
        # Therefore steps 0, 1, and 2 use random actions.
        self.assertEqual(
            env.action_space.sample_count,
            3,
        )

        # Steps 3, 4, 5, and 6 use the stochastic actor.
        self.assertEqual(
            agent.policy_action_count,
            4,
        )

        # Update periods:
        #
        # environment step 2
        # environment step 4
        # environment step 6
        #
        # Two updates at each period:
        #
        # 3 * 2 = 6
        self.assertEqual(
            agent.update_count,
            6,
        )

        self.assertEqual(
            summary.gradient_updates,
            6,
        )

        self.assertEqual(
            summary.total_env_steps,
            7,
        )

        self.assertEqual(
            len(
                replay_buffer.transitions
            ),
            7,
        )

        self.assertIsNotNone(
            summary.last_update_metrics
        )

        self.assertEqual(
            env.action_space.seed_value,
            7,
        )

    def test_manual_time_limit_is_not_stored_as_terminal(
        self,
    ) -> None:
        """A manual episode horizon must store terminated=False."""
        env = FakeEnvironment(
            terminate_at=None
        )

        agent = FakeAgent()

        replay_buffer = (
            FakeReplayBuffer()
        )

        trainer = SACTrainer(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            config=SACTrainerConfig(
                total_steps=3,
                batch_size=2,
                start_steps=100,
                update_after=100,
                update_every=2,
                max_episode_steps=3,
            ),
        )

        summary = trainer.train()

        self.assertEqual(
            summary.completed_episodes,
            1,
        )

        self.assertEqual(
            summary.episode_lengths,
            (3,),
        )

        self.assertEqual(
            summary.episode_returns,
            (3.0,),
        )

        # The episode ended only because of the trainer horizon.
        self.assertFalse(
            replay_buffer
            .transitions[-1][4]
        )

    def test_true_termination_is_stored_as_done(
        self,
    ) -> None:
        """A genuine environment terminal must store terminated=True."""
        env = FakeEnvironment(
            terminate_at=2
        )

        agent = FakeAgent()

        replay_buffer = (
            FakeReplayBuffer()
        )

        trainer = SACTrainer(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            config=SACTrainerConfig(
                total_steps=2,
                batch_size=2,
                start_steps=100,
                update_after=100,
                update_every=2,
                max_episode_steps=200,
            ),
        )

        summary = trainer.train()

        self.assertEqual(
            summary.completed_episodes,
            1,
        )

        self.assertEqual(
            summary.episode_lengths,
            (2,),
        )

        self.assertTrue(
            replay_buffer
            .transitions[-1][4]
        )

    def test_invalid_environment_dimensions_are_rejected(
        self,
    ) -> None:
        """Trainer must reject environment/agent mismatch."""
        env = FakeEnvironment()

        env.observation_space = type(
            "WrongObservationSpace",
            (),
            {
                "shape": (3,),
            },
        )()

        with self.assertRaises(
            ValueError
        ):
            SACTrainer(
                env=env,
                agent=FakeAgent(),
                replay_buffer=(
                    FakeReplayBuffer()
                ),
                config=SACTrainerConfig(
                    total_steps=5,
                ),
            )

    def test_step_callback_receives_completed_progress(
        self,
    ) -> None:
        """Callback must run after every completed environment step."""
        env = FakeEnvironment()
        agent = FakeAgent()

        replay_buffer = (
            FakeReplayBuffer()
        )

        trainer = SACTrainer(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            config=SACTrainerConfig(
                total_steps=3,
                batch_size=2,
                start_steps=100,
                update_after=100,
                update_every=2,
                max_episode_steps=100,
            ),
        )

        callback_records: list[
            tuple[
                int,
                int,
                dict[str, float] | None,
            ]
        ] = []

        def callback(
            completed_steps: int,
            gradient_updates: int,
            metrics: dict[str, float] | None,
        ) -> None:
            callback_records.append(
                (
                    completed_steps,
                    gradient_updates,
                    metrics,
                )
            )

        trainer.train(
            step_callback=callback
        )

        self.assertEqual(
            [
                record[0]
                for record
                in callback_records
            ],
            [
                1,
                2,
                3,
            ],
        )

        self.assertTrue(
            all(
                record[1] == 0
                for record
                in callback_records
            )
        )

        self.assertTrue(
            all(
                record[2] is None
                for record
                in callback_records
            )
        )

    def test_callback_receives_latest_update_metrics(
        self,
    ) -> None:
        """Callback must receive metrics after scheduled updates."""
        env = FakeEnvironment()
        agent = FakeAgent()

        replay_buffer = (
            FakeReplayBuffer()
        )

        trainer = SACTrainer(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            config=SACTrainerConfig(
                total_steps=3,
                batch_size=2,
                start_steps=100,
                update_after=2,
                update_every=2,
                max_episode_steps=100,
            ),
        )

        callback_records: list[
            tuple[
                int,
                int,
                dict[str, float] | None,
            ]
        ] = []

        def callback(
            completed_steps: int,
            gradient_updates: int,
            metrics: dict[str, float] | None,
        ) -> None:
            callback_records.append(
                (
                    completed_steps,
                    gradient_updates,
                    metrics,
                )
            )

        trainer.train(
            step_callback=callback
        )

        # Updates happen at zero-based environment_step=2,
        # meaning after three completed environment interactions.
        self.assertEqual(
            callback_records[-1][0],
            3,
        )

        self.assertEqual(
            callback_records[-1][1],
            2,
        )

        self.assertIsNotNone(
            callback_records[-1][2]
        )

        assert (
            callback_records[-1][2]
            is not None
        )

        self.assertEqual(
            callback_records[-1][2][
                "actor_loss"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()