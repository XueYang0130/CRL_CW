import unittest
from dataclasses import dataclass

import numpy as np

from crl_cw.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    SACEvaluator,
)


OBSERVATION_DIM = 39
ACTION_DIM = 4


class FakeObservationSpace:
    """Minimal observation-space object."""

    shape = (
        OBSERVATION_DIM,
    )


class FakeActionSpace:
    """Minimal action-space object."""

    shape = (
        ACTION_DIM,
    )


@dataclass(frozen=True)
class FakeStep:
    """One predefined fake-environment transition."""

    reward: float
    success: bool = False
    terminated: bool = False
    truncated: bool = False


class FakeEvaluationEnvironment:
    """Environment containing predefined evaluation episodes."""

    observation_space = (
        FakeObservationSpace()
    )

    action_space = (
        FakeActionSpace()
    )

    def __init__(
        self,
        episodes: list[
            list[FakeStep]
        ],
    ) -> None:
        self.episodes = episodes

        self.reset_seeds: list[
            int | None
        ] = []

        self._episode_index = -1
        self._step_index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[
        np.ndarray,
        dict,
    ]:
        """Advance to the next predefined episode."""
        self._episode_index += 1
        self._step_index = 0

        self.reset_seeds.append(
            seed
        )

        if (
            self._episode_index
            >= len(self.episodes)
        ):
            raise RuntimeError(
                "Fake environment has no remaining episode."
            )

        observation = np.full(
            OBSERVATION_DIM,
            float(
                self._episode_index
            ),
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
        """Execute the next predefined transition."""
        if action.shape != (
            ACTION_DIM,
        ):
            raise AssertionError(
                "Evaluator supplied invalid action shape."
            )

        episode = self.episodes[
            self._episode_index
        ]

        if (
            self._step_index
            >= len(episode)
        ):
            raise RuntimeError(
                "Evaluator stepped beyond the predefined episode."
            )

        step = episode[
            self._step_index
        ]

        self._step_index += 1

        observation = np.full(
            OBSERVATION_DIM,
            float(
                self._step_index
            ),
            dtype=np.float32,
        )

        return (
            observation,
            step.reward,
            step.terminated,
            step.truncated,
            {
                "success": float(
                    step.success
                ),
            },
        )


class FakeEvaluationAgent:
    """Agent spy recording deterministic/stochastic evaluation calls."""

    observation_dim = (
        OBSERVATION_DIM
    )

    action_dim = (
        ACTION_DIM
    )

    def __init__(
        self,
    ) -> None:
        self.training = True

        self.deterministic_values: list[
            bool
        ] = []

    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Return one fixed valid action."""
        if observation.shape != (
            OBSERVATION_DIM,
        ):
            raise AssertionError(
                "Evaluator supplied invalid observation."
            )

        self.deterministic_values.append(
            deterministic
        )

        return np.zeros(
            ACTION_DIM,
            dtype=np.float32,
        )

    def train(
        self,
        mode: bool = True,
    ) -> "FakeEvaluationAgent":
        """Match torch.nn.Module.train()."""
        self.training = bool(
            mode
        )

        return self

    def eval(
        self,
    ) -> "FakeEvaluationAgent":
        """Match torch.nn.Module.eval()."""
        return self.train(
            False
        )


class TestEvaluationResult(
    unittest.TestCase
):
    def test_aggregated_metrics(
        self,
    ) -> None:
        """EvaluationResult must calculate all main metrics."""
        result = EvaluationResult(
            episode_returns=(
                3.0,
                1.0,
            ),
            episode_successes=(
                True,
                False,
            ),
            episode_lengths=(
                2,
                4,
            ),
        )

        self.assertEqual(
            result.num_episodes,
            2,
        )

        self.assertEqual(
            result.total_steps,
            6,
        )

        self.assertAlmostEqual(
            result.mean_return,
            2.0,
            places=6,
        )

        self.assertAlmostEqual(
            result.success_rate,
            0.5,
            places=6,
        )

        self.assertAlmostEqual(
            result.mean_episode_length,
            3.0,
            places=6,
        )

        self.assertEqual(
            result.as_dict(),
            {
                "average_return": 2.0,
                "success_rate": 0.5,
                "average_episode_length": 3.0,
            },
        )


class TestSACEvaluator(
    unittest.TestCase
):
    def test_evaluate_returns_correct_metrics(
        self,
    ) -> None:
        """Evaluator must aggregate return, success, and length."""
        env = FakeEvaluationEnvironment(
            episodes=[
                [
                    FakeStep(
                        reward=1.0,
                    ),
                    FakeStep(
                        reward=2.0,
                        success=True,
                        terminated=True,
                    ),
                ],
                [
                    FakeStep(
                        reward=0.5,
                    ),
                    FakeStep(
                        reward=0.5,
                    ),
                    FakeStep(
                        reward=0.5,
                        truncated=True,
                    ),
                ],
            ]
        )

        agent = (
            FakeEvaluationAgent()
        )

        evaluator = SACEvaluator(
            env=env,
            agent=agent,
            config=EvaluationConfig(
                num_episodes=2,
                max_episode_steps=200,
                seed=10,
            ),
        )

        result = evaluator.evaluate(
            deterministic=True
        )

        self.assertEqual(
            result.episode_returns,
            (
                3.0,
                1.5,
            ),
        )

        self.assertEqual(
            result.episode_successes,
            (
                True,
                False,
            ),
        )

        self.assertEqual(
            result.episode_lengths,
            (
                2,
                3,
            ),
        )

        self.assertAlmostEqual(
            result.mean_return,
            2.25,
            places=6,
        )

        self.assertAlmostEqual(
            result.success_rate,
            0.5,
            places=6,
        )

        self.assertAlmostEqual(
            result.mean_episode_length,
            2.5,
            places=6,
        )

        # Each evaluation episode receives a reproducible,
        # but different, reset seed.
        self.assertEqual(
            env.reset_seeds,
            [
                10,
                11,
            ],
        )

        # Every action in this evaluation must use mean action.
        self.assertTrue(
            all(
                agent.deterministic_values
            )
        )

        # Evaluation must restore the previous training mode.
        self.assertTrue(
            agent.training
        )

    def test_success_is_retained_after_first_success_step(
        self,
    ) -> None:
        """One reached task-completion state makes the episode successful."""
        env = FakeEvaluationEnvironment(
            episodes=[
                [
                    FakeStep(
                        reward=0.0,
                        success=True,
                    ),
                    FakeStep(
                        reward=0.0,
                        success=False,
                    ),
                    FakeStep(
                        reward=0.0,
                        success=False,
                        terminated=True,
                    ),
                ],
            ]
        )

        evaluator = SACEvaluator(
            env=env,
            agent=(
                FakeEvaluationAgent()
            ),
            config=EvaluationConfig(
                num_episodes=1,
                max_episode_steps=200,
            ),
        )

        result = evaluator.evaluate(
            deterministic=True
        )

        self.assertEqual(
            result.episode_successes,
            (True,),
        )

        self.assertAlmostEqual(
            result.success_rate,
            1.0,
            places=6,
        )

    def test_manual_episode_horizon_is_respected(
        self,
    ) -> None:
        """Evaluator must stop at max_episode_steps."""
        env = FakeEvaluationEnvironment(
            episodes=[
                [
                    FakeStep(
                        reward=1.0
                    ),
                    FakeStep(
                        reward=1.0
                    ),
                    FakeStep(
                        reward=100.0
                    ),
                ],
            ]
        )

        evaluator = SACEvaluator(
            env=env,
            agent=(
                FakeEvaluationAgent()
            ),
            config=EvaluationConfig(
                num_episodes=1,
                max_episode_steps=2,
            ),
        )

        result = evaluator.evaluate(
            deterministic=True
        )

        # The third predefined reward must not be collected.
        self.assertEqual(
            result.episode_returns,
            (2.0,),
        )

        self.assertEqual(
            result.episode_lengths,
            (2,),
        )

    def test_invalid_environment_dimensions_are_rejected(
        self,
    ) -> None:
        """Evaluator must reject environment/agent mismatch."""
        env = FakeEvaluationEnvironment(
            episodes=[
                [
                    FakeStep(
                        reward=0.0,
                        terminated=True,
                    ),
                ],
            ]
        )

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
            SACEvaluator(
                env=env,
                agent=(
                    FakeEvaluationAgent()
                ),
                config=EvaluationConfig(
                    num_episodes=1,
                ),
            )

    def test_stochastic_evaluation_mode_is_forwarded(
        self,
    ) -> None:
        """Evaluator must forward deterministic=False to the actor."""
        env = FakeEvaluationEnvironment(
            episodes=[
                [
                    FakeStep(
                        reward=1.0,
                        terminated=True,
                    ),
                ],
            ]
        )

        agent = (
            FakeEvaluationAgent()
        )

        evaluator = SACEvaluator(
            env=env,
            agent=agent,
            config=EvaluationConfig(
                num_episodes=1,
                max_episode_steps=200,
            ),
        )

        evaluator.evaluate(
            deterministic=False
        )

        self.assertEqual(
            agent.deterministic_values,
            [
                False,
            ],
        )

        self.assertTrue(
            agent.training
        )

    def test_default_evaluation_is_deterministic(
        self,
    ) -> None:
        """Calling evaluate() without an argument must use mean action."""
        env = FakeEvaluationEnvironment(
            episodes=[
                [
                    FakeStep(
                        reward=1.0,
                        terminated=True,
                    ),
                ],
            ]
        )

        agent = (
            FakeEvaluationAgent()
        )

        evaluator = SACEvaluator(
            env=env,
            agent=agent,
            config=EvaluationConfig(
                num_episodes=1,
            ),
        )

        evaluator.evaluate()

        self.assertEqual(
            agent.deterministic_values,
            [
                True,
            ],
        )


if __name__ == "__main__":
    unittest.main()