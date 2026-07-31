"""Evaluation utilities for the single-task SAC baseline.

Evaluation is separated from training so that:

- no replay transitions are stored;
- no gradient updates are performed;
- deterministic or stochastic actor actions can be used;
- one explicit actor head can be evaluated on the current task;
- return, success rate, and episode length are measured independently.

The evaluator does not close the environment automatically because the
same evaluation environment may be reused multiple times.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from envs.cw_env import extract_success


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for deterministic or stochastic policy evaluation.

    Attributes:
        num_episodes:
            Number of complete evaluation episodes.

        max_episode_steps:
            Maximum number of environment steps in one episode.

            These tasks use an episode horizon of 200.

        seed:
            Base evaluation seed.

            Episode i is reset using:

                seed + i

            This provides reproducible but distinct episode resets.
    """

    num_episodes: int = 10
    max_episode_steps: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate evaluation configuration."""
        if self.num_episodes <= 0:
            raise ValueError(
                "num_episodes must be positive."
            )

        if self.max_episode_steps <= 0:
            raise ValueError(
                "max_episode_steps must be positive."
            )


@dataclass(frozen=True)
class EvaluationResult:
    """Results from one deterministic or stochastic evaluation."""

    episode_returns: tuple[float, ...]
    episode_successes: tuple[bool, ...]
    episode_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate that all episode-result sequences align."""
        result_lengths = {
            len(self.episode_returns),
            len(self.episode_successes),
            len(self.episode_lengths),
        }

        if len(result_lengths) != 1:
            raise ValueError(
                "Evaluation result fields must contain the "
                "same number of episodes."
            )

        if len(self.episode_returns) == 0:
            raise ValueError(
                "EvaluationResult must contain at least one episode."
            )

    @property
    def num_episodes(self) -> int:
        """Return the number of evaluated episodes."""
        return len(self.episode_returns)

    @property
    def total_steps(self) -> int:
        """Return the total number of evaluation environment steps."""
        return int(
            sum(self.episode_lengths)
        )

    @property
    def mean_return(self) -> float:
        """Return average episode return."""
        return float(
            np.mean(self.episode_returns)
        )

    @property
    def success_rate(self) -> float:
        """Return the proportion of successful episodes.

        An episode is successful when the environment reports success
        at least once during that episode.
        """
        return float(
            np.mean(self.episode_successes)
        )

    @property
    def mean_episode_length(self) -> float:
        """Return average episode length."""
        return float(
            np.mean(self.episode_lengths)
        )

    def as_dict(self) -> dict[str, float]:
        """Return the main evaluation metrics as ordinary floats."""
        return {
            "average_return": self.mean_return,
            "success_rate": self.success_rate,
            "average_episode_length": (
                self.mean_episode_length
            ),
        }


class SACEvaluator:
    """Evaluate one SAC policy on one environment.

    The evaluator expects the agent to expose:

    - observation_dim;
    - action_dim;
    - select_action(observation, deterministic=True);
    - select_action_with_head(
          observation,
          head_index=...,
          deterministic=True,
      ) when explicit-head evaluation is requested.

    SACAgent satisfies this interface.
    """

    def __init__(
        self,
        env: Any,
        agent: Any,
        config: EvaluationConfig,
    ) -> None:
        """Initialize the evaluator.

        Args:
            env:
                Gymnasium-compatible evaluation environment.

                A separate environment from the training environment is
                recommended for formal experiments.

            agent:
                SACAgent or another object with the same action-selection
                interface.

            config:
                Evaluation configuration.
        """
        self.env = env
        self.agent = agent
        self.config = config

        self._validate_environment_dimensions()

    def evaluate(
        self,
        *,
        deterministic: bool = True,
        head_index: int | None = None,
    ) -> EvaluationResult:
        """Run evaluation episodes.

        Args:
            deterministic:
                When True, evaluate the Gaussian mean action.

                When False, evaluate stochastic actions sampled from
                the squashed Gaussian policy.

            head_index:
                Optional actor-head index to use explicitly.

                When None, the task ID already appended to the
                observation selects the current task head.

                When provided, the physical observation remains from
                the current environment, but action selection is forced
                through the requested actor head. This is used to
                compare previous heads on a new task during
                best-return exploration.
        """
        if head_index is not None:
            if not isinstance(head_index, int):
                raise TypeError("head_index must be an int or None.")

            num_tasks = getattr(self.agent, "num_tasks", None)
            if num_tasks is not None and not 0 <= head_index < int(num_tasks):
                raise ValueError(
                    f"head_index must be in [0, {int(num_tasks)}), "
                    f"got {head_index}."
                )

        episode_returns: list[float] = []
        episode_successes: list[bool] = []
        episode_lengths: list[int] = []

        # nn.Module objects expose a `training` flag.
        #
        # Save and restore it so evaluation does not permanently alter
        # the state used by later training calls.
        previous_training_mode = getattr(
            self.agent,
            "training",
            None,
        )

        # Stochastic evaluation must be reproducible without changing
        # the random stream used by subsequent training updates.
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        eval_method = getattr(
            self.agent,
            "eval",
            None,
        )

        if callable(eval_method):
            eval_method()

        try:
            for episode_index in range(
                self.config.num_episodes
            ):
                observation = self._reset_environment(
                    seed=(
                        self.config.seed
                        + episode_index
                    )
                )

                episode_return = 0.0
                episode_length = 0
                episode_success = False

                for _ in range(
                    self.config.max_episode_steps
                ):
                    if head_index is None:
                        action = self.agent.select_action(
                            observation,
                            deterministic=deterministic,
                        )
                    else:
                        select_with_head = getattr(
                            self.agent,
                            "select_action_with_head",
                            None,
                        )
                        if not callable(select_with_head):
                            raise AttributeError(
                                "Explicit-head evaluation requires the "
                                "agent to expose select_action_with_head()."
                            )

                        action = select_with_head(
                            observation,
                            head_index=head_index,
                            deterministic=deterministic,
                        )

                    action = self._prepare_action(
                        action
                    )

                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = self._step_environment(
                        action
                    )

                    episode_return += reward
                    episode_length += 1

                    # MetaWorld may report success before the episode
                    # ends. Once achieved, the complete episode is
                    # counted as successful.
                    episode_success = (
                        episode_success
                        or bool(
                            extract_success(info)
                        )
                    )

                    observation = next_observation

                    if terminated or truncated:
                        break

                episode_returns.append(
                    float(episode_return)
                )

                episode_successes.append(
                    bool(episode_success)
                )

                episode_lengths.append(
                    int(episode_length)
                )

        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)

            # Restore the exact previous nn.Module training mode.
            if previous_training_mode is not None:
                train_method = getattr(
                    self.agent,
                    "train",
                    None,
                )

                if callable(train_method):
                    train_method(
                        bool(previous_training_mode)
                    )

        return EvaluationResult(
            episode_returns=tuple(
                episode_returns
            ),
            episode_successes=tuple(
                episode_successes
            ),
            episode_lengths=tuple(
                episode_lengths
            ),
        )

    def _reset_environment(
        self,
        *,
        seed: int,
    ) -> np.ndarray:
        """Reset a Gymnasium or older Gym-style environment."""
        try:
            reset_result = self.env.reset(
                seed=seed
            )
        except TypeError:
            reset_result = self.env.reset()

        if (
            isinstance(reset_result, tuple)
            and len(reset_result) == 2
        ):
            observation = reset_result[0]
        else:
            observation = reset_result

        return self._prepare_observation(
            observation
        )

    def _step_environment(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Normalize Gymnasium five-value and Gym four-value steps."""
        step_result = self.env.step(action)

        if not isinstance(step_result, tuple):
            raise TypeError(
                "env.step() must return a tuple."
            )

        if len(step_result) == 5:
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = step_result

        elif len(step_result) == 4:
            (
                next_observation,
                reward,
                done,
                info,
            ) = step_result

            info_mapping = (
                info
                if isinstance(info, Mapping)
                else {}
            )

            truncated = bool(
                info_mapping.get(
                    "TimeLimit.truncated",
                    False,
                )
            )

            terminated = bool(
                done and not truncated
            )

        else:
            raise ValueError(
                "env.step() must return either 4 or 5 values."
            )

        reward_value = float(reward)

        if not math.isfinite(
            reward_value
        ):
            raise ValueError(
                "Environment reward must be finite."
            )

        normalized_info = (
            dict(info)
            if isinstance(info, Mapping)
            else {}
        )

        return (
            self._prepare_observation(
                next_observation
            ),
            reward_value,
            bool(terminated),
            bool(truncated),
            normalized_info,
        )

    def _prepare_observation(
        self,
        observation: Any,
    ) -> np.ndarray:
        """Validate one observation returned by the environment."""
        observation_array = np.asarray(
            observation,
            dtype=np.float32,
        )

        expected_shape = (
            self.agent.observation_dim,
        )

        if (
            observation_array.shape
            != expected_shape
        ):
            raise ValueError(
                "Evaluation observation has incorrect shape: "
                f"expected {expected_shape}, "
                f"got {observation_array.shape}."
            )

        if not np.all(
            np.isfinite(observation_array)
        ):
            raise ValueError(
                "Evaluation observation contains "
                "non-finite values."
            )

        return observation_array

    def _prepare_action(
        self,
        action: Any,
    ) -> np.ndarray:
        """Validate one policy action."""
        action_array = np.asarray(
            action,
            dtype=np.float32,
        )

        expected_shape = (
            self.agent.action_dim,
        )

        if action_array.shape != expected_shape:
            raise ValueError(
                "Evaluation action has incorrect shape: "
                f"expected {expected_shape}, "
                f"got {action_array.shape}."
            )

        if not np.all(
            np.isfinite(action_array)
        ):
            raise ValueError(
                "Evaluation action contains non-finite values."
            )

        return action_array

    def _validate_environment_dimensions(
        self,
    ) -> None:
        """Check that evaluation environment and agent dimensions match."""
        observation_space = getattr(
            self.env,
            "observation_space",
            None,
        )

        action_space = getattr(
            self.env,
            "action_space",
            None,
        )

        if observation_space is None:
            raise ValueError(
                "Evaluation environment must expose "
                "observation_space."
            )

        if action_space is None:
            raise ValueError(
                "Evaluation environment must expose action_space."
            )

        observation_shape = getattr(
            observation_space,
            "shape",
            None,
        )

        action_shape = getattr(
            action_space,
            "shape",
            None,
        )

        expected_observation_shape = (
            self.agent.observation_dim,
        )

        expected_action_shape = (
            self.agent.action_dim,
        )

        if (
            observation_shape
            != expected_observation_shape
        ):
            raise ValueError(
                "Evaluation environment and agent observation "
                "dimensions do not match: "
                f"environment={observation_shape}, "
                f"agent={expected_observation_shape}."
            )

        if (
            action_shape
            != expected_action_shape
        ):
            raise ValueError(
                "Evaluation environment and agent action "
                "dimensions do not match: "
                f"environment={action_shape}, "
                f"agent={expected_action_shape}."
            )
