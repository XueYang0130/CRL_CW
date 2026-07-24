"""Single-task training loop for the SAC baseline.

This module connects:

- one Gymnasium-compatible environment;
- one standard online SAC replay buffer;
- one SACAgent.

The online replay buffer stores transition data used for SAC gradient
updates. It is different from the future continual-learning task memory,
which may store trained policies, checkpoints, priors, or other knowledge.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.special
import torch

from agents import ReplayBuffer, SACAgent
from envs import extract_success


class ExplorationHeadSelector:
    """Reference-style exploration helper for multi-head continual SAC."""

    def __init__(
        self,
        strategy: str,
        num_available_heads: int,
    ) -> None:
        self.strategy = strategy
        self.num_available_heads = num_available_heads
        self.current_head_index: int | None = None
        self.current_rewards: list[float] = []
        self.current_successes: list[float] = []
        self.episode_returns = [[] for _ in range(self.num_available_heads)]
        self.episode_successes = [[] for _ in range(self.num_available_heads)]

    def tell_results(
        self,
        reward: float,
        success: float,
    ) -> None:
        if self.current_head_index is None:
            raise RuntimeError("Exploration head must be selected before logging results.")
        self.current_rewards.append(reward)
        self.current_successes.append(success)

    def start_new_episode(self) -> int:
        if self.current_head_index is not None:
            self.episode_returns[self.current_head_index].append(
                float(sum(self.current_rewards))
            )
            self.episode_successes[self.current_head_index].append(
                float(any(value > 0.0 for value in self.current_successes))
            )
            self.current_rewards = []
            self.current_successes = []

        self.current_head_index = self._select_head()
        return self.current_head_index

    def _select_head(self) -> int:
        if self.strategy == "current":
            return self.num_available_heads - 1
        if self.strategy == "previous":
            return self.num_available_heads - 2
        if self.strategy == "uniform_previous":
            return random.randint(0, self.num_available_heads - 2)
        if self.strategy == "uniform_previous_or_current":
            return random.randint(0, self.num_available_heads - 1)

        for head_index in range(self.num_available_heads - 1):
            if len(self.episode_returns[head_index]) == 0:
                return head_index

        if self.strategy in {"best_success", "best_return"}:
            scores = []
            for head_index in range(self.num_available_heads - 1):
                if self.strategy == "best_success":
                    score = float(np.mean(self.episode_successes[head_index]))
                else:
                    score = float(np.mean(self.episode_returns[head_index]))
                scores.append(score)
            return int(np.argmax(scores))

        if self.strategy.startswith("softmax_return_"):
            temperature = float(self.strategy[len("softmax_return_"):])
            scores = np.asarray(
                [
                    np.mean(self.episode_returns[head_index])
                    for head_index in range(self.num_available_heads - 1)
                ],
                dtype=np.float64,
            )
            min_score = float(scores.min())
            max_score = float(scores.max())
            scores = (scores - min_score) / (max_score - min_score + 1e-6)
            probabilities = scipy.special.softmax(scores / temperature)
            return int(
                np.random.choice(
                    np.arange(self.num_available_heads - 1),
                    p=probabilities,
                )
            )

        raise ValueError(f"Unsupported exploration strategy: {self.strategy}")


@dataclass(frozen=True)
class SACTrainerConfig:
    """Configuration for one SAC training period.

    Attributes:
        total_steps:
            Total number of environment interactions.

        batch_size:
            Number of replay transitions sampled for one gradient update.

        start_steps:
            Initial environment steps used for exploration.

            If ``exploration_head_index`` is None, actions are sampled
            uniformly from the environment action space.

            If ``exploration_head_index`` is provided, that previous
            actor head generates the exploration actions.

            For compatibility with the reference implementation, the
            condition is:

                environment_step <= start_steps

            where environment_step is zero-based.

        exploration_head_index:
            Optional previous actor head used during the initial
            exploration period.

            This supports best-return exploration. The
            selected old head only generates data. Replay observations
            retain the current task ID, so SAC updates train the current
            task actor and critic heads.

        update_after:
            Zero-based environment step after which learning may begin.

            This schedule is independent from ``start_steps``. For
            example, with ``start_steps=10_000`` and
            ``update_after=1_000``, the trainer still uses random
            actions through step 10,000, but SAC gradient updates begin
            once enough replay data has been collected at step 1,000.

        update_every:
            Number of environment steps between update periods.

            At every update period, the trainer performs
            `update_every` gradient updates.

        max_episode_steps:
            Maximum length of one episode.

        seed:
            Seed used for the environment action space and, when enabled,
            NumPy and Torch.

        reseed_global_rng:
            Whether construction of this trainer should reset the global
            NumPy and Torch random-number generators.

            This should normally be True for an independent single-task
            experiment.

            During one continuous CW10 run, it should be False for each
            new task trainer, because the global random streams should
            continue rather than restart at every task boundary.
    """

    total_steps: int
    batch_size: int = 128
    start_steps: int = 10_000
    exploration_head_index: int | None = None
    exploration_strategy: str | None = None
    exploration_available_heads: int | None = None
    update_after: int = 1_000
    update_every: int = 50
    max_episode_steps: int = 200
    callback_every_steps: int | None = None
    seed: int = 0
    reseed_global_rng: bool = True
    guide_head_index: int | None = None
    guide_steps: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.total_steps <= 0:
            raise ValueError(
                "total_steps must be positive."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.start_steps < 0:
            raise ValueError(
                "start_steps must be non-negative."
            )

        if (
            self.exploration_head_index is not None
            and not isinstance(
                self.exploration_head_index,
                int,
            )
        ):
            raise TypeError(
                "exploration_head_index must be an int or None."
            )

        if (
            self.exploration_head_index is not None
            and self.exploration_head_index < 0
        ):
            raise ValueError(
                "exploration_head_index must be non-negative."
            )

        if (
            self.exploration_strategy is not None
            and not isinstance(self.exploration_strategy, str)
        ):
            raise TypeError("exploration_strategy must be a string or None.")

        if self.exploration_available_heads is not None:
            if self.exploration_available_heads <= 0:
                raise ValueError("exploration_available_heads must be positive.")
            if self.exploration_available_heads == 1 and self.exploration_strategy is not None:
                raise ValueError("Exploration strategy requires at least two available heads.")

        if self.update_after < 0:
            raise ValueError(
                "update_after must be non-negative."
            )

        if self.update_every <= 0:
            raise ValueError(
                "update_every must be positive."
            )

        if self.max_episode_steps <= 0:
            raise ValueError(
                "max_episode_steps must be positive."
            )
        if self.guide_steps < 0 or self.guide_steps > self.max_episode_steps:
            raise ValueError("guide_steps must be within the episode horizon.")
        if self.guide_steps > 0 and self.guide_head_index is None:
            raise ValueError("guide_head_index is required when guide_steps is positive.")

        if (
            self.callback_every_steps is not None
            and self.callback_every_steps <= 0
        ):
            raise ValueError(
                "callback_every_steps must be positive or None."
            )

        if not isinstance(
            self.reseed_global_rng,
            bool,
        ):
            raise TypeError(
                "reseed_global_rng must be a bool."
            )


@dataclass(frozen=True)
class TrainingSummary:
    """Summary returned after one training period."""

    total_env_steps: int
    gradient_updates: int
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    last_update_metrics: dict[str, float] | None

    @property
    def completed_episodes(self) -> int:
        """Return the number of completed episodes."""
        return len(
            self.episode_returns
        )

    @property
    def mean_episode_return(self) -> float:
        """Return the mean completed-episode return."""
        if not self.episode_returns:
            return 0.0

        return float(
            np.mean(
                self.episode_returns
            )
        )

    @property
    def mean_episode_length(self) -> float:
        """Return the mean completed-episode length."""
        if not self.episode_lengths:
            return 0.0

        return float(
            np.mean(
                self.episode_lengths
            )
        )


TrainingStepCallback = Callable[
    [
        int,
        int,
        dict[str, float] | None,
    ],
    None,
]


class SACTrainer:
    """Online SAC interaction and update loop.

    The trainer receives the replay buffer rather than creating it
    internally. This allows the outer experiment runner to decide what
    should happen at task boundaries, for example:

    - clear the online replay buffer;
    - retain all transitions for Perfect Memory;
    - replace the buffer;
    - use a separate task-policy memory.

    This class itself contains no continual-learning task-switch logic.
    """

    def __init__(
        self,
        env: Any,
        agent: SACAgent,
        replay_buffer: ReplayBuffer,
        config: SACTrainerConfig,
    ) -> None:
        """Initialize the trainer."""
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.config = config
        self._exploration_selector: ExplorationHeadSelector | None = None
        if self.config.exploration_strategy is not None:
            available_heads = self.config.exploration_available_heads
            if available_heads is None:
                raise ValueError(
                    "exploration_available_heads must be set when exploration_strategy is used."
                )
            self._exploration_selector = ExplorationHeadSelector(
                self.config.exploration_strategy,
                available_heads,
            )

        self._validate_environment_dimensions()
        self._validate_exploration_head()
        self._set_random_seeds()
        self._guide_steps = self.config.guide_steps

    @property
    def guide_steps(self) -> int:
        return self._guide_steps

    def set_guide_steps(self, guide_steps: int) -> None:
        if not 0 <= guide_steps <= self.config.max_episode_steps:
            raise ValueError("guide_steps must be within the episode horizon.")
        self._guide_steps = guide_steps

    def train(
        self,
        *,
        step_callback: TrainingStepCallback | None = None,
    ) -> TrainingSummary:
        """Run one SAC training period.

        Args:
            step_callback:
                Optional callback executed at configured callback
                boundaries, after all gradient updates scheduled for
                the current environment step.

                It receives:

                    completed environment steps;
                    cumulative gradient updates in this training period;
                    most recent SAC update metrics.

        Returns:
            TrainingSummary containing episode and update statistics.
        """
        observation = self._reset_environment(
            seed=self.config.seed
        )

        episode_return = 0.0
        episode_length = 0

        episode_returns: list[float] = []
        episode_lengths: list[int] = []

        gradient_updates = 0
        last_update_metrics: dict[str, float] | None = None
        for environment_step in range(
            self.config.total_steps
        ):
            # =====================================================
            # 1. Select an exploration action.
            # =====================================================
            action = self._select_action(
                observation=observation,
                environment_step=environment_step,
                episode_step=episode_length,
            )

            # =====================================================
            # 2. Execute the action.
            # =====================================================
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = self._step_environment(
                action
            )

            if (
                self._exploration_selector is not None
                and self._exploration_selector.current_head_index is not None
            ):
                self._exploration_selector.tell_results(
                    reward=reward,
                    success=extract_success(info),
                )

            episode_return += reward
            episode_length += 1

            reached_manual_time_limit = (
                episode_length
                >= self.config.max_episode_steps
            )

            # Only genuine MDP termination is stored as terminated=True.
            #
            # Time-limit truncation should still bootstrap from the next
            # observation in the SAC Bellman target.
            self.replay_buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=bool(
                    terminated
                ),
            )

            observation = next_observation

            episode_ended = bool(
                terminated
                or truncated
                or reached_manual_time_limit
            )

            # =====================================================
            # 3. Record and reset a completed episode.
            # =====================================================
            if episode_ended:
                episode_returns.append(
                    float(
                        episode_return
                    )
                )

                episode_lengths.append(
                    int(
                        episode_length
                    )
                )

                episode_return = 0.0
                episode_length = 0

                # There is no reason to reset after the final requested
                # environment interaction.
                if (
                    environment_step
                    < self.config.total_steps - 1
                ):
                    observation = (
                        self._reset_environment()
                    )
                    if (
                        self._exploration_selector is not None
                        and environment_step < self.config.start_steps
                    ):
                        self._exploration_selector.start_new_episode()

            # =====================================================
            # 4. Perform scheduled SAC gradient updates.
            # =====================================================
            completed_steps = environment_step + 1
            should_notify = self._should_notify(
                completed_steps=completed_steps,
                step_callback=step_callback,
            )

            if self._should_update(environment_step):
                for update_index in range(self.config.update_every):
                    batch = self._sample_replay_batch()
                    collect_metrics = bool(
                        update_index
                        == self.config.update_every - 1
                        and self._should_collect_metrics(
                            completed_steps=completed_steps,
                        )
                    )

                    try:
                        update_metrics = self.agent.update_batch(
                            observations=batch["observations"],
                            actions=batch["actions"],
                            rewards=batch["rewards"],
                            next_observations=batch["next_observations"],
                            dones=batch["terminated"],
                            collect_metrics=collect_metrics,
                        )
                    except TypeError:
                        update_metrics = self.agent.update_batch(
                            observations=batch["observations"],
                            actions=batch["actions"],
                            rewards=batch["rewards"],
                            next_observations=batch["next_observations"],
                            dones=batch["terminated"],
                        )

                    if update_metrics is not None:
                        last_update_metrics = update_metrics

                    gradient_updates += 1

            # =====================================================
            # 5. Notify only at requested evaluation boundaries.
            # =====================================================
            if should_notify and step_callback is not None:
                step_callback(
                    completed_steps,
                    gradient_updates,
                    last_update_metrics,
                )

        return TrainingSummary(
            total_env_steps=(
                self.config.total_steps
            ),
            gradient_updates=(
                gradient_updates
            ),
            episode_returns=tuple(
                episode_returns
            ),
            episode_lengths=tuple(
                episode_lengths
            ),
            last_update_metrics=(
                last_update_metrics
            ),
        )

    def _select_action(
        self,
        *,
        observation: np.ndarray,
        environment_step: int,
        episode_step: int,
    ) -> np.ndarray:
        """Select an exploration or current-policy action."""
        if (
            self.config.guide_head_index is not None
            and episode_step < self._guide_steps
        ):
            action = self.agent.select_guide_action(
                observation,
                guide_task_index=self.config.guide_head_index,
                deterministic=False,
            )
        elif self.config.guide_head_index is not None:
            action = self.agent.select_action(
                observation,
                deterministic=False,
            )
        elif environment_step <= self.config.start_steps:
            if self._exploration_selector is not None:
                head_index = self._exploration_selector.current_head_index
                if head_index is None:
                    head_index = self._exploration_selector.start_new_episode()
                action = self.agent.select_action_with_head(
                    observation,
                    head_index=head_index,
                    deterministic=False,
                )
            elif self.config.exploration_head_index is None:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action_with_head(
                    observation,
                    head_index=self.config.exploration_head_index,
                    deterministic=False,
                )
        else:
            action = self.agent.select_action(
                observation,
                deterministic=False,
            )

        return np.asarray(action, dtype=np.float32)

    def _should_collect_metrics(
        self,
        *,
        completed_steps: int,
    ) -> bool:
        """Collect diagnostics shortly before the next callback."""
        interval = self.config.callback_every_steps
        if interval is None:
            return bool(
                self.config.total_steps - completed_steps
                < self.config.update_every
            )
        next_interval_boundary = (
            ((completed_steps + interval - 1) // interval)
            * interval
        )
        next_callback_step = min(
            next_interval_boundary,
            self.config.total_steps,
        )
        steps_until_callback = (
            next_callback_step - completed_steps
        )
        return bool(
            0 <= steps_until_callback < self.config.update_every
        )

    def _should_notify(
        self,
        *,
        completed_steps: int,
        step_callback: TrainingStepCallback | None,
    ) -> bool:
        """Return whether the runner callback should be invoked."""
        if step_callback is None:
            return False

        interval = self.config.callback_every_steps
        if completed_steps == self.config.total_steps:
            return True
        if interval is None:
            return True
        return bool(
            completed_steps % interval == 0
        )

    def _should_update(
        self,
        environment_step: int,
    ) -> bool:
        """Return whether an update period should start."""
        return bool(
            environment_step
            >= self.config.update_after
            and environment_step
            % self.config.update_every
            == 0
        )

    def _sample_replay_batch(
        self,
    ) -> dict[str, torch.Tensor]:
        """Sample one minibatch using the real ReplayBuffer API."""
        replay_batch = (
            self.replay_buffer.sample(
                batch_size=(
                    self.config.batch_size
                ),
                device=(
                    self.agent.device
                ),
            )
        )

        required_fields = (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminated",
        )

        for field_name in required_fields:
            if not hasattr(
                replay_batch,
                field_name,
            ):
                raise AttributeError(
                    "ReplayBatch is missing required field "
                    f"{field_name!r}."
                )

        return {
            "observations": (
                replay_batch.observations
            ),
            "actions": (
                replay_batch.actions
            ),
            "rewards": (
                replay_batch.rewards
            ),
            "next_observations": (
                replay_batch.next_observations
            ),
            "terminated": (
                replay_batch.terminated
            ),
        }

    def _reset_environment(
        self,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        """Reset a Gymnasium or older Gym-style environment."""
        if seed is None:
            reset_result = (
                self.env.reset()
            )
        else:
            try:
                reset_result = (
                    self.env.reset(
                        seed=seed
                    )
                )
            except TypeError:
                # Compatibility fallback for older Gym wrappers.
                reset_result = (
                    self.env.reset()
                )

        if (
            isinstance(
                reset_result,
                tuple,
            )
            and len(
                reset_result
            )
            == 2
        ):
            observation = (
                reset_result[0]
            )
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
        step_result = self.env.step(
            action
        )

        if not isinstance(
            step_result,
            tuple,
        ):
            raise TypeError(
                "env.step() must return a tuple."
            )

        if len(
            step_result
        ) == 5:
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = step_result

        elif len(
            step_result
        ) == 4:
            (
                next_observation,
                reward,
                done,
                info,
            ) = step_result

            info_mapping = (
                info
                if isinstance(
                    info,
                    Mapping,
                )
                else {}
            )

            truncated = bool(
                info_mapping.get(
                    "TimeLimit.truncated",
                    False,
                )
            )

            terminated = bool(
                done
                and not truncated
            )

        else:
            raise ValueError(
                "env.step() must return either 4 or 5 values."
            )

        reward_value = float(
            reward
        )

        if not math.isfinite(
            reward_value
        ):
            raise ValueError(
                "Environment reward must be finite."
            )

        normalized_info = (
            dict(
                info
            )
            if isinstance(
                info,
                Mapping,
            )
            else {}
        )

        return (
            self._prepare_observation(
                next_observation
            ),
            reward_value,
            bool(
                terminated
            ),
            bool(
                truncated
            ),
            normalized_info,
        )

    def _prepare_observation(
        self,
        observation: Any,
    ) -> np.ndarray:
        """Convert one environment observation to float32."""
        return np.asarray(observation, dtype=np.float32)

    def _validate_environment_dimensions(
        self,
    ) -> None:
        """Check that environment and agent dimensions agree."""
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
                "Environment must expose observation_space."
            )

        if action_space is None:
            raise ValueError(
                "Environment must expose action_space."
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
                "Environment and agent observation dimensions "
                "do not match: "
                f"environment={observation_shape}, "
                f"agent={expected_observation_shape}."
            )

        if (
            action_shape
            != expected_action_shape
        ):
            raise ValueError(
                "Environment and agent action dimensions "
                "do not match: "
                f"environment={action_shape}, "
                f"agent={expected_action_shape}."
            )

    def _validate_exploration_head(
        self,
    ) -> None:
        """Validate the optional previous head used for exploration."""
        head_index = self.config.exploration_head_index
        if head_index is None:
            return

        if not 0 <= head_index < self.agent.num_tasks:
            raise ValueError(
                "exploration_head_index must be in "
                f"[0, {self.agent.num_tasks}), got {head_index}."
            )

        select_with_head = getattr(
            self.agent,
            "select_action_with_head",
            None,
        )
        if not callable(select_with_head):
            raise AttributeError(
                "Using exploration_head_index requires the agent to "
                "expose select_action_with_head()."
            )

    def _set_random_seeds(
        self,
    ) -> None:
        """Seed global RNGs when requested and seed the action space.

        An independent single-task experiment normally sets
        reseed_global_rng=True.

        A continuous CW10 experiment should seed NumPy and Torch once
        before creating the agent, and then use reseed_global_rng=False
        when constructing a trainer for each new task. This prevents the
        random-number streams from restarting at task boundaries.
        """
        if self.config.reseed_global_rng:
            np.random.seed(
                self.config.seed
            )

            torch.manual_seed(
                self.config.seed
            )

        action_space = getattr(
            self.env,
            "action_space",
            None,
        )

        if (
            action_space is not None
            and hasattr(
                action_space,
                "seed",
            )
        ):
            action_space.seed(
                self.config.seed
            )
