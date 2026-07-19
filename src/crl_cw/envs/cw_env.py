"""MetaWorld environment creation for the CW10 benchmark.

Public experiment labels retain the original Continual World ``-v1`` names.
Modern MetaWorld environments are created with their corresponding ``-v3``
names.

For sequential CW10 training, ``append_task_id=True`` appends a ten-way
task one-hot vector to every observation. Single-task scripts remain
backward compatible because the default is ``append_task_id=False``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import metaworld  # noqa: F401  # Registers Meta-World environments.
import numpy as np


CW_TO_METAWORLD_TASK: dict[str, str] = {
    "hammer-v1": "hammer-v3",
    "push-wall-v1": "push-wall-v3",
    "faucet-close-v1": "faucet-close-v3",
    "push-back-v1": "push-back-v3",
    "stick-pull-v1": "stick-pull-v3",
    "handle-press-side-v1": "handle-press-side-v3",
    "push-v1": "push-v3",
    "shelf-place-v1": "shelf-place-v3",
    "window-close-v1": "window-close-v3",
    "peg-unplug-side-v1": "peg-unplug-side-v3",
}

CW10_TASK_NAMES: tuple[str, ...] = tuple(
    CW_TO_METAWORLD_TASK.keys()
)
CW10_NUM_TASKS = len(CW10_TASK_NAMES)
CLONEX_EPISODE_LENGTH = 200


class TaskOneHotObservation(gym.ObservationWrapper):
    """Append a known task ID as a one-hot observation suffix."""

    def __init__(
        self,
        env: gym.Env,
        *,
        task_index: int,
        num_tasks: int,
    ) -> None:
        super().__init__(env)

        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive.")
        if not 0 <= task_index < num_tasks:
            raise ValueError(
                f"task_index must be in [0, {num_tasks}), "
                f"got {task_index}."
            )

        original_space = env.observation_space
        if not isinstance(original_space, gym.spaces.Box):
            raise TypeError(
                "TaskOneHotObservation requires a Box observation space."
            )

        self.task_index = int(task_index)
        self.num_tasks = int(num_tasks)

        original_low = np.asarray(
            original_space.low,
            dtype=np.float32,
        ).reshape(-1)
        original_high = np.asarray(
            original_space.high,
            dtype=np.float32,
        ).reshape(-1)

        low = np.concatenate(
            (
                original_low,
                np.zeros(self.num_tasks, dtype=np.float32),
            )
        )
        high = np.concatenate(
            (
                original_high,
                np.ones(self.num_tasks, dtype=np.float32),
            )
        )

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation_array = np.asarray(
            observation,
            dtype=np.float32,
        ).reshape(-1)

        task_one_hot = np.zeros(
            self.num_tasks,
            dtype=np.float32,
        )
        task_one_hot[self.task_index] = 1.0

        return np.concatenate(
            (observation_array, task_one_hot)
        ).astype(np.float32, copy=False)


def modern_task_name(task_name: str) -> str:
    """Return the modern MetaWorld name corresponding to a CW10 task."""
    try:
        return CW_TO_METAWORLD_TASK[task_name]
    except KeyError as exc:
        supported = ", ".join(CW_TO_METAWORLD_TASK)
        raise ValueError(
            f"Unknown CW10 task {task_name!r}. "
            f"Supported tasks: {supported}"
        ) from exc


def task_index_for_name(task_name: str) -> int:
    """Return the canonical CW10 index for one task name."""
    try:
        return CW10_TASK_NAMES.index(task_name)
    except ValueError as exc:
        supported = ", ".join(CW10_TASK_NAMES)
        raise ValueError(
            f"Unknown CW10 task {task_name!r}. "
            f"Supported tasks: {supported}"
        ) from exc


def make_cw_env(
    task_name: str,
    seed: int = 0,
    render_mode: str | None = None,
    max_episode_steps: int = CLONEX_EPISODE_LENGTH,
    *,
    append_task_id: bool = False,
) -> gym.Env:
    """Create one Gymnasium-compatible CW10 environment.

    Args:
        task_name:
            Original CW10 task label ending in ``-v1``.
        seed:
            Seed passed to MetaWorld environment construction.
        render_mode:
            Optional Gymnasium render mode.
        max_episode_steps:
            Maximum episode length.
        append_task_id:
            Append the canonical ten-way CW10 task one-hot vector.
            Enable this for sequential CW10 training.
    """
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive.")

    env = gym.make(
        "Meta-World/MT1",
        env_name=modern_task_name(task_name),
        seed=seed,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )

    if append_task_id:
        env = TaskOneHotObservation(
            env,
            task_index=task_index_for_name(task_name),
            num_tasks=CW10_NUM_TASKS,
        )

    return env


def extract_success(info: dict[str, Any]) -> float:
    """Extract one step's binary MetaWorld success indicator."""
    raw_success = info.get("success", 0.0)
    values = np.asarray(
        raw_success,
        dtype=np.float32,
    ).reshape(-1)

    if values.size == 0:
        return 0.0
    return float(values[0] > 0.0)
