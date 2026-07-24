from __future__ import annotations

from typing import Any

import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np

CW10_TASKS_V3: tuple[str, ...] = (
    "hammer-v3",
    "push-wall-v3",
    "faucet-close-v3",
    "push-back-v3",
    "stick-pull-v3",
    "handle-press-side-v3",
    "push-v3",
    "shelf-place-v3",
    "window-close-v3",
    "peg-unplug-side-v3",
)
CW10_TASKS_V2: tuple[str, ...] = tuple(
    task_name.replace("-v3", "-v2")
    for task_name in CW10_TASKS_V3
)
CW10_TASKS: tuple[str, ...] = CW10_TASKS_V3
CW10_NUM_TASKS = len(CW10_TASKS_V3)
DEFAULT_EPISODE_LENGTH = 200


class RelaxedObservationSpace(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        original_space = env.observation_space
        if not isinstance(original_space, gym.spaces.Box):
            raise TypeError("RelaxedObservationSpace requires a Box observation space.")

        shape = tuple(np.asarray(original_space.shape, dtype=np.int64))
        low = np.full(shape, -np.inf, dtype=np.float32)
        high = np.full(shape, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)


class TaskOneHotObservation(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        task_index: int,
        num_tasks: int,
    ) -> None:
        super().__init__(env)

        original_space = env.observation_space
        if not isinstance(original_space, gym.spaces.Box):
            raise TypeError("TaskOneHotObservation requires a Box observation space.")
        if not 0 <= task_index < num_tasks:
            raise ValueError(f"task_index must be in [0, {num_tasks}), got {task_index}.")

        self.task_index = task_index
        self.num_tasks = num_tasks

        low = np.asarray(original_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(original_space.high, dtype=np.float32).reshape(-1)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((low, np.zeros(self.num_tasks, dtype=np.float32))),
            high=np.concatenate((high, np.ones(self.num_tasks, dtype=np.float32))),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        task_one_hot = np.zeros(self.num_tasks, dtype=np.float32)
        task_one_hot[self.task_index] = 1.0
        return np.concatenate((observation_array, task_one_hot)).astype(
            np.float32,
            copy=False,
        )


def canonical_cw10_task_name(task_name: str) -> str:
    if task_name.endswith("-v3"):
        return task_name[:-3]
    if task_name.endswith("-v2"):
        return task_name[:-3]
    return task_name


def resolve_env_version(task_name: str, env_version: str | None = None) -> str:
    if env_version is not None:
        if env_version not in {"v2", "v3"}:
            raise ValueError(f"Unsupported env_version {env_version!r}. Expected 'v2' or 'v3'.")
        return env_version
    if task_name.endswith("-v2"):
        return "v2"
    if task_name.endswith("-v3"):
        return "v3"
    return "v3"


def cw10_tasks_for_version(env_version: str = "v3") -> tuple[str, ...]:
    if env_version == "v3":
        return CW10_TASKS_V3
    if env_version == "v2":
        return CW10_TASKS_V2
    raise ValueError(f"Unsupported env_version {env_version!r}. Expected 'v2' or 'v3'.")


def task_index_for_name(task_name: str) -> int:
    canonical_name = canonical_cw10_task_name(task_name)
    canonical_tasks = [canonical_cw10_task_name(name) for name in CW10_TASKS_V3]
    try:
        return canonical_tasks.index(canonical_name)
    except ValueError as exc:
        supported = ", ".join(CW10_TASKS_V3)
        raise ValueError(f"Unknown CW10 task {task_name!r}. Supported tasks: {supported}") from exc


def get_cw10_tasks(env_version: str = "v3") -> list[str]:
    return list(cw10_tasks_for_version(env_version))


def make_cw_env(
    task_name: str,
    seed: int = 0,
    render_mode: str | None = None,
    max_episode_steps: int = DEFAULT_EPISODE_LENGTH,
    *,
    append_task_id: bool = False,
    env_version: str | None = None,
) -> gym.Env:
    resolved_version = resolve_env_version(task_name, env_version)
    supported_tasks = cw10_tasks_for_version(resolved_version)

    if task_name not in supported_tasks:
        supported = ", ".join(supported_tasks)
        raise ValueError(f"Unknown CW10 task {task_name!r}. Supported tasks: {supported}")

    if resolved_version == "v3":
        env = gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            seed=seed,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            use_one_hot=False,
            disable_env_checker=True,
        )
    else:
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{task_name}-goal-observable"]
        env = env_cls(seed=seed, render_mode=render_mode)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = RelaxedObservationSpace(env)

    if append_task_id:
        env = TaskOneHotObservation(
            env,
            task_index=task_index_for_name(task_name),
            num_tasks=CW10_NUM_TASKS,
        )

    return env


def extract_success(info: dict[str, Any]) -> float:
    raw_success = info.get("success", 0.0)
    values = np.asarray(raw_success, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(values[0] > 0.0)
