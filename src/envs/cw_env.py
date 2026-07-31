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


class StickPullV1CompatibleReward(gym.Wrapper):
    """Project-local reward override for stick-pull-v3.

    This keeps the old v1 reward structure but replaces the legacy object
    indexing with explicit v3 positions so we can test learnability without
    modifying the installed Meta-World package.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._target_pos = np.zeros(3, dtype=np.float64)
        self.height_target = 0.0
        self.stick_height = 0.0
        self.max_pull_dist = 0.0
        self.max_place_dist = 0.0
        self.pick_completed = False

    @property
    def _base_env(self) -> Any:
        return self.unwrapped

    def _refresh_task_geometry(self) -> None:
        base_env = self._base_env
        self.pick_completed = False
        self._target_pos = np.asarray(base_env._target_pos, dtype=np.float64).copy()
        self.height_target = float(base_env.heightTarget)
        self.stick_height = float(base_env.stickHeight)
        self.max_pull_dist = float(base_env.maxPullDist)
        self.max_place_dist = float(base_env.maxPlaceDist)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._refresh_task_geometry()
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, _, terminated, truncated, info = self.env.step(action)
        reward = self.compute_reward(np.asarray(action, dtype=np.float64), np.asarray(observation, dtype=np.float64))
        if "unscaled_reward" in info:
            info["unscaled_reward"] = reward
        return observation, float(reward), terminated, truncated, info

    def compute_reward(self, action: np.ndarray, obs: np.ndarray) -> float:
        stick_pos = obs[4:7]
        handle_pos = obs[11:14]
        container_pos = handle_pos + np.array([0.05, 0.0, 0.0], dtype=np.float64)

        right_finger = self._base_env._get_site_pos("rightEndEffector")
        left_finger = self._base_env._get_site_pos("leftEndEffector")
        finger_center = (right_finger + left_finger) / 2.0

        pull_goal = self._target_pos[:-1]
        pull_dist = float(np.linalg.norm(container_pos[:2] - pull_goal))
        place_dist = float(np.linalg.norm(stick_pos - container_pos))
        reach_dist = float(np.linalg.norm(stick_pos - finger_center))

        reach_reward = -reach_dist
        if reach_dist < 0.05:
            reach_reward = -reach_dist + max(float(action[-1]), 0.0) / 50.0

        tolerance = 0.01
        self.pick_completed = bool(stick_pos[2] >= (self.height_target - tolerance))
        object_dropped = bool(
            (stick_pos[2] < (self.stick_height + 0.005))
            and (pull_dist > 0.02)
            and (reach_dist > 0.02)
        )

        h_scale = 100.0
        if self.pick_completed and not object_dropped:
            pick_reward = h_scale * self.height_target
        elif (reach_dist < 0.1) and (stick_pos[2] > (self.stick_height + 0.005)):
            pick_reward = h_scale * min(self.height_target, float(stick_pos[2]))
        else:
            pick_reward = 0.0

        c1 = 1000.0
        c2 = 0.01
        c3 = 0.001
        condition = self.pick_completed and (reach_dist < 0.1) and not object_dropped
        if condition:
            pull_reward = 1000.0 * (self.max_place_dist - place_dist) + c1 * (
                np.exp(-(place_dist**2) / c2) + np.exp(-(place_dist**2) / c3)
            )
            if place_dist < 0.05:
                c4 = 2000.0
                pull_reward += 1000.0 * (self.max_pull_dist - pull_dist) + c4 * (
                    np.exp(-(pull_dist**2) / c2) + np.exp(-(pull_dist**2) / c3)
                )
            pull_reward = max(float(pull_reward), 0.0)
        else:
            pull_reward = 0.0

        return float(reach_reward + pick_reward + pull_reward)


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


def resolve_reward_function_version(
    task_name: str,
    reward_function_version: str,
) -> str:
    if reward_function_version == "cw10_v1":
        if task_name == "stick-pull-v3":
            return "v1_compatible"
        return "v1"
    return reward_function_version


def make_cw_env(
    task_name: str,
    seed: int = 0,
    render_mode: str | None = None,
    max_episode_steps: int = DEFAULT_EPISODE_LENGTH,
    *,
    append_task_id: bool = False,
    env_version: str | None = None,
    reward_function_version: str = "v2",
    terminate_on_success: bool = False,
    num_task_ids: int = CW10_NUM_TASKS,
) -> gym.Env:
    resolved_version = resolve_env_version(task_name, env_version)
    resolved_reward_version = resolve_reward_function_version(
        task_name,
        reward_function_version,
    )
    supported_tasks = cw10_tasks_for_version(resolved_version)

    if task_name not in supported_tasks:
        supported = ", ".join(supported_tasks)
        raise ValueError(f"Unknown CW10 task {task_name!r}. Supported tasks: {supported}")

    if resolved_version == "v3":
        use_stick_pull_v1_compatible = (
            task_name == "stick-pull-v3" and resolved_reward_version == "v1_compatible"
        )
        env = gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            seed=seed,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            use_one_hot=False,
            reward_function_version="v1" if use_stick_pull_v1_compatible else resolved_reward_version,
            terminate_on_success=terminate_on_success,
            disable_env_checker=True,
        )
        if use_stick_pull_v1_compatible:
            env = StickPullV1CompatibleReward(env)
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
            num_tasks=num_task_ids,
        )

    return env


def extract_success(info: dict[str, Any]) -> float:
    raw_success = info.get("success", 0.0)
    values = np.asarray(raw_success, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(values[0] > 0.0)
