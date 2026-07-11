"""MetaWorld environment creation for the CW10 benchmark.

ClonEx-SAC used the original MetaWorld task names ending in ``-v1``.
Modern MetaWorld 3.x exposes corresponding task names ending in ``-v3``.

The public experiment labels remain the original CW10 names. Translation
to modern MetaWorld names happens only when an environment is created.

Important:
    This compatibility layer preserves the CW10 task identities, order,
    episode horizon, and success interface. It does not make MetaWorld
    3.x dynamically identical to the old MetaWorld version used by
    ClonEx-SAC.
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

CLONEX_EPISODE_LENGTH = 200


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


def make_cw_env(
    task_name: str,
    seed: int = 0,
    render_mode: str | None = None,
    max_episode_steps: int = CLONEX_EPISODE_LENGTH,
) -> gym.Env:
    """Create a MetaWorld 3.x environment for one CW10 task.

    Args:
        task_name: Original CW10 task label ending in ``-v1``.
        seed: Seed passed to MetaWorld environment construction.
        render_mode: Optional Gymnasium render mode.
        max_episode_steps: Maximum episode length.

    Returns:
        A Gymnasium-compatible MetaWorld environment.
    """
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive.")

    env_name = modern_task_name(task_name)

    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        seed=seed,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )

    return env


def extract_success(info: dict[str, Any]) -> float:
    """Extract one step's binary MetaWorld success indicator.

    During episode evaluation, success must later be aggregated using
    the maximum success value observed over the entire episode.
    """
    raw_success = info.get("success", 0.0)
    values = np.asarray(raw_success, dtype=np.float32).reshape(-1)

    if values.size == 0:
        return 0.0

    return float(values[0] > 0.0)