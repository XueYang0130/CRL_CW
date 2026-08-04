"""Checkpoint utilities for SAC policy snapshots.

A checkpoint stores trainable SAC parameters and optimizer state:

- actor parameters;
- online critic parameters;
- target critic parameters;
- task-specific log entropy coefficients;
- Adam optimizer state;
- environment-step count;
- optional experiment metadata.

Online replay and method-specific memories, including Full BC episodic memory,
are intentionally not stored. These files support evaluation and diagnostics;
they are not sufficient for lossless continuation of a continual run.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch



CHECKPOINT_VERSION = 1

def _validate_agent_interface(agent: Any) -> None:
    """Validate that the object exposes the SAC checkpoint interface."""

    required_attributes = (
        "state_dict",
        "load_state_dict",
        "optimizer",
        "device",
        "observation_dim",
        "action_dim",
        "num_tasks",
        "task_id_dim",
        "learning_rate",
        "gamma",
        "polyak",
        "target_entropy",
    )

    missing = [
        name
        for name in required_attributes
        if not hasattr(agent, name)
    ]

    if missing:
        raise TypeError(
            "Agent does not expose the SAC checkpoint interface.\n"
            f"Missing attributes: {', '.join(missing)}"
        )


@dataclass(frozen=True)
class CheckpointInfo:
    """Information returned after loading one checkpoint."""

    path: Path
    environment_step: int
    metadata: dict[str, Any]
    optimizer_loaded: bool
    checkpoint_version: int


def save_sac_checkpoint(
    *,
    agent: Any,
    path: str | Path,
    environment_step: int,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save SAC parameters and optimizer state for evaluation or diagnostics."""
    _validate_agent_interface(agent)

    if environment_step < 0:
        raise ValueError("environment_step must be non-negative.")

    checkpoint_path = Path(path).expanduser()

    if checkpoint_path.suffix == "":
        raise ValueError(
            "Checkpoint path must include a file extension, "
            "for example '.pt'."
        )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_metadata = (
        {}
        if metadata is None
        else dict(metadata)
    )

    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "environment_step": int(environment_step),
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "agent_config": {
            "observation_dim": agent.observation_dim,
            "action_dim": agent.action_dim,
            "num_tasks": agent.num_tasks,
            "task_id_dim": agent.task_id_dim,
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "polyak": agent.polyak,
            "target_entropy": agent.target_entropy,
        },
        "metadata": normalized_metadata,
    }

    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )

    try:
        torch.save(
            payload,
            temporary_path,
        )
        os.replace(
            temporary_path,
            checkpoint_path,
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return checkpoint_path


def load_sac_checkpoint(
    *,
    agent: Any,
    path: str | Path,
    map_location: str | torch.device | None = None,
    load_optimizer: bool = True,
) -> CheckpointInfo:
    """Load a checkpoint into an existing SACAgent."""
    _validate_agent_interface(agent)

    checkpoint_path = Path(path).expanduser()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Checkpoint file does not exist: "
            f"{checkpoint_path}"
        )

    resolved_location = (
        agent.device
        if map_location is None
        else map_location
    )

    try:
        payload = torch.load(
            checkpoint_path,
            map_location=resolved_location,
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(
            checkpoint_path,
            map_location=resolved_location,
        )

    if not isinstance(payload, dict):
        raise ValueError(
            "Checkpoint payload must be a dictionary."
        )

    required_keys = {
        "checkpoint_version",
        "environment_step",
        "agent_state_dict",
        "optimizer_state_dict",
        "agent_config",
        "metadata",
    }

    missing_keys = required_keys - set(payload.keys())

    if missing_keys:
        missing_text = ", ".join(
            sorted(missing_keys)
        )
        raise ValueError(
            "Checkpoint is missing required fields: "
            f"{missing_text}."
        )

    checkpoint_version = int(
        payload["checkpoint_version"]
    )

    if checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported checkpoint version: "
            f"expected {CHECKPOINT_VERSION}, "
            f"got {checkpoint_version}."
        )

    environment_step = int(
        payload["environment_step"]
    )

    if environment_step < 0:
        raise ValueError(
            "Checkpoint environment_step must be non-negative."
        )

    agent_config = payload["agent_config"]

    if not isinstance(agent_config, Mapping):
        raise ValueError(
            "Checkpoint agent_config must be a mapping."
        )

    _validate_agent_compatibility(
        agent=agent,
        saved_config=agent_config,
    )

    agent.load_state_dict(
        payload["agent_state_dict"],
        strict=True,
    )

    if load_optimizer:
        agent.optimizer.load_state_dict(
            payload["optimizer_state_dict"]
        )
        _move_optimizer_state(
            optimizer=agent.optimizer,
            device=agent.device,
        )

    metadata = payload["metadata"]

    if not isinstance(metadata, Mapping):
        raise ValueError(
            "Checkpoint metadata must be a mapping."
        )

    return CheckpointInfo(
        path=checkpoint_path,
        environment_step=environment_step,
        metadata=dict(metadata),
        optimizer_loaded=bool(load_optimizer),
        checkpoint_version=checkpoint_version,
    )


def _validate_agent_compatibility(
    *,
    agent: Any,
    saved_config: Mapping[str, Any],
) -> None:
    """Validate that the current agent matches the saved configuration."""
    required_config_keys = {
        "observation_dim",
        "action_dim",
        "num_tasks",
        "task_id_dim",
        "learning_rate",
        "gamma",
        "polyak",
        "target_entropy",
    }

    missing_keys = required_config_keys - set(
        saved_config.keys()
    )

    if missing_keys:
        missing_text = ", ".join(
            sorted(missing_keys)
        )
        raise ValueError(
            "Checkpoint agent_config is missing: "
            f"{missing_text}."
        )

    saved_observation_dim = int(
        saved_config["observation_dim"]
    )
    saved_action_dim = int(
        saved_config["action_dim"]
    )
    saved_num_tasks = int(
        saved_config["num_tasks"]
    )
    saved_task_id_dim = int(
        saved_config["task_id_dim"]
    )

    if saved_observation_dim != agent.observation_dim:
        raise ValueError(
            "Checkpoint observation dimension does not match "
            "the current agent: "
            f"checkpoint={saved_observation_dim}, "
            f"agent={agent.observation_dim}."
        )

    if saved_action_dim != agent.action_dim:
        raise ValueError(
            "Checkpoint action dimension does not match "
            "the current agent: "
            f"checkpoint={saved_action_dim}, "
            f"agent={agent.action_dim}."
        )

    if saved_num_tasks != agent.num_tasks:
        raise ValueError(
            "Checkpoint num_tasks does not match "
            "the current agent: "
            f"checkpoint={saved_num_tasks}, "
            f"agent={agent.num_tasks}."
        )

    if saved_task_id_dim != agent.task_id_dim:
        raise ValueError(
            "Checkpoint task_id_dim does not match "
            "the current agent: "
            f"checkpoint={saved_task_id_dim}, "
            f"agent={agent.task_id_dim}."
        )

    saved_learning_rate = float(
        saved_config["learning_rate"]
    )
    saved_gamma = float(
        saved_config["gamma"]
    )
    saved_polyak = float(
        saved_config["polyak"]
    )
    saved_target_entropy = float(
        saved_config["target_entropy"]
    )

    comparisons = (
        (
            "learning_rate",
            saved_learning_rate,
            agent.learning_rate,
        ),
        (
            "gamma",
            saved_gamma,
            agent.gamma,
        ),
        (
            "polyak",
            saved_polyak,
            agent.polyak,
        ),
        (
            "target_entropy",
            saved_target_entropy,
            agent.target_entropy,
        ),
    )

    for name, saved_value, current_value in comparisons:
        if not math.isfinite(saved_value):
            raise ValueError(
                f"Checkpoint {name} must be finite."
            )

        if not math.isclose(
            saved_value,
            float(current_value),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Checkpoint {name} does not match "
                "the current agent: "
                f"checkpoint={saved_value}, "
                f"agent={current_value}."
            )


def _move_optimizer_state(
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move all optimizer-state tensors to the agent device."""
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if isinstance(value, torch.Tensor):
                parameter_state[key] = value.to(device)
