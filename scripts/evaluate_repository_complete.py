#!/usr/bin/env python3
"""Evaluate either a full-policy repository or a shared-backbone repository.

Two repository interpretations are supported.

1. Full-policy repository (default; --no-shared-backbone)

   Each task-boundary checkpoint contributes one frozen full policy:

       checkpoint 0 -> backbone_0 + head_0
       checkpoint 1 -> backbone_1 + head_1
       checkpoint 2 -> backbone_2 + head_2

   Final oracle retrieval evaluates every frozen full policy on every task and
   selects the best one according to --selection-criterion.

2. Shared-backbone repository (--shared-backbone)

   At boundary b, the checkpoint supplies the current shared backbone B_b and
   all task heads learned so far:

       boundary 0 -> B_0 + {H_0}
       boundary 1 -> B_1 + {H_0, H_1}
       boundary 2 -> B_2 + {H_0, H_1, H_2}

   Final oracle retrieval uses the final checkpoint and every trained head:

       task j -> best of {B_final + H_0, ..., B_final + H_T}

   Repository forgetting compares the oracle repository performance immediately
   after task j with the final oracle repository performance on task j.

The script reports repository/system-side Average Performance and Forgetting.
Forward Transfer is computed from the continual active-task learning curves
stored in evaluations.csv and matched single-task scratch learning curves,
using normalized trapezoidal AUCs of matched deterministic learning curves.

Example: original frozen full-policy repository
-----------------------------------------------
PYTHONPATH=src python scripts/evaluate_repository_complete.py \
  --run-dir outputs/cw3_finetune/cw3_random_500k_seed0 \
  --baseline-curves outputs/cw10_single_task_baselines/cw3_single_task_scratch_500k_seed0_v2/aggregate/baseline_curves.json \
  --episodes 10 \
  --device cpu \
  --no-shared-backbone \
  --selection-criterion return

Example: shared backbone + task-specific heads
----------------------------------------------
PYTHONPATH=src python scripts/evaluate_repository_complete.py \
  --run-dir outputs/cw3_finetune/cw3_random_500k_seed0 \
  --baseline-curves outputs/cw10_single_task_baselines/cw3_single_task_scratch_500k_seed0_v2/aggregate/baseline_curves.json \
  --episodes 10 \
  --device cpu \
  --shared-backbone \
  --selection-criterion return
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from crl_cw.agents import SACAgent
from crl_cw.envs.cw_env import make_cw_env
from crl_cw.evaluation import EvaluationConfig, SACEvaluator


PAIR_FIELDS = [
    "repository_mode",
    "boundary_index",
    "boundary_task",
    "evaluation_task_index",
    "evaluation_task_name",
    "candidate_index",
    "candidate_source_task",
    "candidate_head_index",
    "checkpoint",
    "environment_step",
    "mean_return",
    "std_return",
    "success_rate",
    "mean_episode_length",
    "num_episodes",
    "selected_at_boundary",
    "selected_final",
]

SELECTED_FIELDS = [
    "repository_mode",
    "boundary_index",
    "boundary_task",
    "task_index",
    "task_name",
    "selected_candidate_index",
    "selected_candidate_source_task",
    "selected_head_index",
    "selected_checkpoint",
    "mean_return",
    "std_return",
    "success_rate",
    "mean_episode_length",
    "selection_criterion",
]

PER_TASK_FIELDS = [
    "task_index",
    "task_name",
    "end_of_task_success",
    "final_success",
    "forgetting_end_to_final",
    "maximum_historical_success",
    "forgetting_max_to_final",
    "end_of_task_return",
    "final_return",
    "selected_end_candidate_index",
    "selected_end_candidate_source_task",
    "selected_end_head_index",
    "selected_final_candidate_index",
    "selected_final_candidate_source_task",
    "selected_final_head_index",
    "continual_success_auc",
    "baseline_success_auc",
    "raw_forward_transfer",
    "normalized_forward_transfer",
]

BOUNDARY_MATRIX_FIELDS = [
    "boundary_index",
    "boundary_task",
]

TASK_KEYS = (
    "task_name",
    "task",
    "env_name",
    "environment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen full-policy repository or a shared-backbone "
            "plus task-specific-head repository."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help=(
            "Completed continual run directory containing config.json and "
            "checkpoints/task_*.pt."
        ),
    )
    parser.add_argument(
        "--baseline-curves",
        type=str,
        required=True,
        help=(
            "Matched single-task aggregate/baseline_curves.json. Forward "
            "Transfer is computed from the continual active-task learning "
            "curve versus this scratch learning curve."
        ),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=80_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--shared-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "False: each checkpoint is one frozen full policy using its source "
            "task head. True: each boundary checkpoint provides one current "
            "shared backbone and every head learned by that boundary."
        ),
    )
    parser.add_argument(
        "--selection-criterion",
        choices=("return", "success"),
        default="return",
        help=(
            "Oracle policy-selection criterion. 'return' selects the largest "
            "deterministic mean return. 'success' selects the largest success "
            "rate and uses return as the tie-breaker."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<run-dir>/shared_backbone_repository_eval when "
            "--shared-backbone is enabled, otherwise "
            "<run-dir>/full_policy_repository_eval."
        ),
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive.")
    if args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive.")

    return args


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")

    return value



def load_baseline_success_curves(
    path: Path,
    *,
    tasks: list[str],
) -> tuple[list[int], list[list[float]]]:
    """Load matched deterministic scratch learning curves."""
    data = load_json(path)

    baseline_tasks = data.get("tasks")
    if baseline_tasks != tasks:
        raise ValueError(
            "Baseline task order does not match the continual run.\n"
            f"Continual: {tasks}\n"
            f"Baseline:  {baseline_tasks}"
        )

    raw_steps = data.get("evaluation_steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(
            "baseline_curves.json must contain a non-empty "
            "'evaluation_steps' list."
        )
    evaluation_steps = [int(value) for value in raw_steps]

    if any(step <= 0 for step in evaluation_steps):
        raise ValueError("Baseline evaluation steps must be positive.")
    if evaluation_steps != sorted(set(evaluation_steps)):
        raise ValueError(
            "Baseline evaluation steps must be unique and increasing."
        )

    raw_curves = data.get("deterministic_success_curves")
    if not isinstance(raw_curves, list) or len(raw_curves) != len(tasks):
        raise ValueError(
            "baseline_curves.json must contain one "
            "deterministic_success_curves entry per task."
        )

    curves: list[list[float]] = []
    for task_name, raw_curve in zip(tasks, raw_curves, strict=True):
        if not isinstance(raw_curve, list):
            raise TypeError(
                f"Scratch success curve for {task_name} must be a list."
            )
        curve = [float(value) for value in raw_curve]
        if len(curve) != len(evaluation_steps):
            raise ValueError(
                f"Scratch curve length mismatch for {task_name}: "
                f"{len(curve)} values but {len(evaluation_steps)} steps."
            )

        values = np.asarray(curve, dtype=np.float64)
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(
                f"Scratch success curve for {task_name} must be finite "
                "and lie in [0, 1]."
            )
        curves.append(curve)

    return evaluation_steps, curves


def load_continual_active_task_curves(
    path: Path,
    *,
    tasks: list[str],
) -> tuple[list[int], list[list[float]]]:
    """Read active-task deterministic learning curves from evaluations.csv.

    The ClonEx-style runner evaluates all encountered tasks at each evaluation
    point. Forward Transfer uses only the row where the evaluation task equals
    the currently trained task.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing continual evaluation history: {path}"
        )

    required_columns = {
        "active_task_index",
        "active_task_name",
        "active_task_step",
        "evaluation_task_index",
        "evaluation_task_name",
        "deterministic_success_rate",
    }

    records_by_task: dict[int, list[tuple[int, float]]] = {
        index: [] for index in range(len(tasks))
    }

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing = required_columns - fieldnames
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            active_index = int(row["active_task_index"])
            evaluation_index = int(row["evaluation_task_index"])

            # Keep only the active task's own learning curve.
            if active_index != evaluation_index:
                continue
            if active_index < 0 or active_index >= len(tasks):
                continue

            expected_name = tasks[active_index]
            if row["active_task_name"] != expected_name:
                raise ValueError(
                    "Active task name/index mismatch in evaluations.csv: "
                    f"index={active_index}, name={row['active_task_name']!r}, "
                    f"expected={expected_name!r}."
                )
            if row["evaluation_task_name"] != expected_name:
                raise ValueError(
                    "Evaluation task name/index mismatch in evaluations.csv: "
                    f"index={evaluation_index}, "
                    f"name={row['evaluation_task_name']!r}, "
                    f"expected={expected_name!r}."
                )

            step = int(row["active_task_step"])
            success = float(row["deterministic_success_rate"])
            if step <= 0:
                raise ValueError(
                    f"Active-task evaluation step must be positive: {step}."
                )
            if not math.isfinite(success) or not 0.0 <= success <= 1.0:
                raise ValueError(
                    f"Invalid deterministic success for {expected_name}: "
                    f"{success}."
                )
            records_by_task[active_index].append((step, success))

    curves: list[list[float]] = []
    common_steps: list[int] | None = None

    for task_index, task_name in enumerate(tasks):
        records = sorted(records_by_task[task_index], key=lambda item: item[0])
        if not records:
            raise ValueError(
                f"No active-task learning-curve rows found for {task_name}."
            )

        steps = [item[0] for item in records]
        if steps != sorted(set(steps)):
            raise ValueError(
                f"Duplicate or unordered active-task steps for {task_name}: "
                f"{steps}."
            )

        if common_steps is None:
            common_steps = steps
        elif steps != common_steps:
            raise ValueError(
                "Continual tasks use different evaluation grids.\n"
                f"Expected: {common_steps}\n"
                f"{task_name}: {steps}"
            )

        curves.append([item[1] for item in records])

    assert common_steps is not None
    return common_steps, curves


def normalized_trapezoid_auc(
    *,
    steps: list[int],
    values: list[float],
    label: str,
) -> float:
    """Return trapezoidal AUC divided by the observed step interval.

    Existing CRL_CW runs evaluate at eval_every, 2*eval_every, ..., Delta
    and do not contain an evaluation at task step zero. Therefore this
    function integrates over the common observed interval
    [steps[0], steps[-1]] and divides by its duration.

    No artificial step-zero success value is invented.
    """
    if len(steps) != len(values):
        raise ValueError(
            f"{label} has {len(values)} values but {len(steps)} steps."
        )
    if len(steps) < 2:
        raise ValueError(
            f"{label} needs at least two evaluation points for AUC."
        )

    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)

    if not np.all(np.isfinite(x)):
        raise ValueError(f"{label} steps contain NaN or infinity.")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{label} values contain NaN or infinity.")
    if np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError(f"{label} success values must lie in [0, 1].")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(f"{label} steps must be strictly increasing.")

    duration = float(x[-1] - x[0])
    if duration <= 0.0:
        raise ValueError(f"{label} has a non-positive AUC interval.")

    # np.trapezoid is available in newer NumPy releases. np.trapz is
    # retained as a compatibility fallback for older environments.
    trapezoid = getattr(np, "trapezoid", np.trapz)
    area = float(trapezoid(y, x=x))
    return area / duration


def compute_learning_curve_forward_transfer(
    *,
    tasks: list[str],
    continual_steps: list[int],
    continual_curves: list[list[float]],
    baseline_steps: list[int],
    baseline_curves: list[list[float]],
) -> dict[str, Any]:
    """Compute ClonEx-style learning-curve Forward Transfer.

    For task i, the continual curve is the deterministic success curve
    produced while task i is actively trained, using the current shared
    backbone and task-i head at every evaluation point.

    The repository oracle is deliberately not used for Forward Transfer.
    Repository selection is used only for Average Performance and
    Forgetting.

    For every task:
        AUC_i   = normalized trapezoidal AUC of the continual curve
        AUC_i^b = normalized trapezoidal AUC of the scratch curve
        raw_FT_i = AUC_i - AUC_i^b
        FT_i = raw_FT_i / (1 - AUC_i^b)

    Since existing runs do not store task-step-zero evaluation, both
    curves are integrated over their identical observed interval:
    [evaluation_steps[0], evaluation_steps[-1]].
    """
    if continual_steps != baseline_steps:
        raise ValueError(
            "Continual and scratch evaluation grids do not match.\n"
            f"Continual: {continual_steps}\n"
            f"Scratch:   {baseline_steps}"
        )
    if len(continual_curves) != len(tasks):
        raise ValueError("Missing continual curve for one or more tasks.")
    if len(baseline_curves) != len(tasks):
        raise ValueError("Missing baseline curve for one or more tasks.")

    continual_auc: dict[str, float] = {}
    baseline_auc: dict[str, float] = {}
    raw_per_task: dict[str, float] = {}
    normalized_per_task: dict[str, float | None] = {}

    for task_name, continual_curve, baseline_curve in zip(
        tasks,
        continual_curves,
        baseline_curves,
        strict=True,
    ):
        continual_value = normalized_trapezoid_auc(
            steps=continual_steps,
            values=continual_curve,
            label=f"continual curve for {task_name}",
        )
        baseline_value = normalized_trapezoid_auc(
            steps=baseline_steps,
            values=baseline_curve,
            label=f"scratch curve for {task_name}",
        )

        raw_value = continual_value - baseline_value
        remaining_headroom = 1.0 - baseline_value

        continual_auc[task_name] = continual_value
        baseline_auc[task_name] = baseline_value
        raw_per_task[task_name] = raw_value

        if remaining_headroom <= 1e-12:
            normalized_per_task[task_name] = None
        else:
            normalized_per_task[task_name] = float(
                raw_value / remaining_headroom
            )

    raw_values = list(raw_per_task.values())
    normalized_values = [
        float(value)
        for value in normalized_per_task.values()
        if value is not None
    ]
    undefined_tasks = [
        task_name
        for task_name, value in normalized_per_task.items()
        if value is None
    ]

    return {
        "forward_transfer": (
            None
            if not normalized_values
            else float(np.mean(normalized_values))
        ),
        "raw_forward_transfer": float(np.mean(raw_values)),
        "forward_transfer_status": (
            "available"
            if normalized_values
            else "unavailable_all_normalized_ft_undefined"
        ),
        "continual_success_auc_per_task": continual_auc,
        "baseline_success_auc_per_task": baseline_auc,
        "raw_forward_transfer_per_task": raw_per_task,
        "normalized_forward_transfer_per_task": normalized_per_task,
        "normalized_forward_transfer_defined_tasks": len(
            normalized_values
        ),
        "normalized_forward_transfer_undefined_tasks": undefined_tasks,
        "evaluation_steps": continual_steps,
        "auc_interval_start": int(continual_steps[0]),
        "auc_interval_end": int(continual_steps[-1]),
        "auc_includes_task_step_zero": False,
        "forward_transfer_definition": (
            "For each task, use the active-task deterministic success "
            "learning curve from the continual learner and the matched "
            "single-task scratch curve. Compute each normalized AUC with "
            "trapezoidal integration over the common observed evaluation "
            "interval, then FT_i=(AUC_i-AUC_i^b)/(1-AUC_i^b). "
            "The repository oracle is not used for FWT."
        ),
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(payload, dict):
        raise TypeError(
            f"Checkpoint must contain a dictionary: {path}"
        )

    required = {
        "environment_step",
        "agent_state_dict",
        "agent_config",
        "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"{path} is missing checkpoint fields: "
            f"{', '.join(sorted(missing))}"
        )

    if not isinstance(payload["agent_config"], Mapping):
        raise TypeError(
            f"agent_config must be a mapping: {path}"
        )

    return payload


def close_env(env: Any | None) -> None:
    if env is None:
        return

    close_method = getattr(env, "close", None)
    if callable(close_method):
        close_method()


def checkpoint_task_number(path: Path) -> int:
    match = re.match(r"task_(\d+)_", path.name)
    if match is None:
        raise ValueError(
            "Task checkpoint filename must match "
            f"task_NN_<task-name>.pt: {path.name}"
        )

    task_number = int(match.group(1))
    if task_number <= 0:
        raise ValueError(
            f"Checkpoint task number must be positive: {path.name}"
        )

    return task_number


def resolve_tasks(
    configuration: dict[str, Any],
    checkpoint_paths: list[Path],
) -> list[str]:
    raw_tasks = configuration.get("tasks")

    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError(
            "config.json must contain a non-empty 'tasks' list."
        )

    tasks = [str(task) for task in raw_tasks]

    if len(checkpoint_paths) > len(tasks):
        raise ValueError(
            f"Found {len(checkpoint_paths)} task checkpoints but only "
            f"{len(tasks)} tasks in config.json."
        )

    tasks = tasks[:len(checkpoint_paths)]

    if len(tasks) != len(checkpoint_paths):
        raise RuntimeError(
            "Task/checkpoint count mismatch."
        )

    return tasks


def build_agent(
    *,
    payload: dict[str, Any],
    task_name: str,
    seed: int,
    device: str,
) -> SACAgent:
    saved = dict(payload["agent_config"])

    env = make_cw_env(
        task_name=task_name,
        seed=seed,
        append_task_id=True,
    )
    try:
        action_low = np.asarray(
            env.action_space.low,
            dtype=np.float32,
        )
        action_high = np.asarray(
            env.action_space.high,
            dtype=np.float32,
        )
    finally:
        close_env(env)

    agent = SACAgent(
        observation_dim=int(saved["observation_dim"]),
        action_dim=int(saved["action_dim"]),
        action_low=action_low,
        action_high=action_high,
        num_tasks=int(saved["num_tasks"]),
        task_id_dim=int(saved["task_id_dim"]),
        learning_rate=float(saved["learning_rate"]),
        gamma=float(saved["gamma"]),
        polyak=float(saved["polyak"]),
        target_entropy=float(saved["target_entropy"]),
        initial_log_alpha=0.0,
        device=device,
    )

    agent.load_state_dict(
        payload["agent_state_dict"],
        strict=True,
    )
    agent.eval()
    return agent


def evaluate_candidate(
    *,
    payload: dict[str, Any],
    checkpoint_path: Path,
    boundary_index: int,
    boundary_task: str,
    candidate_index: int,
    candidate_source_task: str,
    head_index: int,
    evaluation_task_index: int,
    evaluation_task_name: str,
    episodes: int,
    max_episode_steps: int,
    seed: int,
    device: str,
    repository_mode: str,
) -> dict[str, Any]:
    agent = build_agent(
        payload=payload,
        task_name=evaluation_task_name,
        seed=seed,
        device=device,
    )

    if head_index < 0 or head_index >= agent.num_tasks:
        raise ValueError(
            f"Head index {head_index} is outside checkpoint head range "
            f"[0, {agent.num_tasks - 1}] for {checkpoint_path.name}."
        )

    env = make_cw_env(
        task_name=evaluation_task_name,
        seed=seed,
        append_task_id=True,
    )

    try:
        evaluator = SACEvaluator(
            env=env,
            agent=agent,
            config=EvaluationConfig(
                num_episodes=episodes,
                max_episode_steps=max_episode_steps,
                seed=seed,
            ),
        )
        result = evaluator.evaluate(
            deterministic=True,
            head_index=head_index,
        )
    finally:
        close_env(env)
        del agent

    returns = np.asarray(
        result.episode_returns,
        dtype=np.float64,
    )

    if returns.size != episodes:
        raise RuntimeError(
            f"{checkpoint_path.name} on {evaluation_task_name} returned "
            f"{returns.size} episodes; expected {episodes}."
        )

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=0))
    success_rate = float(result.success_rate)

    if not math.isfinite(mean_return):
        raise RuntimeError(
            "Evaluation produced a non-finite mean return."
        )
    if not 0.0 <= success_rate <= 1.0:
        raise RuntimeError(
            "Evaluation produced a success rate outside [0, 1]."
        )

    return {
        "repository_mode": repository_mode,
        "boundary_index": int(boundary_index),
        "boundary_task": boundary_task,
        "evaluation_task_index": int(evaluation_task_index),
        "evaluation_task_name": evaluation_task_name,
        "candidate_index": int(candidate_index),
        "candidate_source_task": candidate_source_task,
        "candidate_head_index": int(head_index),
        "checkpoint": str(checkpoint_path),
        "environment_step": int(payload["environment_step"]),
        "mean_return": mean_return,
        "std_return": std_return,
        "success_rate": success_rate,
        "mean_episode_length": float(result.mean_episode_length),
        "num_episodes": int(result.num_episodes),
        "selected_at_boundary": False,
        "selected_final": False,
    }


def select_best(
    rows: list[dict[str, Any]],
    *,
    criterion: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(
            "Cannot select from an empty candidate set."
        )

    if criterion == "return":
        return max(
            rows,
            key=lambda row: (
                float(row["mean_return"]),
                float(row["success_rate"]),
                -int(row["candidate_index"]),
            ),
        )

    if criterion == "success":
        return max(
            rows,
            key=lambda row: (
                float(row["success_rate"]),
                float(row["mean_return"]),
                -int(row["candidate_index"]),
            ),
        )

    raise ValueError(
        f"Unsupported selection criterion: {criterion}"
    )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_full_policy_repository(
    *,
    checkpoint_paths: list[Path],
    payloads: list[dict[str, Any]],
    tasks: list[str],
    episodes: int,
    max_episode_steps: int,
    seed: int,
    device: str,
) -> list[dict[str, Any]]:
    """Evaluate every frozen checkpoint policy on every task.

    Policy i means checkpoint i with source head i.
    """
    rows: list[dict[str, Any]] = []
    repository_mode = "full_policy_repository"

    for evaluation_task_index, evaluation_task_name in enumerate(tasks):
        evaluation_seed = seed + evaluation_task_index * 1_000

        for candidate_index, (
            checkpoint_path,
            payload,
        ) in enumerate(
            zip(
                checkpoint_paths,
                payloads,
                strict=True,
            )
        ):
            print(
                f"Evaluating full policy {candidate_index + 1}/{len(tasks)} "
                f"({tasks[candidate_index]}) on task "
                f"{evaluation_task_index + 1}/{len(tasks)} "
                f"({evaluation_task_name})",
                flush=True,
            )

            rows.append(
                evaluate_candidate(
                    payload=payload,
                    checkpoint_path=checkpoint_path,
                    boundary_index=candidate_index,
                    boundary_task=tasks[candidate_index],
                    candidate_index=candidate_index,
                    candidate_source_task=tasks[candidate_index],
                    head_index=candidate_index,
                    evaluation_task_index=evaluation_task_index,
                    evaluation_task_name=evaluation_task_name,
                    episodes=episodes,
                    max_episode_steps=max_episode_steps,
                    seed=evaluation_seed,
                    device=device,
                    repository_mode=repository_mode,
                )
            )

    return rows


def evaluate_shared_backbone_repository(
    *,
    checkpoint_paths: list[Path],
    payloads: list[dict[str, Any]],
    tasks: list[str],
    episodes: int,
    max_episode_steps: int,
    seed: int,
    device: str,
) -> list[dict[str, Any]]:
    """Evaluate each current shared backbone with all heads learned so far.

    At boundary b:
        checkpoint b supplies backbone B_b
        candidate heads are H_0 ... H_b

    Every learned task is evaluated at every later boundary so the output can
    be used for final AP and repository forgetting.
    """
    rows: list[dict[str, Any]] = []
    repository_mode = "shared_backbone_task_specific_heads"

    for boundary_index, (
        checkpoint_path,
        payload,
    ) in enumerate(
        zip(
            checkpoint_paths,
            payloads,
            strict=True,
        )
    ):
        boundary_task = tasks[boundary_index]

        # At boundary b only tasks 0..b have been encountered.
        evaluation_task_indices = range(boundary_index + 1)
        available_head_indices = range(boundary_index + 1)

        for evaluation_task_index in evaluation_task_indices:
            evaluation_task_name = tasks[evaluation_task_index]
            evaluation_seed = (
                seed
                + boundary_index * 10_000
                + evaluation_task_index * 1_000
            )

            for head_index in available_head_indices:
                print(
                    f"Evaluating boundary {boundary_index + 1}/{len(tasks)} "
                    f"backbone ({boundary_task}), "
                    f"head {head_index + 1}/{boundary_index + 1} "
                    f"({tasks[head_index]}) on task "
                    f"{evaluation_task_index + 1}/{len(tasks)} "
                    f"({evaluation_task_name})",
                    flush=True,
                )

                rows.append(
                    evaluate_candidate(
                        payload=payload,
                        checkpoint_path=checkpoint_path,
                        boundary_index=boundary_index,
                        boundary_task=boundary_task,
                        candidate_index=head_index,
                        candidate_source_task=tasks[head_index],
                        head_index=head_index,
                        evaluation_task_index=evaluation_task_index,
                        evaluation_task_name=evaluation_task_name,
                        episodes=episodes,
                        max_episode_steps=max_episode_steps,
                        seed=evaluation_seed,
                        device=device,
                        repository_mode=repository_mode,
                    )
                )

    return rows


def select_boundary_oracle_rows(
    *,
    pair_rows: list[dict[str, Any]],
    tasks: list[str],
    criterion: str,
    shared_backbone: bool,
) -> list[dict[str, Any]]:
    """Select the best candidate for each valid boundary/task pair."""
    selected_rows: list[dict[str, Any]] = []

    if shared_backbone:
        boundary_indices = range(len(tasks))

        for boundary_index in boundary_indices:
            for task_index in range(boundary_index + 1):
                candidates = [
                    row
                    for row in pair_rows
                    if int(row["boundary_index"]) == boundary_index
                    and int(row["evaluation_task_index"]) == task_index
                ]
                selected = select_best(
                    candidates,
                    criterion=criterion,
                )
                selected["selected_at_boundary"] = True

                selected_rows.append(
                    {
                        "repository_mode": selected["repository_mode"],
                        "boundary_index": boundary_index,
                        "boundary_task": tasks[boundary_index],
                        "task_index": task_index,
                        "task_name": tasks[task_index],
                        "selected_candidate_index": int(
                            selected["candidate_index"]
                        ),
                        "selected_candidate_source_task": str(
                            selected["candidate_source_task"]
                        ),
                        "selected_head_index": int(
                            selected["candidate_head_index"]
                        ),
                        "selected_checkpoint": str(
                            selected["checkpoint"]
                        ),
                        "mean_return": float(
                            selected["mean_return"]
                        ),
                        "std_return": float(
                            selected["std_return"]
                        ),
                        "success_rate": float(
                            selected["success_rate"]
                        ),
                        "mean_episode_length": float(
                            selected["mean_episode_length"]
                        ),
                        "selection_criterion": criterion,
                    }
                )
    else:
        # In the full-policy repository, repository boundary b contains
        # frozen policies 0..b. We can derive every boundary/task oracle from
        # the complete policy-task evaluation table.
        for boundary_index in range(len(tasks)):
            for task_index in range(boundary_index + 1):
                candidates = [
                    row
                    for row in pair_rows
                    if int(row["evaluation_task_index"]) == task_index
                    and int(row["candidate_index"]) <= boundary_index
                ]
                selected = select_best(
                    candidates,
                    criterion=criterion,
                )
                selected["selected_at_boundary"] = True

                selected_rows.append(
                    {
                        "repository_mode": selected["repository_mode"],
                        "boundary_index": boundary_index,
                        "boundary_task": tasks[boundary_index],
                        "task_index": task_index,
                        "task_name": tasks[task_index],
                        "selected_candidate_index": int(
                            selected["candidate_index"]
                        ),
                        "selected_candidate_source_task": str(
                            selected["candidate_source_task"]
                        ),
                        "selected_head_index": int(
                            selected["candidate_head_index"]
                        ),
                        "selected_checkpoint": str(
                            selected["checkpoint"]
                        ),
                        "mean_return": float(
                            selected["mean_return"]
                        ),
                        "std_return": float(
                            selected["std_return"]
                        ),
                        "success_rate": float(
                            selected["success_rate"]
                        ),
                        "mean_episode_length": float(
                            selected["mean_episode_length"]
                        ),
                        "selection_criterion": criterion,
                    }
                )

    final_boundary_index = len(tasks) - 1
    final_lookup = {
        int(row["task_index"]): row
        for row in selected_rows
        if int(row["boundary_index"]) == final_boundary_index
    }

    if len(final_lookup) != len(tasks):
        raise RuntimeError(
            "Final boundary does not contain one oracle selection per task."
        )

    # Mark the underlying pair rows selected at the final boundary.
    for task_index, final_selected in final_lookup.items():
        for row in pair_rows:
            if shared_backbone:
                same_candidate = (
                    int(row["boundary_index"]) == final_boundary_index
                    and int(row["evaluation_task_index"]) == task_index
                    and int(row["candidate_head_index"])
                    == int(final_selected["selected_head_index"])
                )
            else:
                same_candidate = (
                    int(row["evaluation_task_index"]) == task_index
                    and int(row["candidate_index"])
                    == int(final_selected["selected_candidate_index"])
                )

            if same_candidate:
                row["selected_final"] = True

    return selected_rows


def compute_metrics(
    *,
    tasks: list[str],
    selected_rows: list[dict[str, Any]],
    shared_backbone: bool,
    selection_criterion: str,
    forward_transfer_metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    final_boundary_index = len(tasks) - 1

    selected_lookup = {
        (
            int(row["boundary_index"]),
            int(row["task_index"]),
        ): row
        for row in selected_rows
    }

    final_rows = [
        selected_lookup[(final_boundary_index, task_index)]
        for task_index in range(len(tasks))
    ]

    final_success = {
        tasks[task_index]: float(
            final_rows[task_index]["success_rate"]
        )
        for task_index in range(len(tasks))
    }
    final_return = {
        tasks[task_index]: float(
            final_rows[task_index]["mean_return"]
        )
        for task_index in range(len(tasks))
    }

    end_success: dict[str, float] = {}
    end_return: dict[str, float] = {}
    forgetting_end_to_final: dict[str, float] = {}
    maximum_historical_success: dict[str, float] = {}
    forgetting_max_to_final: dict[str, float] = {}

    per_task_rows: list[dict[str, Any]] = []

    for task_index, task_name in enumerate(tasks):
        end_row = selected_lookup[(task_index, task_index)]
        final_row = selected_lookup[
            (final_boundary_index, task_index)
        ]

        historical_rows = [
            row
            for row in selected_rows
            if int(row["task_index"]) == task_index
            and int(row["boundary_index"]) >= task_index
        ]

        end_success_value = float(end_row["success_rate"])
        final_success_value = float(final_row["success_rate"])
        end_return_value = float(end_row["mean_return"])
        final_return_value = float(final_row["mean_return"])
        max_success_value = max(
            float(row["success_rate"])
            for row in historical_rows
        )

        end_success[task_name] = end_success_value
        end_return[task_name] = end_return_value
        forgetting_end_to_final[task_name] = (
            end_success_value - final_success_value
        )
        maximum_historical_success[task_name] = max_success_value
        forgetting_max_to_final[task_name] = (
            max_success_value - final_success_value
        )

        per_task_rows.append(
            {
                "task_index": task_index,
                "task_name": task_name,
                "end_of_task_success": end_success_value,
                "final_success": final_success_value,
                "forgetting_end_to_final": (
                    forgetting_end_to_final[task_name]
                ),
                "maximum_historical_success": max_success_value,
                "forgetting_max_to_final": (
                    forgetting_max_to_final[task_name]
                ),
                "end_of_task_return": end_return_value,
                "final_return": final_return_value,
                "selected_end_candidate_index": int(
                    end_row["selected_candidate_index"]
                ),
                "selected_end_candidate_source_task": str(
                    end_row["selected_candidate_source_task"]
                ),
                "selected_end_head_index": int(
                    end_row["selected_head_index"]
                ),
                "selected_final_candidate_index": int(
                    final_row["selected_candidate_index"]
                ),
                "selected_final_candidate_source_task": str(
                    final_row["selected_candidate_source_task"]
                ),
                "selected_final_head_index": int(
                    final_row["selected_head_index"]
                ),
                "continual_success_auc": (
                    forward_transfer_metrics[
                        "continual_success_auc_per_task"
                    ][task_name]
                ),
                "baseline_success_auc": (
                    forward_transfer_metrics[
                        "baseline_success_auc_per_task"
                    ][task_name]
                ),
                "raw_forward_transfer": (
                    forward_transfer_metrics[
                        "raw_forward_transfer_per_task"
                    ][task_name]
                ),
                "normalized_forward_transfer": (
                    ""
                    if forward_transfer_metrics[
                        "normalized_forward_transfer_per_task"
                    ][task_name] is None
                    else forward_transfer_metrics[
                        "normalized_forward_transfer_per_task"
                    ][task_name]
                ),
            }
        )

    # For consistency with the previous evaluator, average_forgetting uses all
    # tasks. The final task contributes zero because its end and final boundary
    # are the same.
    average_forgetting = float(
        np.mean(
            list(
                forgetting_end_to_final.values()
            )
        )
    )
    average_forgetting_max = float(
        np.mean(
            list(
                forgetting_max_to_final.values()
            )
        )
    )

    repository_mode = (
        "shared_backbone_task_specific_heads"
        if shared_backbone
        else "full_policy_repository"
    )

    metrics = {
        "evaluation_protocol": (
            "oracle_repository_final_ap_and_boundary_forgetting"
        ),
        "repository_mode": repository_mode,
        "shared_backbone": bool(shared_backbone),
        "selection_criterion": selection_criterion,
        "average_performance": float(
            np.mean(list(final_success.values()))
        ),
        "average_success_rate": float(
            np.mean(list(final_success.values()))
        ),
        "average_final_return": float(
            np.mean(list(final_return.values()))
        ),
        "average_forgetting": average_forgetting,
        "average_forgetting_end_to_final": average_forgetting,
        "average_forgetting_max_to_final": (
            average_forgetting_max
        ),
        "final_success_rate_per_task": final_success,
        "final_return_per_task": final_return,
        "end_of_task_success_per_task": end_success,
        "end_of_task_return_per_task": end_return,
        "forgetting_per_task": forgetting_end_to_final,
        "forgetting_end_to_final_per_task": (
            forgetting_end_to_final
        ),
        "maximum_historical_success_per_task": (
            maximum_historical_success
        ),
        "forgetting_max_to_final_per_task": (
            forgetting_max_to_final
        ),
        "average_performance_definition": (
            "Mean final oracle-selected deterministic success over all tasks."
        ),
        "forgetting_definition": (
            "For task j, oracle repository success immediately after learning "
            "task j minus final oracle repository success on task j."
        ),
        "max_history_forgetting_definition": (
            "For task j, maximum oracle repository success observed from "
            "boundary j through the final boundary minus final oracle "
            "repository success."
        ),
        "full_policy_repository_definition": (
            "Each task checkpoint contributes one frozen full policy using "
            "the source task head with the same index."
        ),
        "shared_backbone_repository_definition": (
            "At boundary b, checkpoint b contributes the current shared "
            "backbone and heads 0..b. Final AP uses the final checkpoint with "
            "all trained heads."
        ),
        **forward_transfer_metrics,
    }

    return metrics, per_task_rows


def write_boundary_matrix(
    *,
    path: Path,
    tasks: list[str],
    selected_rows: list[dict[str, Any]],
    value_key: str,
) -> None:
    lookup = {
        (
            int(row["boundary_index"]),
            int(row["task_index"]),
        ): row[value_key]
        for row in selected_rows
    }

    fields = [
        *BOUNDARY_MATRIX_FIELDS,
        *tasks,
    ]

    rows: list[dict[str, Any]] = []

    for boundary_index, boundary_task in enumerate(tasks):
        row: dict[str, Any] = {
            "boundary_index": boundary_index,
            "boundary_task": boundary_task,
        }

        for task_index, task_name in enumerate(tasks):
            row[task_name] = (
                lookup[(boundary_index, task_index)]
                if task_index <= boundary_index
                else ""
            )

        rows.append(row)

    write_csv(path, fields, rows)


def write_final_candidate_matrix(
    *,
    path: Path,
    tasks: list[str],
    pair_rows: list[dict[str, Any]],
    value_key: str,
    shared_backbone: bool,
) -> None:
    final_boundary_index = len(tasks) - 1

    if shared_backbone:
        final_rows = [
            row
            for row in pair_rows
            if int(row["boundary_index"]) == final_boundary_index
        ]
    else:
        final_rows = pair_rows

    lookup = {
        (
            int(row["candidate_index"]),
            int(row["evaluation_task_index"]),
        ): row[value_key]
        for row in final_rows
    }

    fields = [
        "candidate_index",
        "candidate_source_task",
        *tasks,
    ]
    rows: list[dict[str, Any]] = []

    for candidate_index, source_task in enumerate(tasks):
        row: dict[str, Any] = {
            "candidate_index": candidate_index,
            "candidate_source_task": source_task,
        }

        for task_index, task_name in enumerate(tasks):
            row[task_name] = lookup[
                (candidate_index, task_index)
            ]

        rows.append(row)

    write_csv(path, fields, rows)


def main() -> None:
    args = parse_args()

    run_directory = Path(
        args.run_dir
    ).expanduser()

    if not run_directory.is_dir():
        raise NotADirectoryError(
            f"Run directory does not exist: {run_directory}"
        )

    configuration = load_json(
        run_directory / "config.json"
    )

    baseline_curves_path = Path(
        args.baseline_curves
    ).expanduser()
    if not baseline_curves_path.is_file():
        raise FileNotFoundError(
            f"Baseline curves file does not exist: {baseline_curves_path}"
        )

    checkpoint_directory = (
        run_directory / "checkpoints"
    )
    checkpoint_paths = sorted(
        checkpoint_directory.glob("task_*.pt"),
        key=checkpoint_task_number,
    )

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No task_*.pt checkpoints found under "
            f"{checkpoint_directory}"
        )

    tasks = resolve_tasks(
        configuration,
        checkpoint_paths,
    )

    baseline_steps, baseline_curves = (
        load_baseline_success_curves(
            baseline_curves_path,
            tasks=tasks,
        )
    )
    continual_steps, continual_curves = (
        load_continual_active_task_curves(
            run_directory / "evaluations.csv",
            tasks=tasks,
        )
    )
    forward_transfer_metrics = (
        compute_learning_curve_forward_transfer(
            tasks=tasks,
            continual_steps=continual_steps,
            continual_curves=continual_curves,
            baseline_steps=baseline_steps,
            baseline_curves=baseline_curves,
        )
    )

    output_directory = (
        Path(args.output_dir).expanduser()
        if args.output_dir is not None
        else (
            run_directory
            / (
                "shared_backbone_repository_eval"
                if args.shared_backbone
                else "full_policy_repository_eval"
            )
        )
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    payloads = [
        load_checkpoint(path)
        for path in checkpoint_paths
    ]

    if args.shared_backbone:
        pair_rows = evaluate_shared_backbone_repository(
            checkpoint_paths=checkpoint_paths,
            payloads=payloads,
            tasks=tasks,
            episodes=args.episodes,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            device=args.device,
        )
    else:
        pair_rows = evaluate_full_policy_repository(
            checkpoint_paths=checkpoint_paths,
            payloads=payloads,
            tasks=tasks,
            episodes=args.episodes,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            device=args.device,
        )

    selected_rows = select_boundary_oracle_rows(
        pair_rows=pair_rows,
        tasks=tasks,
        criterion=args.selection_criterion,
        shared_backbone=args.shared_backbone,
    )

    metrics, per_task_rows = compute_metrics(
        tasks=tasks,
        selected_rows=selected_rows,
        shared_backbone=args.shared_backbone,
        selection_criterion=args.selection_criterion,
        forward_transfer_metrics=forward_transfer_metrics,
    )

    metrics.update(
        {
            "run_directory": str(run_directory),
            "tasks": tasks,
            "checkpoint_count": len(checkpoint_paths),
            "episodes_per_evaluation": args.episodes,
            "max_episode_steps": args.max_episode_steps,
            "seed": args.seed,
            "device": args.device,
            "output_directory": str(output_directory),
            "baseline_curves_path": str(baseline_curves_path),
            "continual_evaluations_path": str(
                run_directory / "evaluations.csv"
            ),
        }
    )

    write_csv(
        output_directory / "candidate_task_results.csv",
        PAIR_FIELDS,
        pair_rows,
    )

    write_csv(
        output_directory / "oracle_selection_by_boundary.csv",
        SELECTED_FIELDS,
        selected_rows,
    )

    write_csv(
        output_directory / "per_task_metrics.csv",
        PER_TASK_FIELDS,
        per_task_rows,
    )

    write_boundary_matrix(
        path=(
            output_directory
            / "oracle_success_by_boundary.csv"
        ),
        tasks=tasks,
        selected_rows=selected_rows,
        value_key="success_rate",
    )

    write_boundary_matrix(
        path=(
            output_directory
            / "oracle_return_by_boundary.csv"
        ),
        tasks=tasks,
        selected_rows=selected_rows,
        value_key="mean_return",
    )

    write_boundary_matrix(
        path=(
            output_directory
            / "selected_head_by_boundary.csv"
        ),
        tasks=tasks,
        selected_rows=selected_rows,
        value_key="selected_head_index",
    )

    write_final_candidate_matrix(
        path=(
            output_directory
            / "final_candidate_success_matrix.csv"
        ),
        tasks=tasks,
        pair_rows=pair_rows,
        value_key="success_rate",
        shared_backbone=args.shared_backbone,
    )

    write_final_candidate_matrix(
        path=(
            output_directory
            / "final_candidate_return_matrix.csv"
        ),
        tasks=tasks,
        pair_rows=pair_rows,
        value_key="mean_return",
        shared_backbone=args.shared_backbone,
    )

    with (
        output_directory / "repository_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            sort_keys=True,
        )

    print()
    print("=" * 72)
    print("Repository evaluation completed")
    print("=" * 72)
    print(
        "Repository mode      : "
        f"{metrics['repository_mode']}"
    )
    print(
        "Selection criterion  : "
        f"{metrics['selection_criterion']}"
    )
    print(
        "Average performance  : "
        f"{metrics['average_performance']:.6f}"
    )
    print(
        "Average forgetting   : "
        f"{metrics['average_forgetting']:.6f}"
    )
    print(
        "Max-history forgetting: "
        f"{metrics['average_forgetting_max_to_final']:.6f}"
    )
    print(
        "Average final return : "
        f"{metrics['average_final_return']:.6f}"
    )
    print(
        "Raw forward transfer : "
        f"{metrics['raw_forward_transfer']:.6f}"
    )
    normalized_fwt = metrics["forward_transfer"]
    print(
        "Forward transfer     : "
        + (
            "undefined"
            if normalized_fwt is None
            else f"{normalized_fwt:.6f}"
        )
    )
    print(
        "Results              : "
        f"{output_directory}"
    )


if __name__ == "__main__":
    main()
