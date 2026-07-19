"""Run ClonEx-style multi-head SAC on a CW10 task subsequence.

The runner can train the first N tasks of the official CW10 sequence while
retaining the original 10-dimensional task identity and all 10 actor/critic
heads. Use ``--sequence-task-count 3`` for CW3 development experiments and
``--sequence-task-count 10`` for the complete CW10 sequence.

The default exploration strategy is best-return:

- task 0 uses uniform random initial exploration;
- before task i > 0, every previous actor head is evaluated on task i;
- the previous head with the highest mean return generates the initial
  exploration data;
- SAC updates still train the current task's fresh actor and critic heads.

The shared actor and critic backbones continue across tasks. The online replay
buffer and optimizer state are reset at task boundaries by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from crl_cw.agents import ReplayBuffer, SACAgent
from crl_cw.envs.cw_env import make_cw_env
from crl_cw.envs.tasks import get_cw10_tasks
from crl_cw.evaluation import EvaluationConfig, EvaluationResult, SACEvaluator
from crl_cw.training import SACTrainer, SACTrainerConfig
from crl_cw.utils import save_sac_checkpoint
from crl_cw.utils.wandb_logger import WandbLogger


METHOD_NAME_PREFIX = "clonex_sac_multi_head"

EVALUATION_FIELDS = [
    "evaluation_index",
    "global_step",
    "gradient_updates",
    "active_task_index",
    "active_task_name",
    "active_task_step",
    "evaluation_task_index",
    "evaluation_task_name",
    "deterministic_average_return",
    "deterministic_success_rate",
    "deterministic_average_episode_length",
    "stochastic_average_return",
    "stochastic_success_rate",
    "stochastic_average_episode_length",
    "actor_loss",
    "q1_loss",
    "q2_loss",
    "alpha_loss",
    "alpha",
    "q1_mean",
    "q2_mean",
    "q_target_mean",
    "log_prob_mean",
    "elapsed_seconds",
]

TASK_SUMMARY_FIELDS = [
    "task_index",
    "task_name",
    "exploration_strategy",
    "selected_exploration_head",
    "selected_exploration_source_task",
    "selected_exploration_mean_return",
    "global_step_end",
    "task_gradient_updates",
    "cumulative_gradient_updates",
    "completed_training_episodes",
    "mean_training_return",
    "final_alpha",
    "replay_buffer_size",
    "elapsed_seconds",
]

PER_TASK_METRIC_FIELDS = [
    "task_index",
    "task_name",
    "end_of_task_success",
    "final_success",
    "forgetting",
    "final_mean_return",
    "raw_forward_transfer",
    "normalized_forward_transfer",
]

FINAL_EVALUATION_FIELDS = [
    "task_index",
    "task_name",
    "mean_return",
    "success_rate",
    "mean_episode_length",
    "num_episodes",
]

BEST_RETURN_SELECTION_FIELDS = [
    "current_task_index",
    "current_task_name",
    "candidate_head_index",
    "candidate_source_task",
    "mean_return",
    "success_rate",
    "mean_episode_length",
    "selected",
]


def parse_args() -> argparse.Namespace:
    """Parse CW10 and W&B command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run ClonEx-SAC-style multi-head SAC on the first N tasks "
            "of the official CW10 sequence."
        )
    )

    parser.add_argument(
        "--sequence-task-count",
        type=int,
        default=10,
        help=(
            "Number of tasks trained from the start of the official "
            "CW10 sequence. The network still retains all 10 heads and "
            "the original 10-dimensional task ID."
        ),
    )
    parser.add_argument("--steps-per-task", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument(
        "--det-eval-episodes",
        type=int,
        default=3,
        help=(
            "Deterministic episodes per intermediate evaluation. "
            "These results define formal learning curves and metrics."
        ),
    )
    parser.add_argument(
        "--stoch-eval-episodes",
        type=int,
        default=0,
        help=(
            "Optional stochastic diagnostic episodes per evaluation. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=5,
        help=(
            "Episodes per task in the final all-task evaluation."
        ),
    )
    parser.add_argument(
        "--best-return-eval-episodes",
        type=int,
        default=2,
        help=(
            "Number of stochastic episodes used to evaluate each "
            "previous actor head during best-return selection. "
            "This is independent of regular performance evaluation."
        ),
    )
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument(
        "--exploration-strategy",
        type=str,
        choices=("random", "best-return"),
        default="best-return",
        help=(
            "Initial exploration for each task. With best-return, task 0 "
            "uses random actions and later tasks use the previous actor "
            "head with the highest return on the current task."
        ),
    )
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--target-output-std", type=float, default=0.089)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        "--reset-buffer-on-task-change",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reset-optimizer-on-task-change",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--initial-log-alpha",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--final-metric-tail-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--baseline-curves",
        type=str,
        default=None,
        help=(
            "Optional path to matched single-task baseline_curves.json. "
            "When provided, Forward Transfer is computed automatically. "
            "When omitted, the runner searches under outputs/ for a "
            "matching baseline file."
        ),
    )
    parser.add_argument(
        "--resume-run-dir",
        type=str,
        default=None,
        help=(
            "Optional previous run directory created by this runner. "
            "The agent, completed-task count, evaluation history, and "
            "metric history are loaded so training can continue at the "
            "next task boundary."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory used for local results. When omitted, results "
            "are saved automatically under outputs/cwN_finetune, where "
            "N is --sequence-task-count."
        ),
    )
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="crl-cw",
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help=(
            "Optional W&B group. When omitted, a cwN group is generated "
            "automatically from --sequence-task-count and the exploration "
            "strategy."
        ),
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument(
        "--wandb-log-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--wandb-upload-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()

    for name in (
        "steps_per_task",
        "eval_every",
        "det_eval_episodes",
        "final_eval_episodes",
        "best_return_eval_episodes",
        "replay_size",
        "batch_size",
        "update_every",
        "max_episode_steps",
        "final_metric_tail_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(
                f"--{name.replace('_', '-')} must be positive."
            )

    if not 1 <= args.sequence_task_count <= 10:
        parser.error(
            "--sequence-task-count must be between 1 and 10."
        )

    if args.stoch_eval_episodes < 0:
        parser.error(
            "--stoch-eval-episodes must be non-negative."
        )

    if args.start_steps < 0:
        parser.error("--start-steps must be non-negative.")
    if args.update_after < 0:
        parser.error("--update-after must be non-negative.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning-rate must be finite and positive.")
    if not math.isfinite(args.gamma) or not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be finite and in [0, 1].")
    if not math.isfinite(args.polyak) or not 0.0 <= args.polyak <= 1.0:
        parser.error("--polyak must be finite and in [0, 1].")
    if (
        not math.isfinite(args.target_output_std)
        or args.target_output_std <= 0.0
    ):
        parser.error(
            "--target-output-std must be finite and positive."
        )
    if not math.isfinite(args.initial_log_alpha):
        parser.error("--initial-log-alpha must be finite.")

    return args


def resolve_method_name(
    exploration_strategy: str,
) -> str:
    """Return a stable method label for saved results."""
    normalized_strategy = exploration_strategy.replace("-", "_")
    return (
        f"{METHOD_NAME_PREFIX}_{normalized_strategy}_exploration"
    )


def compute_target_entropy(
    action_dim: int,
    target_output_std: float,
) -> float:
    """Convert desired policy output std to target entropy."""
    one_dimensional_entropy = math.log(
        target_output_std * math.sqrt(2.0 * math.pi * math.e)
    )
    return float(action_dim * one_dimensional_entropy)


def create_run_directory(args: argparse.Namespace) -> Path:
    """Create one unique local output directory."""
    sequence_label = f"cw{args.sequence_task_count}"

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        strategy = args.exploration_strategy.replace("-", "_")
        run_name = (
            f"{sequence_label}_clonex_multi_head_{strategy}_"
            f"seed{args.seed}_{timestamp}"
        )
    else:
        run_name = args.run_name.replace("/", "_").replace("\\", "_")

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir is not None
        else Path("outputs") / f"{sequence_label}_finetune"
    )

    run_directory = output_dir / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "checkpoints").mkdir()
    (run_directory / "metrics").mkdir()
    (run_directory / "plots").mkdir()
    return run_directory


def append_csv_row(
    path: Path,
    fields: list[str],
    row: dict[str, Any],
) -> None:
    """Append one CSV row and create a header when needed."""
    file_exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def metric_or_empty(
    metrics: dict[str, float] | None,
    name: str,
) -> float | str:
    """Read one finite metric or return an empty CSV field."""
    if metrics is None or name not in metrics:
        return ""

    value = float(metrics[name])
    if not math.isfinite(value):
        raise RuntimeError(
            f"Training metric {name!r} is non-finite: {value}."
        )
    return value


def validate_env_compatibility(
    env: Any,
    *,
    observation_dim: int,
    action_dim: int,
    task_name: str,
) -> None:
    """Check that all CW10 tasks share expected dimensions."""
    if env.observation_space.shape != (observation_dim,):
        raise ValueError(
            f"{task_name} observation shape is "
            f"{env.observation_space.shape}; "
            f"expected {(observation_dim,)}."
        )
    if env.action_space.shape != (action_dim,):
        raise ValueError(
            f"{task_name} action shape is "
            f"{env.action_space.shape}; expected {(action_dim,)}."
        )


def close_env(env: Any | None) -> None:
    """Close an environment when possible."""
    if env is None:
        return
    close_method = getattr(env, "close", None)
    if callable(close_method):
        close_method()


def reset_optimizer_state(agent: SACAgent) -> None:
    """Reset Adam moments while retaining network parameters."""
    agent.optimizer.state.clear()


def validate_evaluation_result(
    result: EvaluationResult,
    *,
    expected_episodes: int,
    label: str,
) -> None:
    """Validate one evaluator result."""
    if result.num_episodes != expected_episodes:
        raise RuntimeError(
            f"{label} returned {result.num_episodes} episodes; "
            f"expected {expected_episodes}."
        )
    if not math.isfinite(result.mean_return):
        raise RuntimeError(f"{label} mean return is non-finite.")
    if not 0.0 <= result.success_rate <= 1.0:
        raise RuntimeError(
            f"{label} success rate is outside [0, 1]."
        )


def select_best_return_head(
    *,
    evaluator: SACEvaluator,
    current_task_index: int,
    tasks: list[str] | tuple[str, ...],
    expected_episodes: int,
) -> tuple[int, dict[int, EvaluationResult]]:
    """Evaluate every previous head and select the highest-return one.

    All candidate heads are evaluated on the same current-task
    environment resets. Ties are resolved in favour of the lowest head
    index because candidates are visited in ascending order.
    """
    if current_task_index <= 0:
        raise ValueError(
            "Best-return selection requires at least one previous task."
        )

    if current_task_index >= len(tasks):
        raise ValueError(
            "current_task_index is outside the task sequence."
        )

    results: dict[int, EvaluationResult] = {}

    for head_index in range(current_task_index):
        result = evaluator.evaluate(
            deterministic=False,
            head_index=head_index,
        )
        validate_evaluation_result(
            result,
            expected_episodes=expected_episodes,
            label=(
                "best-return/"
                f"{tasks[head_index]}-to-{tasks[current_task_index]}"
            ),
        )
        results[head_index] = result

    best_head_index = max(
        results,
        key=lambda index: results[index].mean_return,
    )
    return best_head_index, results



def _copy_if_present(
    source_directory: Path,
    destination_directory: Path,
    filename: str,
) -> None:
    """Copy one previous result file into a resumed run directory."""
    source_path = source_directory / filename
    destination_path = destination_directory / filename
    if source_path.is_file():
        shutil.copy2(source_path, destination_path)


def save_resume_state(
    *,
    path: Path,
    agent: SACAgent,
    completed_task_count: int,
    cumulative_gradient_updates: int,
    evaluation_index: int,
    evaluation_history: list[dict[str, Any]],
    tasks: list[str] | tuple[str, ...],
    steps_per_task: int,
    eval_every: int,
    seed: int,
    exploration_strategy: str,
    run_directory: Path,
) -> None:
    """Save task-boundary state required to continue a later task."""
    payload = {
        "schema_version": 1,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "completed_task_count": int(completed_task_count),
        "cumulative_gradient_updates": int(
            cumulative_gradient_updates
        ),
        "evaluation_index": int(evaluation_index),
        "evaluation_history": list(evaluation_history),
        "tasks": list(tasks),
        "steps_per_task": int(steps_per_task),
        "eval_every": int(eval_every),
        "seed": int(seed),
        "exploration_strategy": str(exploration_strategy),
        "source_run_directory": str(run_directory),
    }
    torch.save(payload, path)


def load_resume_state(path: Path) -> dict[str, Any]:
    """Load a resume checkpoint created by save_resume_state."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {path}"
        )

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(payload, dict):
        raise TypeError(
            "Resume checkpoint must contain a dictionary."
        )

    required_fields = (
        "agent_state_dict",
        "completed_task_count",
        "cumulative_gradient_updates",
        "evaluation_index",
        "evaluation_history",
        "tasks",
        "steps_per_task",
        "eval_every",
        "seed",
        "exploration_strategy",
    )
    for field_name in required_fields:
        if field_name not in payload:
            raise ValueError(
                "Resume checkpoint is missing required field "
                f"{field_name!r}."
            )

    return payload


def find_matching_baseline_curves(
    *,
    requested_path: str | None,
    tasks: list[str],
    steps_per_task: int,
    eval_every: int,
    seed: int,
) -> Path | None:
    """Find a deterministic baseline matching this continual run.

    An explicitly requested path is preferred. If that path does not
    exist, the function warns and falls back to automatic discovery
    instead of aborting a completed training run. A missing match is
    represented by ``None`` so AP, forgetting, checkpoints, plots, and
    reports can still be produced while FT is marked unavailable.
    """

    def load_matching_candidate(candidate: Path) -> bool:
        if not candidate.is_file():
            return False

        try:
            with candidate.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return False

        baseline_tasks = data.get("tasks")
        if not isinstance(baseline_tasks, list):
            return False
        if baseline_tasks[: len(tasks)] != list(tasks):
            return False
        if int(data.get("steps_per_task", -1)) != steps_per_task:
            return False
        if int(data.get("eval_every", -1)) != eval_every:
            return False
        if int(data.get("seed", -1)) != seed:
            return False

        protocol = data.get("evaluation_protocol")
        if protocol not in (None, "deterministic"):
            return False

        curves = data.get("deterministic_success_curves")
        if not isinstance(curves, list):
            return False
        if len(curves) < len(tasks):
            return False

        expected_points = steps_per_task // eval_every
        for task_curve in curves[: len(tasks)]:
            if not isinstance(task_curve, list):
                return False
            if len(task_curve) != expected_points:
                return False

        return True

    if requested_path is not None:
        requested_candidate = Path(requested_path).expanduser()
        if requested_candidate.is_file():
            if load_matching_candidate(requested_candidate):
                print(
                    "Using requested baseline curves: "
                    f"{requested_candidate}",
                    flush=True,
                )
                return requested_candidate

            print(
                "Warning: requested baseline exists but does not match "
                "this run's tasks, steps, evaluation interval, seed, or "
                f"deterministic curve shape: {requested_candidate}",
                flush=True,
            )
        else:
            print(
                "Warning: requested baseline file does not exist; "
                "falling back to automatic discovery: "
                f"{requested_candidate}",
                flush=True,
            )

    baseline_root = Path("outputs") / "cw10_single_task_baselines"
    if not baseline_root.is_dir():
        print(
            "Warning: no single-task baseline directory was found at "
            f"{baseline_root}. Forward transfer will be unavailable.",
            flush=True,
        )
        return None

    candidates = list(
        baseline_root.rglob("aggregate/baseline_curves.json")
    )
    candidates.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for candidate in candidates:
        if load_matching_candidate(candidate):
            print(
                "Automatically matched baseline curves: "
                f"{candidate}",
                flush=True,
            )
            return candidate

    print(
        "Warning: no matching deterministic single-task baseline was "
        "found. AP, forgetting, returns, checkpoints, plots, and reports "
        "will still be saved; forward transfer will be unavailable.",
        flush=True,
    )
    return None

def compute_forward_transfer_from_baseline(
    *,
    evaluation_history: list[dict[str, Any]],
    tasks: list[str],
    baseline_path: Path | None,
    steps_per_task: int,
    eval_every: int,
) -> dict[str, Any]:
    """Compute raw and normalized FT from matched success curves."""
    if baseline_path is None:
        return {
            "forward_transfer": None,
            "raw_forward_transfer": None,
            "forward_transfer_status": (
                "unavailable_no_matching_single_task_baseline"
            ),
            "raw_forward_transfer_per_task": {},
            "normalized_forward_transfer_per_task": {},
            "normalized_forward_transfer_defined_tasks": 0,
            "normalized_forward_transfer_undefined_tasks": [],
            "baseline_curves_path": None,
        }

    with baseline_path.open("r", encoding="utf-8") as file:
        baseline_data = json.load(file)

    baseline_tasks = list(baseline_data["tasks"])
    baseline_curves = list(
        baseline_data["deterministic_success_curves"]
    )
    baseline_steps = list(baseline_data["evaluation_steps"])

    expected_steps = list(
        range(
            eval_every,
            steps_per_task + 1,
            eval_every,
        )
    )
    if expected_steps[-1] != steps_per_task:
        expected_steps.append(steps_per_task)

    if baseline_steps != expected_steps:
        raise ValueError(
            "Baseline evaluation grid does not match this run: "
            f"baseline={baseline_steps}, expected={expected_steps}."
        )

    raw_per_task: dict[str, float] = {}
    normalized_per_task: dict[str, float] = {}

    for task_index, task_name in enumerate(tasks):
        if baseline_tasks[task_index] != task_name:
            raise ValueError(
                "Baseline task order does not match the continual run."
            )

        active_records = [
            record
            for record in evaluation_history
            if int(record["active_task_index"]) == task_index
        ]
        active_records.sort(
            key=lambda record: int(record["active_task_step"])
        )

        continual_curve = [
            float(record["deterministic_success"][task_name])
            for record in active_records
        ]
        baseline_curve = [
            float(value)
            for value in baseline_curves[task_index]
        ]

        if len(continual_curve) != len(baseline_curve):
            raise ValueError(
                "Continual and baseline curves have different lengths "
                f"for {task_name}: continual={len(continual_curve)}, "
                f"baseline={len(baseline_curve)}."
            )

        continual_mean = float(np.mean(continual_curve))
        baseline_mean = float(np.mean(baseline_curve))
        raw_value = continual_mean - baseline_mean
        remaining_headroom = 1.0 - baseline_mean

        raw_per_task[task_name] = float(raw_value)

        if remaining_headroom <= 1e-12:
            # Normalized FT is mathematically undefined when the
            # independent baseline already has mean success 1.0.
            # Preserve raw FT and represent only this per-task
            # normalized value as null in JSON/CSV outputs.
            normalized_per_task[task_name] = None
        else:
            normalized_value = raw_value / remaining_headroom
            normalized_per_task[task_name] = float(
                normalized_value
            )

    defined_normalized_values = [
        float(value)
        for value in normalized_per_task.values()
        if value is not None
    ]

    if defined_normalized_values:
        average_normalized_transfer = float(
            np.mean(defined_normalized_values)
        )
        forward_transfer_status = "available"
    else:
        average_normalized_transfer = None
        forward_transfer_status = (
            "unavailable_all_normalized_ft_undefined"
        )

    return {
        "forward_transfer": average_normalized_transfer,
        "raw_forward_transfer": float(
            np.mean(list(raw_per_task.values()))
        ),
        "forward_transfer_status": forward_transfer_status,
        "raw_forward_transfer_per_task": raw_per_task,
        "normalized_forward_transfer_per_task": normalized_per_task,
        "normalized_forward_transfer_defined_tasks": int(
            len(defined_normalized_values)
        ),
        "normalized_forward_transfer_undefined_tasks": [
            task_name
            for task_name, value in normalized_per_task.items()
            if value is None
        ],
        "baseline_curves_path": str(baseline_path),
    }


def compute_final_metrics(
    *,
    evaluation_history: list[dict[str, Any]],
    final_success_per_task: dict[str, float],
    final_return_per_task: dict[str, float],
    tasks: list[str] | tuple[str, ...],
    tail_size: int,
    baseline_path: Path | None,
    steps_per_task: int,
    eval_every: int,
) -> dict[str, Any]:
    """Compute continual metrics from active curves and final evaluation."""
    if not evaluation_history:
        raise ValueError("evaluation_history must not be empty.")

    task_names = list(tasks)
    final_per_task = {
        task_name: float(final_success_per_task[task_name])
        for task_name in task_names
    }
    final_return_per_task = {
        task_name: float(final_return_per_task[task_name])
        for task_name in task_names
    }

    end_of_task_per_task: dict[str, float] = {}
    forgetting_per_task: dict[str, float] = {}

    for task_index, task_name in enumerate(task_names):
        active_records = [
            record
            for record in evaluation_history
            if int(record["active_task_index"]) == task_index
        ]
        active_records.sort(
            key=lambda record: int(record["active_task_step"])
        )
        if not active_records:
            raise RuntimeError(
                f"No active-task evaluations for {task_name}."
            )

        active_tail = active_records[
            -min(tail_size, len(active_records)):
        ]
        end_value = float(
            np.mean(
                [
                    float(record["deterministic_success"][task_name])
                    for record in active_tail
                ]
            )
        )
        end_of_task_per_task[task_name] = end_value
        forgetting_per_task[task_name] = float(
            end_value - final_per_task[task_name]
        )

    forward_transfer = compute_forward_transfer_from_baseline(
        evaluation_history=evaluation_history,
        tasks=task_names,
        baseline_path=baseline_path,
        steps_per_task=steps_per_task,
        eval_every=eval_every,
    )

    average_performance = float(
        np.mean(list(final_per_task.values()))
    )

    return {
        "tail_size": int(tail_size),
        "average_performance": average_performance,
        "average_success_rate": average_performance,
        "average_forgetting": float(
            np.mean(list(forgetting_per_task.values()))
        ),
        "average_final_return": float(
            np.mean(list(final_return_per_task.values()))
        ),
        "final_per_task": final_per_task,
        "final_success_rate_per_task": final_per_task,
        "end_of_task_per_task": end_of_task_per_task,
        "forgetting_per_task": forgetting_per_task,
        "final_return_per_task": final_return_per_task,
        **forward_transfer,
    }


def save_final_metrics(
    *,
    metrics_directory: Path,
    tasks: list[str] | tuple[str, ...],
    final_metrics: dict[str, Any],
) -> tuple[Path, Path]:
    """Save scalar and per-task continual metrics."""
    metrics_json_path = metrics_directory / "continual_metrics.json"
    per_task_csv_path = metrics_directory / "per_task_metrics.csv"

    with metrics_json_path.open("w", encoding="utf-8") as file:
        json.dump(final_metrics, file, indent=2, sort_keys=True)

    for task_index, task_name in enumerate(tasks):
        append_csv_row(
            per_task_csv_path,
            PER_TASK_METRIC_FIELDS,
            {
                "task_index": task_index,
                "task_name": task_name,
                "end_of_task_success": final_metrics[
                    "end_of_task_per_task"
                ][task_name],
                "final_success": final_metrics[
                    "final_per_task"
                ][task_name],
                "forgetting": final_metrics[
                    "forgetting_per_task"
                ][task_name],
                "final_mean_return": final_metrics[
                    "final_return_per_task"
                ][task_name],
                "raw_forward_transfer": final_metrics[
                    "raw_forward_transfer_per_task"
                ].get(task_name, ""),
                "normalized_forward_transfer": final_metrics[
                    "normalized_forward_transfer_per_task"
                ].get(task_name, ""),
            },
        )

    return metrics_json_path, per_task_csv_path



def plot_learning_curves(
    *,
    evaluation_history: list[dict[str, Any]],
    tasks: list[str] | tuple[str, ...],
    plots_directory: Path,
) -> dict[str, Path]:
    """Save success and return learning curves for one run."""
    if not evaluation_history:
        raise ValueError(
            "Cannot plot learning curves without evaluation history."
        )

    plots_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    task_names = list(tasks)
    saved_paths: dict[str, Path] = {}

    success_path = (
        plots_directory / "active_task_success_curve.png"
    )
    figure = plt.figure(figsize=(9, 6))
    axes = figure.add_subplot(1, 1, 1)

    for task_index, task_name in enumerate(task_names):
        records = [
            record
            for record in evaluation_history
            if int(record["active_task_index"]) == task_index
        ]
        records.sort(
            key=lambda record: int(record["active_task_step"])
        )
        if not records:
            continue

        x_values = [
            int(record["active_task_step"])
            for record in records
        ]
        y_values = [
            float(record["deterministic_success"][task_name])
            for record in records
        ]
        axes.plot(
            x_values,
            y_values,
            marker="o",
            label=task_name,
        )

    axes.set_title("Active-task deterministic success")
    axes.set_xlabel("Environment steps on active task")
    axes.set_ylabel("Success rate")
    axes.set_ylim(-0.02, 1.02)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(success_path, dpi=180)
    plt.close(figure)
    saved_paths["active_task_deterministic_success_curve"] = success_path

    return_path = (
        plots_directory / "active_task_return_curve.png"
    )
    figure = plt.figure(figsize=(9, 6))
    axes = figure.add_subplot(1, 1, 1)

    for task_index, task_name in enumerate(task_names):
        records = [
            record
            for record in evaluation_history
            if int(record["active_task_index"]) == task_index
        ]
        records.sort(
            key=lambda record: int(record["active_task_step"])
        )
        if not records:
            continue

        x_values = [
            int(record["active_task_step"])
            for record in records
        ]
        y_values = [
            float(record["deterministic_return"][task_name])
            for record in records
        ]
        axes.plot(
            x_values,
            y_values,
            marker="o",
            label=task_name,
        )

    axes.set_title("Active-task mean return")
    axes.set_xlabel("Environment steps on active task")
    axes.set_ylabel("Mean episode return")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(return_path, dpi=180)
    plt.close(figure)
    saved_paths["active_task_deterministic_return_curve"] = return_path

    return saved_paths


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as a readable duration."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    return (
        f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"
    )


def save_experiment_report(
    *,
    path: Path,
    run_name: str,
    started_at: str,
    completed_at: str,
    elapsed_seconds: float,
    configuration: dict[str, Any],
    final_metrics: dict[str, Any],
    plot_paths: dict[str, Path],
    run_directory: Path,
    summary_path: Path,
    metrics_json_path: Path,
    per_task_metrics_path: Path,
    final_checkpoint: Path,
    resume_checkpoint_path: Path,
    wandb_run_url: str | None,
) -> None:
    """Write one human-readable experiment report."""
    forward_transfer = final_metrics["forward_transfer"]
    forward_transfer_text = (
        "Unavailable "
        f"({final_metrics['forward_transfer_status']})"
        if forward_transfer is None
        else f"{float(forward_transfer):.6f}"
    )

    lines = [
        f"# Experiment summary: {run_name}",
        "",
        "## Timing",
        "",
        f"- Started: {started_at}",
        f"- Completed: {completed_at}",
        (
            "- Duration: "
            f"{format_duration(elapsed_seconds)} "
            f"({elapsed_seconds:.2f} seconds)"
        ),
        "",
        "## Experiment configuration",
        "",
        f"- Method: {configuration['method']}",
        (
            "- Exploration strategy: "
            f"{configuration['exploration_strategy']}"
        ),
        f"- Seed: {configuration['seed']}",
        (
            "- Trained task count: "
            f"{configuration['sequence_task_count']}"
        ),
        (
            "- Network task heads: "
            f"{configuration['network_num_tasks']}"
        ),
        (
            "- Tasks: "
            + ", ".join(configuration["tasks"])
        ),
        (
            "- Steps per task: "
            f"{configuration['steps_per_task']:,}"
        ),
        (
            "- Evaluation interval: "
            f"{configuration['eval_every']:,}"
        ),
        (
            "- Optional stochastic diagnostic episodes: "
            f"{configuration['stoch_eval_episodes']}"
        ),
        (
            "- Final all-task evaluation episodes: "
            f"{configuration['final_eval_episodes']}"
        ),
        (
            "- Best-return selection episodes: "
            f"{configuration['best_return_eval_episodes']}"
        ),
        (
            "- Formal deterministic evaluation episodes: "
            f"{configuration['det_eval_episodes']}"
        ),
        (
            "- Initial exploration steps: "
            f"{configuration['start_steps']:,}"
        ),
        (
            "- Gradient updates start after: "
            f"{configuration['update_after']:,} steps"
        ),
        (
            "- Update interval: "
            f"{configuration['update_every']}"
        ),
        f"- Batch size: {configuration['batch_size']}",
        f"- Device: {configuration['device']}",
        "",
        "## Final results",
        "",
        (
            "- Average performance: "
            f"{final_metrics['average_performance']:.6f}"
        ),
        (
            "- Average success rate: "
            f"{final_metrics['average_success_rate']:.6f}"
        ),
        (
            "- Average forgetting: "
            f"{final_metrics['average_forgetting']:.6f}"
        ),
        (
            "- Average final return: "
            f"{final_metrics['average_final_return']:.6f}"
        ),
        f"- Forward transfer: {forward_transfer_text}",
    ]

    if final_metrics["raw_forward_transfer"] is not None:
        lines.append(
            "- Raw forward transfer: "
            f"{final_metrics['raw_forward_transfer']:.6f}"
        )

    lines.extend(
        [
            "",
            "## Per-task final results",
            "",
        ]
    )
    for task_name, success_value in final_metrics[
        "final_success_rate_per_task"
    ].items():
        return_value = final_metrics[
            "final_return_per_task"
        ][task_name]
        lines.append(
            f"- {task_name}: "
            f"success={float(success_value):.6f}, "
            f"return={float(return_value):.6f}"
        )

    lines.extend(
        [
            "",
            "## Saved learning curves",
            "",
        ]
    )
    for plot_name, plot_path in plot_paths.items():
        relative_path = plot_path.relative_to(
            run_directory
        )
        lines.append(
            f"- {plot_name}: `{relative_path}`"
        )

    lines.extend(
        [
            "",
            "## Saved files",
            "",
            f"- Run directory: `{run_directory}`",
            f"- Summary: `{summary_path}`",
            f"- Metrics JSON: `{metrics_json_path}`",
            (
                "- Per-task metrics: "
                f"`{per_task_metrics_path}`"
            ),
            f"- Final checkpoint: `{final_checkpoint}`",
            (
                "- Resume checkpoint: "
                f"`{resume_checkpoint_path}`"
            ),
        ]
    )

    if wandb_run_url is not None:
        lines.append(f"- W&B run: {wandb_run_url}")

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run one CW10-subsequence continual-learning experiment."""
    args = parse_args()
    method_name = resolve_method_name(
        args.exploration_strategy
    )

    all_tasks = list(get_cw10_tasks())
    tasks = all_tasks[:args.sequence_task_count]

    # The environment still appends the original 10-dimensional CW10
    # one-hot task identity. Therefore the agent must retain all 10
    # actor heads, critic heads, and task-specific entropy coefficients,
    # even when this run trains only the first N tasks.
    network_num_tasks = len(all_tasks)
    sequence_task_count = len(tasks)
    task_id_dim = network_num_tasks

    run_directory = create_run_directory(args)

    evaluations_path = run_directory / "evaluations.csv"
    task_summaries_path = run_directory / "task_summaries.csv"
    best_return_selection_path = (
        run_directory / "best_return_selection.csv"
    )
    final_evaluation_path = (
        run_directory / "final_evaluation.csv"
    )
    summary_path = run_directory / "summary.json"
    experiment_report_path = (
        run_directory / "summary_report.md"
    )
    configuration_path = run_directory / "config.json"
    plots_directory = run_directory / "plots"
    checkpoints_directory = run_directory / "checkpoints"
    metrics_directory = run_directory / "metrics"
    resume_checkpoint_path = (
        checkpoints_directory / "resume_state.pt"
    )
    resume_manifest_path = (
        checkpoints_directory / "resume_manifest.json"
    )

    prototype_env = make_cw_env(
        task_name=tasks[0],
        seed=args.seed,
        append_task_id=True,
    )
    try:
        observation_dim = int(
            prototype_env.observation_space.shape[0]
        )
        action_dim = int(prototype_env.action_space.shape[0])
        action_low = np.asarray(
            prototype_env.action_space.low,
            dtype=np.float32,
        )
        action_high = np.asarray(
            prototype_env.action_space.high,
            dtype=np.float32,
        )
    finally:
        close_env(prototype_env)

    physical_observation_dim = observation_dim - task_id_dim
    target_entropy = compute_target_entropy(
        action_dim,
        args.target_output_std,
    )

    configuration = dict(vars(args))
    configuration.update(
        {
            "tasks": tasks,
            "all_cw10_tasks": all_tasks,
            "sequence_task_count": sequence_task_count,
            "network_num_tasks": network_num_tasks,
            "task_id_dim": task_id_dim,
            "physical_observation_dim": physical_observation_dim,
            "total_steps": (
                args.steps_per_task * sequence_task_count
            ),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "resolved_target_entropy": target_entropy,
            "method": method_name,
        }
    )
    with configuration_path.open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2, sort_keys=True)

    wandb_logger = WandbLogger(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=(
            args.wandb_group
            if args.wandb_group is not None
            else (
                f"cw{sequence_task_count}-clonex-multi-head-"
                f"{args.exploration_strategy}"
            )
        ),
        run_name=run_directory.name,
        config=configuration,
        run_directory=run_directory,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        log_code=args.wandb_log_code,
        code_root=".",
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = SACAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        num_tasks=network_num_tasks,
        task_id_dim=task_id_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        polyak=args.polyak,
        target_entropy=target_entropy,
        initial_log_alpha=args.initial_log_alpha,
        device=args.device,
    )

    replay_buffer = ReplayBuffer(
        observation_dim=observation_dim,
        action_dim=action_dim,
        capacity=args.replay_size,
        seed=args.seed,
    )

    evaluation_envs: list[Any] = []
    deterministic_evaluators: list[SACEvaluator] = []
    stochastic_evaluators: list[SACEvaluator | None] = []
    final_evaluators: list[SACEvaluator] = []
    best_return_evaluators: dict[int, SACEvaluator] = {}

    for task_index, task_name in enumerate(tasks):
        deterministic_env = make_cw_env(
            task_name=task_name,
            seed=args.seed + 10_000 + task_index * 100,
            append_task_id=True,
        )
        validate_env_compatibility(
            deterministic_env, observation_dim=observation_dim,
            action_dim=action_dim, task_name=task_name,
        )
        evaluation_envs.append(deterministic_env)
        deterministic_evaluators.append(
            SACEvaluator(
                env=deterministic_env, agent=agent,
                config=EvaluationConfig(
                    num_episodes=args.det_eval_episodes,
                    max_episode_steps=args.max_episode_steps,
                    seed=args.seed + 10_000 + task_index * 100,
                ),
            )
        )

        if args.stoch_eval_episodes > 0:
            stochastic_env = make_cw_env(
                task_name=task_name,
                seed=args.seed + 20_000 + task_index * 100,
                append_task_id=True,
            )
            validate_env_compatibility(
                stochastic_env, observation_dim=observation_dim,
                action_dim=action_dim, task_name=task_name,
            )
            evaluation_envs.append(stochastic_env)
            stochastic_evaluators.append(
                SACEvaluator(
                    env=stochastic_env, agent=agent,
                    config=EvaluationConfig(
                        num_episodes=args.stoch_eval_episodes,
                        max_episode_steps=args.max_episode_steps,
                        seed=args.seed + 20_000 + task_index * 100,
                    ),
                )
            )
        else:
            stochastic_evaluators.append(None)

        final_env = make_cw_env(
            task_name=task_name,
            seed=args.seed + 40_000 + task_index * 100,
            append_task_id=True,
        )
        validate_env_compatibility(
            final_env, observation_dim=observation_dim,
            action_dim=action_dim, task_name=task_name,
        )
        evaluation_envs.append(final_env)
        final_evaluators.append(
            SACEvaluator(
                env=final_env, agent=agent,
                config=EvaluationConfig(
                    num_episodes=args.final_eval_episodes,
                    max_episode_steps=args.max_episode_steps,
                    seed=args.seed + 40_000 + task_index * 100,
                ),
            )
        )

        if task_index > 0:
            best_return_env = make_cw_env(
                task_name=task_name,
                seed=args.seed + 30_000 + task_index * 100,
                append_task_id=True,
            )
            validate_env_compatibility(
                best_return_env, observation_dim=observation_dim,
                action_dim=action_dim, task_name=task_name,
            )
            evaluation_envs.append(best_return_env)
            best_return_evaluators[task_index] = SACEvaluator(
                env=best_return_env, agent=agent,
                config=EvaluationConfig(
                    num_episodes=args.best_return_eval_episodes,
                    max_episode_steps=args.max_episode_steps,
                    seed=args.seed + 30_000 + task_index * 100,
                ),
            )

    experiment_start = time.perf_counter()
    experiment_started_at = (
        datetime.now().astimezone().isoformat()
    )
    cumulative_gradient_updates = 0
    evaluation_index = 0
    start_task_index = 0
    current_training_env: Any | None = None
    evaluation_history: list[dict[str, Any]] = []
    exit_code = 0

    if args.resume_run_dir is not None:
        previous_run_directory = Path(
            args.resume_run_dir
        ).expanduser()
        previous_resume_path = (
            previous_run_directory
            / "checkpoints"
            / "resume_state.pt"
        )
        resume_state = load_resume_state(
            previous_resume_path
        )

        if int(resume_state["steps_per_task"]) != args.steps_per_task:
            raise ValueError(
                "Resume run and current run use different "
                "steps_per_task values."
            )
        if int(resume_state["eval_every"]) != args.eval_every:
            raise ValueError(
                "Resume run and current run use different "
                "eval_every values."
            )
        if int(resume_state["seed"]) != args.seed:
            raise ValueError(
                "Resume run and current run use different seeds."
            )
        if (
            str(resume_state["exploration_strategy"])
            != args.exploration_strategy
        ):
            raise ValueError(
                "Resume run and current run use different "
                "exploration strategies."
            )

        completed_task_count = int(
            resume_state["completed_task_count"]
        )
        if completed_task_count >= sequence_task_count:
            raise ValueError(
                "The resume run has already completed at least as many "
                "tasks as requested by --sequence-task-count."
            )

        previous_tasks = list(resume_state["tasks"])
        if previous_tasks != tasks[:completed_task_count]:
            raise ValueError(
                "Resume task history does not match the requested "
                "CW10 prefix."
            )

        agent.load_state_dict(
            resume_state["agent_state_dict"]
        )

        if (
            not args.reset_optimizer_on_task_change
            and "optimizer_state_dict" in resume_state
        ):
            agent.optimizer.load_state_dict(
                resume_state["optimizer_state_dict"]
            )
        else:
            reset_optimizer_state(agent)

        start_task_index = completed_task_count
        cumulative_gradient_updates = int(
            resume_state["cumulative_gradient_updates"]
        )
        evaluation_index = int(
            resume_state["evaluation_index"]
        )
        evaluation_history = list(
            resume_state["evaluation_history"]
        )

        for filename in (
            "evaluations.csv",
            "task_summaries.csv",
            "best_return_selection.csv",
        ):
            _copy_if_present(
                previous_run_directory,
                run_directory,
                filename,
            )

    sequence_label = f"CW{sequence_task_count}"

    print("=" * 76)
    print(f"{sequence_label} ClonEx-style multi-head SAC")
    print("=" * 76)
    print(
        f"Sequence tasks trained   : "
        f"{sequence_task_count}"
    )
    print(
        f"Network task heads       : "
        f"{network_num_tasks}"
    )
    print(f"Steps per task           : {args.steps_per_task:,}")
    print(
        "Total environment steps  : "
        f"{args.steps_per_task * sequence_task_count:,}"
    )
    print(f"Observation dim          : {observation_dim}")
    print(f"Physical observation dim : {physical_observation_dim}")
    print(f"Task ID dim              : {task_id_dim}")
    print(f"Exploration strategy     : {args.exploration_strategy}")
    print(f"Formal deterministic eps : {args.det_eval_episodes}")
    print(f"Stochastic diagnostic eps: {args.stoch_eval_episodes}")
    print(f"Final deterministic eps  : {args.final_eval_episodes}")
    print(
        "Best-return eval episodes: "
        f"{args.best_return_eval_episodes}"
    )
    print(f"Initial exploration steps: {args.start_steps:,}")
    print(f"Reset replay buffer      : {args.reset_buffer_on_task_change}")
    print(f"Reset optimizer state    : {args.reset_optimizer_on_task_change}")
    print(f"W&B mode                 : {args.wandb_mode}")
    print(f"Output directory         : {run_directory}")
    print(f"Start task index         : {start_task_index}")
    if args.resume_run_dir is not None:
        print(f"Resumed from             : {args.resume_run_dir}")
    if wandb_logger.run_url is not None:
        print(f"W&B run                  : {wandb_logger.run_url}")

    try:
        for task_index in range(
            start_task_index,
            sequence_task_count,
        ):
            task_name = tasks[task_index]
            if task_index > 0 and args.reset_buffer_on_task_change:
                replay_buffer.clear()

            if task_index > 0 and args.reset_optimizer_on_task_change:
                reset_optimizer_state(agent)

            current_training_env = make_cw_env(
                task_name=task_name,
                seed=args.seed + task_index,
                append_task_id=True,
            )
            validate_env_compatibility(
                current_training_env,
                observation_dim=observation_dim,
                action_dim=action_dim,
                task_name=task_name,
            )

            print("\n" + "-" * 76)
            print(
                f"Task {task_index + 1:02d}/"
                f"{sequence_task_count}: "
                f"{task_name}"
            )
            print("-" * 76, flush=True)

            selected_exploration_head: int | None = None
            selected_exploration_return: float | None = None

            if (
                args.exploration_strategy == "best-return"
                and task_index > 0
            ):
                (
                    selected_exploration_head,
                    candidate_results,
                ) = select_best_return_head(
                    evaluator=best_return_evaluators[task_index],
                    current_task_index=task_index,
                    tasks=tasks,
                    expected_episodes=(
                        args.best_return_eval_episodes
                    ),
                )

                selected_exploration_return = float(
                    candidate_results[
                        selected_exploration_head
                    ].mean_return
                )

                print(
                    "Best-return candidate heads "
                    f"on {task_name}:"
                )
                for candidate_head_index in range(task_index):
                    candidate_result = candidate_results[
                        candidate_head_index
                    ]
                    is_selected = (
                        candidate_head_index
                        == selected_exploration_head
                    )
                    marker = "*" if is_selected else " "
                    print(
                        f" {marker} head {candidate_head_index}: "
                        f"{tasks[candidate_head_index]} | "
                        f"mean return "
                        f"{candidate_result.mean_return:.6f} | "
                        f"success "
                        f"{candidate_result.success_rate:.3f}",
                        flush=True,
                    )

                    append_csv_row(
                        best_return_selection_path,
                        BEST_RETURN_SELECTION_FIELDS,
                        {
                            "current_task_index": task_index,
                            "current_task_name": task_name,
                            "candidate_head_index": (
                                candidate_head_index
                            ),
                            "candidate_source_task": (
                                tasks[candidate_head_index]
                            ),
                            "mean_return": (
                                candidate_result.mean_return
                            ),
                            "success_rate": (
                                candidate_result.success_rate
                            ),
                            "mean_episode_length": (
                                candidate_result.mean_episode_length
                            ),
                            "selected": int(is_selected),
                        },
                    )

                print(
                    "Selected exploration head: "
                    f"{selected_exploration_head} "
                    f"({tasks[selected_exploration_head]})",
                    flush=True,
                )
            else:
                print(
                    "Initial exploration policy: uniform random",
                    flush=True,
                )

            trainer = SACTrainer(
                env=current_training_env,
                agent=agent,
                replay_buffer=replay_buffer,
                config=SACTrainerConfig(
                    total_steps=args.steps_per_task,
                    batch_size=args.batch_size,
                    start_steps=args.start_steps,
                    exploration_head_index=(
                        selected_exploration_head
                    ),
                    update_after=args.update_after,
                    update_every=args.update_every,
                    max_episode_steps=args.max_episode_steps,
                    callback_every_steps=args.eval_every,
                    seed=args.seed + task_index,
                    reseed_global_rng=False,
                ),
            )

            updates_before_task = cumulative_gradient_updates

            def step_callback(
                completed_task_steps: int,
                task_gradient_updates: int,
                metrics: dict[str, float] | None,
                *,
                _task_index: int = task_index,
                _task_name: str = task_name,
            ) -> None:
                nonlocal evaluation_index

                global_step = (
                    _task_index * args.steps_per_task
                    + completed_task_steps
                )
                should_evaluate = (
                    global_step % args.eval_every == 0
                    or completed_task_steps == args.steps_per_task
                )
                if not should_evaluate:
                    return

                evaluation_index += 1
                elapsed_seconds = (
                    time.perf_counter() - experiment_start
                )
                global_updates = (
                    updates_before_task + task_gradient_updates
                )

                deterministic_result = deterministic_evaluators[
                    _task_index
                ].evaluate(deterministic=True)
                validate_evaluation_result(
                    deterministic_result,
                    expected_episodes=args.det_eval_episodes,
                    label=f"deterministic/{_task_name}",
                )
                stochastic_evaluator = stochastic_evaluators[_task_index]
                stochastic_result = (
                    stochastic_evaluator.evaluate(deterministic=False)
                    if stochastic_evaluator is not None else None
                )
                if stochastic_result is not None:
                    validate_evaluation_result(
                        stochastic_result,
                        expected_episodes=args.stoch_eval_episodes,
                        label=f"stochastic/{_task_name}",
                    )
                deterministic_success = {
                    _task_name: float(deterministic_result.success_rate)
                }
                deterministic_return = {
                    _task_name: float(deterministic_result.mean_return)
                }
                stochastic_success = (
                    {_task_name: float(stochastic_result.success_rate)}
                    if stochastic_result is not None else {}
                )
                stochastic_return = (
                    {_task_name: float(stochastic_result.mean_return)}
                    if stochastic_result is not None else {}
                )
                append_csv_row(
                    evaluations_path, EVALUATION_FIELDS,
                    {
                        "evaluation_index": evaluation_index,
                        "global_step": global_step,
                        "gradient_updates": global_updates,
                        "active_task_index": _task_index,
                        "active_task_name": _task_name,
                        "active_task_step": completed_task_steps,
                        "evaluation_task_index": _task_index,
                        "evaluation_task_name": _task_name,
                        "deterministic_average_return": deterministic_result.mean_return,
                        "deterministic_success_rate": deterministic_result.success_rate,
                        "deterministic_average_episode_length": deterministic_result.mean_episode_length,
                        "stochastic_average_return": "" if stochastic_result is None else stochastic_result.mean_return,
                        "stochastic_success_rate": "" if stochastic_result is None else stochastic_result.success_rate,
                        "stochastic_average_episode_length": "" if stochastic_result is None else stochastic_result.mean_episode_length,
                        "actor_loss": metric_or_empty(metrics, "actor_loss"),
                        "q1_loss": metric_or_empty(metrics, "q1_loss"),
                        "q2_loss": metric_or_empty(metrics, "q2_loss"),
                        "alpha_loss": metric_or_empty(metrics, "alpha_loss"),
                        "alpha": metric_or_empty(metrics, "alpha"),
                        "q1_mean": metric_or_empty(metrics, "q1_mean"),
                        "q2_mean": metric_or_empty(metrics, "q2_mean"),
                        "q_target_mean": metric_or_empty(metrics, "q_target_mean"),
                        "log_prob_mean": metric_or_empty(metrics, "log_prob_mean"),
                        "elapsed_seconds": elapsed_seconds,
                    },
                )
                evaluation_history.append(
                    {
                        "evaluation_index": evaluation_index,
                        "global_step": global_step,
                        "active_task_index": _task_index,
                        "active_task_name": _task_name,
                        "active_task_step": completed_task_steps,
                        "deterministic_success": deterministic_success,
                        "deterministic_return": deterministic_return,
                        "stochastic_success": stochastic_success,
                        "stochastic_return": stochastic_return,
                    }
                )

                wandb_logger.log_evaluation(
                    global_step=global_step,
                    gradient_updates=global_updates,
                    evaluation_index=evaluation_index,
                    active_task_index=_task_index,
                    active_task_name=_task_name,
                    active_task_step=completed_task_steps,
                    training_metrics=metrics,
                    stochastic_success=stochastic_success,
                    deterministic_success=deterministic_success,
                    stochastic_return=stochastic_return,
                    deterministic_return=deterministic_return,
                    elapsed_seconds=elapsed_seconds,
                )

                print(
                    f"Evaluation {evaluation_index:03d} | "
                    f"task {_task_index + 1:02d}/"
                    f"{sequence_task_count} {_task_name} | "
                    f"task step {completed_task_steps:,}/"
                    f"{args.steps_per_task:,} | "
                    f"return {deterministic_result.mean_return:.3f} | "
                    f"success {deterministic_result.success_rate:.3f}",
                    flush=True,
                )

            task_summary = trainer.train(
                step_callback=step_callback
            )
            cumulative_gradient_updates += (
                task_summary.gradient_updates
            )

            global_step_end = (
                task_index + 1
            ) * args.steps_per_task
            elapsed_seconds = (
                time.perf_counter() - experiment_start
            )
            final_alpha = agent.alpha_value_for_task(task_index)

            append_csv_row(
                task_summaries_path,
                TASK_SUMMARY_FIELDS,
                {
                    "task_index": task_index,
                    "task_name": task_name,
                    "exploration_strategy": (
                        "random"
                        if selected_exploration_head is None
                        else "best-return"
                    ),
                    "selected_exploration_head": (
                        ""
                        if selected_exploration_head is None
                        else selected_exploration_head
                    ),
                    "selected_exploration_source_task": (
                        ""
                        if selected_exploration_head is None
                        else tasks[selected_exploration_head]
                    ),
                    "selected_exploration_mean_return": (
                        ""
                        if selected_exploration_return is None
                        else selected_exploration_return
                    ),
                    "global_step_end": global_step_end,
                    "task_gradient_updates": (
                        task_summary.gradient_updates
                    ),
                    "cumulative_gradient_updates": (
                        cumulative_gradient_updates
                    ),
                    "completed_training_episodes": (
                        task_summary.completed_episodes
                    ),
                    "mean_training_return": (
                        task_summary.mean_episode_return
                    ),
                    "final_alpha": final_alpha,
                    "replay_buffer_size": len(replay_buffer),
                    "elapsed_seconds": elapsed_seconds,
                },
            )

            wandb_logger.log_task_summary(
                global_step=global_step_end,
                task_index=task_index,
                task_name=task_name,
                task_gradient_updates=(
                    task_summary.gradient_updates
                ),
                cumulative_gradient_updates=(
                    cumulative_gradient_updates
                ),
                completed_training_episodes=(
                    task_summary.completed_episodes
                ),
                mean_training_return=(
                    task_summary.mean_episode_return
                ),
                final_alpha=final_alpha,
                replay_buffer_size=len(replay_buffer),
                elapsed_seconds=elapsed_seconds,
            )

            save_sac_checkpoint(
                agent=agent,
                path=(
                    checkpoints_directory
                    / f"task_{task_index + 1:02d}_{task_name}.pt"
                ),
                environment_step=global_step_end,
                metadata={
                    "method": method_name,
                    "task_index": task_index,
                    "task_name": task_name,
                    "seed": args.seed,
                    "exploration_strategy": (
                        "random"
                        if selected_exploration_head is None
                        else "best-return"
                    ),
                    "selected_exploration_head": (
                        selected_exploration_head
                    ),
                    "selected_exploration_source_task": (
                        None
                        if selected_exploration_head is None
                        else tasks[selected_exploration_head]
                    ),
                    "selected_exploration_mean_return": (
                        selected_exploration_return
                    ),
                    "all_alpha_values": (
                        agent.all_alpha_values()
                    ),
                },
            )

            save_resume_state(
                path=resume_checkpoint_path,
                agent=agent,
                completed_task_count=task_index + 1,
                cumulative_gradient_updates=(
                    cumulative_gradient_updates
                ),
                evaluation_index=evaluation_index,
                evaluation_history=evaluation_history,
                tasks=tasks[:task_index + 1],
                steps_per_task=args.steps_per_task,
                eval_every=args.eval_every,
                seed=args.seed,
                exploration_strategy=args.exploration_strategy,
                run_directory=run_directory,
            )

            with resume_manifest_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "resume_checkpoint": str(
                            resume_checkpoint_path
                        ),
                        "completed_task_count": task_index + 1,
                        "next_task_index": task_index + 1,
                        "next_task_name": (
                            tasks[task_index + 1]
                            if task_index + 1 < sequence_task_count
                            else None
                        ),
                        "source_run_directory": str(run_directory),
                    },
                    file,
                    indent=2,
                    sort_keys=True,
                )

            close_env(current_training_env)
            current_training_env = None

        final_checkpoint = checkpoints_directory / "final.pt"
        total_steps = (
            args.steps_per_task * sequence_task_count
        )

        save_sac_checkpoint(
            agent=agent,
            path=final_checkpoint,
            environment_step=total_steps,
            metadata={
                "method": method_name,
                "tasks": tasks,
                "seed": args.seed,
                "all_alpha_values": agent.all_alpha_values(),
            },
        )

        print("\n" + "-" * 76)
        print("Final all-task deterministic evaluation")
        print("-" * 76)

        final_success_per_task: dict[str, float] = {}
        final_return_per_task: dict[str, float] = {}

        for final_task_index, final_task_name in enumerate(tasks):
            final_result = final_evaluators[
                final_task_index
            ].evaluate(deterministic=True)

            validate_evaluation_result(
                final_result,
                expected_episodes=args.final_eval_episodes,
                label=f"final/{final_task_name}",
            )

            final_success_per_task[final_task_name] = float(
                final_result.success_rate
            )
            final_return_per_task[final_task_name] = float(
                final_result.mean_return
            )

            append_csv_row(
                final_evaluation_path,
                FINAL_EVALUATION_FIELDS,
                {
                    "task_index": final_task_index,
                    "task_name": final_task_name,
                    "mean_return": final_result.mean_return,
                    "success_rate": final_result.success_rate,
                    "mean_episode_length": (
                        final_result.mean_episode_length
                    ),
                    "num_episodes": final_result.num_episodes,
                },
            )

            print(
                f"{final_task_name:<24} | "
                f"return {final_result.mean_return:>10.3f} | "
                f"success {final_result.success_rate:.3f}",
                flush=True,
            )

        baseline_path = find_matching_baseline_curves(
            requested_path=args.baseline_curves,
            tasks=tasks,
            steps_per_task=args.steps_per_task,
            eval_every=args.eval_every,
            seed=args.seed,
        )

        final_metrics = compute_final_metrics(
            evaluation_history=evaluation_history,
            final_success_per_task=final_success_per_task,
            final_return_per_task=final_return_per_task,
            tasks=tasks,
            tail_size=args.final_metric_tail_size,
            baseline_path=baseline_path,
            steps_per_task=args.steps_per_task,
            eval_every=args.eval_every,
        )
        metrics_json_path, per_task_metrics_path = (
            save_final_metrics(
                metrics_directory=metrics_directory,
                tasks=tasks,
                final_metrics=final_metrics,
            )
        )

        plot_paths = plot_learning_curves(
            evaluation_history=evaluation_history,
            tasks=tasks,
            plots_directory=plots_directory,
        )

        wandb_logger.log_final_metrics(
            global_step=total_steps,
            average_performance=(
                final_metrics["average_performance"]
            ),
            average_forgetting=(
                final_metrics["average_forgetting"]
            ),
            final_per_task=final_metrics["final_per_task"],
            end_of_task_per_task=(
                final_metrics["end_of_task_per_task"]
            ),
            forgetting_per_task=(
                final_metrics["forgetting_per_task"]
            ),
        )

        elapsed_seconds = (
            time.perf_counter() - experiment_start
        )
        experiment_completed_at = (
            datetime.now().astimezone().isoformat()
        )

        summary = {
            "run_name": run_directory.name,
            "started_at": experiment_started_at,
            "completed_at": experiment_completed_at,
            "duration_hhmmss": format_duration(
                elapsed_seconds
            ),
            "method": method_name,
            "exploration_strategy": args.exploration_strategy,
            "seed": args.seed,
            "tasks": tasks,
            "steps_per_task": args.steps_per_task,
            "total_steps": total_steps,
            "gradient_updates": cumulative_gradient_updates,
            "elapsed_seconds": elapsed_seconds,
            "average_performance": (
                final_metrics["average_performance"]
            ),
            "average_success_rate": (
                final_metrics["average_success_rate"]
            ),
            "average_forgetting": (
                final_metrics["average_forgetting"]
            ),
            "average_final_return": (
                final_metrics["average_final_return"]
            ),
            "forward_transfer": (
                final_metrics["forward_transfer"]
            ),
            "raw_forward_transfer": (
                final_metrics["raw_forward_transfer"]
            ),
            "forward_transfer_status": (
                final_metrics["forward_transfer_status"]
            ),
            "baseline_curves_path": (
                final_metrics["baseline_curves_path"]
            ),
            "all_alpha_values": agent.all_alpha_values(),
            "evaluations_csv": str(evaluations_path),
            "task_summaries_csv": str(task_summaries_path),
            "final_evaluation_csv": str(final_evaluation_path),
            "best_return_selection_csv": (
                str(best_return_selection_path)
                if best_return_selection_path.is_file()
                else None
            ),
            "continual_metrics_json": str(metrics_json_path),
            "per_task_metrics_csv": str(per_task_metrics_path),
            "final_checkpoint": str(final_checkpoint),
            "resume_checkpoint": str(resume_checkpoint_path),
            "learning_curve_plots": {
                name: str(path)
                for name, path in plot_paths.items()
            },
            "summary_report": str(
                experiment_report_path
            ),
            "wandb_mode": args.wandb_mode,
            "wandb_run_url": wandb_logger.run_url,
        }
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)

        save_experiment_report(
            path=experiment_report_path,
            run_name=run_directory.name,
            started_at=experiment_started_at,
            completed_at=experiment_completed_at,
            elapsed_seconds=elapsed_seconds,
            configuration=configuration,
            final_metrics=final_metrics,
            plot_paths=plot_paths,
            run_directory=run_directory,
            summary_path=summary_path,
            metrics_json_path=metrics_json_path,
            per_task_metrics_path=per_task_metrics_path,
            final_checkpoint=final_checkpoint,
            resume_checkpoint_path=resume_checkpoint_path,
            wandb_run_url=wandb_logger.run_url,
        )

        artifact_files = [
            configuration_path,
            evaluations_path,
            task_summaries_path,
            final_evaluation_path,
            metrics_json_path,
            per_task_metrics_path,
            summary_path,
            experiment_report_path,
            resume_manifest_path,
            *plot_paths.values(),
        ]
        if best_return_selection_path.is_file():
            artifact_files.append(
                best_return_selection_path
            )

        wandb_logger.log_results_artifact(
            name=f"{run_directory.name}-results",
            files=artifact_files,
            metadata={
                "method": method_name,
                "seed": args.seed,
                "steps_per_task": args.steps_per_task,
                "exploration_strategy": args.exploration_strategy,
            },
        )

        if args.wandb_upload_final_checkpoint:
            wandb_logger.log_model_artifact(
                name=f"{run_directory.name}-model",
                checkpoint_path=final_checkpoint,
                metadata={
                    "method": method_name,
                    "seed": args.seed,
                    "total_steps": total_steps,
                },
            )

        print("\n" + "=" * 76)
        print(f"{sequence_label} run completed")
        print("=" * 76)
        print(
            "Average performance : "
            f"{final_metrics['average_performance']:.6f}"
        )
        print(
            "Average success rate: "
            f"{final_metrics['average_success_rate']:.6f}"
        )
        print(
            "Average forgetting  : "
            f"{final_metrics['average_forgetting']:.6f}"
        )
        print(
            "Average final return: "
            f"{final_metrics['average_final_return']:.6f}"
        )
        if final_metrics["forward_transfer"] is None:
            print(
                "Forward transfer    : unavailable "
                f"({final_metrics['forward_transfer_status']})"
            )
        else:
            print(
                "Forward transfer    : "
                f"{final_metrics['forward_transfer']:.6f}"
            )
            print(
                "Raw forward transfer: "
                f"{final_metrics['raw_forward_transfer']:.6f}"
            )
        print(f"Evaluations CSV    : {evaluations_path}")
        print(f"Task summaries     : {task_summaries_path}")
        print(f"Final evaluation   : {final_evaluation_path}")
        if best_return_selection_path.is_file():
            print(
                "Best-return choices : "
                f"{best_return_selection_path}"
            )
        print(f"Continual metrics  : {metrics_json_path}")
        print(f"Per-task metrics   : {per_task_metrics_path}")
        print(f"Final checkpoint   : {final_checkpoint}")
        print(f"Resume checkpoint  : {resume_checkpoint_path}")
        print(f"Summary report     : {experiment_report_path}")
        print(f"Learning curves    : {plots_directory}")
        print(
            "Elapsed time        : "
            f"{format_duration(elapsed_seconds)} "
            f"({elapsed_seconds:.2f} seconds)"
        )
        if wandb_logger.run_url is not None:
            print(f"W&B run            : {wandb_logger.run_url}")

    except BaseException:
        exit_code = 1
        raise

    finally:
        close_env(current_training_env)
        for env in evaluation_envs:
            close_env(env)
        wandb_logger.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
