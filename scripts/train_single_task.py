"""Train SAC from scratch on one MetaWorld task.

This script connects the complete single-task baseline:

1. a real MetaWorld training environment;
2. one standard online SAC replay buffer;
3. SACAgent;
4. SACTrainer;
5. stochastic and deterministic evaluation;
6. CSV logging;
7. model and optimizer checkpoints.

It deliberately contains no continual-learning task switching, no
task-specific policy memory, and no transfer logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from crl_cw.agents import ReplayBuffer, SACAgent
from crl_cw.envs import make_cw_env
from crl_cw.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    SACEvaluator,
)
from crl_cw.training import (
    SACTrainer,
    SACTrainerConfig,
)
from crl_cw.utils import save_sac_checkpoint
from crl_cw.utils.wandb_logger import WandbLogger


METHOD_NAME = "single_task_sac_from_scratch"

CSV_FIELDS = [
    "environment_step",
    "gradient_updates",
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the single-task SAC baseline on one MetaWorld task."
        )
    )

    parser.add_argument(
        "--task",
        type=str,
        default="hammer-v1",
        help="MetaWorld task name.",
    )

    parser.add_argument(
        "--total-steps",
        type=int,
        default=1_000_000,
        help="Total number of training-environment interactions.",
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=20_000,
        help="Number of training steps between evaluations.",
    )

    parser.add_argument(
        "--det-eval-episodes",
        type=int,
        default=1,
        help=(
            "Number of deterministic evaluation episodes. "
            "Deterministic evaluation uses the Gaussian mean action."
        ),
    )

    parser.add_argument(
        "--stoch-eval-episodes",
        type=int,
        default=0,
        help=(
            "Number of stochastic evaluation episodes. "
            "Stochastic evaluation samples actions from the policy."
        ),
    )

    parser.add_argument(
        "--replay-size",
        type=int,
        default=1_000_000,
        help="Capacity of the online SAC replay buffer.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of replay transitions in one gradient update.",
    )

    parser.add_argument(
        "--start-steps",
        type=int,
        default=10_000,
        help="Initial steps using uniformly random actions.",
    )

    parser.add_argument(
        "--update-after",
        type=int,
        default=1_000,
        help="Environment step after which gradient updates may begin.",
    )

    parser.add_argument(
        "--update-every",
        type=int,
        default=50,
        help=(
            "Number of steps between update periods. "
            "The same number of updates is performed in each period."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Reward discount factor.",
    )

    parser.add_argument(
        "--polyak",
        type=float,
        default=0.995,
        help="Target-network retention coefficient.",
    )

    parser.add_argument(
        "--target-output-std",
        type=float,
        default=None,
        help=(
            "Optional desired action-distribution standard deviation. "
            "When omitted, target entropy defaults to -action_dim."
        ),
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Maximum number of steps in one episode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Training seed.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, such as cpu or mps.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/single_task",
        help="Parent directory for experiment outputs.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional run-directory name. "
            "A timestamped name is generated when omitted."
        ),
    )

    parser.add_argument(
        "--initial-log-alpha",
        type=float,
        default=1.0,
        help="Initial value of SAC log alpha.",
    )

    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("disabled", "online", "offline"),
        default="disabled",
        help=(
            "W&B mode: disabled, online, or offline. "
            "Offline runs can be synced later."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="crl-cw",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="cw10-single-task-baselines",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--wandb-notes",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb-log-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--wandb-upload-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Upload final.pt as a W&B model artifact. "
            "Disabled by default to avoid unnecessary storage."
        ),
    )

    args = parser.parse_args()

    positive_integer_fields = (
        "total_steps",
        "eval_every",
        "det_eval_episodes",
        "replay_size",
        "batch_size",
        "update_every",
        "max_episode_steps",
    )

    for field_name in positive_integer_fields:
        value = getattr(
            args,
            field_name,
        )

        if value <= 0:
            command_name = field_name.replace(
                "_",
                "-",
            )

            parser.error(
                f"--{command_name} must be positive."
            )

    if args.stoch_eval_episodes < 0:
        parser.error("--stoch-eval-episodes must be non-negative.")

    if args.start_steps < 0:
        parser.error(
            "--start-steps must be non-negative."
        )

    if args.update_after < 0:
        parser.error(
            "--update-after must be non-negative."
        )

    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0.0
    ):
        parser.error(
            "--learning-rate must be finite and positive."
        )

    if (
        not math.isfinite(args.gamma)
        or not 0.0 <= args.gamma <= 1.0
    ):
        parser.error(
            "--gamma must be finite and in [0, 1]."
        )

    if (
        not math.isfinite(args.polyak)
        or not 0.0 <= args.polyak <= 1.0
    ):
        parser.error(
            "--polyak must be finite and in [0, 1]."
        )

    if args.target_output_std is not None:
        if (
            not math.isfinite(
                args.target_output_std
            )
            or args.target_output_std <= 0.0
        ):
            parser.error(
                "--target-output-std must be finite and positive."
            )

    if not math.isfinite(args.initial_log_alpha):
        parser.error(
            "--initial-log-alpha must be finite."
        )

    return args


def compute_target_entropy(
    *,
    action_dim: int,
    target_output_std: float | None,
) -> float | None:
    """Convert desired output standard deviation to target entropy.

    When target_output_std is omitted, returning None allows SACAgent
    to use its default:

        target_entropy = -action_dim

    Otherwise, use the same Gaussian entropy relationship employed by
    the Continual World SAC implementation:

        one_dimensional_entropy =
            log(std * sqrt(2 * pi * e))

        target_entropy =
            action_dim * one_dimensional_entropy
    """
    if target_output_std is None:
        return None

    one_dimensional_entropy = math.log(
        target_output_std
        * math.sqrt(
            2.0
            * math.pi
            * math.e
        )
    )

    return float(
        action_dim
        * one_dimensional_entropy
    )


def create_run_directory(
    args: argparse.Namespace,
) -> Path:
    """Create a new output directory for this experiment."""
    safe_task_name = (
        args.task
        .replace("/", "_")
        .replace("\\", "_")
    )

    if args.run_name is None:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        run_name = (
            f"{safe_task_name}"
            f"_seed{args.seed}"
            f"_{timestamp}"
        )
    else:
        run_name = (
            args.run_name
            .replace("/", "_")
            .replace("\\", "_")
        )

    run_directory = (
        Path(args.output_dir)
        .expanduser()
        / run_name
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        run_directory
        / "checkpoints"
    ).mkdir(
        parents=False,
        exist_ok=False,
    )

    return run_directory


def save_configuration(
    *,
    args: argparse.Namespace,
    run_directory: Path,
    target_entropy: float | None,
    observation_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    """Save and return the resolved experiment configuration."""
    configuration = dict(
        vars(args)
    )

    configuration.update(
        {
            "method": METHOD_NAME,
            "resolved_target_entropy": target_entropy,
            "run_directory": str(run_directory),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
        }
    )

    configuration_path = (
        run_directory
        / "config.json"
    )

    with configuration_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
            sort_keys=True,
        )

    return configuration


def validate_environment_spaces(
    env: Any,
) -> tuple[
    int,
    int,
    np.ndarray,
    np.ndarray,
]:
    """Read and validate flat observation and action spaces."""
    observation_space = getattr(
        env,
        "observation_space",
        None,
    )

    action_space = getattr(
        env,
        "action_space",
        None,
    )

    if observation_space is None:
        raise ValueError(
            "Training environment must expose observation_space."
        )

    if action_space is None:
        raise ValueError(
            "Training environment must expose action_space."
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

    if (
        not isinstance(
            observation_shape,
            tuple,
        )
        or len(observation_shape) != 1
    ):
        raise ValueError(
            "Observation space must be a flat vector."
        )

    if (
        not isinstance(
            action_shape,
            tuple,
        )
        or len(action_shape) != 1
    ):
        raise ValueError(
            "Action space must be a flat vector."
        )

    observation_dim = int(
        observation_shape[0]
    )

    action_dim = int(
        action_shape[0]
    )

    if observation_dim <= 0:
        raise ValueError(
            "Observation dimension must be positive."
        )

    if action_dim <= 0:
        raise ValueError(
            "Action dimension must be positive."
        )

    action_low = np.asarray(
        action_space.low,
        dtype=np.float32,
    )

    action_high = np.asarray(
        action_space.high,
        dtype=np.float32,
    )

    expected_action_shape = (
        action_dim,
    )

    if (
        action_low.shape
        != expected_action_shape
    ):
        raise ValueError(
            "Action lower bounds have incorrect shape: "
            f"expected {expected_action_shape}, "
            f"got {action_low.shape}."
        )

    if (
        action_high.shape
        != expected_action_shape
    ):
        raise ValueError(
            "Action upper bounds have incorrect shape: "
            f"expected {expected_action_shape}, "
            f"got {action_high.shape}."
        )

    if (
        not np.all(
            np.isfinite(action_low)
        )
        or not np.all(
            np.isfinite(action_high)
        )
    ):
        raise ValueError(
            "Action-space bounds must be finite."
        )

    if not np.all(
        action_low < action_high
    ):
        raise ValueError(
            "Every action lower bound must be less "
            "than its upper bound."
        )

    return (
        observation_dim,
        action_dim,
        action_low,
        action_high,
    )


def validate_matching_environment(
    *,
    env: Any,
    observation_dim: int,
    action_dim: int,
    name: str,
) -> None:
    """Ensure an evaluation environment matches the training spaces."""
    expected_observation_shape = (
        observation_dim,
    )

    expected_action_shape = (
        action_dim,
    )

    observation_shape = getattr(
        env.observation_space,
        "shape",
        None,
    )

    action_shape = getattr(
        env.action_space,
        "shape",
        None,
    )

    if (
        observation_shape
        != expected_observation_shape
    ):
        raise ValueError(
            f"{name} observation shape does not match training: "
            f"expected {expected_observation_shape}, "
            f"got {observation_shape}."
        )

    if (
        action_shape
        != expected_action_shape
    ):
        raise ValueError(
            f"{name} action shape does not match training: "
            f"expected {expected_action_shape}, "
            f"got {action_shape}."
        )


def append_csv_row(
    *,
    path: Path,
    row: dict[str, Any],
) -> None:
    """Append one evaluation record to the CSV file."""
    file_exists = path.is_file()

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            row
        )


def evaluation_fields(
    *,
    prefix: str,
    result: EvaluationResult,
) -> dict[str, float]:
    """Convert evaluation results to prefixed CSV fields."""
    return {
        f"{prefix}_average_return": (
            result.mean_return
        ),
        f"{prefix}_success_rate": (
            result.success_rate
        ),
        f"{prefix}_average_episode_length": (
            result.mean_episode_length
        ),
    }


def metric_or_empty(
    metrics: dict[str, float] | None,
    name: str,
) -> float | str:
    """Return one finite training metric or an empty CSV field."""
    if metrics is None:
        return ""

    if name not in metrics:
        return ""

    value = float(
        metrics[name]
    )

    if not math.isfinite(value):
        raise RuntimeError(
            f"Training metric {name!r} is non-finite: {value}."
        )

    return value


def validate_evaluation_result(
    *,
    result: EvaluationResult,
    expected_episodes: int,
    label: str,
) -> None:
    """Validate one evaluation result before saving it."""
    if (
        result.num_episodes
        != expected_episodes
    ):
        raise RuntimeError(
            f"{label} evaluation returned the wrong "
            "number of episodes: "
            f"expected {expected_episodes}, "
            f"got {result.num_episodes}."
        )

    if not math.isfinite(
        result.mean_return
    ):
        raise RuntimeError(
            f"{label} average return is non-finite."
        )

    if not (
        0.0
        <= result.success_rate
        <= 1.0
    ):
        raise RuntimeError(
            f"{label} success rate is outside [0, 1]."
        )

    if result.total_steps <= 0:
        raise RuntimeError(
            f"{label} evaluation performed no steps."
        )


def print_evaluation(
    *,
    completed_env_steps: int,
    gradient_updates: int,
    deterministic_result: EvaluationResult,
    stochastic_result: EvaluationResult | None,
    elapsed_seconds: float,
) -> None:
    """Print one evaluation report."""
    print(
        "\n"
        + "=" * 72
    )

    print(
        "Evaluation at environment step "
        f"{completed_env_steps:,}"
    )

    print(
        "=" * 72
    )

    print(
        "Gradient updates              : "
        f"{gradient_updates:,}"
    )

    print(
        "Deterministic episodes        : "
        f"{deterministic_result.num_episodes}"
    )

    print(
        "Deterministic average return  : "
        f"{deterministic_result.mean_return:.6f}"
    )

    print(
        "Deterministic success rate    : "
        f"{deterministic_result.success_rate:.6f}"
    )

    if stochastic_result is not None:
        print(
            "Stochastic episodes           : "
            f"{stochastic_result.num_episodes}"
        )
        print(
            "Stochastic average return     : "
            f"{stochastic_result.mean_return:.6f}"
        )
        print(
            "Stochastic success rate       : "
            f"{stochastic_result.success_rate:.6f}"
        )

    print(
        "Elapsed seconds               : "
        f"{elapsed_seconds:.2f}"
    )


def save_final_summary(
    *,
    run_directory: Path,
    task_name: str,
    seed: int,
    total_env_steps: int,
    gradient_updates: int,
    completed_episodes: int,
    mean_training_return: float,
    final_alpha: float,
    elapsed_seconds: float,
    last_evaluation: dict[str, Any] | None,
) -> None:
    """Save a compact final experiment summary."""
    summary = {
        "task_name": task_name,
        "seed": seed,
        "total_env_steps": total_env_steps,
        "gradient_updates": gradient_updates,
        "completed_training_episodes": completed_episodes,
        "mean_training_return": mean_training_return,
        "final_alpha": final_alpha,
        "elapsed_seconds": elapsed_seconds,
        "last_evaluation": last_evaluation,
    }

    summary_path = (
        run_directory
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )


def close_environment(
    env: Any | None,
) -> None:
    """Close an environment if it exposes close()."""
    if env is None:
        return

    close_method = getattr(
        env,
        "close",
        None,
    )

    if callable(
        close_method
    ):
        close_method()


def main() -> None:
    """Run one complete single-task SAC experiment."""
    args = parse_args()

    run_directory = create_run_directory(
        args
    )

    csv_path = (
        run_directory
        / "evaluations.csv"
    )

    checkpoint_directory = (
        run_directory
        / "checkpoints"
    )

    training_env: Any | None = None
    deterministic_eval_env: Any | None = None
    stochastic_eval_env: Any | None = None
    wandb_logger: WandbLogger | None = None
    exit_code = 0

    try:
        training_env = make_cw_env(
            task_name=args.task,
            seed=args.seed,
        )

        deterministic_eval_env = make_cw_env(
            task_name=args.task,
            seed=args.seed + 10_000,
        )

        if args.stoch_eval_episodes > 0:
            stochastic_eval_env = make_cw_env(
                task_name=args.task,
                seed=args.seed + 20_000,
            )

        (
            observation_dim,
            action_dim,
            action_low,
            action_high,
        ) = validate_environment_spaces(
            training_env
        )

        validate_matching_environment(
            env=deterministic_eval_env,
            observation_dim=observation_dim,
            action_dim=action_dim,
            name=(
                "Deterministic evaluation environment"
            ),
        )

        if stochastic_eval_env is not None:
            validate_matching_environment(
                env=stochastic_eval_env,
                observation_dim=observation_dim,
                action_dim=action_dim,
                name=(
                    "Stochastic evaluation environment"
                ),
            )

        target_entropy = compute_target_entropy(
            action_dim=action_dim,
            target_output_std=(
                args.target_output_std
            ),
        )

        configuration = save_configuration(
            args=args,
            run_directory=run_directory,
            target_entropy=target_entropy,
            observation_dim=observation_dim,
            action_dim=action_dim,
        )

        wandb_logger = WandbLogger(
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
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
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            polyak=args.polyak,
            target_entropy=target_entropy,
            initial_log_alpha=args.initial_log_alpha,
            device=args.device,
        )

        # This is the online SAC replay buffer.
        #
        # It stores transition data for the current single-task
        # training run. It is not the future task-policy memory.
        replay_buffer = ReplayBuffer(
            observation_dim=observation_dim,
            action_dim=action_dim,
            capacity=args.replay_size,
            seed=args.seed,
        )

        trainer = SACTrainer(
            env=training_env,
            agent=agent,
            replay_buffer=replay_buffer,
            config=SACTrainerConfig(
                total_steps=args.total_steps,
                batch_size=args.batch_size,
                start_steps=args.start_steps,
                update_after=args.update_after,
                update_every=args.update_every,
                max_episode_steps=(
                    args.max_episode_steps
                ),
                callback_every_steps=args.eval_every,
                seed=args.seed,
            ),
        )

        deterministic_evaluator = SACEvaluator(
            env=deterministic_eval_env,
            agent=agent,
            config=EvaluationConfig(
                num_episodes=(
                    args.det_eval_episodes
                ),
                max_episode_steps=(
                    args.max_episode_steps
                ),
                seed=args.seed + 10_000,
            ),
        )

        stochastic_evaluator = (
            SACEvaluator(
                env=stochastic_eval_env,
                agent=agent,
                config=EvaluationConfig(
                    num_episodes=(
                        args.stoch_eval_episodes
                    ),
                    max_episode_steps=(
                        args.max_episode_steps
                    ),
                    seed=args.seed + 20_000,
                ),
            )
            if stochastic_eval_env is not None
            else None
        )

        last_evaluation_row: dict[
            str,
            Any,
        ] | None = None
        evaluation_index = 0

        experiment_start_time = (
            time.perf_counter()
        )

        print(
            "=" * 72
        )

        print(
            "Single-task SAC experiment"
        )

        print(
            "=" * 72
        )

        print(
            f"Task                    : {args.task}"
        )

        print(
            f"Seed                    : {args.seed}"
        )

        print(
            f"Device                  : {args.device}"
        )

        print(
            "Observation dimension   : "
            f"{observation_dim}"
        )

        print(
            "Action dimension        : "
            f"{action_dim}"
        )

        print(
            "Total training steps    : "
            f"{args.total_steps:,}"
        )

        print(
            "Evaluation interval     : "
            f"{args.eval_every:,}"
        )

        print(
            "Deterministic episodes  : "
            f"{args.det_eval_episodes}"
        )

        print(
            "Stochastic episodes     : "
            f"{args.stoch_eval_episodes}"
        )

        print(
            "Replay capacity         : "
            f"{args.replay_size:,}"
        )

        print(
            "Batch size              : "
            f"{args.batch_size}"
        )

        print(
            "Start random steps      : "
            f"{args.start_steps:,}"
        )

        print(
            "Update after            : "
            f"{args.update_after:,}"
        )

        print(
            "Update every            : "
            f"{args.update_every}"
        )

        print(
            "Target entropy          : "
            f"{agent.target_entropy:.6f}"
        )

        print(
            f"Output directory        : {run_directory}"
        )

        print(
            "\nTraining started.",
            flush=True,
        )

        def step_callback(
            completed_env_steps: int,
            gradient_updates: int,
            metrics: dict[str, float] | None,
        ) -> None:
            """Evaluate, log, and checkpoint at scheduled steps."""
            nonlocal last_evaluation_row
            nonlocal evaluation_index

            should_evaluate = (
                completed_env_steps
                % args.eval_every
                == 0
                or completed_env_steps
                == args.total_steps
            )

            if not should_evaluate:
                return

            evaluation_index += 1

            deterministic_result = (
                deterministic_evaluator.evaluate(
                    deterministic=True
                )
            )

            validate_evaluation_result(
                result=deterministic_result,
                expected_episodes=(
                    args.det_eval_episodes
                ),
                label="Deterministic",
            )

            stochastic_result = (
                stochastic_evaluator.evaluate(
                    deterministic=False
                )
                if stochastic_evaluator is not None
                else None
            )

            if stochastic_result is not None:
                validate_evaluation_result(
                    result=stochastic_result,
                    expected_episodes=(
                        args.stoch_eval_episodes
                    ),
                    label="Stochastic",
                )

            elapsed_seconds = (
                time.perf_counter()
                - experiment_start_time
            )

            row: dict[str, Any] = {
                "environment_step": (
                    completed_env_steps
                ),
                "gradient_updates": (
                    gradient_updates
                ),
                **evaluation_fields(
                    prefix="deterministic",
                    result=deterministic_result,
                ),
                **(
                    evaluation_fields(
                        prefix="stochastic",
                        result=stochastic_result,
                    )
                    if stochastic_result is not None
                    else {
                        "stochastic_average_return": "",
                        "stochastic_success_rate": "",
                        "stochastic_average_episode_length": "",
                    }
                ),
                "actor_loss": metric_or_empty(
                    metrics,
                    "actor_loss",
                ),
                "q1_loss": metric_or_empty(
                    metrics,
                    "q1_loss",
                ),
                "q2_loss": metric_or_empty(
                    metrics,
                    "q2_loss",
                ),
                "alpha_loss": metric_or_empty(
                    metrics,
                    "alpha_loss",
                ),
                "alpha": metric_or_empty(
                    metrics,
                    "alpha",
                ),
                "q1_mean": metric_or_empty(
                    metrics,
                    "q1_mean",
                ),
                "q2_mean": metric_or_empty(
                    metrics,
                    "q2_mean",
                ),
                "q_target_mean": metric_or_empty(
                    metrics,
                    "q_target_mean",
                ),
                "log_prob_mean": metric_or_empty(
                    metrics,
                    "log_prob_mean",
                ),
                "elapsed_seconds": (
                    elapsed_seconds
                ),
            }

            append_csv_row(
                path=csv_path,
                row=row,
            )

            last_evaluation_row = dict(
                row
            )

            if wandb_logger is not None:
                wandb_logger.log_evaluation(
                    global_step=completed_env_steps,
                    gradient_updates=gradient_updates,
                    evaluation_index=evaluation_index,
                    active_task_index=0,
                    active_task_name=args.task,
                    active_task_step=completed_env_steps,
                    training_metrics=metrics,
                    stochastic_success=(
                        {args.task: stochastic_result.success_rate}
                        if stochastic_result is not None
                        else {}
                    ),
                    deterministic_success={
                        args.task: deterministic_result.success_rate
                    },
                    stochastic_return=(
                        {args.task: stochastic_result.mean_return}
                        if stochastic_result is not None
                        else {}
                    ),
                    deterministic_return={
                        args.task: deterministic_result.mean_return
                    },
                    elapsed_seconds=elapsed_seconds,
                )

            # latest.pt is overwritten after every evaluation.
            #
            # This saves model and optimizer state, but does not save
            # the online replay buffer or exact environment state.
            save_sac_checkpoint(
                agent=agent,
                path=(
                    checkpoint_directory
                    / "latest.pt"
                ),
                environment_step=(
                    completed_env_steps
                ),
                metadata={
                    "task_name": args.task,
                    "seed": args.seed,
                    "evaluation": row,
                },
            )

            print_evaluation(
                completed_env_steps=(
                    completed_env_steps
                ),
                gradient_updates=(
                    gradient_updates
                ),
                deterministic_result=(
                    deterministic_result
                ),
                stochastic_result=(
                    stochastic_result
                ),
                elapsed_seconds=(
                    elapsed_seconds
                ),
            )

        training_summary = trainer.train(
            step_callback=step_callback
        )

        total_elapsed_seconds = (
            time.perf_counter()
            - experiment_start_time
        )

        final_checkpoint_path = (
            checkpoint_directory
            / "final.pt"
        )

        save_sac_checkpoint(
            agent=agent,
            path=final_checkpoint_path,
            environment_step=(
                training_summary.total_env_steps
            ),
            metadata={
                "task_name": args.task,
                "seed": args.seed,
                "final_evaluation": (
                    last_evaluation_row
                ),
            },
        )

        if wandb_logger is not None:
            wandb_logger.log_task_summary(
                global_step=training_summary.total_env_steps,
                task_index=0,
                task_name=args.task,
                task_gradient_updates=training_summary.gradient_updates,
                cumulative_gradient_updates=training_summary.gradient_updates,
                completed_training_episodes=(
                    training_summary.completed_episodes
                ),
                mean_training_return=(
                    training_summary.mean_episode_return
                ),
                final_alpha=agent.alpha_value,
                replay_buffer_size=len(replay_buffer),
                elapsed_seconds=total_elapsed_seconds,
            )

        save_final_summary(
            run_directory=run_directory,
            task_name=args.task,
            seed=args.seed,
            total_env_steps=(
                training_summary.total_env_steps
            ),
            gradient_updates=(
                training_summary.gradient_updates
            ),
            completed_episodes=(
                training_summary.completed_episodes
            ),
            mean_training_return=(
                training_summary.mean_episode_return
            ),
            final_alpha=(
                agent.alpha_value
            ),
            elapsed_seconds=(
                total_elapsed_seconds
            ),
            last_evaluation=(
                last_evaluation_row
            ),
        )

        summary_path = run_directory / "summary.json"
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        summary.update(
            {
                "method": METHOD_NAME,
                "evaluations_csv": str(csv_path),
                "final_checkpoint": str(final_checkpoint_path),
                "wandb_mode": args.wandb_mode,
                "wandb_run_url": (
                    None
                    if wandb_logger is None
                    else wandb_logger.run_url
                ),
            }
        )
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)

        if wandb_logger is not None:
            wandb_logger.log_results_artifact(
                name=f"{run_directory.name}-results",
                files=[
                    run_directory / "config.json",
                    csv_path,
                    summary_path,
                ],
                metadata={
                    "method": METHOD_NAME,
                    "task_name": args.task,
                    "seed": args.seed,
                    "total_steps": training_summary.total_env_steps,
                },
            )

            if args.wandb_upload_final_checkpoint:
                wandb_logger.log_model_artifact(
                    name=f"{run_directory.name}-model",
                    checkpoint_path=final_checkpoint_path,
                    metadata={
                        "method": METHOD_NAME,
                        "task_name": args.task,
                        "seed": args.seed,
                        "total_steps": training_summary.total_env_steps,
                    },
                )

        print(
            "\n"
            + "=" * 72
        )

        print(
            "Training completed"
        )

        print(
            "=" * 72
        )

        print(
            "Environment steps       : "
            f"{training_summary.total_env_steps:,}"
        )

        print(
            "Gradient updates        : "
            f"{training_summary.gradient_updates:,}"
        )

        print(
            "Completed train episodes: "
            f"{training_summary.completed_episodes:,}"
        )

        print(
            "Mean training return    : "
            f"{training_summary.mean_episode_return:.6f}"
        )

        print(
            "Final alpha             : "
            f"{agent.alpha_value:.6f}"
        )

        print(
            "Elapsed seconds         : "
            f"{total_elapsed_seconds:.2f}"
        )

        print(
            f"Evaluation CSV          : {csv_path}"
        )

        print(
            f"Final checkpoint        : {final_checkpoint_path}"
        )

        print(
            "Summary JSON            : "
            f"{run_directory / 'summary.json'}"
        )

        if (
            wandb_logger is not None
            and wandb_logger.run_url is not None
        ):
            print(
                f"W&B run                : {wandb_logger.run_url}"
            )

    except BaseException:
        exit_code = 1
        raise

    finally:
        close_environment(
            training_env
        )

        close_environment(
            deterministic_eval_env
        )

        close_environment(
            stochastic_eval_env
        )

        if wandb_logger is not None:
            wandb_logger.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()