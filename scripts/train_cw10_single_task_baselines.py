"""Run independent SAC baselines for the fixed CW10 task sequence.

Each task is launched in a separate Python process through
``scripts/train_single_task.py``. This guarantees that every task starts with
new network parameters, new optimizer state, a new entropy parameter, and an
empty replay buffer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from crl_cw.envs import get_cw10_tasks
from crl_cw.evaluation.single_task_baselines import (
    aggregate_single_task_baselines,
)


METHOD_NAME = "cw10_independent_single_task_sac_baselines"

_MANIFEST_FIELDS = (
    "task_index",
    "task_name",
    "status",
    "run_name",
    "run_directory",
    "log_path",
    "return_code",
    "started_at_utc",
    "finished_at_utc",
    "elapsed_seconds",
    "error",
)


def parse_args() -> argparse.Namespace:
    """Parse batch, SAC, storage, and W&B arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train SAC independently from scratch on the fixed CW10 tasks."
        )
    )

    parser.add_argument("--steps-per-task", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--det-eval-episodes", type=int, default=3)
    parser.add_argument("--stoch-eval-episodes", type=int, default=0)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--target-output-std", type=float, default=0.089)
    parser.add_argument("--initial-log-alpha", type=float, default=1.0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=3,
        help=(
            "Use the first N tasks of the fixed CW10 order. The default "
            "is 3 for the current CW3 development experiments. Use 10 "
            "later for the complete CW10 baseline."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/cw10_single_task_baselines",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--aggregate-tail-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--keep-latest-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep checkpoints/latest.pt after a task completes. The final "
            "checkpoint is always retained."
        ),
    )

    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", type=str, default="crl-cw")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="cw10-single-task-baselines",
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
        "replay_size",
        "batch_size",
        "update_every",
        "max_episode_steps",
        "aggregate_tail_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")

    if args.stoch_eval_episodes < 0:
        parser.error("--stoch-eval-episodes must be non-negative.")

    if not 1 <= args.num_tasks <= 10:
        parser.error("--num-tasks must be between 1 and 10.")
    if args.start_steps < 0:
        parser.error("--start-steps must be non-negative.")
    if args.update_after < 0:
        parser.error("--update-after must be non-negative.")
    if args.steps_per_task % args.eval_every != 0:
        parser.error(
            "--steps-per-task must be divisible by --eval-every so the "
            "single-task and continual active-task curves align exactly."
        )
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
        parser.error("--target-output-std must be finite and positive.")
    if not math.isfinite(args.initial_log_alpha):
        parser.error("--initial-log-alpha must be finite.")

    return args


def create_batch_directory(
    *,
    output_directory: str | Path,
    run_name: str | None,
    seed: int,
) -> Path:
    """Create one new batch directory and its child folders."""
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        resolved_name = f"cw10_single_task_seed{seed}_{timestamp}"
    else:
        resolved_name = _safe_name(run_name)

    batch_directory = Path(output_directory).expanduser() / resolved_name
    batch_directory.mkdir(parents=True, exist_ok=False)
    (batch_directory / "runs").mkdir()
    (batch_directory / "logs").mkdir()
    return batch_directory


def build_single_task_command(
    *,
    args: argparse.Namespace,
    repository_root: Path,
    task_name: str,
    task_run_name: str,
    task_runs_directory: Path,
) -> list[str]:
    """Build one explicit child-process command."""
    command = [
        sys.executable,
        str(repository_root / "scripts" / "train_single_task.py"),
        "--task",
        task_name,
        "--total-steps",
        str(args.steps_per_task),
        "--eval-every",
        str(args.eval_every),
        "--det-eval-episodes",
        str(args.det_eval_episodes),
        "--stoch-eval-episodes",
        str(args.stoch_eval_episodes),
        "--replay-size",
        str(args.replay_size),
        "--batch-size",
        str(args.batch_size),
        "--start-steps",
        str(args.start_steps),
        "--update-after",
        str(args.update_after),
        "--update-every",
        str(args.update_every),
        "--learning-rate",
        str(args.learning_rate),
        "--gamma",
        str(args.gamma),
        "--polyak",
        str(args.polyak),
        "--target-output-std",
        str(args.target_output_std),
        "--initial-log-alpha",
        str(args.initial_log_alpha),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--output-dir",
        str(task_runs_directory),
        "--run-name",
        task_run_name,
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-project",
        args.wandb_project,
        "--wandb-group",
        args.wandb_group,
        (
            "--wandb-log-code"
            if args.wandb_log_code
            else "--no-wandb-log-code"
        ),
        (
            "--wandb-upload-final-checkpoint"
            if args.wandb_upload_final_checkpoint
            else "--no-wandb-upload-final-checkpoint"
        ),
    ]

    if args.wandb_entity is not None:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.wandb_notes is not None:
        command.extend(["--wandb-notes", args.wandb_notes])
    if args.wandb_tags:
        command.append("--wandb-tags")
        command.extend(str(tag) for tag in args.wandb_tags)

    return command


def run_and_tee(
    *,
    command: Sequence[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> int:
    """Run a child command while streaming output to terminal and a log."""
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is None:
            raise RuntimeError("Child process stdout pipe was not created.")

        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return int(process.wait())


def write_manifest(
    *,
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    """Rewrite the small manifest after each task boundary."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the selected fixed-prefix CW10 task batch."""
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    tasks = get_cw10_tasks()[: args.num_tasks]

    batch_directory = create_batch_directory(
        output_directory=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
    )
    runs_directory = batch_directory / "runs"
    logs_directory = batch_directory / "logs"
    manifest_path = batch_directory / "batch_manifest.csv"
    configuration_path = batch_directory / "batch_config.json"
    summary_path = batch_directory / "summary.json"

    configuration = dict(vars(args))
    configuration.update(
        {
            "method": METHOD_NAME,
            "tasks": tasks,
            "num_tasks": len(tasks),
            "batch_directory": str(batch_directory),
            "python_executable": sys.executable,
        }
    )
    with configuration_path.open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2, sort_keys=True)

    child_environment = dict(os.environ)
    source_directory = str(repository_root / "src")
    existing_python_path = child_environment.get("PYTHONPATH", "")
    child_environment["PYTHONPATH"] = (
        source_directory
        if not existing_python_path
        else source_directory + os.pathsep + existing_python_path
    )

    print("=" * 76)
    print("CW10 independent single-task SAC baselines")
    print("=" * 76)
    print(f"Tasks                    : {len(tasks)}")
    print(f"Steps per task           : {args.steps_per_task:,}")
    print(f"Evaluation interval      : {args.eval_every:,}")
    print(f"Deterministic eval eps   : {args.det_eval_episodes}")
    print(f"Stochastic eval eps      : {args.stoch_eval_episodes}")
    print(
        "FT evaluation protocol   : deterministic success curves"
    )
    print(f"Seed                     : {args.seed}")
    print(f"Device                   : {args.device}")
    print(f"W&B mode                 : {args.wandb_mode}")
    print(f"Batch directory          : {batch_directory}")
    print(
        "Isolation                 : one new Python process and one new "
        "SACAgent per task"
    )

    manifest_rows: list[dict[str, Any]] = []
    batch_start = time.perf_counter()

    for task_index, task_name in enumerate(tasks):
        safe_task_name = _safe_name(task_name)
        task_run_name = (
            f"task_{task_index + 1:02d}_{safe_task_name}_seed{args.seed}"
        )
        task_run_directory = runs_directory / task_run_name
        log_path = logs_directory / f"{task_run_name}.log"

        started_at = _utc_now()
        task_start = time.perf_counter()
        manifest_row: dict[str, Any] = {
            "task_index": task_index,
            "task_name": task_name,
            "status": "running",
            "run_name": task_run_name,
            "run_directory": str(
                task_run_directory.relative_to(batch_directory)
            ),
            "log_path": str(log_path.relative_to(batch_directory)),
            "return_code": "",
            "started_at_utc": started_at,
            "finished_at_utc": "",
            "elapsed_seconds": "",
            "error": "",
        }
        manifest_rows.append(manifest_row)
        write_manifest(path=manifest_path, rows=manifest_rows)

        print("\n" + "-" * 76)
        print(f"Task {task_index + 1:02d}/{len(tasks)}: {task_name}")
        print("-" * 76, flush=True)

        command = build_single_task_command(
            args=args,
            repository_root=repository_root,
            task_name=task_name,
            task_run_name=task_run_name,
            task_runs_directory=runs_directory,
        )

        return_code = run_and_tee(
            command=command,
            cwd=repository_root,
            environment=child_environment,
            log_path=log_path,
        )
        elapsed_seconds = time.perf_counter() - task_start

        manifest_row["return_code"] = return_code
        manifest_row["finished_at_utc"] = _utc_now()
        manifest_row["elapsed_seconds"] = elapsed_seconds

        if return_code != 0:
            manifest_row["status"] = "failed"
            manifest_row["error"] = (
                f"train_single_task.py exited with code {return_code}."
            )
            write_manifest(path=manifest_path, rows=manifest_rows)
            _write_batch_summary(
                path=summary_path,
                status="failed",
                tasks=tasks,
                completed_tasks=sum(
                    row["status"] == "completed" for row in manifest_rows
                ),
                batch_directory=batch_directory,
                elapsed_seconds=time.perf_counter() - batch_start,
                aggregate_summary_path=None,
                failed_task=task_name,
            )
            raise RuntimeError(
                f"Single-task baseline failed for {task_name!r}. "
                f"See {log_path}."
            )

        try:
            _validate_task_outputs(task_run_directory)

            if not args.keep_latest_checkpoint:
                latest_checkpoint = (
                    task_run_directory / "checkpoints" / "latest.pt"
                )
                if latest_checkpoint.is_file():
                    latest_checkpoint.unlink()
        except Exception as error:
            manifest_row["status"] = "failed"
            manifest_row["error"] = str(error)
            write_manifest(path=manifest_path, rows=manifest_rows)
            _write_batch_summary(
                path=summary_path,
                status="failed",
                tasks=tasks,
                completed_tasks=sum(
                    row["status"] == "completed" for row in manifest_rows
                ),
                batch_directory=batch_directory,
                elapsed_seconds=time.perf_counter() - batch_start,
                aggregate_summary_path=None,
                failed_task=task_name,
            )
            raise

        manifest_row["status"] = "completed"
        write_manifest(path=manifest_path, rows=manifest_rows)

    aggregate_result = aggregate_single_task_baselines(
        batch_directory=batch_directory,
        tail_size=args.aggregate_tail_size,
    )
    total_elapsed_seconds = time.perf_counter() - batch_start

    _write_batch_summary(
        path=summary_path,
        status="completed",
        tasks=tasks,
        completed_tasks=len(tasks),
        batch_directory=batch_directory,
        elapsed_seconds=total_elapsed_seconds,
        aggregate_summary_path=aggregate_result.summary_json_path,
        failed_task=None,
    )

    print("\n" + "=" * 76)
    print("CW10 single-task baseline batch completed")
    print("=" * 76)
    print(f"Completed tasks          : {len(tasks)}")
    print(f"Elapsed seconds          : {total_elapsed_seconds:.2f}")
    print(f"Manifest                 : {manifest_path}")
    print(f"Batch summary            : {summary_path}")
    print(f"Baseline curves JSON     : {aggregate_result.curves_json_path}")
    print(
        f"Deterministic success CSV: "
        f"{aggregate_result.deterministic_success_csv_path}"
    )
    print(
        f"Deterministic return CSV : "
        f"{aggregate_result.deterministic_return_csv_path}"
    )
    print(
        "FT baseline ready        : yes "
        "(baseline_curves.json contains deterministic_success_curves)"
    )


def _validate_task_outputs(run_directory: Path) -> None:
    required_paths = (
        run_directory / "config.json",
        run_directory / "evaluations.csv",
        run_directory / "summary.json",
        run_directory / "checkpoints" / "final.pt",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Single-task process returned success but required outputs are "
            f"missing: {missing}"
        )


def _write_batch_summary(
    *,
    path: Path,
    status: str,
    tasks: Sequence[str],
    completed_tasks: int,
    batch_directory: Path,
    elapsed_seconds: float,
    aggregate_summary_path: Path | None,
    failed_task: str | None,
) -> None:
    payload = {
        "method": METHOD_NAME,
        "status": status,
        "tasks": list(tasks),
        "num_tasks": len(tasks),
        "completed_tasks": completed_tasks,
        "failed_task": failed_task,
        "batch_directory": str(batch_directory),
        "elapsed_seconds": elapsed_seconds,
        "aggregate_summary": (
            None
            if aggregate_summary_path is None
            else str(aggregate_summary_path)
        ),
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def _safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
