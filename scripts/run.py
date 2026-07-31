from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import DEFAULT_EPISODE_LENGTH, get_cw10_tasks
from evaluation import aggregate_single_task_baselines
from methods import (
    available_method_ids,
    default_method_for_mode,
    is_method_compatible,
    method_defaults,
)
from training import run_cw10_experiment, run_single_task_experiment
from utils import write_json


def load_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return dict(payload)


def load_dotenv_values(path: str | Path) -> dict[str, str]:
    dotenv_path = Path(path).expanduser().resolve()
    if not dotenv_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", choices=("single", "single-batch", "continual"), default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--task", type=str, default="hammer-v3")
    parser.add_argument("--env-version", choices=("v2", "v3"), default="v3")
    parser.add_argument(
        "--reward-function-version",
        choices=("v1", "v2", "v1_compatible", "cw10_v1"),
        default="v2",
    )
    parser.add_argument("--steps-per-task", type=int, default=1_000_000)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--sequence-task-count", type=int, default=10)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--det-eval-episodes", type=int, default=1)
    parser.add_argument("--stoch-eval-episodes", type=int, default=10)
    parser.add_argument("--best-return-eval-episodes", type=int, default=2)
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
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_EPISODE_LENGTH)
    parser.add_argument("--exploration-strategy", type=str, default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--reset-buffer-on-task-change", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-optimizer-on-task-change", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-curves", type=str, default=None)
    parser.add_argument("--packnet-retrain-steps", type=int, default=0)
    parser.add_argument("--episodic-memory-per-task", type=int, default=0)
    parser.add_argument("--episodic-batch-size", type=int, default=0)
    parser.add_argument("--actor-cloning-coefficient", type=float, default=0.0)
    parser.add_argument("--full-bc-reference-episodes", type=int, default=20)
    parser.add_argument("--full-bc-reference-max-attempts", type=int, default=80)
    parser.add_argument("--semantic-segments", nargs="+", default=None)
    parser.add_argument(
        "--semantic-segment-scheme",
        choices=("heuristic_v1", "task_aware_v2", "task_aware_v3"),
        default="heuristic_v1",
    )
    parser.add_argument("--semantic-post-success-steps", type=int, default=10)
    parser.add_argument("--background-segment-ratio", type=float, default=0.0)
    parser.add_argument("--task-specific-segment-ratio", type=float, default=0.0)
    parser.add_argument(
        "--segment-selection-mode",
        choices=("fixed", "task_adaptive", "llm_online"),
        default="fixed",
    )
    parser.add_argument("--segment-selection-manifest", type=str, default=None)
    parser.add_argument("--task-specific-segment-manifest", type=str, default=None)
    parser.add_argument("--llm-controller-model", type=str, default="gpt-5")
    parser.add_argument("--llm-controller-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--llm-controller-update-every-evals", type=int, default=1)
    parser.add_argument("--llm-controller-max-output-tokens", type=int, default=400)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--wsrl-backbone-source", choices=("current", "best_return"), default="current")
    parser.add_argument("--wsrl-head-source", choices=("reset", "current", "best_return"), default="reset")
    parser.add_argument("--wsrl-warmup-steps", type=int, default=10_000)
    parser.add_argument("--jsrl-initial-guide-steps", type=int, default=180)
    parser.add_argument("--jsrl-curriculum-stages", type=int, default=10)
    parser.add_argument("--jsrl-evaluation-interval", type=int, default=20_000)
    parser.add_argument("--jsrl-moving-average-window", type=int, default=5)
    parser.add_argument("--jsrl-stage-tolerance", type=float, default=0.10)
    parser.add_argument("--jsrl-min-evaluations-per-stage", type=int, default=1)
    parser.add_argument("--jsrl-max-evaluations-without-advance", type=int, default=5)
    parser.add_argument("--aggregate-tail-size", type=int, default=5)
    parser.add_argument("--tail-size", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    explicit_destinations = {
        action.dest
        for action in parser._actions
        if any(
            argument == option or argument.startswith(f"{option}=")
            for argument in sys.argv[1:]
            for option in action.option_strings
        )
    }
    dotenv_values = load_dotenv_values(PROJECT_ROOT / ".env")
    env_model = (
        os.environ.get("OPENAI_MODEL", "").strip()
        or dotenv_values.get("OPENAI_MODEL", "").strip()
    )
    if env_model and "llm_controller_model" not in explicit_destinations:
        args.llm_controller_model = env_model

    if args.config is not None:
        config = load_config(args.config)
        unknown_keys = [key for key in config if not hasattr(args, key)]
        if unknown_keys:
            parser.error(f"Unknown config keys: {', '.join(sorted(unknown_keys))}")
        for key, value in config.items():
            if key not in explicit_destinations:
                setattr(args, key, value)

    if args.mode is None:
        parser.error("--mode is required unless provided by --config.")
    if args.method is None:
        args.method = default_method_for_mode(args.mode)
    if args.method not in available_method_ids():
        parser.error(f"Unsupported method '{args.method}'. Available methods: {', '.join(available_method_ids())}")
    configured_keys = set(config) if args.config is not None else set()
    for key, value in method_defaults(args.method).items():
        if key not in explicit_destinations and key not in configured_keys:
            setattr(args, key, value)
    if not is_method_compatible(args.mode, args.method):
        parser.error(f"Method '{args.method}' is not compatible with mode '{args.mode}'.")
    if args.total_steps is None:
        args.total_steps = args.steps_per_task
    if args.semantic_segments is not None:
        args.semantic_segments = list(args.semantic_segments)
    if args.background_segment_ratio < 0.0:
        parser.error("--background-segment-ratio must be non-negative.")
    if args.task_specific_segment_ratio < 0.0:
        parser.error("--task-specific-segment-ratio must be non-negative.")
    if args.semantic_post_success_steps < 0:
        parser.error("--semantic-post-success-steps must be non-negative.")
    if args.llm_controller_update_every_evals <= 0:
        parser.error("--llm-controller-update-every-evals must be positive.")
    if not 1 <= args.sequence_task_count <= 10:
        parser.error("--sequence-task-count must be between 1 and 10.")
    if not 1 <= args.num_tasks <= 10:
        parser.error("--num-tasks must be between 1 and 10.")
    if args.steps_per_task <= 0 or args.total_steps <= 0:
        parser.error("Training steps must be positive.")
    if args.eval_every <= 0:
        parser.error("--eval-every must be positive.")
    if args.stoch_eval_episodes <= 0:
        parser.error("--stoch-eval-episodes must be positive.")
    if args.det_eval_episodes < 0:
        parser.error("--det-eval-episodes must be non-negative.")
    if args.tail_size <= 0 or args.aggregate_tail_size <= 0:
        parser.error("Metric tail sizes must be positive.")
    if (
        args.task_specific_segment_ratio > 0.0
        and args.task_specific_segment_manifest is None
    ):
        parser.error(
            "--task-specific-segment-ratio requires "
            "--task-specific-segment-manifest."
        )
    if args.segment_selection_mode == "llm_online" and args.mode != "continual":
        parser.error("llm_online segment selection requires --mode continual.")
    for option_name, path_value in (
        ("--baseline-curves", args.baseline_curves),
        ("--segment-selection-manifest", args.segment_selection_manifest),
        ("--task-specific-segment-manifest", args.task_specific_segment_manifest),
    ):
        if path_value is not None and not Path(path_value).expanduser().is_file():
            parser.error(f"{option_name} file does not exist: {path_value}")
    return args


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_directory(base_output_dir: str | Path, run_name: str) -> Path:
    run_dir = Path(base_output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    return run_dir


def run_single(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or "outputs/single_task"
    run_name = args.run_name or f"{args.method}_{args.task}_seed{args.seed}_{timestamp()}"
    run_dir = make_run_directory(output_dir, run_name)
    result = run_single_task_experiment(args, run_dir)
    print(
        json.dumps(
            {
                "method": args.method,
                "task": args.task,
                "final_success": result["summary"]["final_stochastic_success_rate"],
                "final_return": result["summary"]["final_stochastic_average_return"],
                "run_directory": str(run_dir),
            },
            indent=2,
        )
    )


def run_single_batch(args: argparse.Namespace) -> None:
    tasks = get_cw10_tasks(args.env_version)[: args.num_tasks]
    output_dir = args.output_dir or "outputs/cw10_single_task_baselines"
    run_name = args.run_name or f"{args.method}_cw10_seed{args.seed}_{timestamp()}"
    batch_dir = Path(output_dir) / run_name
    runs_dir = batch_dir / "runs"
    logs_dir = batch_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir()

    write_json(
        batch_dir / "batch_config.json",
        {
            "method": args.method,
            "tasks": tasks,
            "seed": args.seed,
            "env_version": args.env_version,
            "reward_function_version": args.reward_function_version,
            "steps_per_task": args.steps_per_task,
            "eval_every": args.eval_every,
            "det_eval_episodes": args.det_eval_episodes,
            "stoch_eval_episodes": args.stoch_eval_episodes,
            "max_episode_steps": args.max_episode_steps,
        },
    )

    manifest_rows: list[dict[str, object]] = []
    script_path = Path(__file__).resolve()

    for task_index, task_name in enumerate(tasks):
        task_run_name = f"{task_name}_seed{args.seed}"
        log_path = logs_dir / f"{task_run_name}.log"
        command = [
            sys.executable,
            str(script_path),
            "--mode",
            "single",
            "--method",
            args.method,
            "--task",
            task_name,
            "--env-version",
            args.env_version,
            "--reward-function-version",
            args.reward_function_version,
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
            str(runs_dir),
            "--run-name",
            task_run_name,
        ]
        started_at = time.time()
        started_iso = datetime.now(timezone.utc).isoformat()
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
        finished_iso = datetime.now(timezone.utc).isoformat()
        manifest_rows.append(
            {
                "task_index": task_index,
                "task_name": task_name,
                "status": "completed" if result.returncode == 0 else "failed",
                "run_name": task_run_name,
                "run_directory": str(runs_dir / task_run_name),
                "log_path": str(log_path),
                "return_code": result.returncode,
                "started_at_utc": started_iso,
                "finished_at_utc": finished_iso,
                "elapsed_seconds": time.time() - started_at,
                "error": "" if result.returncode == 0 else f"process exited with {result.returncode}",
            }
        )
        if result.returncode != 0:
            break

    manifest_path = batch_dir / "batch_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    if any(row["status"] != "completed" for row in manifest_rows):
        write_json(batch_dir / "summary.json", {"status": "failed", "batch_directory": str(batch_dir)})
        raise SystemExit(1)

    aggregate = aggregate_single_task_baselines(
        batch_directory=batch_dir,
        output_directory=batch_dir / "aggregate",
        tail_size=args.aggregate_tail_size,
    )
    write_json(
        batch_dir / "summary.json",
        {
            "status": "completed",
            "method": args.method,
            "batch_directory": str(batch_dir),
            "aggregate_summary": str(aggregate.summary_json_path),
            "aggregate_curves": str(aggregate.curves_json_path),
        },
    )
    print(
        json.dumps(
            {
                "method": args.method,
                "batch_directory": str(batch_dir),
                "aggregate": str(aggregate.summary_json_path),
            },
            indent=2,
        )
    )


def run_continual(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or "outputs/cw10_continual"
    run_name = args.run_name or f"{args.method}_cw10_seed{args.seed}_{timestamp()}"
    run_dir = make_run_directory(output_dir, run_name)
    result = run_cw10_experiment(args, run_dir)
    payload = dict(result["summary"])
    payload["method"] = args.method
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "single":
        run_single(args)
        return
    if args.mode == "single-batch":
        run_single_batch(args)
        return
    run_continual(args)


if __name__ == "__main__":
    main()
