"""Compare deterministic and stochastic evaluation of SAC checkpoints.

Example:

PYTHONPATH=src python scripts/checkpoint_diagnostic.py \
  --run-dir outputs/cw3_finetune/cw3_random_1m_seed0_optimized \
  --episodes 20 \
  --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from crl_cw.agents import SACAgent
from crl_cw.envs.cw_env import make_cw_env
from crl_cw.evaluation import EvaluationConfig, SACEvaluator


CSV_FIELDS = [
    "checkpoint",
    "environment_step",
    "task_index",
    "task_name",
    "evaluation_mode",
    "num_episodes",
    "mean_return",
    "std_return",
    "min_return",
    "max_return",
    "success_rate",
    "mean_episode_length",
    "alpha",
    "seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate each task-boundary SAC checkpoint using both "
            "deterministic and stochastic policies."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help=(
            "Completed continual-run directory containing config.json "
            "and checkpoints/task_*.pt."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Episodes per policy mode and task checkpoint.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument(
        "--seed",
        type=int,
        default=70_000,
        help="Base diagnostic evaluation seed.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional result directory. By default, files are saved "
            "under RUN_DIR/checkpoint_diagnostic."
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
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dictionary: {path}")

    required = {
        "environment_step",
        "agent_state_dict",
        "agent_config",
        "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"Checkpoint {path} is missing fields: "
            + ", ".join(sorted(missing))
        )
    if not isinstance(payload["agent_config"], Mapping):
        raise TypeError("checkpoint agent_config must be a mapping.")
    if not isinstance(payload["metadata"], Mapping):
        raise TypeError("checkpoint metadata must be a mapping.")
    return payload


def close_env(env: Any | None) -> None:
    if env is None:
        return
    close_method = getattr(env, "close", None)
    if callable(close_method):
        close_method()


def build_agent(
    *,
    payload: dict[str, Any],
    task_name: str,
    seed: int,
    device: str,
) -> SACAgent:
    saved_config = dict(payload["agent_config"])
    env = make_cw_env(
        task_name=task_name,
        seed=seed,
        append_task_id=True,
    )
    try:
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)
    finally:
        close_env(env)

    agent = SACAgent(
        observation_dim=int(saved_config["observation_dim"]),
        action_dim=int(saved_config["action_dim"]),
        action_low=action_low,
        action_high=action_high,
        num_tasks=int(saved_config["num_tasks"]),
        task_id_dim=int(saved_config["task_id_dim"]),
        learning_rate=float(saved_config["learning_rate"]),
        gamma=float(saved_config["gamma"]),
        polyak=float(saved_config["polyak"]),
        target_entropy=float(saved_config["target_entropy"]),
        initial_log_alpha=0.0,
        device=device,
    )
    agent.load_state_dict(payload["agent_state_dict"], strict=True)
    agent.eval()
    return agent


def evaluate_mode(
    *,
    agent: SACAgent,
    task_name: str,
    task_index: int,
    checkpoint_path: Path,
    environment_step: int,
    deterministic: bool,
    episodes: int,
    max_episode_steps: int,
    seed: int,
) -> dict[str, Any]:
    env = make_cw_env(
        task_name=task_name,
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
        result = evaluator.evaluate(deterministic=deterministic)
    finally:
        close_env(env)

    returns = np.asarray(result.episode_returns, dtype=np.float64)
    return {
        "checkpoint": str(checkpoint_path),
        "environment_step": int(environment_step),
        "task_index": int(task_index),
        "task_name": task_name,
        "evaluation_mode": "deterministic" if deterministic else "stochastic",
        "num_episodes": int(result.num_episodes),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns, ddof=0)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "success_rate": float(result.success_rate),
        "mean_episode_length": float(result.mean_episode_length),
        "alpha": float(agent.alpha_value_for_task(task_index)),
        "seed": int(seed),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_directory = Path(args.run_dir).expanduser()
    if not run_directory.is_dir():
        raise NotADirectoryError(
            f"Run directory does not exist: {run_directory}"
        )

    configuration = load_json(run_directory / "config.json")
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_paths = sorted(checkpoint_directory.glob("task_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            "No task-boundary checkpoints were found under "
            f"{checkpoint_directory}"
        )

    output_directory = (
        Path(args.output_dir).expanduser()
        if args.output_dir is not None
        else run_directory / "checkpoint_diagnostic"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    comparison_by_task: dict[str, dict[str, Any]] = {}

    print("=" * 92)
    print("SAC checkpoint diagnostic")
    print("=" * 92)
    print(f"Run directory : {run_directory}")
    print(f"Checkpoints   : {len(checkpoint_paths)}")
    print(f"Episodes/mode : {args.episodes}")
    print(f"Device        : {args.device}")
    print()

    for checkpoint_number, checkpoint_path in enumerate(checkpoint_paths):
        payload = load_checkpoint_payload(checkpoint_path)
        metadata = dict(payload["metadata"])
        if "task_name" not in metadata or "task_index" not in metadata:
            raise ValueError(
                f"{checkpoint_path} metadata must contain task_name and task_index."
            )

        task_name = str(metadata["task_name"])
        task_index = int(metadata["task_index"])
        environment_step = int(payload["environment_step"])

        agent = build_agent(
            payload=payload,
            task_name=task_name,
            seed=args.seed + checkpoint_number * 1_000,
            device=args.device,
        )

        deterministic_seed = args.seed + checkpoint_number * 1_000
        stochastic_seed = deterministic_seed + 500

        deterministic_row = evaluate_mode(
            agent=agent,
            task_name=task_name,
            task_index=task_index,
            checkpoint_path=checkpoint_path,
            environment_step=environment_step,
            deterministic=True,
            episodes=args.episodes,
            max_episode_steps=args.max_episode_steps,
            seed=deterministic_seed,
        )
        stochastic_row = evaluate_mode(
            agent=agent,
            task_name=task_name,
            task_index=task_index,
            checkpoint_path=checkpoint_path,
            environment_step=environment_step,
            deterministic=False,
            episodes=args.episodes,
            max_episode_steps=args.max_episode_steps,
            seed=stochastic_seed,
        )

        rows.extend([deterministic_row, stochastic_row])
        comparison_by_task[task_name] = {
            "task_index": task_index,
            "checkpoint": str(checkpoint_path),
            "environment_step": environment_step,
            "alpha": deterministic_row["alpha"],
            "deterministic": {
                key: deterministic_row[key]
                for key in (
                    "mean_return",
                    "std_return",
                    "success_rate",
                    "mean_episode_length",
                )
            },
            "stochastic": {
                key: stochastic_row[key]
                for key in (
                    "mean_return",
                    "std_return",
                    "success_rate",
                    "mean_episode_length",
                )
            },
            "deterministic_minus_stochastic_return": float(
                deterministic_row["mean_return"]
                - stochastic_row["mean_return"]
            ),
            "deterministic_minus_stochastic_success": float(
                deterministic_row["success_rate"]
                - stochastic_row["success_rate"]
            ),
        }

        print(f"{task_name} | checkpoint step {environment_step:,}")
        print(
            "  deterministic | "
            f"return {deterministic_row['mean_return']:.3f} "
            f"± {deterministic_row['std_return']:.3f} | "
            f"success {deterministic_row['success_rate']:.3f}"
        )
        print(
            "  stochastic    | "
            f"return {stochastic_row['mean_return']:.3f} "
            f"± {stochastic_row['std_return']:.3f} | "
            f"success {stochastic_row['success_rate']:.3f}"
        )
        print(
            "  difference    | "
            f"return {comparison_by_task[task_name]['deterministic_minus_stochastic_return']:+.3f} | "
            f"success {comparison_by_task[task_name]['deterministic_minus_stochastic_success']:+.3f} | "
            f"alpha {deterministic_row['alpha']:.6f}"
        )
        print()

    csv_path = output_directory / "checkpoint_diagnostic.csv"
    summary_path = output_directory / "checkpoint_diagnostic_summary.json"
    write_csv(csv_path, rows)

    summary = {
        "run_directory": str(run_directory),
        "source_configuration": configuration,
        "episodes_per_mode": int(args.episodes),
        "max_episode_steps": int(args.max_episode_steps),
        "device": args.device,
        "tasks": comparison_by_task,
        "csv_path": str(csv_path),
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    print("=" * 92)
    print("Diagnostic completed")
    print("=" * 92)
    print(f"CSV     : {csv_path}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
