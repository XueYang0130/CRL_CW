from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for import_path in (SRC_DIR, PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from agents import ReplayBuffer
from envs import DEFAULT_EPISODE_LENGTH, make_cw_env
from methods import get_method
from scripts.probe_stickpull_with_memory import (
    build_agent_from_config,
    evaluate_task,
    load_filtered_memory,
    load_json,
)
from training.sac_trainer import SACTrainer, SACTrainerConfig, seed_global_rngs
from utils import load_sac_checkpoint, save_sac_checkpoint, write_csv, write_json


EVAL_FIELDS = [
    "environment_step",
    "gradient_updates",
    "stochastic_average_return",
    "stochastic_success_rate",
    "stochastic_average_episode_length",
    "alpha",
    "actor_cloning_coefficient",
    "bc_gate_phase",
    "bc_gate_transition",
    "bc_gradient_cosine",
    "bc_gradient_norm_ratio",
    "bc_applied_norm_ratio",
    "bc_gradient_conflict",
    "active_reference_states",
    "elapsed_seconds",
]


@dataclass(frozen=True)
class GateDecision:
    coefficient: float
    phase: str
    transition: str


class ThreePhaseBCGate:
    """Weak transfer, exploration fallback, then retention recovery."""

    def __init__(
        self,
        *,
        initial_coefficient: float,
        fallback_coefficient: float,
        recovery_coefficient: float,
        fallback_after_evals: int,
        fallback_success_threshold: float,
        recovery_success_threshold: float,
        progress_window: int,
        return_progress_threshold: float,
    ) -> None:
        self.initial_coefficient = initial_coefficient
        self.fallback_coefficient = fallback_coefficient
        self.recovery_coefficient = recovery_coefficient
        self.fallback_after_evals = fallback_after_evals
        self.fallback_success_threshold = fallback_success_threshold
        self.recovery_success_threshold = recovery_success_threshold
        self.return_progress_threshold = return_progress_threshold
        self.phase = "initial_transfer"
        self.evaluations = 0
        self.successes: deque[float] = deque(maxlen=progress_window)
        self.returns: deque[float] = deque(maxlen=progress_window)

    @property
    def coefficient(self) -> float:
        if self.phase == "initial_transfer":
            return self.initial_coefficient
        if self.phase == "exploration_fallback":
            return self.fallback_coefficient
        return self.recovery_coefficient

    def update(self, *, success: float, average_return: float) -> GateDecision:
        self.evaluations += 1
        self.successes.append(float(success))
        self.returns.append(float(average_return))
        transition = ""

        if self.phase == "initial_transfer" and self.evaluations >= self.fallback_after_evals:
            recent_success = max(self.successes, default=0.0)
            return_progress = self._relative_return_progress()
            if (
                recent_success <= self.fallback_success_threshold
                and return_progress <= self.return_progress_threshold
            ):
                self.phase = "exploration_fallback"
                transition = (
                    "initial_transfer->exploration_fallback:"
                    f"success={recent_success:.3f},return_progress={return_progress:.3f}"
                )

        # Recovery is allowed from either early phase. A real success signal is
        # stronger evidence than a noisy return increase on stick-pull.
        if self.phase != "retention_recovery" and success >= self.recovery_success_threshold:
            old_phase = self.phase
            self.phase = "retention_recovery"
            transition = (
                f"{old_phase}->retention_recovery:success={success:.3f}"
            )

        return GateDecision(
            coefficient=self.coefficient,
            phase=self.phase,
            transition=transition,
        )

    def _relative_return_progress(self) -> float:
        if len(self.returns) < self.returns.maxlen or len(self.returns) < 2:
            return float("inf")
        first = self.returns[0]
        last = self.returns[-1]
        scale = max(abs(first), abs(last), 1.0)
        return (last - first) / scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone stick-pull probe with a three-phase BC gate."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--memory-manifest", required=True)
    parser.add_argument("--new-task-index", type=int, default=4)
    parser.add_argument("--initial-bc-coefficient", type=float, default=10.0)
    parser.add_argument("--fallback-bc-coefficient", type=float, default=0.0)
    parser.add_argument("--recovery-bc-coefficient", type=float, default=100.0)
    parser.add_argument("--fallback-after-evals", type=int, default=5)
    parser.add_argument("--fallback-success-threshold", type=float, default=0.0)
    parser.add_argument("--recovery-success-threshold", type=float, default=0.2)
    parser.add_argument("--progress-window", type=int, default=5)
    parser.add_argument("--return-progress-threshold", type=float, default=0.10)
    parser.add_argument("--gradient-aware-max-norm-ratio", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--stoch-eval-episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    coefficients = (
        args.initial_bc_coefficient,
        args.fallback_bc_coefficient,
        args.recovery_bc_coefficient,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in coefficients):
        raise ValueError("All BC coefficients must be finite and non-negative.")
    if args.fallback_after_evals < args.progress_window:
        raise ValueError("--fallback-after-evals must be >= --progress-window.")
    if args.progress_window < 2:
        raise ValueError("--progress-window must be at least 2.")
    for name, value in (
        ("fallback-success-threshold", args.fallback_success_threshold),
        ("recovery-success-threshold", args.recovery_success_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1].")
    if args.recovery_success_threshold <= args.fallback_success_threshold:
        raise ValueError(
            "--recovery-success-threshold must exceed --fallback-success-threshold."
        )
    if args.gradient_aware_max_norm_ratio <= 0.0:
        raise ValueError("--gradient-aware-max-norm-ratio must be positive.")
    if args.steps <= 0 or args.eval_every <= 0 or args.steps % args.eval_every != 0:
        raise ValueError("--steps must be positive and divisible by --eval-every.")


def make_output_dir(args: argparse.Namespace) -> Path:
    base = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "stickpull_three_phase_probe"
    )
    output_dir = base / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    return output_dir


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_global_rngs(args.seed)

    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = Path(args.memory_manifest).expanduser().resolve()
    config = load_json(run_dir / "config.json")
    output_dir = make_output_dir(args)
    tasks = list(config["tasks"])
    if not 0 < args.new_task_index < len(tasks):
        raise ValueError("--new-task-index must identify a non-first configured task.")

    previous_task_index = args.new_task_index - 1
    new_task_name = tasks[args.new_task_index]
    method = get_method(str(config["method"]))
    append_task_id = bool(method.append_task_id)
    max_episode_steps = int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH))

    agent = build_agent_from_config(config=config, device=args.device)
    load_sac_checkpoint(
        agent=agent,
        path=run_dir / "checkpoints" / f"task_{previous_task_index}.pt",
        map_location=args.device,
        load_optimizer=False,
    )
    replay_buffer = ReplayBuffer(
        observation_dim=agent.observation_dim,
        action_dim=agent.action_dim,
        capacity=int(config.get("replay_size", 1_000_000)),
        seed=args.seed,
    )
    agent.on_task_start(task_index=args.new_task_index, replay_buffer=replay_buffer)
    replay_buffer.clear()
    agent.rebuild_optimizer()
    agent.copy_alpha(
        source_task_index=previous_task_index,
        target_task_index=args.new_task_index,
    )

    observations, target_means, target_log_stds, _, reference_state_count = load_filtered_memory(
        manifest_path=manifest_path,
        source_task_indices=None,
        allowed_segments=None,
    )
    if reference_state_count == 0:
        raise ValueError("The supplied memory manifest contains no reference states.")
    agent.set_reference_memory_with_targets(
        observations=observations,
        target_means=target_means,
        target_log_stds=target_log_stds,
    )
    agent.configure_gradient_aware_bc(
        enabled=True,
        max_norm_ratio=args.gradient_aware_max_norm_ratio,
    )

    gate = ThreePhaseBCGate(
        initial_coefficient=args.initial_bc_coefficient,
        fallback_coefficient=args.fallback_bc_coefficient,
        recovery_coefficient=args.recovery_bc_coefficient,
        fallback_after_evals=args.fallback_after_evals,
        fallback_success_threshold=args.fallback_success_threshold,
        recovery_success_threshold=args.recovery_success_threshold,
        progress_window=args.progress_window,
        return_progress_threshold=args.return_progress_threshold,
    )
    agent.actor_cloning_coefficient = gate.coefficient

    train_env = make_cw_env(
        new_task_name,
        seed=args.seed,
        max_episode_steps=max_episode_steps,
        append_task_id=append_task_id,
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
    )
    trainer = SACTrainer(
        env=train_env,
        agent=agent,
        replay_buffer=replay_buffer,
        config=SACTrainerConfig(
            total_steps=args.steps,
            batch_size=args.batch_size,
            start_steps=args.start_steps,
            update_after=args.update_after,
            update_every=args.update_every,
            max_episode_steps=max_episode_steps,
            callback_every_steps=args.eval_every,
            seed=args.seed + args.new_task_index,
            reseed_global_rng=False,
        ),
    )

    run_config = {
        **config,
        "probe_type": "stickpull_three_phase_bc",
        "probe_new_task_index": args.new_task_index,
        "probe_new_task_name": new_task_name,
        "probe_reference_state_count": reference_state_count,
        "probe_memory_manifest": str(manifest_path),
        "probe_initial_bc_coefficient": args.initial_bc_coefficient,
        "probe_fallback_bc_coefficient": args.fallback_bc_coefficient,
        "probe_recovery_bc_coefficient": args.recovery_bc_coefficient,
        "probe_fallback_after_evals": args.fallback_after_evals,
        "probe_fallback_success_threshold": args.fallback_success_threshold,
        "probe_recovery_success_threshold": args.recovery_success_threshold,
        "probe_progress_window": args.progress_window,
        "probe_return_progress_threshold": args.return_progress_threshold,
        "probe_gradient_aware_max_norm_ratio": args.gradient_aware_max_norm_ratio,
        "probe_seed": args.seed,
        "probe_steps": args.steps,
    }
    write_json(output_dir / "config.json", run_config)

    started_at = time.time()
    eval_rows: list[dict[str, float | int | str]] = []
    transitions: list[dict[str, float | int | str]] = []
    best_success = float("-inf")
    best_return = float("-inf")

    def evaluate(step: int, gradient_updates: int, update_metrics: dict[str, float] | None) -> None:
        del update_metrics
        nonlocal best_success, best_return
        result = evaluate_task(
            agent=agent,
            task_name=new_task_name,
            env_version=str(config["env_version"]),
            reward_function_version=str(config["reward_function_version"]),
            episodes=args.stoch_eval_episodes,
            deterministic=False,
            max_episode_steps=max_episode_steps,
            seed=args.seed + 50_000 + step,
            append_task_id=append_task_id,
        )
        decision = gate.update(
            success=result["success_rate"],
            average_return=result["average_return"],
        )
        agent.actor_cloning_coefficient = decision.coefficient
        if decision.transition:
            transitions.append(
                {
                    "environment_step": step,
                    "phase": decision.phase,
                    "coefficient": decision.coefficient,
                    "reason": decision.transition,
                }
            )
            print(
                "[stickpull-three-phase] gate transition "
                f"step={step:,} phase={decision.phase} "
                f"coefficient={decision.coefficient:g} reason={decision.transition}"
            )

        row = {
            "environment_step": step,
            "gradient_updates": gradient_updates,
            "stochastic_average_return": result["average_return"],
            "stochastic_success_rate": result["success_rate"],
            "stochastic_average_episode_length": result["average_episode_length"],
            "alpha": agent.diagnostic_alpha_value(args.new_task_index),
            "actor_cloning_coefficient": agent.actor_cloning_coefficient,
            "bc_gate_phase": decision.phase,
            "bc_gate_transition": decision.transition,
            "bc_gradient_cosine": agent.last_bc_gradient_cosine,
            "bc_gradient_norm_ratio": agent.last_bc_gradient_norm_ratio,
            "bc_applied_norm_ratio": agent.last_bc_applied_norm_ratio,
            "bc_gradient_conflict": int(agent.last_bc_conflict),
            "active_reference_states": agent.reference_state_count,
            "elapsed_seconds": time.time() - started_at,
        }
        eval_rows.append(row)

        if result["success_rate"] > best_success:
            best_success = result["success_rate"]
            save_sac_checkpoint(
                agent=agent,
                path=output_dir / "checkpoints" / "best_success.pt",
                environment_step=step,
                metadata={"task_name": new_task_name, "metric": "success"},
            )
        if result["average_return"] > best_return:
            best_return = result["average_return"]
            save_sac_checkpoint(
                agent=agent,
                path=output_dir / "checkpoints" / "best_return.pt",
                environment_step=step,
                metadata={"task_name": new_task_name, "metric": "return"},
            )
        print(
            f"[stickpull-three-phase] step={step:,}/{args.steps:,} "
            f"success={result['success_rate']:.3f} "
            f"return={result['average_return']:.3f} "
            f"phase={decision.phase} bc={decision.coefficient:g}"
        )

    summary = trainer.train(step_callback=evaluate)
    train_env.close()
    save_sac_checkpoint(
        agent=agent,
        path=output_dir / "checkpoints" / "final.pt",
        environment_step=summary.total_env_steps,
        metadata={"task_name": new_task_name, "metric": "final"},
    )
    write_csv(output_dir / "evaluations.csv", EVAL_FIELDS, eval_rows)
    write_json(output_dir / "gate_transitions.json", {"transitions": transitions})

    final_eval = eval_rows[-1]
    write_json(
        output_dir / "summary.json",
        {
            "probe_type": "stickpull_three_phase_bc",
            "new_task_name": new_task_name,
            "reference_state_count": reference_state_count,
            "final_success_rate": final_eval["stochastic_success_rate"],
            "final_average_return": final_eval["stochastic_average_return"],
            "best_success_rate": best_success,
            "best_average_return": best_return,
            "final_bc_gate_phase": gate.phase,
            "final_actor_cloning_coefficient": gate.coefficient,
            "gate_transitions": transitions,
            "completed_episodes": summary.completed_episodes,
            "gradient_updates": summary.gradient_updates,
            "elapsed_seconds": time.time() - started_at,
            "run_directory": str(output_dir),
        },
    )


if __name__ == "__main__":
    main()
