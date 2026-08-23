from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import (
    FullBehaviorCloningSACAgent,
    LayerwiseAdaptivePCGradAgent,
    OptimisticEnsembleFullBCAgent,
    PrioritizedNStepReplayBuffer,
    ReplayBuffer,
)
from envs import DEFAULT_EPISODE_LENGTH, make_cw_env
from methods import get_method
from scripts.build_stickpull_matched_memory import build_agent_from_config
from scripts.diagnose_success_cluster_balancing import (
    extract_bc_feature_gradients,
    kmeans_labels,
    project_and_standardize_features,
    state_dict_digest,
)
from scripts.probe_stickpull_with_memory import (
    evaluate_task,
    load_json,
)
from training.sac_trainer import SACTrainer, SACTrainerConfig, seed_global_rngs
from utils import (
    append_csv_rows,
    load_sac_checkpoint,
    save_sac_checkpoint,
    write_csv,
    write_json,
)


EVAL_FIELDS = (
    "environment_step",
    "gradient_updates",
    "stochastic_average_return",
    "stochastic_success_rate",
    "stochastic_average_episode_length",
    "actor_loss",
    "q1_loss",
    "q2_loss",
    "q1_mean",
    "q2_mean",
    "q_target_mean",
    "ensemble_critic_loss",
    "q_ensemble_std_mean",
    "td_error_abs_mean",
    "priority_beta",
    "importance_weight_mean",
    "alpha",
    "reference_states",
    "guide_mode",
    "guide_steps",
    "elapsed_seconds",
)

GUIDE_EVENT_FIELDS = (
    "environment_step",
    "event",
    "from_guide_steps",
    "to_guide_steps",
    "mixed_average_return",
    "reference_return",
    "reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled stick-pull acquisition ablation from one fixed "
            "task-3 checkpoint."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--memory-manifest",
        default=None,
        help="Required for memory-backed conditions; omitted for --memory-mode none.",
    )
    parser.add_argument("--new-task-index", type=int, default=4)
    parser.add_argument(
        "--memory-mode",
        choices=("success", "broad", "mixed50", "mixed80_dynamic", "none"),
        required=True,
    )
    parser.add_argument(
        "--integration",
        choices=(
            "adaptive_pcgrad",
            "layerwise_adaptive_pcgrad",
            "kl_budget_adaptive_pcgrad",
            "optimistic_nstep_pcgrad",
            "plain",
        ),
        required=True,
    )
    parser.add_argument(
        "--memory-sampler",
        choices=("uniform", "gradient_cluster_routed"),
        default="uniform",
    )
    parser.add_argument("--clusters-per-task", type=int, default=8)
    parser.add_argument("--cluster-projection-dim", type=int, default=16)
    parser.add_argument("--cluster-kmeans-iterations", type=int, default=25)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--stoch-eval-episodes", type=int, default=5)
    parser.add_argument("--retention-eval-episodes", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--bc-update-interval", type=int, default=1)
    parser.add_argument("--bc-adaptive-target-ratio", type=float, default=0.2)
    parser.add_argument("--bc-adaptive-conflict-ratio", type=float, default=0.05)
    parser.add_argument("--critic-ensemble-size", type=int, default=4)
    parser.add_argument("--critic-bootstrap-probability", type=float, default=0.8)
    parser.add_argument("--optimistic-ucb-beta", type=float, default=0.5)
    parser.add_argument("--optimistic-action-candidates", type=int, default=8)
    parser.add_argument("--n-step-return", type=int, default=3)
    parser.add_argument("--prioritized-replay-fraction", type=float, default=0.2)
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument("--priority-beta-start", type=float, default=0.4)
    parser.add_argument("--priority-beta-end", type=float, default=1.0)
    parser.add_argument("--priority-epsilon", type=float, default=1e-6)
    parser.add_argument("--kl-budget-low", type=float, default=0.05)
    parser.add_argument("--kl-budget-high", type=float, default=0.5)
    parser.add_argument("--kl-budget-ema-beta", type=float, default=0.99)
    parser.add_argument(
        "--guide-mode",
        choices=("none", "fixed_short", "fast_curriculum"),
        default="none",
    )
    parser.add_argument("--guide-initial-steps", type=int, default=100)
    parser.add_argument("--guide-fixed-env-steps", type=int, default=40_000)
    parser.add_argument("--guide-max-env-steps", type=int, default=80_000)
    parser.add_argument(
        "--guide-curriculum-horizons",
        type=int,
        nargs="+",
        default=(100, 50, 25, 0),
    )
    parser.add_argument("--guide-stage-tolerance", type=float, default=0.10)
    parser.add_argument("--guide-eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def make_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    base = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "stickpull_causal_ablation"
    )
    name = args.run_name or (
        f"{run_dir.name}_{args.memory_mode}_{args.integration}_seed{args.seed}"
    )
    output_dir = base / name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    return output_dir


def resolve_manifest_file(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path.resolve()
    local_path = manifest_path.parent / path.name
    if local_path.is_file():
        return local_path.resolve()
    raise FileNotFoundError(f"Memory payload does not exist: {raw_path}")


def configure_integration(
    agent: FullBehaviorCloningSACAgent,
    *,
    integration: str,
    bc_update_interval: int,
    adaptive_target_ratio: float,
    adaptive_conflict_ratio: float,
) -> None:
    if integration in {
        "adaptive_pcgrad",
        "layerwise_adaptive_pcgrad",
        "kl_budget_adaptive_pcgrad",
        "optimistic_nstep_pcgrad",
    }:
        agent.bc_gradient_strategy = "pcgrad_sac_priority"
        agent.bc_combination_strategy = "adaptive_additive"
    elif integration == "plain":
        agent.bc_gradient_strategy = "standard"
        agent.bc_combination_strategy = "average"
    else:
        raise ValueError(f"Unsupported integration mode: {integration}")
    if not math.isfinite(adaptive_target_ratio) or adaptive_target_ratio < 0.0:
        raise ValueError("Adaptive target ratio must be finite and non-negative.")
    if not math.isfinite(adaptive_conflict_ratio) or adaptive_conflict_ratio < 0.0:
        raise ValueError("Adaptive conflict ratio must be finite and non-negative.")
    if adaptive_conflict_ratio > adaptive_target_ratio:
        raise ValueError("Adaptive conflict ratio cannot exceed target ratio.")
    agent.bc_adaptive_target_ratio = float(adaptive_target_ratio)
    agent.bc_adaptive_conflict_ratio = float(adaptive_conflict_ratio)
    configure_interval = getattr(agent, "configure_bc_update_interval", None)
    if configure_interval is not None:
        configure_interval(bc_update_interval)
    elif bc_update_interval != 1:
        raise RuntimeError(
            "This agent version does not support sparse BC updates; "
            "use --bc-update-interval 1."
        )


def preserve_global_rngs() -> tuple[object, tuple[Any, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def restore_global_rngs(
    state: tuple[object, tuple[Any, ...], torch.Tensor],
) -> None:
    python_state, numpy_state, torch_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)


def evaluate_frozen_guide_return(
    *,
    agent: FullBehaviorCloningSACAgent,
    env: Any,
    guide_task_index: int,
    episodes: int,
    max_episode_steps: int,
    seed: int,
    guide_steps: int,
) -> float:
    """Evaluate a frozen guide prefix without perturbing the training RNG stream."""
    if episodes <= 0:
        raise ValueError("Guide evaluation episodes must be positive.")
    rng_state = preserve_global_rngs()
    returns: list[float] = []
    try:
        for episode_index in range(episodes):
            reset_result = env.reset(seed=seed + episode_index)
            observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            episode_return = 0.0
            for episode_step in range(max_episode_steps):
                if episode_step < guide_steps:
                    action = agent.select_guide_action(
                        observation,
                        guide_task_index=guide_task_index,
                        deterministic=False,
                    )
                else:
                    action = agent.select_action(observation, deterministic=False)
                step_result = env.step(action)
                if len(step_result) == 5:
                    observation, reward, terminated, truncated, _ = step_result
                    done = bool(terminated or truncated)
                else:
                    observation, reward, done, _ = step_result
                episode_return += float(reward)
                if done:
                    break
            returns.append(episode_return)
    finally:
        restore_global_rngs(rng_state)
    return float(np.mean(returns))


def configure_frozen_guides(
    *,
    agent: FullBehaviorCloningSACAgent,
    run_dir: Path,
    task_index: int,
    env: Any,
    episodes: int,
    max_episode_steps: int,
    seed: int,
) -> tuple[int, float]:
    set_guide_state = getattr(agent, "set_task_guide_actor_state", None)
    initialize_guide = getattr(agent, "initialize_task_from_guide", None)
    if not callable(set_guide_state) or not callable(initialize_guide):
        raise RuntimeError("Guide mode requires a demonstration-guided agent.")

    guide_returns: list[float] = []
    for source_task_index in range(task_index):
        snapshot_path = (
            run_dir
            / "teacher_snapshots"
            / f"task_{source_task_index}_best_actor.pt"
        )
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"Missing frozen guide snapshot: {snapshot_path}")
        snapshot = torch.load(snapshot_path, map_location=agent.device, weights_only=False)
        actor_state = snapshot.get("actor_state_dict")
        if not isinstance(actor_state, dict):
            raise ValueError(f"Invalid actor snapshot: {snapshot_path}")
        set_guide_state(task_index=source_task_index, actor_state=actor_state)
        guide_returns.append(
            evaluate_frozen_guide_return(
                agent=agent,
                env=env,
                guide_task_index=source_task_index,
                episodes=episodes,
                max_episode_steps=max_episode_steps,
                seed=seed + source_task_index * episodes,
                guide_steps=max_episode_steps,
            )
        )
    selected = max(range(task_index), key=guide_returns.__getitem__)
    initialize_guide(task_index=task_index, guide_task_index=selected)
    return selected, guide_returns[selected]


def load_memory(
    *,
    agent: FullBehaviorCloningSACAgent,
    manifest_path: Path | None,
    memory_mode: str,
    expected_source_tasks: list[str],
    seed: int,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    if memory_mode == "none" and manifest_path is None:
        agent.clear_reference_memory()
        return 0, {
            "schema_version": 1,
            "purpose": "no_memory_control",
            "source_tasks": list(expected_source_tasks),
            "teacher": None,
        }, []
    if manifest_path is None:
        raise ValueError("--memory-manifest is required unless --memory-mode none.")
    manifest = load_json(manifest_path)
    if manifest.get("purpose") != "matched_stickpull_memory_causal_ablation":
        raise ValueError("The supplied manifest is not a matched causal-memory manifest.")
    if list(manifest.get("source_tasks", [])) != expected_source_tasks:
        raise ValueError(
            "Memory source tasks do not match the tasks preceding stick-pull."
        )
    if manifest.get("teacher") != "per_task_best_actor_snapshot":
        raise ValueError("Both memory conditions must use best-actor relabelled targets.")
    if memory_mode == "none":
        agent.clear_reference_memory()
        return 0, manifest, []

    files = manifest.get("files")
    mixed_modes = {"mixed50", "mixed80_dynamic"}
    required_files = (
        ("success", "broad") if memory_mode in mixed_modes else (memory_mode,)
    )
    if not isinstance(files, dict) or any(name not in files for name in required_files):
        raise ValueError(f"Manifest does not define the {memory_mode} memory payload.")
    payloads: dict[str, dict[str, np.ndarray]] = {}
    for name in required_files:
        payload_path = resolve_manifest_file(manifest_path, str(files[name]))
        with np.load(payload_path) as payload:
            payloads[name] = {
                "observations": payload["observations"].astype(np.float32, copy=True),
                "target_means": payload["target_means"].astype(np.float32, copy=True),
                "target_log_stds": payload["target_log_stds"].astype(
                    np.float32, copy=True
                ),
            }
    memory_rows: list[dict[str, Any]] = []
    if memory_mode == "mixed50":
        observations, target_means, target_log_stds, memory_rows = (
            build_nonoverlapping_mixed_memory(
                success_payload=payloads["success"],
                broad_payload=payloads["broad"],
                task_id_dim=agent.task_id_dim,
                expected_source_tasks=expected_source_tasks,
                states_per_task=int(manifest["memory_states_per_task"]),
                seed=seed,
            )
        )
    elif memory_mode == "mixed80_dynamic":
        observations, target_means, target_log_stds, memory_rows = (
            build_dynamic_success_dominant_memory(
                success_payload=payloads["success"],
                broad_payload=payloads["broad"],
                task_id_dim=agent.task_id_dim,
                expected_source_tasks=expected_source_tasks,
                states_per_task=int(manifest["memory_states_per_task"]),
                success_fraction=0.8,
                seed=seed,
            )
        )
    else:
        observations = payloads[memory_mode]["observations"]
        target_means = payloads[memory_mode]["target_means"]
        target_log_stds = payloads[memory_mode]["target_log_stds"]
    agent.set_reference_memory_with_targets(
        observations=observations,
        target_means=target_means,
        target_log_stds=target_log_stds,
    )
    expected_count = (
        len(expected_source_tasks) * int(manifest["memory_states_per_task"])
        if memory_mode in mixed_modes
        else int(manifest["counts"][memory_mode])
    )
    if agent.reference_state_count != expected_count:
        raise RuntimeError(
            f"Loaded memory count mismatch: expected {expected_count}, "
            f"found {agent.reference_state_count}."
        )
    return agent.reference_state_count, manifest, memory_rows


def build_nonoverlapping_mixed_memory(
    *,
    success_payload: dict[str, np.ndarray],
    broad_payload: dict[str, np.ndarray],
    task_id_dim: int,
    expected_source_tasks: list[str],
    states_per_task: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build an exact 50/50 task-stratified memory with no duplicate states."""
    if states_per_task <= 0 or states_per_task % 2 != 0:
        raise ValueError("mixed50 requires a positive, even states-per-task capacity.")
    for name, payload in (("success", success_payload), ("broad", broad_payload)):
        observations = payload["observations"]
        if observations.ndim != 2 or observations.shape[1] < task_id_dim:
            raise ValueError(f"{name} observations have an invalid shape.")
        if payload["target_means"].shape[0] != observations.shape[0]:
            raise ValueError(f"{name} target means do not align with observations.")
        if payload["target_log_stds"].shape != payload["target_means"].shape:
            raise ValueError(f"{name} target log stds have an invalid shape.")

    def source_indices(observations: np.ndarray) -> np.ndarray:
        task_vectors = observations[:, -task_id_dim:]
        indices = np.argmax(task_vectors, axis=1)
        expected = np.zeros_like(task_vectors)
        expected[np.arange(indices.size), indices] = 1.0
        if not np.allclose(task_vectors, expected, atol=1e-5, rtol=0.0):
            raise ValueError("Mixed-memory observations lack valid one-hot task IDs.")
        return indices

    success_sources = source_indices(success_payload["observations"])
    broad_sources = source_indices(broad_payload["observations"])
    half = states_per_task // 2
    rng = np.random.default_rng(seed + 910_000)
    selected_parts: list[dict[str, np.ndarray]] = []
    rows: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(expected_source_tasks):
        success_candidates = np.flatnonzero(success_sources == task_index)
        broad_candidates = np.flatnonzero(broad_sources == task_index)
        if success_candidates.size < half or broad_candidates.size < half:
            raise RuntimeError(f"Task {task_name} cannot fill the mixed50 memory quota.")

        broad_indices = rng.choice(broad_candidates, size=half, replace=False)
        broad_hashes = {
            broad_payload["observations"][index].tobytes()
            for index in broad_indices
        }
        eligible_success = np.asarray(
            [
                index
                for index in success_candidates
                if success_payload["observations"][index].tobytes() not in broad_hashes
            ],
            dtype=np.int64,
        )
        if eligible_success.size < half:
            raise RuntimeError(
                f"Task {task_name} has only {eligible_success.size} non-overlapping "
                f"success states; {half} are required."
            )
        success_indices = rng.choice(eligible_success, size=half, replace=False)
        for source_name, payload, indices in (
            ("success", success_payload, success_indices),
            ("broad", broad_payload, broad_indices),
        ):
            selected_parts.append(
                {
                    "observations": payload["observations"][indices],
                    "target_means": payload["target_means"][indices],
                    "target_log_stds": payload["target_log_stds"][indices],
                }
            )
            rows.append(
                {
                    "source_task_index": task_index,
                    "source_task_name": task_name,
                    "memory_source": source_name,
                    "states": int(indices.size),
                }
            )

    observations = np.concatenate(
        [part["observations"] for part in selected_parts], axis=0
    )
    target_means = np.concatenate(
        [part["target_means"] for part in selected_parts], axis=0
    )
    target_log_stds = np.concatenate(
        [part["target_log_stds"] for part in selected_parts], axis=0
    )
    permutation = rng.permutation(observations.shape[0])
    return (
        observations[permutation].copy(),
        target_means[permutation].copy(),
        target_log_stds[permutation].copy(),
        rows,
    )


def build_dynamic_success_dominant_memory(
    *,
    success_payload: dict[str, np.ndarray],
    broad_payload: dict[str, np.ndarray],
    task_id_dim: int,
    expected_source_tasks: list[str],
    states_per_task: int,
    success_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build a task-balanced success-preferred memory with broad fallback."""
    if states_per_task <= 0:
        raise ValueError("states_per_task must be positive.")
    if not 0.0 < success_fraction < 1.0:
        raise ValueError("success_fraction must be strictly between zero and one.")
    for name, payload in (("success", success_payload), ("broad", broad_payload)):
        observations = payload["observations"]
        if observations.ndim != 2 or observations.shape[1] < task_id_dim:
            raise ValueError(f"{name} observations have an invalid shape.")
        if payload["target_means"].shape[0] != observations.shape[0]:
            raise ValueError(f"{name} target means do not align with observations.")
        if payload["target_log_stds"].shape != payload["target_means"].shape:
            raise ValueError(f"{name} target log stds have an invalid shape.")

    def source_indices(observations: np.ndarray) -> np.ndarray:
        task_vectors = observations[:, -task_id_dim:]
        indices = np.argmax(task_vectors, axis=1)
        expected = np.zeros_like(task_vectors)
        expected[np.arange(indices.size), indices] = 1.0
        if not np.allclose(task_vectors, expected, atol=1e-5, rtol=0.0):
            raise ValueError("Dynamic mixed-memory observations lack valid task IDs.")
        return indices

    success_sources = source_indices(success_payload["observations"])
    broad_sources = source_indices(broad_payload["observations"])
    nominal_success = int(round(states_per_task * success_fraction))
    rng = np.random.default_rng(seed + 920_000)
    selected_parts: list[dict[str, np.ndarray]] = []
    rows: list[dict[str, Any]] = []

    for task_index, task_name in enumerate(expected_source_tasks):
        success_candidates = np.flatnonzero(success_sources == task_index)
        broad_candidates = np.flatnonzero(broad_sources == task_index)
        initial_success_count = min(nominal_success, success_candidates.size)
        success_indices = rng.choice(
            success_candidates,
            size=initial_success_count,
            replace=False,
        )
        success_hashes = {
            success_payload["observations"][index].tobytes()
            for index in success_indices
        }
        eligible_broad = np.asarray(
            [
                index
                for index in broad_candidates
                if broad_payload["observations"][index].tobytes()
                not in success_hashes
            ],
            dtype=np.int64,
        )
        broad_count = min(
            states_per_task - initial_success_count,
            eligible_broad.size,
        )
        broad_indices = rng.choice(
            eligible_broad,
            size=broad_count,
            replace=False,
        )
        broad_hashes = {
            broad_payload["observations"][index].tobytes()
            for index in broad_indices
        }
        selected_success = set(int(index) for index in success_indices.tolist())
        extra_success_candidates = np.asarray(
            [
                index
                for index in success_candidates
                if int(index) not in selected_success
                and success_payload["observations"][index].tobytes()
                not in broad_hashes
            ],
            dtype=np.int64,
        )
        extra_success_count = min(
            states_per_task - initial_success_count - broad_count,
            extra_success_candidates.size,
        )
        if extra_success_count:
            extra_success = rng.choice(
                extra_success_candidates,
                size=extra_success_count,
                replace=False,
            )
            success_indices = np.concatenate((success_indices, extra_success))

        total_selected = int(success_indices.size + broad_indices.size)
        if total_selected != states_per_task:
            raise RuntimeError(
                f"Task {task_name} cannot fill dynamic mixed memory: "
                f"selected {total_selected}/{states_per_task} unique states."
            )
        for source_name, payload, indices in (
            ("success", success_payload, success_indices),
            ("broad", broad_payload, broad_indices),
        ):
            if indices.size:
                selected_parts.append(
                    {
                        "observations": payload["observations"][indices],
                        "target_means": payload["target_means"][indices],
                        "target_log_stds": payload["target_log_stds"][indices],
                    }
                )
            rows.append(
                {
                    "source_task_index": task_index,
                    "source_task_name": task_name,
                    "memory_source": source_name,
                    "states": int(indices.size),
                }
            )

    observations = np.concatenate(
        [part["observations"] for part in selected_parts], axis=0
    )
    target_means = np.concatenate(
        [part["target_means"] for part in selected_parts], axis=0
    )
    target_log_stds = np.concatenate(
        [part["target_log_stds"] for part in selected_parts], axis=0
    )
    permutation = rng.permutation(observations.shape[0])
    return (
        observations[permutation].copy(),
        target_means[permutation].copy(),
        target_log_stds[permutation].copy(),
        rows,
    )


class GradientClusterRoutedSampler:
    """Cycle uniformly over task-gradient clusters, one cluster per BC update."""

    def __init__(
        self,
        *,
        candidates_by_group: dict[tuple[int, int], np.ndarray],
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Routed sampler batch_size must be positive.")
        if not candidates_by_group:
            raise ValueError("Routed sampler requires at least one cluster.")
        self.candidates_by_group = {
            group: np.asarray(indices, dtype=np.int64)
            for group, indices in candidates_by_group.items()
        }
        for group, candidates in self.candidates_by_group.items():
            if candidates.size < batch_size:
                raise ValueError(
                    f"Gradient cluster {group} has {candidates.size} states, "
                    f"but BC batch size is {batch_size}."
                )
        self.groups = tuple(sorted(self.candidates_by_group))
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self._pending_groups: list[tuple[int, int]] = []
        self.draw_counts = {group: 0 for group in self.groups}

    def sample_indices(self) -> tuple[np.ndarray, tuple[int, int]]:
        if not self._pending_groups:
            permutation = self.rng.permutation(len(self.groups))
            self._pending_groups = [self.groups[int(index)] for index in permutation]
        group = self._pending_groups.pop()
        candidates = self.candidates_by_group[group]
        indices = self.rng.choice(
            candidates,
            size=self.batch_size,
            replace=False,
        )
        self.draw_counts[group] += 1
        return indices, group


def configure_gradient_cluster_routed_sampling(
    *,
    agent: FullBehaviorCloningSACAgent,
    clusters_per_task: int,
    projection_dim: int,
    kmeans_iterations: int,
    seed: int,
) -> tuple[GradientClusterRoutedSampler, list[dict[str, Any]]]:
    if agent.reference_state_count == 0:
        raise ValueError("Gradient-cluster routing requires non-empty BC memory.")
    observations = agent._episodic_observations.numpy()
    target_means = agent._episodic_target_means.numpy()
    target_log_stds = agent._episodic_target_log_stds.numpy()
    source_indices = agent._episodic_source_task_indices.numpy()
    actor_digest = state_dict_digest(agent.actor)
    signatures = extract_bc_feature_gradients(
        agent=agent,
        observations=observations,
        target_means=target_means,
        target_log_stds=target_log_stds,
    )
    rng = np.random.default_rng(seed + 830_000)
    projected = project_and_standardize_features(
        signatures,
        projection_dim=projection_dim,
        rng=rng,
    )
    candidates_by_group: dict[tuple[int, int], np.ndarray] = {}
    cluster_rows: list[dict[str, Any]] = []
    for task_index in sorted(np.unique(source_indices).astype(int).tolist()):
        positions = np.flatnonzero(source_indices == task_index)
        labels, _ = kmeans_labels(
            projected[positions],
            clusters=clusters_per_task,
            iterations=kmeans_iterations,
            rng=rng,
        )
        for cluster_id in range(clusters_per_task):
            candidates = positions[labels == cluster_id]
            candidates_by_group[(task_index, cluster_id)] = candidates
            cluster_rows.append(
                {
                    "source_task_index": task_index,
                    "cluster_id": cluster_id,
                    "states": int(candidates.size),
                    "state_fraction_within_task": float(candidates.size / positions.size),
                }
            )
    after_digest = state_dict_digest(agent.actor)
    if after_digest != actor_digest:
        raise RuntimeError("Gradient clustering unexpectedly changed actor parameters.")

    sampler = GradientClusterRoutedSampler(
        candidates_by_group=candidates_by_group,
        batch_size=agent.episodic_batch_size,
        seed=seed + 840_000,
    )

    def sample_routed_batch(
        self: FullBehaviorCloningSACAgent,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, _group = sampler.sample_indices()
        return (
            self._episodic_observations[indices].to(self.device),
            self._episodic_target_means[indices].to(self.device),
            self._episodic_target_log_stds[indices].to(self.device),
        )

    agent._sample_episodic_batch = MethodType(sample_routed_batch, agent)
    return sampler, cluster_rows


def write_gradient_diagnostics(
    *,
    agent: FullBehaviorCloningSACAgent,
    output_dir: Path,
) -> dict[str, float | int | bool]:
    diagnostic_dir = output_dir / "gradient_diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    rows_by_category = agent.drain_gradient_diagnostics()
    for category, rows in rows_by_category.items():
        if rows:
            write_csv(
                diagnostic_dir / f"{category}.csv",
                rows[0].keys(),
                rows,
            )
    summary = agent.gradient_diagnostics_summary()
    write_json(diagnostic_dir / "summary.json", summary)
    return summary


def metric(metrics: dict[str, float] | None, name: str) -> float:
    if metrics is None:
        return float("nan")
    return float(metrics.get(name, float("nan")))


def evaluate_source_task_retention(
    *,
    agent: FullBehaviorCloningSACAgent,
    tasks: list[str],
    source_task_count: int,
    env_version: str,
    reward_function_version: str,
    episodes: int,
    max_episode_steps: int,
    append_task_id: bool,
    seed: int,
) -> list[dict[str, float | int | str]]:
    """Evaluate old tasks without perturbing the subsequent training RNG stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    rows: list[dict[str, float | int | str]] = []
    try:
        for task_index in range(source_task_count):
            result = evaluate_task(
                agent=agent,
                task_name=tasks[task_index],
                env_version=env_version,
                reward_function_version=reward_function_version,
                episodes=episodes,
                deterministic=False,
                max_episode_steps=max_episode_steps,
                seed=seed + 1_000 * task_index,
                append_task_id=append_task_id,
            )
            rows.append(
                {
                    "source_task_index": task_index,
                    "source_task_name": tasks[task_index],
                    "success_rate": result["success_rate"],
                    "average_return": result["average_return"],
                }
            )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
    return rows


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.eval_every <= 0:
        raise ValueError("--steps and --eval-every must be positive.")
    if args.steps % args.eval_every != 0:
        raise ValueError("--steps must be divisible by --eval-every.")
    if args.start_steps < 0 or args.update_after < 0 or args.update_every <= 0:
        raise ValueError("Invalid SAC update schedule.")
    if args.bc_update_interval <= 0:
        raise ValueError("--bc-update-interval must be positive.")
    if args.retention_eval_episodes <= 0:
        raise ValueError("--retention-eval-episodes must be positive.")
    if args.critic_ensemble_size < 2:
        raise ValueError("--critic-ensemble-size must be at least two.")
    if (
        not math.isfinite(args.critic_bootstrap_probability)
        or not 0.0 < args.critic_bootstrap_probability <= 1.0
    ):
        raise ValueError("--critic-bootstrap-probability must be in (0, 1].")
    if not math.isfinite(args.optimistic_ucb_beta) or args.optimistic_ucb_beta < 0.0:
        raise ValueError("--optimistic-ucb-beta must be finite and non-negative.")
    if args.optimistic_action_candidates <= 0 or args.n_step_return <= 0:
        raise ValueError("Optimistic candidates and n-step return must be positive.")
    if (
        not math.isfinite(args.prioritized_replay_fraction)
        or not 0.0 <= args.prioritized_replay_fraction <= 1.0
    ):
        raise ValueError("--prioritized-replay-fraction must be in [0, 1].")
    if not math.isfinite(args.priority_alpha) or args.priority_alpha < 0.0:
        raise ValueError("--priority-alpha must be finite and non-negative.")
    if (
        not math.isfinite(args.priority_beta_start)
        or not math.isfinite(args.priority_beta_end)
        or not 0.0 <= args.priority_beta_start <= args.priority_beta_end <= 1.0
    ):
        raise ValueError("PER betas must satisfy 0 <= start <= end <= 1.")
    if not math.isfinite(args.priority_epsilon) or args.priority_epsilon <= 0.0:
        raise ValueError("--priority-epsilon must be finite and positive.")
    if not math.isfinite(args.kl_budget_low) or args.kl_budget_low < 0.0:
        raise ValueError("--kl-budget-low must be finite and non-negative.")
    if (
        not math.isfinite(args.kl_budget_high)
        or args.kl_budget_high <= args.kl_budget_low
    ):
        raise ValueError("--kl-budget-high must be finite and exceed --kl-budget-low.")
    if (
        not math.isfinite(args.kl_budget_ema_beta)
        or not 0.0 <= args.kl_budget_ema_beta < 1.0
    ):
        raise ValueError("--kl-budget-ema-beta must be finite and in [0, 1).")
    if args.guide_eval_episodes <= 0:
        raise ValueError("--guide-eval-episodes must be positive.")
    if not 0.0 <= args.guide_stage_tolerance < 1.0:
        raise ValueError("--guide-stage-tolerance must be in [0, 1).")
    if args.guide_mode != "none":
        if args.integration == "layerwise_adaptive_pcgrad":
            raise ValueError(
                "Layer-wise PCGrad validation does not support a rollout guide; "
                "use --guide-mode none to preserve the controlled comparison."
            )
        if args.integration == "optimistic_nstep_pcgrad":
            raise ValueError(
                "Optimistic n-step validation must use --guide-mode none so "
                "the discovery mechanism is isolated from rollout guidance."
            )
        if args.integration == "kl_budget_adaptive_pcgrad":
            raise ValueError(
                "KL-budget validation must use --guide-mode none so the "
                "controller is isolated from rollout guidance."
            )
        if not 0 < args.guide_initial_steps < DEFAULT_EPISODE_LENGTH:
            raise ValueError("--guide-initial-steps must be within the episode horizon.")
        if args.guide_fixed_env_steps <= 0 or args.guide_max_env_steps <= 0:
            raise ValueError("Guide duration limits must be positive.")
        if args.guide_fixed_env_steps % args.eval_every != 0:
            raise ValueError("--guide-fixed-env-steps must be divisible by --eval-every.")
        if args.guide_max_env_steps % args.eval_every != 0:
            raise ValueError("--guide-max-env-steps must be divisible by --eval-every.")
    curriculum_horizons = tuple(args.guide_curriculum_horizons)
    if args.guide_mode == "fast_curriculum":
        if len(curriculum_horizons) < 2 or curriculum_horizons[-1] != 0:
            raise ValueError("Fast curriculum horizons must end in 0.")
        if any(
            left <= right
            for left, right in zip(curriculum_horizons, curriculum_horizons[1:])
        ):
            raise ValueError("Fast curriculum horizons must be strictly decreasing.")
        if curriculum_horizons[0] >= DEFAULT_EPISODE_LENGTH:
            raise ValueError("Guide horizon must leave learner-controlled episode steps.")
    if args.clusters_per_task <= 1:
        raise ValueError("--clusters-per-task must exceed one.")
    if args.cluster_projection_dim <= 0 or args.cluster_kmeans_iterations <= 0:
        raise ValueError("Gradient-cluster parameters must be positive.")
    if args.memory_sampler == "gradient_cluster_routed":
        if args.memory_mode != "success":
            raise ValueError(
                "Gradient-cluster routing is defined only for successful-state memory."
            )
        if args.integration != "adaptive_pcgrad":
            raise ValueError(
                "This validation isolates routing under adaptive PCGrad integration."
            )

    seed_global_rngs(args.seed)
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = (
        Path(args.memory_manifest).expanduser().resolve()
        if args.memory_manifest is not None
        else None
    )
    config = load_json(run_dir / "config.json")
    tasks = list(config["tasks"])
    if not 0 < args.new_task_index < len(tasks):
        raise ValueError("--new-task-index must identify a non-first task.")
    if tasks[args.new_task_index] != "stick-pull-v3":
        raise ValueError("This causal runner is intentionally restricted to stick-pull-v3.")

    method = get_method(str(config["method"]))
    if method.method_id != "success_replay_best_adaptive_pcgrad":
        raise ValueError(
            "Use a success_replay_best_adaptive_pcgrad run so every condition "
            "starts from the same method checkpoint."
        )
    output_dir = make_output_dir(args, run_dir)
    agent_config = dict(config)
    if args.guide_mode != "none":
        # The demonstration-guided subclass is parameter-compatible with the
        # source checkpoint and only adds frozen rollout-guide storage.
        agent_config["method"] = "success_replay_best_jumpstart_adaptive_pcgrad"
    elif args.integration == "kl_budget_adaptive_pcgrad":
        agent_config.update(
            {
                "method": "success_replay_best_kl_budget_pcgrad",
                "kl_budget_low": args.kl_budget_low,
                "kl_budget_high": args.kl_budget_high,
                "kl_budget_ema_beta": args.kl_budget_ema_beta,
            }
        )
    elif args.integration == "layerwise_adaptive_pcgrad":
        agent_config["method"] = "success_replay_best_layerwise_adaptive_pcgrad"
    previous_task_index = args.new_task_index - 1
    source_checkpoint = run_dir / "checkpoints" / f"task_{previous_task_index}.pt"
    if args.integration == "optimistic_nstep_pcgrad":
        base_agent = build_agent_from_config(config=config, device=args.device)
        load_sac_checkpoint(
            agent=base_agent,
            path=source_checkpoint,
            map_location=args.device,
            load_optimizer=False,
        )
        agent_config.update(
            {
                "method": "success_replay_mixed80_optimistic_nstep_pcgrad",
                "critic_ensemble_size": args.critic_ensemble_size,
                "critic_bootstrap_probability": args.critic_bootstrap_probability,
                "optimistic_ucb_beta": args.optimistic_ucb_beta,
                "optimistic_action_candidates": args.optimistic_action_candidates,
            }
        )
        # Constructing the larger module initializes temporary parameters that
        # are immediately overwritten by the source checkpoint. Preserve the
        # global RNG stream so this upgrade does not change matched training
        # actions or actor-noise samples in zero-component controls.
        upgrade_rng_state = preserve_global_rngs()
        try:
            agent = build_agent_from_config(config=agent_config, device=args.device)
            if not isinstance(agent, OptimisticEnsembleFullBCAgent):
                raise RuntimeError("Optimistic integration built the wrong agent class.")
            agent.initialize_from_base_agent(base_agent)
        finally:
            restore_global_rngs(upgrade_rng_state)
        del base_agent
    else:
        agent = build_agent_from_config(config=agent_config, device=args.device)
        load_sac_checkpoint(
            agent=agent,
            path=source_checkpoint,
            map_location=args.device,
            load_optimizer=False,
        )
    if args.integration == "layerwise_adaptive_pcgrad" and not isinstance(
        agent, LayerwiseAdaptivePCGradAgent
    ):
        raise RuntimeError("Layer-wise integration built the wrong agent class.")
    configure_integration(
        agent,
        integration=args.integration,
        bc_update_interval=args.bc_update_interval,
        adaptive_target_ratio=args.bc_adaptive_target_ratio,
        adaptive_conflict_ratio=args.bc_adaptive_conflict_ratio,
    )

    if args.integration == "optimistic_nstep_pcgrad":
        replay_buffer = PrioritizedNStepReplayBuffer(
            observation_dim=agent.observation_dim,
            action_dim=agent.action_dim,
            capacity=int(config.get("replay_size", 1_000_000)),
            seed=args.seed,
            gamma=float(config.get("gamma", 0.99)),
            n_step=args.n_step_return,
            prioritized_fraction=args.prioritized_replay_fraction,
            priority_alpha=args.priority_alpha,
            priority_beta_start=args.priority_beta_start,
            priority_beta_end=args.priority_beta_end,
            priority_beta_steps=args.steps,
            priority_epsilon=args.priority_epsilon,
        )
    else:
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
    reference_states, memory_manifest, mixed_memory_rows = load_memory(
        agent=agent,
        manifest_path=manifest_path,
        memory_mode=args.memory_mode,
        expected_source_tasks=tasks[: args.new_task_index],
        seed=args.seed,
    )
    if args.memory_mode == "none" and reference_states != 0:
        raise RuntimeError("No-memory control unexpectedly contains reference states.")
    if args.memory_mode != "none" and reference_states == 0:
        raise RuntimeError("Memory condition contains no reference states.")
    if mixed_memory_rows:
        write_csv(
            output_dir / "mixed_memory_composition.csv",
            ("source_task_index", "source_task_name", "memory_source", "states"),
            mixed_memory_rows,
        )

    routed_sampler: GradientClusterRoutedSampler | None = None
    cluster_rows: list[dict[str, Any]] = []
    if args.memory_sampler == "gradient_cluster_routed":
        routed_sampler, cluster_rows = configure_gradient_cluster_routed_sampling(
            agent=agent,
            clusters_per_task=args.clusters_per_task,
            projection_dim=args.cluster_projection_dim,
            kmeans_iterations=args.cluster_kmeans_iterations,
            seed=args.seed,
        )
        task_names_by_index = {index: name for index, name in enumerate(tasks)}
        for row in cluster_rows:
            row["source_task_name"] = task_names_by_index[
                int(row["source_task_index"])
            ]
        write_csv(
            output_dir / "memory_gradient_clusters.csv",
            (
                "source_task_index",
                "source_task_name",
                "cluster_id",
                "states",
                "state_fraction_within_task",
            ),
            cluster_rows,
        )

    task_name = tasks[args.new_task_index]
    max_episode_steps = int(config.get("max_episode_steps", DEFAULT_EPISODE_LENGTH))
    env = make_cw_env(
        task_name,
        seed=args.seed + args.new_task_index,
        max_episode_steps=max_episode_steps,
        append_task_id=bool(method.append_task_id),
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
        num_task_ids=len(tasks),
    )
    guide_eval_env = None
    selected_guide_index: int | None = None
    selected_guide_return: float | None = None
    guide_events: list[dict[str, float | int | str]] = []
    curriculum_stage = 0
    initial_guide_steps = 0
    if args.guide_mode != "none":
        guide_eval_env = make_cw_env(
            task_name,
            seed=args.seed + 40_000,
            max_episode_steps=max_episode_steps,
            append_task_id=bool(method.append_task_id),
            env_version=str(config["env_version"]),
            reward_function_version=str(config["reward_function_version"]),
            num_task_ids=len(tasks),
        )
        selected_guide_index, selected_guide_return = configure_frozen_guides(
            agent=agent,
            run_dir=run_dir,
            task_index=args.new_task_index,
            env=guide_eval_env,
            episodes=args.guide_eval_episodes,
            max_episode_steps=max_episode_steps,
            seed=args.seed + 60_000,
        )
        initial_guide_steps = (
            args.guide_initial_steps
            if args.guide_mode == "fixed_short"
            else curriculum_horizons[0]
        )
        guide_events.append(
            {
                "environment_step": 0,
                "event": "guide_selected",
                "from_guide_steps": 0,
                "to_guide_steps": initial_guide_steps,
                "mixed_average_return": selected_guide_return,
                "reference_return": selected_guide_return,
                "reason": f"source_task_{selected_guide_index}",
            }
        )
    trainer = SACTrainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        config=SACTrainerConfig(
            total_steps=args.steps,
            batch_size=args.batch_size,
            start_steps=args.start_steps,
            exploration_strategy=("best_return" if args.guide_mode == "none" else None),
            exploration_available_heads=(
                args.new_task_index + 1 if args.guide_mode == "none" else None
            ),
            update_after=args.update_after,
            update_every=args.update_every,
            max_episode_steps=max_episode_steps,
            callback_every_steps=args.eval_every,
            seed=args.seed + args.new_task_index,
            reseed_global_rng=False,
            guide_head_index=selected_guide_index,
            guide_steps=initial_guide_steps,
        ),
    )

    run_config = {
        "source_run_dir": str(run_dir),
        "source_checkpoint": str(
            run_dir / "checkpoints" / f"task_{previous_task_index}.pt"
        ),
        "memory_manifest": str(manifest_path) if manifest_path is not None else None,
        "memory_manifest_schema_version": memory_manifest["schema_version"],
        "memory_mode": args.memory_mode,
        "memory_sampler": args.memory_sampler,
        "integration": args.integration,
        "new_task_index": args.new_task_index,
        "new_task_name": task_name,
        "reference_states": reference_states,
        "retention_eval_episodes": args.retention_eval_episodes,
        "actor_cloning_coefficient": agent.actor_cloning_coefficient,
        "bc_gradient_strategy": agent.bc_gradient_strategy,
        "bc_combination_strategy": agent.bc_combination_strategy,
        "bc_adaptive_target_ratio": agent.bc_adaptive_target_ratio,
        "bc_adaptive_conflict_ratio": agent.bc_adaptive_conflict_ratio,
        "bc_update_interval": args.bc_update_interval,
        "discovery_critic": {
            "enabled": args.integration == "optimistic_nstep_pcgrad",
            "ensemble_size": args.critic_ensemble_size,
            "bootstrap_probability": args.critic_bootstrap_probability,
            "ucb_beta": args.optimistic_ucb_beta,
            "action_candidates": args.optimistic_action_candidates,
        },
        "current_replay": {
            "n_step": args.n_step_return,
            "prioritized_fraction": args.prioritized_replay_fraction,
            "priority_alpha": args.priority_alpha,
            "priority_beta_start": args.priority_beta_start,
            "priority_beta_end": args.priority_beta_end,
            "priority_beta_steps": args.steps,
            "priority_epsilon": args.priority_epsilon,
        },
        "kl_budget": {
            "enabled": args.integration == "kl_budget_adaptive_pcgrad",
            "low": args.kl_budget_low,
            "high": args.kl_budget_high,
            "ema_beta": args.kl_budget_ema_beta,
        },
        "guide": {
            "mode": args.guide_mode,
            "selected_source_task_index": selected_guide_index,
            "selected_source_task_name": (
                tasks[selected_guide_index] if selected_guide_index is not None else None
            ),
            "selected_source_return": selected_guide_return,
            "initial_steps": initial_guide_steps,
            "fixed_env_steps": args.guide_fixed_env_steps,
            "max_env_steps": args.guide_max_env_steps,
            "curriculum_horizons": list(curriculum_horizons),
            "stage_tolerance": args.guide_stage_tolerance,
            "evaluation_episodes": args.guide_eval_episodes,
            "self_guide_replacement": False,
        },
        "gradient_cluster_routing": {
            "enabled": routed_sampler is not None,
            "clusters_per_task": args.clusters_per_task,
            "projection_dim": args.cluster_projection_dim,
            "kmeans_iterations": args.cluster_kmeans_iterations,
            "groups": len(routed_sampler.groups) if routed_sampler is not None else 0,
            "minimum_group_states": (
                min(len(values) for values in routed_sampler.candidates_by_group.values())
                if routed_sampler is not None
                else 0
            ),
        },
        "exploration_strategy": (
            "best_return" if args.guide_mode == "none" else "frozen_policy_guide"
        ),
        "exploration_available_heads": (
            args.new_task_index + 1 if args.guide_mode == "none" else None
        ),
        "steps": args.steps,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "device": args.device,
        "environment": {
            "env_version": config["env_version"],
            "reward_function_version": config["reward_function_version"],
        },
    }
    write_json(output_dir / "config.json", run_config)

    retention_before = evaluate_source_task_retention(
        agent=agent,
        tasks=tasks,
        source_task_count=args.new_task_index,
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
        episodes=args.retention_eval_episodes,
        max_episode_steps=max_episode_steps,
        append_task_id=bool(method.append_task_id),
        seed=args.seed + 700_000,
    )

    started_at = time.time()
    eval_rows: list[dict[str, float | int]] = []
    best_success = float("-inf")
    best_return = float("-inf")

    def evaluate(
        step: int,
        gradient_updates: int,
        update_metrics: dict[str, float] | None,
    ) -> None:
        nonlocal best_success, best_return, curriculum_stage
        applied_guide_steps = trainer.guide_steps
        result = evaluate_task(
            agent=agent,
            task_name=task_name,
            env_version=str(config["env_version"]),
            reward_function_version=str(config["reward_function_version"]),
            episodes=args.stoch_eval_episodes,
            deterministic=False,
            max_episode_steps=max_episode_steps,
            seed=args.seed + 50_000 + step,
            append_task_id=bool(method.append_task_id),
        )
        best_success = max(best_success, result["success_rate"])
        best_return = max(best_return, result["average_return"])
        eval_rows.append(
            {
                "environment_step": step,
                "gradient_updates": gradient_updates,
                "stochastic_average_return": result["average_return"],
                "stochastic_success_rate": result["success_rate"],
                "stochastic_average_episode_length": result["average_episode_length"],
                "actor_loss": metric(update_metrics, "actor_loss"),
                "q1_loss": metric(update_metrics, "q1_loss"),
                "q2_loss": metric(update_metrics, "q2_loss"),
                "q1_mean": metric(update_metrics, "q1_mean"),
                "q2_mean": metric(update_metrics, "q2_mean"),
                "q_target_mean": metric(update_metrics, "q_target_mean"),
                "ensemble_critic_loss": metric(update_metrics, "ensemble_critic_loss"),
                "q_ensemble_std_mean": metric(update_metrics, "q_ensemble_std_mean"),
                "td_error_abs_mean": metric(update_metrics, "td_error_abs_mean"),
                "priority_beta": metric(update_metrics, "priority_beta"),
                "importance_weight_mean": metric(update_metrics, "importance_weight_mean"),
                "alpha": agent.diagnostic_alpha_value(args.new_task_index),
                "reference_states": reference_states,
                "guide_mode": args.guide_mode,
                "guide_steps": applied_guide_steps,
                "elapsed_seconds": time.time() - started_at,
            }
        )
        append_csv_rows(
            output_dir / "evaluations_live.csv",
            EVAL_FIELDS,
            [eval_rows[-1]],
        )
        print(
            f"[stickpull-causal] memory={args.memory_mode} "
            f"sampler={args.memory_sampler} integration={args.integration} "
            f"guide={args.guide_mode}:{applied_guide_steps} "
            f"step={step:,}/{args.steps:,} "
            f"success={result['success_rate']:.3f} "
            f"return={result['average_return']:.3f}"
            + (
                f" kl_excess={agent.kl_budget_excess:.4f} "
                f"kl_multiplier={agent.kl_budget_multiplier:.3f}"
                if args.integration == "kl_budget_adaptive_pcgrad"
                else ""
            ),
            flush=True,
        )

        if args.guide_mode == "fixed_short" and step >= args.guide_fixed_env_steps:
            if trainer.guide_steps > 0:
                previous_steps = trainer.guide_steps
                trainer.set_guide_steps(0)
                guide_events.append(
                    {
                        "environment_step": step,
                        "event": "horizon_advanced",
                        "from_guide_steps": previous_steps,
                        "to_guide_steps": 0,
                        "mixed_average_return": float("nan"),
                        "reference_return": selected_guide_return,
                        "reason": "fixed_short_cutoff",
                    }
                )
        elif args.guide_mode == "fast_curriculum" and trainer.guide_steps > 0:
            if guide_eval_env is None or selected_guide_index is None:
                raise RuntimeError("Fast curriculum is missing its frozen guide.")
            mixed_return = evaluate_frozen_guide_return(
                agent=agent,
                env=guide_eval_env,
                guide_task_index=selected_guide_index,
                episodes=args.guide_eval_episodes,
                max_episode_steps=max_episode_steps,
                seed=args.seed + 80_000 + step,
                guide_steps=trainer.guide_steps,
            )
            reference = float(selected_guide_return)
            threshold = reference - args.guide_stage_tolerance * abs(reference)
            passed = mixed_return >= threshold
            forced = step >= args.guide_max_env_steps
            if passed or forced:
                previous_steps = trainer.guide_steps
                if forced:
                    curriculum_stage = len(curriculum_horizons) - 1
                else:
                    curriculum_stage = min(
                        curriculum_stage + 1,
                        len(curriculum_horizons) - 1,
                    )
                next_steps = curriculum_horizons[curriculum_stage]
                trainer.set_guide_steps(next_steps)
                guide_events.append(
                    {
                        "environment_step": step,
                        "event": "horizon_advanced",
                        "from_guide_steps": previous_steps,
                        "to_guide_steps": next_steps,
                        "mixed_average_return": mixed_return,
                        "reference_return": reference,
                        "reason": "hard_cutoff" if forced else "performance",
                    }
                )
                print(
                    f"[stickpull-guide] step={step:,} h={previous_steps}->{next_steps} "
                    f"mixed_return={mixed_return:.3f} reference={reference:.3f} "
                    f"reason={'hard_cutoff' if forced else 'performance'}",
                    flush=True,
                )

    try:
        training_summary = trainer.train(step_callback=evaluate)
    finally:
        env.close()
        if guide_eval_env is not None:
            guide_eval_env.close()

    if not eval_rows:
        raise RuntimeError("Training completed without evaluation rows.")
    save_sac_checkpoint(
        agent=agent,
        path=output_dir / "checkpoints" / "final.pt",
        environment_step=args.steps,
        metadata={
            "task_name": task_name,
            "memory_mode": args.memory_mode,
            "memory_sampler": args.memory_sampler,
            "integration": args.integration,
        },
    )
    write_csv(output_dir / "evaluations.csv", EVAL_FIELDS, eval_rows)
    if guide_events:
        write_csv(output_dir / "guide_events.csv", GUIDE_EVENT_FIELDS, guide_events)
    gradient_summary = write_gradient_diagnostics(agent=agent, output_dir=output_dir)

    retention_after = evaluate_source_task_retention(
        agent=agent,
        tasks=tasks,
        source_task_count=args.new_task_index,
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
        episodes=args.retention_eval_episodes,
        max_episode_steps=max_episode_steps,
        append_task_id=bool(method.append_task_id),
        seed=args.seed + 700_000,
    )
    retention_rows = []
    for before, after in zip(retention_before, retention_after, strict=True):
        retention_rows.append(
            {
                "source_task_index": before["source_task_index"],
                "source_task_name": before["source_task_name"],
                "before_success": before["success_rate"],
                "after_success": after["success_rate"],
                "forgetting": float(before["success_rate"]) - float(after["success_rate"]),
                "before_return": before["average_return"],
                "after_return": after["average_return"],
            }
        )
    write_csv(
        output_dir / "source_task_retention.csv",
        (
            "source_task_index",
            "source_task_name",
            "before_success",
            "after_success",
            "forgetting",
            "before_return",
            "after_return",
        ),
        retention_rows,
    )

    routing_summary: dict[str, Any] = {"enabled": routed_sampler is not None}
    if routed_sampler is not None:
        routing_rows = [
            {
                "source_task_index": group[0],
                "cluster_id": group[1],
                "bc_updates": routed_sampler.draw_counts[group],
            }
            for group in routed_sampler.groups
        ]
        write_csv(
            output_dir / "memory_gradient_cluster_usage.csv",
            ("source_task_index", "cluster_id", "bc_updates"),
            routing_rows,
        )
        draw_counts = [int(row["bc_updates"]) for row in routing_rows]
        if max(draw_counts) - min(draw_counts) > 1:
            raise RuntimeError("Routed cluster usage is not uniformly cycled.")
        routing_summary = {
            "enabled": True,
            "groups": len(routing_rows),
            "total_bc_updates": sum(draw_counts),
            "minimum_updates_per_group": min(draw_counts),
            "maximum_updates_per_group": max(draw_counts),
        }

    successes = [float(row["stochastic_success_rate"]) for row in eval_rows]
    returns = [float(row["stochastic_average_return"]) for row in eval_rows]
    first_nonzero_step = next(
        (
            int(row["environment_step"])
            for row in eval_rows
            if float(row["stochastic_success_rate"]) > 0.0
        ),
        None,
    )
    tail_size = min(5, len(successes))
    diagnostic_bc_updates = int(gradient_summary.get("bc_updates_seen", 0))
    bc_updates_applied = int(
        getattr(agent, "bc_updates_applied", diagnostic_bc_updates)
    )
    bc_update_opportunities = int(
        getattr(agent, "bc_update_opportunities", bc_updates_applied)
    )
    summary = {
        "memory_mode": args.memory_mode,
        "memory_sampler": args.memory_sampler,
        "integration": args.integration,
        "guide_mode": args.guide_mode,
        "selected_guide_task_index": selected_guide_index,
        "selected_guide_return": selected_guide_return,
        "final_guide_steps": trainer.guide_steps,
        "new_task_name": task_name,
        "reference_states": reference_states,
        "learning_curve_mean_success": float(np.mean(successes)),
        "learning_curve_mean_return": float(np.mean(returns)),
        "tail5_success": float(np.mean(successes[-tail_size:])),
        "final_success": successes[-1],
        "final_return": returns[-1],
        "best_success": best_success,
        "best_return": best_return,
        "first_nonzero_success_step": first_nonzero_step,
        "completed_episodes": training_summary.completed_episodes,
        "gradient_updates": training_summary.gradient_updates,
        "mean_training_return": training_summary.mean_episode_return,
        "elapsed_seconds": time.time() - started_at,
        "gradient_diagnostics": gradient_summary,
        "bc_update_opportunities": bc_update_opportunities,
        "bc_updates_applied": bc_updates_applied,
        "bc_realized_update_fraction": (
            bc_updates_applied / bc_update_opportunities
            if bc_update_opportunities
            else 0.0
        ),
        "gradient_cluster_routing": routing_summary,
        "method_diagnostics": agent.task_diagnostics(),
        "source_task_retention": {
            "episodes_per_task": args.retention_eval_episodes,
            "mean_success_before": float(
                np.mean([float(row["before_success"]) for row in retention_rows])
            ),
            "mean_success_after": float(
                np.mean([float(row["after_success"]) for row in retention_rows])
            ),
            "mean_forgetting": float(
                np.mean([float(row["forgetting"]) for row in retention_rows])
            ),
        },
        "run_directory": str(output_dir),
    }
    if not all(
        math.isfinite(float(summary[key]))
        for key in (
            "learning_curve_mean_success",
            "learning_curve_mean_return",
            "tail5_success",
            "final_success",
            "final_return",
        )
    ):
        raise RuntimeError("Non-finite primary metrics detected.")
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
