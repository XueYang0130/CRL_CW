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

from agents import FullBehaviorCloningSACAgent, ReplayBuffer
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
    "alpha",
    "reference_states",
    "elapsed_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled stick-pull acquisition ablation from one fixed "
            "task-3 checkpoint."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--memory-manifest", required=True)
    parser.add_argument("--new-task-index", type=int, default=4)
    parser.add_argument(
        "--memory-mode",
        choices=("success", "broad", "mixed50", "mixed80_dynamic", "none"),
        required=True,
    )
    parser.add_argument(
        "--integration",
        choices=("adaptive_pcgrad", "plain"),
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
    if integration == "adaptive_pcgrad":
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
    agent.configure_bc_update_interval(bc_update_interval)


def load_memory(
    *,
    agent: FullBehaviorCloningSACAgent,
    manifest_path: Path,
    memory_mode: str,
    expected_source_tasks: list[str],
    seed: int,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
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
    manifest_path = Path(args.memory_manifest).expanduser().resolve()
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
    agent = build_agent_from_config(config=config, device=args.device)
    previous_task_index = args.new_task_index - 1
    load_sac_checkpoint(
        agent=agent,
        path=run_dir / "checkpoints" / f"task_{previous_task_index}.pt",
        map_location=args.device,
        load_optimizer=False,
    )
    configure_integration(
        agent,
        integration=args.integration,
        bc_update_interval=args.bc_update_interval,
        adaptive_target_ratio=args.bc_adaptive_target_ratio,
        adaptive_conflict_ratio=args.bc_adaptive_conflict_ratio,
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
    trainer = SACTrainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        config=SACTrainerConfig(
            total_steps=args.steps,
            batch_size=args.batch_size,
            start_steps=args.start_steps,
            exploration_strategy="best_return",
            exploration_available_heads=args.new_task_index + 1,
            update_after=args.update_after,
            update_every=args.update_every,
            max_episode_steps=max_episode_steps,
            callback_every_steps=args.eval_every,
            seed=args.seed + args.new_task_index,
            reseed_global_rng=False,
        ),
    )

    run_config = {
        "source_run_dir": str(run_dir),
        "source_checkpoint": str(
            run_dir / "checkpoints" / f"task_{previous_task_index}.pt"
        ),
        "memory_manifest": str(manifest_path),
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
        "bc_update_interval": agent.bc_update_interval,
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
        "exploration_strategy": "best_return",
        "exploration_available_heads": args.new_task_index + 1,
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
        nonlocal best_success, best_return
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
                "alpha": agent.diagnostic_alpha_value(args.new_task_index),
                "reference_states": reference_states,
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
            f"step={step:,}/{args.steps:,} "
            f"success={result['success_rate']:.3f} "
            f"return={result['average_return']:.3f}",
            flush=True,
        )

    try:
        training_summary = trainer.train(step_callback=evaluate)
    finally:
        env.close()

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
    summary = {
        "memory_mode": args.memory_mode,
        "memory_sampler": args.memory_sampler,
        "integration": args.integration,
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
        "bc_update_opportunities": agent.bc_update_opportunities,
        "bc_updates_applied": agent.bc_updates_applied,
        "bc_realized_update_fraction": (
            agent.bc_updates_applied / agent.bc_update_opportunities
            if agent.bc_update_opportunities
            else 0.0
        ),
        "gradient_cluster_routing": routing_summary,
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
