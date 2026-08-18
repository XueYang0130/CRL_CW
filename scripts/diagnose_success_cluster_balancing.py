from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_stickpull_matched_memory import build_agent_from_config
from scripts.diagnose_bc_gradient_concentration import (
    compute_bc_gradient_matrix,
    gradient_concentration_metrics,
    infer_source_tasks,
    load_memory_payload,
    sample_microbatch_indices,
    state_dict_digest,
)
from scripts.probe_stickpull_with_memory import load_json
from training.sac_trainer import seed_global_rngs
from utils import load_sac_checkpoint, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether task-wise actor-feature cluster balancing reduces "
            "successful-memory BC gradient concentration."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--memory-manifest", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--clusters-per-task", type=int, default=8)
    parser.add_argument(
        "--cluster-representation",
        choices=("actor_feature", "bc_feature_gradient"),
        default="actor_feature",
    )
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--microbatches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


@torch.inference_mode()
def extract_actor_features(
    *,
    agent: Any,
    observations: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    actor_was_training = bool(agent.actor.training)
    agent.actor.eval()
    features: list[np.ndarray] = []
    try:
        for start in range(0, observations.shape[0], batch_size):
            batch = torch.as_tensor(
                observations[start : start + batch_size],
                dtype=torch.float32,
                device=agent.device,
            )
            if agent.actor.hide_task_id:
                batch = batch[:, : -agent.task_id_dim]
            encoded = agent.actor.backbone(batch)
            features.append(encoded.cpu().numpy().astype(np.float32, copy=False))
    finally:
        agent.actor.train(actor_was_training)
    return np.concatenate(features, axis=0)


def extract_bc_feature_gradients(
    *,
    agent: Any,
    observations: np.ndarray,
    target_means: np.ndarray,
    target_log_stds: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Return each state's normalized BC gradient at the backbone output."""
    if observations.shape[0] != target_means.shape[0]:
        raise ValueError("Observations and target means must have equal length.")
    if observations.shape[0] != target_log_stds.shape[0]:
        raise ValueError("Observations and target log stds must have equal length.")
    actor_was_training = bool(agent.actor.training)
    agent.actor.eval()
    signatures: list[np.ndarray] = []
    captured_features: list[torch.Tensor] = []

    def capture_output(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured_features.append(output)

    hook = agent.actor.backbone.register_forward_hook(capture_output)
    try:
        for start in range(0, observations.shape[0], batch_size):
            stop = start + batch_size
            observation_batch = torch.as_tensor(
                observations[start:stop],
                dtype=torch.float32,
                device=agent.device,
            )
            target_mean_batch = torch.as_tensor(
                target_means[start:stop],
                dtype=torch.float32,
                device=agent.device,
            )
            target_log_std_batch = torch.as_tensor(
                target_log_stds[start:stop],
                dtype=torch.float32,
                device=agent.device,
            )
            captured_features.clear()
            current_means, current_log_stds = agent.actor.distribution_parameters(
                observation_batch
            )
            if len(captured_features) != 1:
                raise RuntimeError("Expected exactly one actor-backbone forward output.")
            features = captured_features[0]
            per_state_loss = agent._gaussian_kl(
                target_mean_batch,
                target_log_std_batch,
                current_means,
                current_log_stds,
            )
            feature_gradients = torch.autograd.grad(
                per_state_loss.sum(),
                features,
            )[0]
            norms = feature_gradients.norm(dim=1, keepdim=True)
            normalized = feature_gradients / norms.clamp_min(1e-12)
            signatures.append(normalized.detach().cpu().numpy().astype(np.float32))
    finally:
        hook.remove()
        agent.actor.train(actor_was_training)
    if any(parameter.grad is not None for parameter in agent.actor.parameters()):
        raise RuntimeError("Feature-gradient extraction unexpectedly populated .grad.")
    return np.concatenate(signatures, axis=0)


def project_and_standardize_features(
    features: np.ndarray,
    *,
    projection_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("Features must be a non-empty rank-2 array.")
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive.")
    output_dim = min(projection_dim, features.shape[1])
    projection = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(output_dim),
        size=(features.shape[1], output_dim),
    )
    projected = features.astype(np.float64) @ projection
    projected -= projected.mean(axis=0, keepdims=True)
    scale = projected.std(axis=0, keepdims=True)
    projected /= np.maximum(scale, 1e-8)
    return projected.astype(np.float32)


def kmeans_labels(
    features: np.ndarray,
    *,
    clusters: int,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] < clusters:
        raise ValueError("K-means requires at least one state per cluster.")
    if clusters <= 1 or iterations <= 0:
        raise ValueError("clusters must exceed one and iterations must be positive.")

    centers = np.empty((clusters, features.shape[1]), dtype=np.float32)
    first = int(rng.integers(features.shape[0]))
    centers[0] = features[first]
    closest_distance = np.square(features - centers[0]).sum(axis=1)
    for cluster in range(1, clusters):
        total = float(closest_distance.sum())
        if total <= 1e-12:
            candidate = int(rng.integers(features.shape[0]))
        else:
            candidate = int(rng.choice(features.shape[0], p=closest_distance / total))
        centers[cluster] = features[candidate]
        distance = np.square(features - centers[cluster]).sum(axis=1)
        closest_distance = np.minimum(closest_distance, distance)

    labels = np.zeros(features.shape[0], dtype=np.int64)
    for _ in range(iterations):
        distances = np.square(features[:, None, :] - centers[None, :, :]).sum(axis=2)
        next_labels = np.argmin(distances, axis=1).astype(np.int64)
        next_centers = centers.copy()
        nearest_distance = distances[np.arange(features.shape[0]), next_labels]
        for cluster in range(clusters):
            members = features[next_labels == cluster]
            if members.shape[0] == 0:
                replacement = int(np.argmax(nearest_distance))
                next_centers[cluster] = features[replacement]
                next_labels[replacement] = cluster
                nearest_distance[replacement] = 0.0
            else:
                next_centers[cluster] = members.mean(axis=0)
        converged = np.array_equal(labels, next_labels)
        labels, centers = next_labels, next_centers
        if converged:
            break
    counts = np.bincount(labels, minlength=clusters)
    if np.any(counts == 0):
        raise RuntimeError("K-means produced an empty cluster.")
    return labels, centers


def sample_balanced_groups(
    *,
    candidates_by_group: dict[tuple[int, int], np.ndarray],
    microbatches: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    groups = sorted(candidates_by_group)
    if not groups:
        raise ValueError("No feature clusters were provided.")
    base, remainder = divmod(batch_size, len(groups))
    if base == 0:
        raise ValueError("Batch size must be at least the number of feature clusters.")
    allocations = {
        group: base + int(position < remainder)
        for position, group in enumerate(groups)
    }
    batches: list[np.ndarray] = []
    for _ in range(microbatches):
        selected = []
        for group in groups:
            candidates = candidates_by_group[group]
            count = allocations[group]
            if candidates.size < count:
                raise ValueError(
                    f"Cluster {group} has {candidates.size} states, need {count}."
                )
            selected.append(rng.choice(candidates, size=count, replace=False))
        batch = np.concatenate(selected)
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def sample_one_group_per_microbatch(
    *,
    candidates_by_group: dict[tuple[int, int], np.ndarray],
    microbatches: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    groups = sorted(candidates_by_group)
    if not groups:
        raise ValueError("No feature clusters were provided.")
    batches: list[np.ndarray] = []
    order: list[tuple[int, int]] = []
    while len(order) < microbatches:
        cycle = list(groups)
        rng.shuffle(cycle)
        order.extend(cycle)
    for group in order[:microbatches]:
        candidates = candidates_by_group[group]
        if candidates.size < batch_size:
            raise ValueError(
                f"Cluster {group} has {candidates.size} states, need {batch_size}."
            )
        batches.append(rng.choice(candidates, size=batch_size, replace=False))
    return batches


def make_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    base = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "diagnostics"
    )
    name = args.run_name or f"success_cluster_balancing_{run_dir.name}_seed{args.seed}"
    output_dir = base / name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main() -> None:
    args = parse_args()
    if args.clusters_per_task <= 1:
        raise ValueError("--clusters-per-task must exceed one.")
    if args.microbatches < 2 or args.batch_size <= 0:
        raise ValueError("Invalid microbatch configuration.")
    seed_global_rngs(args.seed)

    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = Path(args.memory_manifest).expanduser().resolve()
    config = load_json(run_dir / "config.json")
    manifest = load_json(manifest_path)
    source_tasks = list(manifest.get("source_tasks", []))
    if not source_tasks:
        raise ValueError("Manifest contains no source tasks.")
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint is not None
        else run_dir / "checkpoints" / f"task_{len(source_tasks) - 1}.pt"
    )
    output_dir = make_output_dir(args, run_dir)
    agent = build_agent_from_config(config=config, device=args.device)
    load_sac_checkpoint(
        agent=agent,
        path=checkpoint_path,
        map_location=args.device,
        load_optimizer=False,
    )
    actor_digest = state_dict_digest(agent.actor)
    success = load_memory_payload(
        manifest=manifest,
        manifest_path=manifest_path,
        mode="success",
    )
    broad = load_memory_payload(
        manifest=manifest,
        manifest_path=manifest_path,
        mode="broad",
    )
    source_indices = infer_source_tasks(
        success["observations"],
        num_tasks=agent.num_tasks,
    )
    broad_source_indices = infer_source_tasks(
        broad["observations"],
        num_tasks=agent.num_tasks,
    )
    rng = np.random.default_rng(args.seed + 820_000)

    if args.cluster_representation == "actor_feature":
        features = extract_actor_features(
            agent=agent,
            observations=success["observations"],
        )
    else:
        features = extract_bc_feature_gradients(
            agent=agent,
            observations=success["observations"],
            target_means=success["target_means"],
            target_log_stds=success["target_log_stds"],
        )
    projected = project_and_standardize_features(
        features,
        projection_dim=args.projection_dim,
        rng=rng,
    )
    cluster_ids = np.full(success["observations"].shape[0], -1, dtype=np.int64)
    cluster_rows: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(source_tasks):
        task_positions = np.flatnonzero(source_indices == task_index)
        labels, _ = kmeans_labels(
            projected[task_positions],
            clusters=args.clusters_per_task,
            iterations=args.kmeans_iterations,
            rng=rng,
        )
        cluster_ids[task_positions] = labels
        counts = np.bincount(labels, minlength=args.clusters_per_task)
        for cluster_id, count in enumerate(counts.tolist()):
            cluster_rows.append(
                {
                    "source_task_index": task_index,
                    "source_task_name": task_name,
                    "cluster_id": cluster_id,
                    "states": count,
                    "state_fraction": count / task_positions.size,
                }
            )
    if np.any(cluster_ids < 0):
        raise RuntimeError("Some successful states were not assigned to a cluster.")

    task_candidates = {
        task: np.flatnonzero(source_indices == task)
        for task in range(len(source_tasks))
    }
    broad_task_candidates = {
        task: np.flatnonzero(broad_source_indices == task)
        for task in range(len(source_tasks))
    }
    cluster_candidates = {
        (task, cluster): np.flatnonzero(
            (source_indices == task) & (cluster_ids == cluster)
        )
        for task in range(len(source_tasks))
        for cluster in range(args.clusters_per_task)
    }
    conditions = (
        (
            "success_uniform",
            success,
            sample_microbatch_indices(
                candidates_by_task=task_candidates,
                microbatches=args.microbatches,
                batch_size=args.batch_size,
                rng=rng,
            ),
        ),
        (
            "success_cluster_balanced",
            success,
            sample_balanced_groups(
                candidates_by_group=cluster_candidates,
                microbatches=args.microbatches,
                batch_size=args.batch_size,
                rng=rng,
            ),
        ),
        (
            "success_cluster_routed",
            success,
            sample_one_group_per_microbatch(
                candidates_by_group=cluster_candidates,
                microbatches=args.microbatches,
                batch_size=args.batch_size,
                rng=rng,
            ),
        ),
        (
            "broad_uniform",
            broad,
            sample_microbatch_indices(
                candidates_by_task=broad_task_candidates,
                microbatches=args.microbatches,
                batch_size=args.batch_size,
                rng=rng,
            ),
        ),
    )
    metric_rows: list[dict[str, Any]] = []
    for condition, payload, batches in conditions:
        gradients, losses = compute_bc_gradient_matrix(
            agent=agent,
            payload=payload,
            batches=batches,
        )
        metrics = gradient_concentration_metrics(gradients)
        row = {
            "condition": condition,
            "mean_raw_bc_loss": float(losses.mean()),
            "std_raw_bc_loss": float(losses.std(ddof=0)),
            **metrics,
        }
        metric_rows.append(row)
        print(
            f"[cluster-balance] condition={condition} "
            f"cosine={row['mean_pairwise_cosine']:.4f} "
            f"coherence={row['directional_coherence']:.4f} "
            f"effective_rank={row['effective_rank']:.2f}"
        )

    if state_dict_digest(agent.actor) != actor_digest:
        raise RuntimeError("Read-only cluster diagnostic changed actor parameters.")
    write_csv(output_dir / "clusters.csv", cluster_rows[0].keys(), cluster_rows)
    write_csv(output_dir / "concentration_metrics.csv", metric_rows[0].keys(), metric_rows)
    by_condition = {row["condition"]: row for row in metric_rows}
    uniform = by_condition["success_uniform"]
    balanced = by_condition["success_cluster_balanced"]
    routed = by_condition["success_cluster_routed"]
    broad_metrics = by_condition["broad_uniform"]
    summary = {
        "purpose": "successful_memory_feature_cluster_balancing_diagnostic",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "memory_manifest": str(manifest_path),
        "clusters_per_task": args.clusters_per_task,
        "cluster_representation": args.cluster_representation,
        "projection_dim": args.projection_dim,
        "microbatches": args.microbatches,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "actor_unchanged": True,
        "metrics": metric_rows,
        "cluster_balancing_delta": {
            "pairwise_cosine": (
                balanced["mean_pairwise_cosine"] - uniform["mean_pairwise_cosine"]
            ),
            "directional_coherence": (
                balanced["directional_coherence"] - uniform["directional_coherence"]
            ),
            "effective_rank": balanced["effective_rank"] - uniform["effective_rank"],
            "distance_to_broad_effective_rank_before": abs(
                uniform["effective_rank"] - broad_metrics["effective_rank"]
            ),
            "distance_to_broad_effective_rank_after": abs(
                balanced["effective_rank"] - broad_metrics["effective_rank"]
            ),
            "routed_pairwise_cosine_delta": (
                routed["mean_pairwise_cosine"] - uniform["mean_pairwise_cosine"]
            ),
            "routed_directional_coherence_delta": (
                routed["directional_coherence"] - uniform["directional_coherence"]
            ),
            "routed_effective_rank_delta": (
                routed["effective_rank"] - uniform["effective_rank"]
            ),
        },
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "actor_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
