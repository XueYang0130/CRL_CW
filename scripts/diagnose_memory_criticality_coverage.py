from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_stickpull_matched_memory import build_agent_from_config
from scripts.diagnose_bc_gradient_concentration import (
    infer_source_tasks,
    load_memory_payload,
    state_dict_digest,
)
from scripts.diagnose_success_cluster_balancing import extract_bc_feature_gradients
from scripts.probe_stickpull_with_memory import load_json
from scripts.run_stickpull_causal_ablation import (
    build_nonoverlapping_mixed_memory,
    evaluate_source_task_retention,
)
from training.sac_trainer import seed_global_rngs
from utils import load_sac_checkpoint, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure critical-success and broad-coverage geometry of matched "
            "old-task memories without changing or retraining the policy."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--memory-manifest", required=True)
    parser.add_argument("--success-run", required=True)
    parser.add_argument("--broad-run", required=True)
    parser.add_argument("--mixed50-run", required=True)
    parser.add_argument("--subspace-rank", type=int, default=16)
    parser.add_argument("--retention-eval-episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def make_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    base = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "diagnostics"
    )
    name = args.run_name or f"memory_criticality_coverage_{run_dir.name}"
    output_dir = base / name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def covariance_matrix(signatures: np.ndarray) -> np.ndarray:
    if signatures.ndim != 2 or signatures.shape[0] < 2:
        raise ValueError("Gradient signatures must be a rank-2 matrix with two rows.")
    if not np.isfinite(signatures).all():
        raise ValueError("Gradient signatures contain non-finite values.")
    values = signatures.astype(np.float64, copy=False)
    return values.T @ values / values.shape[0]


def eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], a_min=0.0, a_max=None)
    return eigenvalues, eigenvectors[:, order]


def effective_rank(eigenvalues: np.ndarray) -> float:
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    probabilities = eigenvalues[eigenvalues > total * 1e-12] / total
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def normalized_covariance_error(matrix: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference, ord="fro"))
    if denominator <= 0.0:
        raise ValueError("Reference covariance has zero Frobenius norm.")
    return float(np.linalg.norm(matrix - reference, ord="fro") / denominator)


def captured_reference_energy(
    reference_signatures: np.ndarray,
    basis: np.ndarray,
) -> float:
    total = float(np.square(reference_signatures).sum())
    if total <= 0.0:
        raise ValueError("Reference signatures have zero energy.")
    projected = reference_signatures.astype(np.float64, copy=False) @ basis
    return float(np.square(projected).sum() / total)


def subspace_overlap(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape[0] != second.shape[0] or first.shape[1] != second.shape[1]:
        raise ValueError("Subspace bases must have matching shapes.")
    singular_values = np.linalg.svd(first.T @ second, compute_uv=False)
    return float(np.square(singular_values).mean())


def spectral_geometry_metrics(
    *,
    memory_signatures: np.ndarray,
    critical_signatures: np.ndarray,
    coverage_signatures: np.ndarray,
    rank: int,
) -> dict[str, float | int]:
    dimension = memory_signatures.shape[1]
    if not 0 < rank <= dimension:
        raise ValueError("Subspace rank must lie within the signature dimension.")
    memory_covariance = covariance_matrix(memory_signatures)
    critical_covariance = covariance_matrix(critical_signatures)
    coverage_covariance = covariance_matrix(coverage_signatures)
    memory_values, memory_vectors = eigensystem(memory_covariance)
    critical_values, critical_vectors = eigensystem(critical_covariance)
    coverage_values, coverage_vectors = eigensystem(coverage_covariance)
    memory_basis = memory_vectors[:, :rank]
    critical_basis = critical_vectors[:, :rank]
    coverage_basis = coverage_vectors[:, :rank]
    scaled_logdet = float(np.log1p(dimension * memory_values).sum())
    return {
        "states": int(memory_signatures.shape[0]),
        "signature_dimension": int(dimension),
        "subspace_rank": int(rank),
        "effective_rank": effective_rank(memory_values),
        "stable_rank": float(memory_values.sum() / max(memory_values[0], 1e-12)),
        "top_eigenvalue_fraction": float(
            memory_values[0] / max(memory_values.sum(), 1e-12)
        ),
        "scaled_logdet": scaled_logdet,
        "critical_energy_coverage": captured_reference_energy(
            critical_signatures, memory_basis
        ),
        "candidate_energy_coverage": captured_reference_energy(
            coverage_signatures, memory_basis
        ),
        "critical_topk_overlap": subspace_overlap(memory_basis, critical_basis),
        "candidate_topk_overlap": subspace_overlap(memory_basis, coverage_basis),
        "critical_covariance_error": normalized_covariance_error(
            memory_covariance, critical_covariance
        ),
        "candidate_covariance_error": normalized_covariance_error(
            memory_covariance, coverage_covariance
        ),
    }


def unique_union_indices(
    first_observations: np.ndarray,
    second_observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    seen: set[bytes] = set()
    first_indices: list[int] = []
    second_indices: list[int] = []
    for index, observation in enumerate(first_observations):
        key = observation.tobytes()
        if key not in seen:
            seen.add(key)
            first_indices.append(index)
    for index, observation in enumerate(second_observations):
        key = observation.tobytes()
        if key not in seen:
            seen.add(key)
            second_indices.append(index)
    return np.asarray(first_indices, dtype=np.int64), np.asarray(
        second_indices, dtype=np.int64
    )


def pearson_correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 3:
        return None
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    if args.subspace_rank <= 0 or args.retention_eval_episodes <= 0:
        raise ValueError("Rank and retention evaluation episodes must be positive.")
    seed_global_rngs(args.seed)
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = Path(args.memory_manifest).expanduser().resolve()
    condition_runs = {
        "success": Path(args.success_run).expanduser().resolve(),
        "broad": Path(args.broad_run).expanduser().resolve(),
        "mixed50": Path(args.mixed50_run).expanduser().resolve(),
    }
    config = load_json(run_dir / "config.json")
    manifest = load_json(manifest_path)
    source_tasks = list(manifest.get("source_tasks", []))
    if source_tasks != list(config["tasks"][: len(source_tasks)]):
        raise ValueError("Manifest source tasks do not match the continual run.")
    for name, condition_run in condition_runs.items():
        if not (condition_run / "checkpoints" / "final.pt").is_file():
            raise FileNotFoundError(f"Missing final checkpoint for {name}: {condition_run}")
        if not (condition_run / "summary.json").is_file():
            raise FileNotFoundError(f"Missing summary for {name}: {condition_run}")

    output_dir = make_output_dir(args, run_dir)
    agent = build_agent_from_config(config=config, device=args.device)
    source_checkpoint = run_dir / "checkpoints" / f"task_{len(source_tasks) - 1}.pt"
    load_sac_checkpoint(
        agent=agent,
        path=source_checkpoint,
        map_location=args.device,
        load_optimizer=False,
    )
    actor_digest = state_dict_digest(agent.actor)
    success = load_memory_payload(
        manifest=manifest, manifest_path=manifest_path, mode="success"
    )
    broad = load_memory_payload(
        manifest=manifest, manifest_path=manifest_path, mode="broad"
    )
    mixed_observations, mixed_means, mixed_log_stds, _ = (
        build_nonoverlapping_mixed_memory(
            success_payload=success,
            broad_payload=broad,
            task_id_dim=agent.task_id_dim,
            expected_source_tasks=source_tasks,
            states_per_task=int(manifest["memory_states_per_task"]),
            seed=args.seed,
        )
    )
    payloads = {
        "success": success,
        "broad": broad,
        "mixed50": {
            "observations": mixed_observations,
            "target_means": mixed_means,
            "target_log_stds": mixed_log_stds,
        },
    }
    signatures = {
        name: extract_bc_feature_gradients(
            agent=agent,
            observations=payload["observations"],
            target_means=payload["target_means"],
            target_log_stds=payload["target_log_stds"],
        )
        for name, payload in payloads.items()
    }
    if state_dict_digest(agent.actor) != actor_digest:
        raise RuntimeError("Read-only geometry extraction changed actor parameters.")

    source_indices = {
        name: infer_source_tasks(payload["observations"], num_tasks=agent.num_tasks)
        for name, payload in payloads.items()
    }
    geometry_rows: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(source_tasks):
        success_positions = np.flatnonzero(source_indices["success"] == task_index)
        broad_positions = np.flatnonzero(source_indices["broad"] == task_index)
        first_unique, second_unique = unique_union_indices(
            success["observations"][success_positions],
            broad["observations"][broad_positions],
        )
        critical_reference = signatures["success"][success_positions]
        coverage_reference = np.concatenate(
            (
                signatures["success"][success_positions[first_unique]],
                signatures["broad"][broad_positions[second_unique]],
            ),
            axis=0,
        )
        success_hashes = {
            observation.tobytes()
            for observation in success["observations"][success_positions]
        }
        for condition in ("success", "broad", "mixed50"):
            positions = np.flatnonzero(source_indices[condition] == task_index)
            condition_observations = payloads[condition]["observations"][positions]
            exact_success_overlap = sum(
                observation.tobytes() in success_hashes
                for observation in condition_observations
            ) / positions.size
            metrics = spectral_geometry_metrics(
                memory_signatures=signatures[condition][positions],
                critical_signatures=critical_reference,
                coverage_signatures=coverage_reference,
                rank=args.subspace_rank,
            )
            row = {
                "condition": condition,
                "source_task_index": task_index,
                "source_task_name": task_name,
                "exact_success_reference_overlap": exact_success_overlap,
                "coverage_reference_states": int(coverage_reference.shape[0]),
                **metrics,
            }
            geometry_rows.append(row)
            print(
                f"[criticality-coverage] condition={condition} task={task_name} "
                f"critical={row['critical_energy_coverage']:.3f} "
                f"coverage={row['candidate_energy_coverage']:.3f} "
                f"rank={row['effective_rank']:.2f}",
                flush=True,
            )

    before_rows = evaluate_source_task_retention(
        agent=agent,
        tasks=list(config["tasks"]),
        source_task_count=len(source_tasks),
        env_version=str(config["env_version"]),
        reward_function_version=str(config["reward_function_version"]),
        episodes=args.retention_eval_episodes,
        max_episode_steps=int(config.get("max_episode_steps", 200)),
        append_task_id=True,
        seed=args.seed + 700_000,
    )
    retention_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    for condition, condition_run in condition_runs.items():
        load_sac_checkpoint(
            agent=agent,
            path=condition_run / "checkpoints" / "final.pt",
            map_location=args.device,
            load_optimizer=False,
        )
        after_rows = evaluate_source_task_retention(
            agent=agent,
            tasks=list(config["tasks"]),
            source_task_count=len(source_tasks),
            env_version=str(config["env_version"]),
            reward_function_version=str(config["reward_function_version"]),
            episodes=args.retention_eval_episodes,
            max_episode_steps=int(config.get("max_episode_steps", 200)),
            append_task_id=True,
            seed=args.seed + 700_000,
        )
        forgetting_values = []
        for before, after in zip(before_rows, after_rows, strict=True):
            forgetting = float(before["success_rate"]) - float(after["success_rate"])
            forgetting_values.append(forgetting)
            retention_rows.append(
                {
                    "condition": condition,
                    "source_task_index": before["source_task_index"],
                    "source_task_name": before["source_task_name"],
                    "before_success": before["success_rate"],
                    "after_success": after["success_rate"],
                    "forgetting": forgetting,
                }
            )
        run_summary = load_json(condition_run / "summary.json")
        condition_geometry = [
            row for row in geometry_rows if row["condition"] == condition
        ]
        condition_rows.append(
            {
                "condition": condition,
                "learning_curve_mean_success": run_summary[
                    "learning_curve_mean_success"
                ],
                "tail5_success": run_summary["tail5_success"],
                "first_nonzero_success_step": run_summary[
                    "first_nonzero_success_step"
                ],
                "mean_forgetting": float(np.mean(forgetting_values)),
                "mean_critical_energy_coverage": float(
                    np.mean(
                        [row["critical_energy_coverage"] for row in condition_geometry]
                    )
                ),
                "mean_candidate_energy_coverage": float(
                    np.mean(
                        [row["candidate_energy_coverage"] for row in condition_geometry]
                    )
                ),
                "mean_effective_rank": float(
                    np.mean([row["effective_rank"] for row in condition_geometry])
                ),
                "mean_critical_covariance_error": float(
                    np.mean(
                        [row["critical_covariance_error"] for row in condition_geometry]
                    )
                ),
                "mean_candidate_covariance_error": float(
                    np.mean(
                        [row["candidate_covariance_error"] for row in condition_geometry]
                    )
                ),
            }
        )

    joined_rows = []
    retention_index = {
        (row["condition"], int(row["source_task_index"])): row
        for row in retention_rows
    }
    for row in geometry_rows:
        joined_rows.append(
            {
                **row,
                "forgetting": retention_index[
                    (row["condition"], int(row["source_task_index"]))
                ]["forgetting"],
            }
        )
    correlation_metrics = (
        "critical_energy_coverage",
        "candidate_energy_coverage",
        "effective_rank",
        "critical_covariance_error",
        "candidate_covariance_error",
        "exact_success_reference_overlap",
    )
    correlation_rows = []
    forgetting = [float(row["forgetting"]) for row in joined_rows]
    for metric in correlation_metrics:
        correlation_rows.append(
            {
                "metric": metric,
                "pearson_with_forgetting": pearson_correlation(
                    [float(row[metric]) for row in joined_rows], forgetting
                ),
                "observations": len(joined_rows),
            }
        )

    write_csv(output_dir / "geometry_by_task.csv", geometry_rows[0].keys(), geometry_rows)
    write_csv(output_dir / "retention_by_task.csv", retention_rows[0].keys(), retention_rows)
    write_csv(output_dir / "geometry_and_forgetting.csv", joined_rows[0].keys(), joined_rows)
    write_csv(output_dir / "condition_summary.csv", condition_rows[0].keys(), condition_rows)
    write_csv(output_dir / "correlations.csv", correlation_rows[0].keys(), correlation_rows)
    summary = {
        "purpose": "criticality_coverage_memory_geometry_diagnostic",
        "run_dir": str(run_dir),
        "source_checkpoint": str(source_checkpoint),
        "memory_manifest": str(manifest_path),
        "subspace_rank": args.subspace_rank,
        "retention_eval_episodes": args.retention_eval_episodes,
        "actor_unchanged_during_geometry_extraction": True,
        "conditions": condition_rows,
        "correlations": correlation_rows,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "conditions": condition_rows}, indent=2))


if __name__ == "__main__":
    main()
