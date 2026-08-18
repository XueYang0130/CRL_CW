from __future__ import annotations

import argparse
import hashlib
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
from scripts.probe_stickpull_with_memory import load_json
from training.sac_trainer import seed_global_rngs
from utils import load_sac_checkpoint, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether success-only BC memories produce more coherent, "
            "lower-rank shared-backbone gradients than matched broad memories."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--memory-manifest", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--microbatches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def resolve_manifest_file(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path.resolve()
    local_path = manifest_path.parent / path.name
    if local_path.is_file():
        return local_path.resolve()
    raise FileNotFoundError(f"Memory payload does not exist: {raw_path}")


def array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_memory_payload(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    mode: str,
) -> dict[str, np.ndarray]:
    files = manifest.get("files")
    if not isinstance(files, dict) or mode not in files:
        raise ValueError(f"Manifest does not contain a {mode} memory payload.")
    payload_path = resolve_manifest_file(manifest_path, str(files[mode]))
    with np.load(payload_path) as payload:
        required = {"observations", "target_means", "target_log_stds"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(
                f"{mode} memory is missing arrays: {', '.join(sorted(missing))}."
            )
        arrays = {
            key: payload[key].astype(np.float32, copy=True)
            for key in sorted(required)
        }
    count = arrays["observations"].shape[0]
    if count != int(manifest["counts"][mode]):
        raise ValueError(
            f"{mode} count mismatch: manifest={manifest['counts'][mode]}, "
            f"payload={count}."
        )
    if any(array.shape[0] != count for array in arrays.values()):
        raise ValueError(f"{mode} memory arrays have inconsistent lengths.")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError(f"{mode} memory contains non-finite values.")
    expected_digest = manifest.get("observation_sha256", {}).get(mode)
    if expected_digest and array_digest(arrays["observations"]) != expected_digest:
        raise ValueError(f"{mode} observation digest does not match the manifest.")
    return arrays


def infer_source_tasks(observations: np.ndarray, *, num_tasks: int) -> np.ndarray:
    if observations.ndim != 2 or observations.shape[1] < num_tasks:
        raise ValueError("Observations cannot contain the expected task one-hot vector.")
    task_ids = observations[:, -num_tasks:]
    if not np.allclose(task_ids.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Reference observations contain invalid task one-hot vectors.")
    if not np.all((task_ids >= -1e-6) & (task_ids <= 1.0 + 1e-6)):
        raise ValueError("Reference task IDs must lie in [0, 1].")
    return np.argmax(task_ids, axis=1).astype(np.int64)


def gradient_concentration_metrics(gradients: np.ndarray) -> dict[str, float | int]:
    if gradients.ndim != 2 or gradients.shape[0] < 2:
        raise ValueError("At least two flattened gradients are required.")
    if not np.isfinite(gradients).all():
        raise ValueError("Gradients must be finite.")
    gradients64 = gradients.astype(np.float64, copy=False)
    norms = np.linalg.norm(gradients64, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("Every diagnostic gradient must have positive norm.")

    normalized = gradients64 / norms[:, None]
    cosine = normalized @ normalized.T
    upper = cosine[np.triu_indices(cosine.shape[0], k=1)]
    directional_coherence = float(
        np.linalg.norm(gradients64.sum(axis=0)) / norms.sum()
    )

    gram = gradients64 @ gradients64.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    spectral_mass = float(eigenvalues.sum())
    if spectral_mass <= 0.0:
        raise ValueError("Gradient Gram matrix has no positive spectral mass.")
    probabilities = eigenvalues[eigenvalues > spectral_mass * 1e-12]
    probabilities = probabilities / probabilities.sum()
    effective_rank = float(np.exp(-(probabilities * np.log(probabilities)).sum()))
    largest = float(eigenvalues.max())
    stable_rank = float(spectral_mass / largest)

    return {
        "microbatches": int(gradients.shape[0]),
        "parameter_dimension": int(gradients.shape[1]),
        "mean_gradient_norm": float(norms.mean()),
        "std_gradient_norm": float(norms.std(ddof=0)),
        "gradient_norm_cv": float(norms.std(ddof=0) / norms.mean()),
        "mean_pairwise_cosine": float(upper.mean()),
        "std_pairwise_cosine": float(upper.std(ddof=0)),
        "p10_pairwise_cosine": float(np.quantile(upper, 0.10)),
        "median_pairwise_cosine": float(np.quantile(upper, 0.50)),
        "p90_pairwise_cosine": float(np.quantile(upper, 0.90)),
        "directional_coherence": directional_coherence,
        "effective_rank": effective_rank,
        "normalized_effective_rank": effective_rank / gradients.shape[0],
        "stable_rank": stable_rank,
        "top_eigenvalue_fraction": largest / spectral_mass,
    }


def sample_microbatch_indices(
    *,
    candidates_by_task: dict[int, np.ndarray],
    microbatches: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    task_indices = sorted(candidates_by_task)
    if not task_indices:
        raise ValueError("No source-task candidates were provided.")
    base, remainder = divmod(batch_size, len(task_indices))
    if base == 0:
        raise ValueError("Batch size must be at least the number of source tasks.")
    allocations = {
        task: base + int(position < remainder)
        for position, task in enumerate(task_indices)
    }
    batches: list[np.ndarray] = []
    for _ in range(microbatches):
        selected = []
        for task in task_indices:
            candidates = candidates_by_task[task]
            count = allocations[task]
            if candidates.size < count:
                raise ValueError(
                    f"Source task {task} has {candidates.size} states, need {count}."
                )
            selected.append(rng.choice(candidates, size=count, replace=False))
        batch = np.concatenate(selected)
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def compute_bc_gradient_matrix(
    *,
    agent: Any,
    payload: dict[str, np.ndarray],
    batches: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    parameters = tuple(agent.actor.backbone.parameters())
    if not parameters:
        raise RuntimeError("Actor backbone exposes no trainable parameters.")
    gradients: list[np.ndarray] = []
    losses: list[float] = []
    agent.actor.eval()
    for indices in batches:
        observations = torch.as_tensor(
            payload["observations"][indices],
            dtype=torch.float32,
            device=agent.device,
        )
        target_means = torch.as_tensor(
            payload["target_means"][indices],
            dtype=torch.float32,
            device=agent.device,
        )
        target_log_stds = torch.as_tensor(
            payload["target_log_stds"][indices],
            dtype=torch.float32,
            device=agent.device,
        )
        current_means, current_log_stds = agent.actor.distribution_parameters(
            observations
        )
        loss = agent._gaussian_kl(
            target_means,
            target_log_stds,
            current_means,
            current_log_stds,
        ).mean()
        batch_gradients = torch.autograd.grad(loss, parameters)
        flattened = torch.cat(
            [gradient.detach().reshape(-1).cpu() for gradient in batch_gradients]
        )
        gradients.append(flattened.numpy().astype(np.float32, copy=False))
        losses.append(float(loss.detach().cpu().item()))
    if any(parameter.grad is not None for parameter in agent.actor.parameters()):
        raise RuntimeError("Read-only diagnostic unexpectedly populated .grad buffers.")
    return np.stack(gradients), np.asarray(losses, dtype=np.float64)


def state_dict_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).view(np.uint8))
    return digest.hexdigest()


def make_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    base = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "diagnostics"
    )
    name = args.run_name or f"bc_gradient_concentration_{run_dir.name}_seed{args.seed}"
    output_dir = base / name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main() -> None:
    args = parse_args()
    if args.microbatches < 2 or args.batch_size <= 0:
        raise ValueError("--microbatches must be >= 2 and --batch-size must be positive.")
    seed_global_rngs(args.seed)

    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = Path(args.memory_manifest).expanduser().resolve()
    config = load_json(run_dir / "config.json")
    manifest = load_json(manifest_path)
    if manifest.get("purpose") != "matched_stickpull_memory_causal_ablation":
        raise ValueError("Expected a matched stick-pull causal-memory manifest.")
    if manifest.get("teacher") != "per_task_best_actor_snapshot":
        raise ValueError("Memory targets must use per-task best actor snapshots.")
    source_tasks = list(manifest.get("source_tasks", []))
    if source_tasks != list(config["tasks"][: len(source_tasks)]):
        raise ValueError("Manifest source tasks do not match the continual run.")

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
    before_digest = state_dict_digest(agent.actor)
    rng = np.random.default_rng(args.seed + 810_000)
    rows: list[dict[str, Any]] = []

    for mode in ("success", "broad"):
        payload = load_memory_payload(
            manifest=manifest,
            manifest_path=manifest_path,
            mode=mode,
        )
        source_indices = infer_source_tasks(
            payload["observations"],
            num_tasks=agent.num_tasks,
        )
        unexpected = sorted(set(source_indices.tolist()) - set(range(len(source_tasks))))
        if unexpected:
            raise ValueError(f"{mode} memory contains unexpected task IDs: {unexpected}.")
        scopes: list[tuple[str, dict[int, np.ndarray]]] = [
            (
                "all_tasks_balanced",
                {
                    task: np.flatnonzero(source_indices == task)
                    for task in range(len(source_tasks))
                },
            )
        ]
        scopes.extend(
            (
                f"task_{task}_{source_tasks[task]}",
                {task: np.flatnonzero(source_indices == task)},
            )
            for task in range(len(source_tasks))
        )
        for scope, candidates_by_task in scopes:
            batches = sample_microbatch_indices(
                candidates_by_task=candidates_by_task,
                microbatches=args.microbatches,
                batch_size=args.batch_size,
                rng=rng,
            )
            gradients, losses = compute_bc_gradient_matrix(
                agent=agent,
                payload=payload,
                batches=batches,
            )
            metrics = gradient_concentration_metrics(gradients)
            row: dict[str, Any] = {
                "memory_mode": mode,
                "scope": scope,
                "mean_raw_bc_loss": float(losses.mean()),
                "std_raw_bc_loss": float(losses.std(ddof=0)),
                **metrics,
            }
            rows.append(row)
            print(
                f"[bc-concentration] memory={mode} scope={scope} "
                f"cosine={row['mean_pairwise_cosine']:.4f} "
                f"coherence={row['directional_coherence']:.4f} "
                f"effective_rank={row['effective_rank']:.2f}"
            )

    after_digest = state_dict_digest(agent.actor)
    if after_digest != before_digest:
        raise RuntimeError("Read-only gradient diagnostic changed actor parameters.")

    fieldnames = list(rows[0].keys())
    write_csv(output_dir / "concentration_metrics.csv", fieldnames, rows)
    indexed = {(row["memory_mode"], row["scope"]): row for row in rows}
    comparisons = []
    for scope in [row["scope"] for row in rows if row["memory_mode"] == "success"]:
        success = indexed[("success", scope)]
        broad = indexed[("broad", scope)]
        comparisons.append(
            {
                "scope": scope,
                "pairwise_cosine_delta_success_minus_broad": (
                    success["mean_pairwise_cosine"] - broad["mean_pairwise_cosine"]
                ),
                "coherence_delta_success_minus_broad": (
                    success["directional_coherence"] - broad["directional_coherence"]
                ),
                "effective_rank_delta_success_minus_broad": (
                    success["effective_rank"] - broad["effective_rank"]
                ),
                "effective_rank_ratio_success_over_broad": (
                    success["effective_rank"] / broad["effective_rank"]
                ),
                "top_eigenvalue_fraction_delta_success_minus_broad": (
                    success["top_eigenvalue_fraction"]
                    - broad["top_eigenvalue_fraction"]
                ),
            }
        )
    write_csv(
        output_dir / "success_vs_broad.csv",
        list(comparisons[0].keys()),
        comparisons,
    )
    summary = {
        "purpose": "matched_bc_gradient_concentration_diagnostic",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "memory_manifest": str(manifest_path),
        "microbatches": args.microbatches,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "actor_unchanged": True,
        "metrics": rows,
        "comparisons": comparisons,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "actor_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
