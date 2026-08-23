#!/usr/bin/env python3
"""Offline mechanism analysis for completed continual-learning runs.

The analysis deliberately treats each task-seed pair as an observation and
also reports task-demeaned correlations. The latter remove average task
difficulty, which otherwise confounds gradient/learning correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr


METHODS = {
    "adaptive_pcgrad": "success_replay_best_adaptive_pcgrad_cw10_v3_v1_500k_seed{seed}",
    "frozen_transfer": "semantic_routed_frozen_transfer_pcgrad_cw10_v3_v1_500k_seed{seed}",
}

GRADIENT_COLUMNS = (
    "conflict",
    "conflict_mass",
    "cosine_similarity",
    "sac_gradient_norm",
    "bc_gradient_norm",
    "bc_to_sac_norm_ratio",
    "applied_bc_gradient_norm",
    "applied_bc_to_sac_norm_ratio",
    "bc_norm_scale",
    "bc_combination_scale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("outputs/cw10_continual")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/mechanism_analysis_seeds1_5"),
    )
    parser.add_argument("--analysis-steps", type=int, default=500_000)
    parser.add_argument("--bootstrap-reps", type=int, default=50_000)
    parser.add_argument("--random-seed", type=int, default=20260820)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return float("nan")
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return 1.0 if value.lower() == "true" else 0.0
    result = float(value)
    return result if math.isfinite(result) else float("nan")


def mean_finite(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def coefficient_of_variation(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan")
    mean = float(np.mean(array))
    return float(np.std(array) / abs(mean)) if abs(mean) > 1e-12 else float("nan")


def auc_until(points: list[tuple[int, float]], horizon: int) -> float:
    selected = sorted((step, value) for step, value in points if step <= horizon)
    if not selected or selected[-1][0] != horizon:
        raise ValueError(f"Missing evaluation at step {horizon}.")
    x = np.asarray([0, *[step for step, _ in selected]], dtype=np.float64)
    y = np.asarray([0.0, *[value for _, value in selected]], dtype=np.float64)
    return float(np.trapz(y, x) / horizon)


def first_threshold_step(
    points: list[tuple[int, float]], threshold: float, censor_step: int
) -> int:
    return next((step for step, value in sorted(points) if value >= threshold), censor_step)


def aggregate_gradient_rows(rows: list[dict[str, str]], prefix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in GRADIENT_COLUMNS:
        values = [finite_float(row[column]) for row in rows]
        result[f"{prefix}_{column}_mean"] = mean_finite(values)
        if column in {"bc_gradient_norm", "bc_to_sac_norm_ratio"}:
            result[f"{prefix}_{column}_cv"] = coefficient_of_variation(values)
    effective_norms = [
        finite_float(row["applied_bc_gradient_norm"])
        * finite_float(row["bc_combination_scale"])
        for row in rows
    ]
    effective_ratios = [
        finite_float(row["applied_bc_to_sac_norm_ratio"])
        * finite_float(row["bc_combination_scale"])
        for row in rows
    ]
    interference_indices = [
        finite_float(row["bc_to_sac_norm_ratio"])
        * max(0.0, -finite_float(row["cosine_similarity"]))
        for row in rows
    ]
    cooperation_indices = [
        finite_float(row["bc_to_sac_norm_ratio"])
        * max(0.0, finite_float(row["cosine_similarity"]))
        for row in rows
    ]
    result[f"{prefix}_effective_bc_gradient_norm_mean"] = mean_finite(effective_norms)
    result[f"{prefix}_effective_bc_to_sac_norm_ratio_mean"] = mean_finite(
        effective_ratios
    )
    result[f"{prefix}_effective_bc_to_sac_norm_ratio_cv"] = coefficient_of_variation(
        effective_ratios
    )
    result[f"{prefix}_interference_index_mean"] = mean_finite(interference_indices)
    result[f"{prefix}_cooperation_index_mean"] = mean_finite(cooperation_indices)
    return result


def load_task_seed_rows(
    input_dir: Path, method_name: str, pattern: str, seeds: list[int], analysis_steps: int
) -> list[dict[str, float | int | str]]:
    output: list[dict[str, float | int | str]] = []
    for seed in seeds:
        run_dir = input_dir / pattern.format(seed=seed)
        with (run_dir / "summary.json").open() as handle:
            summary = json.load(handle)
        evaluations = read_csv(run_dir / "evaluations.csv")
        gradients = read_csv(run_dir / "gradient_diagnostics" / "gradient_windows.csv")

        current_evaluations: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in evaluations:
            if int(row["active_task_index"]) == int(row["evaluation_task_index"]):
                current_evaluations[int(row["active_task_index"])].append(row)

        shared_gradients: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in gradients:
            if row["scope"] == "shared_actor_backbone":
                shared_gradients[int(row["current_task_index"])].append(row)

        task_names = [
            name
            for name, _ in sorted(
                (
                    (row["active_task_name"], int(row["active_task_index"]))
                    for row in evaluations
                ),
                key=lambda item: item[1],
            )
        ]
        # Preserve order while removing repeated evaluation rows.
        task_names = list(dict.fromkeys(task_names))
        for task_index, task_name in enumerate(task_names):
            eval_rows = current_evaluations[task_index]
            success_points = [
                (int(row["active_task_step"]), finite_float(row["stochastic_success_rate"]))
                for row in eval_rows
            ]
            budget_eval_rows = [
                row for row in eval_rows if int(row["active_task_step"]) <= analysis_steps
            ]
            grad_rows = shared_gradients.get(task_index, [])
            budget_grad_rows = [
                row for row in grad_rows if int(row["active_task_step"]) <= analysis_steps
            ]
            censor_step = max(step for step, _ in success_points) + 20_000
            row_data: dict[str, float | int | str] = {
                "method": method_name,
                "seed": seed,
                "task_index": task_index,
                "task_name": task_name,
                "delay_success_0_2": first_threshold_step(success_points, 0.2, censor_step),
                "delay_success_0_8": first_threshold_step(success_points, 0.8, censor_step),
                "reached_success_0_2": int(any(v >= 0.2 for _, v in success_points)),
                "reached_success_0_8": int(any(v >= 0.8 for _, v in success_points)),
                "success_at_budget": next(
                    value for step, value in success_points if step == analysis_steps
                ),
                "success_auc_budget": auc_until(success_points, analysis_steps),
                "budget_td_loss_mean": mean_finite(
                    (
                        finite_float(row["q1_loss"]) + finite_float(row["q2_loss"])
                    )
                    / 2.0
                    for row in budget_eval_rows
                ),
                "budget_q_disagreement_mean": mean_finite(
                    abs(finite_float(row["q1_mean"]) - finite_float(row["q2_mean"]))
                    for row in budget_eval_rows
                ),
                "budget_q_target_gap_mean": mean_finite(
                    abs(
                        (
                            finite_float(row["q1_mean"])
                            + finite_float(row["q2_mean"])
                        )
                        / 2.0
                        - finite_float(row["q_target_mean"])
                    )
                    for row in budget_eval_rows
                ),
                "raw_forward_transfer": finite_float(
                    summary["raw_forward_transfer_per_task"].get(task_name)
                ),
                "forgetting": finite_float(summary["forgetting_per_task"].get(task_name)),
                "end_of_task_success": finite_float(
                    summary["end_of_task_per_task"].get(task_name)
                ),
                "final_success": finite_float(
                    summary["final_per_task_success"].get(task_name)
                ),
                "gradient_samples": len(grad_rows),
                "budget_gradient_samples": len(budget_grad_rows),
            }
            row_data.update(aggregate_gradient_rows(budget_grad_rows, "budget"))
            row_data.update(aggregate_gradient_rows(grad_rows, "all"))
            output.append(row_data)
    return output


def demean_by_task(rows: list[dict], field: str) -> np.ndarray:
    values = np.asarray([finite_float(row[field]) for row in rows], dtype=np.float64)
    tasks = np.asarray([int(row["task_index"]) for row in rows])
    result = values.copy()
    for task in np.unique(tasks):
        mask = (tasks == task) & np.isfinite(values)
        result[mask] -= np.mean(values[mask])
    return result


def correlation_record(
    rows: list[dict], predictor: str, outcome: str, *, demean: bool
) -> dict[str, float | int | str]:
    x = demean_by_task(rows, predictor) if demean else np.asarray(
        [finite_float(row[predictor]) for row in rows]
    )
    y = demean_by_task(rows, outcome) if demean else np.asarray(
        [finite_float(row[outcome]) for row in rows]
    )
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        pearson = spearman = pearson_p = spearman_p = float("nan")
    else:
        pearson, pearson_p = pearsonr(x, y)
        spearman, spearman_p = spearmanr(x, y)
    return {
        "predictor": predictor,
        "outcome": outcome,
        "analysis": "task_demeaned" if demean else "raw",
        "observations": int(x.size),
        "pearson_r": float(pearson),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman),
        "spearman_p": float(spearman_p),
    }


def percentile_iqm(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(np.floor(0.25 * values.size))
    return float(np.mean(values[trim : values.size - trim])) if trim else float(np.mean(values))


def rowwise_iqm(values: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=np.float64), axis=1)
    trim = int(np.floor(0.25 * values.shape[1]))
    selected = values[:, trim : values.shape[1] - trim] if trim else values
    return np.mean(selected, axis=1)


def paired_bootstrap_difference(
    left: np.ndarray, right: np.ndarray, reps: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    if left.shape != right.shape:
        raise ValueError("Paired arrays must have equal shape.")
    samples = rng.integers(0, left.size, size=(reps, left.size))
    estimates = rowwise_iqm(left[samples]) - rowwise_iqm(right[samples])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return percentile_iqm(left) - percentile_iqm(right), float(low), float(high)


def critic_comparisons(
    rows: list[dict], seeds: list[int], reps: int, random_seed: int
) -> list[dict[str, float | int | str]]:
    by_key = {
        (row["method"], int(row["seed"]), int(row["task_index"])): row for row in rows
    }
    metrics = (
        "budget_td_loss_mean",
        "budget_q_disagreement_mean",
        "budget_q_target_gap_mean",
        "success_auc_budget",
        "delay_success_0_2",
        "raw_forward_transfer",
        "forgetting",
        "final_success",
    )
    rng = np.random.default_rng(random_seed)
    output = []
    for task_index in range(10):
        for metric in metrics:
            adaptive = np.asarray(
                [by_key[("adaptive_pcgrad", seed, task_index)][metric] for seed in seeds],
                dtype=np.float64,
            )
            frozen = np.asarray(
                [by_key[("frozen_transfer", seed, task_index)][metric] for seed in seeds],
                dtype=np.float64,
            )
            valid = np.isfinite(adaptive) & np.isfinite(frozen)
            if not np.any(valid):
                continue
            delta, low, high = paired_bootstrap_difference(
                frozen[valid], adaptive[valid], reps, rng
            )
            output.append(
                {
                    "task_index": task_index,
                    "task_name": by_key[("adaptive_pcgrad", seeds[0], task_index)]["task_name"],
                    "metric": metric,
                    "seeds": int(np.sum(valid)),
                    "adaptive_mean": float(np.mean(adaptive[valid])),
                    "frozen_mean": float(np.mean(frozen[valid])),
                    "frozen_minus_adaptive_iqm": delta,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_key_relationships(output_dir: Path, rows: list[dict], analysis_steps: int) -> None:
    adaptive = [
        row for row in rows if row["method"] == "adaptive_pcgrad" and row["task_index"] > 0
    ]
    settings = (
        ("budget_conflict_mass_mean", "delay_success_0_2", "Conflict mass", "Steps to success >= 0.2"),
        ("budget_effective_bc_to_sac_norm_ratio_mean", "success_auc_budget", "Effective BC/SAC norm", f"Success AUC (0-{analysis_steps // 1000}k)"),
        ("all_effective_bc_to_sac_norm_ratio_mean", "forgetting", "Effective BC/SAC norm", "Final forgetting"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=180)
    colors = plt.get_cmap("tab10")
    for ax, (x_field, y_field, xlabel, ylabel) in zip(axes, settings):
        for row in adaptive:
            x, y = finite_float(row[x_field]), finite_float(row[y_field])
            if np.isfinite(x) and np.isfinite(y):
                ax.scatter(x, y, s=24, alpha=0.7, color=colors(int(row["task_index"])))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_outcome_scatter.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    all_rows: list[dict] = []
    for method_name, pattern in METHODS.items():
        all_rows.extend(
            load_task_seed_rows(
                args.input_dir, method_name, pattern, args.seeds, args.analysis_steps
            )
        )
    write_csv(args.output_dir / "task_seed_metrics.csv", all_rows)

    adaptive_rows = [
        row for row in all_rows if row["method"] == "adaptive_pcgrad" and row["task_index"] > 0
    ]
    predictors = (
        "budget_conflict_mean",
        "budget_conflict_mass_mean",
        "budget_cosine_similarity_mean",
        "budget_bc_to_sac_norm_ratio_mean",
        "budget_effective_bc_to_sac_norm_ratio_mean",
        "budget_effective_bc_to_sac_norm_ratio_cv",
        "budget_bc_gradient_norm_cv",
        "budget_interference_index_mean",
        "budget_cooperation_index_mean",
        "all_effective_bc_to_sac_norm_ratio_mean",
    )
    outcomes = (
        "delay_success_0_2",
        "delay_success_0_8",
        "success_auc_budget",
        "raw_forward_transfer",
        "forgetting",
    )
    correlations = [
        correlation_record(adaptive_rows, predictor, outcome, demean=demean)
        for predictor in predictors
        for outcome in outcomes
        for demean in (False, True)
    ]
    for record in correlations:
        leave_one_seed_out = []
        for omitted_seed in args.seeds:
            subset = [
                row for row in adaptive_rows if int(row["seed"]) != omitted_seed
            ]
            estimate = correlation_record(
                subset,
                str(record["predictor"]),
                str(record["outcome"]),
                demean=record["analysis"] == "task_demeaned",
            )
            leave_one_seed_out.append(finite_float(estimate["spearman_r"]))
        finite_estimates = np.asarray(
            [value for value in leave_one_seed_out if np.isfinite(value)],
            dtype=np.float64,
        )
        record["loo_spearman_min"] = (
            float(np.min(finite_estimates)) if finite_estimates.size else float("nan")
        )
        record["loo_spearman_max"] = (
            float(np.max(finite_estimates)) if finite_estimates.size else float("nan")
        )
        full_sign = np.sign(finite_float(record["spearman_r"]))
        record["loo_same_sign_fraction"] = (
            float(np.mean(np.sign(finite_estimates) == full_sign))
            if finite_estimates.size and full_sign != 0.0
            else float("nan")
        )
    write_csv(args.output_dir / "gradient_outcome_correlations.csv", correlations)

    critic_rows = critic_comparisons(
        all_rows, args.seeds, args.bootstrap_reps, args.random_seed
    )
    write_csv(args.output_dir / "frozen_vs_adaptive_critic.csv", critic_rows)
    plot_key_relationships(args.output_dir, all_rows, args.analysis_steps)

    strongest = sorted(
        (
            row for row in correlations if row["analysis"] == "task_demeaned"
        ),
        key=lambda row: abs(finite_float(row["spearman_r"])),
        reverse=True,
    )[:10]
    routed = [row for row in critic_rows if row["task_index"] in {4, 5}]
    summary = {
        "purpose": "offline_gradient_and_critic_mechanism_analysis",
        "seeds": args.seeds,
        "methods": list(METHODS),
        "analysis_steps": args.analysis_steps,
        "task_seed_observations_per_method": len(all_rows) // len(METHODS),
        "adaptive_transfer_task_observations": len(adaptive_rows),
        "strongest_task_demeaned_correlations": strongest,
        "routed_task_and_immediate_downstream_comparisons": routed,
        "limitations": [
            "Correlations are observational and task-seed pairs within a seed are not independent.",
            "Gradient norm variability is not directional gradient concentration.",
            "Existing directional concentration diagnostics cover seeds 0 and 4 only.",
            "Existing memory coverage diagnostics are task-specific and do not establish CW10-wide causality.",
            "Logged TD loss and Q gaps are proxies, not direct measurements of Bellman representation shift.",
        ],
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
