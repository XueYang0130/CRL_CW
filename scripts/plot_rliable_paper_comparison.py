#!/usr/bin/env python3
"""Plot current CW10 methods with official rliable IQM confidence intervals."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics


METHODS = {
    "CloneX-SAC": {
        "result_key": "CloneX-SAC",
        "color": "#D2B55B",
        "marker": "o",
    },
    "Full BC (Ours)": {
        "color": "#D979B5",
        "marker": "D",
    },
    "Adaptive PCGrad (Ours)": {
        "result_key": "Adaptive PCGrad",
        "color": "#2F6FAE",
        "marker": "D",
    },
    "Frozen Transfer PCGrad (Ours)": {
        "result_key": "Frozen Transfer PCGrad",
        "color": "#2E8B57",
        "marker": "D",
    },
}

METRICS = ("average_performance", "forward_transfer", "forgetting")
FULL_BC_SEEDS = (0, 1, 2)


def asymmetric_error(value: float, low: float, high: float) -> list[list[float]]:
    return [[max(0.0, value - low)], [max(0.0, high - value)]]


def draw_point(
    ax,
    x: tuple[float, float, float],
    y: tuple[float, float, float],
    *,
    label: str,
    color: str,
    marker: str,
) -> None:
    ax.errorbar(
        x[0],
        y[0],
        xerr=asymmetric_error(*x),
        yerr=asymmetric_error(*y),
        fmt=marker,
        markersize=11 if marker == "o" else 10,
        color=color,
        ecolor=color,
        elinewidth=1.8,
        capsize=5,
        capthick=1.3,
        markeredgecolor="#454545",
        markeredgewidth=0.9,
        label=label,
        zorder=3,
    )


def unpack_rliable_result(
    payload: dict, result_key: str, metric: str
) -> tuple[float, float, float]:
    aggregate = payload["aggregate_metrics"][metric]
    point = float(np.asarray(aggregate["point_estimates"][result_key]).reshape(-1)[0])
    interval = np.asarray(aggregate["confidence_intervals"][result_key]).reshape(2, -1)
    return point, float(interval[0, 0]), float(interval[1, 0])


def full_bc_rliable_results(
    root: Path, *, bootstrap_reps: int = 50_000, random_seed: int = 20260818
) -> dict[str, tuple[float, float, float]]:
    fields = {
        "average_performance": "average_performance",
        "forward_transfer": "raw_forward_transfer",
        "forgetting": "average_forgetting",
    }
    values = {metric: [] for metric in fields}
    for seed in FULL_BC_SEEDS:
        summary_path = (
            root
            / "outputs/cw10_continual"
            / f"full_bc_cw10_v3_v1_500k_seed{seed}"
            / "summary.json"
        )
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing Full BC result: {summary_path}")
        with summary_path.open() as handle:
            summary = json.load(handle)
        for metric, field in fields.items():
            values[metric].append(float(summary[field]))

    results = {}
    for index, metric in enumerate(fields):
        scores = {"Full BC (Ours)": np.asarray(values[metric], dtype=np.float64)[:, None]}
        points, intervals = rly.get_interval_estimates(
            scores,
            metrics.aggregate_iqm,
            reps=bootstrap_reps,
            confidence_interval_size=0.95,
            random_state=np.random.RandomState(random_seed + index),
        )
        point = float(np.asarray(points["Full BC (Ours)"]).reshape(-1)[0])
        interval = np.asarray(intervals["Full BC (Ours)"]).reshape(2, -1)
        results[metric] = (point, float(interval[0, 0]), float(interval[1, 0]))
    return results


def load_results(root: Path) -> dict[str, dict[str, tuple[float, float, float]]]:
    source = root / "outputs/analysis/rliable_cw10_seeds0_5/rliable_results.json"
    with source.open() as handle:
        payload = json.load(handle)

    results = {}
    for display, config in METHODS.items():
        if display == "Full BC (Ours)":
            results[display] = full_bc_rliable_results(root)
            continue
        result_key = config["result_key"]
        results[display] = {
            metric: unpack_rliable_result(payload, result_key, metric) for metric in METRICS
        }
    return results


def save_values(
    path: Path, results: dict[str, dict[str, tuple[float, float, float]]]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "seeds", "metric", "iqm", "ci95_low", "ci95_high"])
        for method, method_results in results.items():
            seeds = "0,1,2" if method == "Full BC (Ours)" else "0,1,2,3,4,5"
            for metric, (point, low, high) in method_results.items():
                writer.writerow([method, seeds, metric, point, low, high])


def draw_comparison(
    results: dict[str, dict[str, tuple[float, float, float]]],
    *,
    x_metric: str,
    xlabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 7.4), dpi=200)
    for method, config in METHODS.items():
        draw_point(
            ax,
            results[method][x_metric],
            results[method]["average_performance"],
            label=method,
            color=config["color"],
            marker=config["marker"],
        )

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Average performance", fontsize=18)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, color="#C9C9C9", linewidth=1.0, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#BDBDBD")
        spine.set_linewidth(1.0)

    ax.set_ylim(0.68, 0.96)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    if x_metric == "forward_transfer":
        ax.set_xlim(0.03, 0.27)
        ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25])
    else:
        ax.set_xlim(-0.04, 0.08)
        ax.set_xticks([-0.04, -0.02, 0.00, 0.02, 0.04, 0.06, 0.08])
        ax.axvline(0.0, color="#777777", linewidth=0.8, zorder=1)

    ax.legend(
        title="CL method",
        loc="lower left",
        fontsize=11,
        title_fontsize=12,
        framealpha=0.94,
        edgecolor="#CCCCCC",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs/analysis/rliable_cw10_seeds0_5"
    results = load_results(root)
    save_values(output_dir / "paper_style_current_methods_iqm.csv", results)

    draw_comparison(
        results,
        x_metric="forward_transfer",
        xlabel="Forward transfer",
        output_path=output_dir / "paper_style_forward_transfer_vs_performance.png",
    )
    draw_comparison(
        results,
        x_metric="forgetting",
        xlabel="Average forgetting (lower is better)",
        output_path=output_dir / "paper_style_forgetting_vs_performance.png",
    )
    print(f"Plots written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
