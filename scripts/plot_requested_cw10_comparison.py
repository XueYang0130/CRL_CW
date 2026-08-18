#!/usr/bin/env python3
"""Plot the requested ClonEx-style CW10 comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# CW10 values reported in Table 6 of Disentangling Transfer in Continual RL.
# Each tuple is (value, lower CI, upper CI), with the paper's 90% intervals.
PAPER = {
    "Fine-tuning": ((0.31, 0.27, 0.34), (0.10, 0.10, 0.10), (0.75, 0.73, 0.76)),
    "Fine-tuning, best-return exploration": ((0.30, 0.25, 0.34), (0.10, 0.10, 0.11), (0.73, 0.71, 0.75)),
    "A-GEM": ((0.26, 0.22, 0.29), (0.13, 0.12, 0.14), (0.68, 0.66, 0.70)),
    "Behavioral cloning": ((0.41, 0.38, 0.43), (0.84, 0.81, 0.86), (0.02, 0.01, 0.03)),
    "EWC": ((0.04, -0.04, 0.12), (0.64, 0.60, 0.68), (0.06, 0.03, 0.09)),
    "L2": ((-0.34, -0.47, -0.21), (0.53, 0.49, 0.58), (0.02, -0.00, 0.04)),
    "MAS": ((-0.06, -0.14, -0.00), (0.53, 0.50, 0.57), (0.11, 0.09, 0.13)),
    "PackNet": ((0.26, 0.22, 0.29), (0.84, 0.81, 0.86), (-0.01, -0.02, 0.00)),
    "Perfect memory": ((-1.13, -1.23, -1.04), (0.27, 0.24, 0.30), (0.03, 0.00, 0.05)),
    "VCL": ((-0.37, -0.47, -0.28), (0.55, 0.51, 0.59), (-0.03, -0.05, 0.01)),
}

PAPER_COLORS = {
    "Fine-tuning": "#4C78A8",
    "Fine-tuning, best-return exploration": "#F28E2B",
    "A-GEM": "#9C755F",
    "Behavioral cloning": "#777777",
    "EWC": "#59A14F",
    "L2": "#F28E2B",
    "MAS": "#C44E52",
    "PackNet": "#CC79A7",
    "Perfect memory": "#8C564B",
    "VCL": "#8172B2",
}

CURRENT = {
    "Adaptive PCGrad": ("Adaptive PCGrad", "#A24E9A"),
    "Frozen Transfer PCGrad": ("Frozen Transfer PCGrad", "#2E8B57"),
    "CloneX-SAC": ("CloneX-SAC", "#B58B16"),
}


def err(value: float, interval: list[float]) -> list[list[float]]:
    return [[value - interval[0]], [interval[1] - value]]


def point(ax, x, y, x_ci, y_ci, color, label, marker, alpha, zorder):
    ax.errorbar(
        x,
        y,
        xerr=err(x, x_ci),
        yerr=err(y, y_ci),
        fmt=marker,
        ms=9,
        color=color,
        ecolor=color,
        elinewidth=1.35,
        capsize=4,
        markeredgecolor="#555555",
        markeredgewidth=0.65,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )


def base_axis(ax, xlabel: str, ylabel: str):
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(True, color="#D0D0D0", linewidth=0.9, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)


def add_paper(ax, x_metric: int, y_metric: int):
    for label, values in PAPER.items():
        x, y = values[x_metric], values[y_metric]
        point(ax, x[0], y[0], [x[1], x[2]], [y[1], y[2]], PAPER_COLORS[label], label, "o", 0.58, 1)


def add_current(ax, payload: dict, x_metric: str, y_metric: str):
    # Full BC is excluded, so the three current methods use their shared seeds 0--5.
    methods = payload["common_core_methods"]["methods"]
    for display, (method, color) in CURRENT.items():
        item = methods[method]
        x = item[x_metric]
        y = item[y_metric]
        point(ax, x["iqm"], y["iqm"], x["ci95_iqm"], y["ci95_iqm"], color, display, "D", 0.95, 3)


def finish(ax, output_path: Path, legend_loc: str, xlim=None, xticks=None):
    ax.legend(loc=legend_loc, fontsize=9, ncol=2, framealpha=0.92, edgecolor="#D0D0D0")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, bbox_inches="tight")
    plt.close(ax.figure)


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs/analysis/matched_seed_iqm_ci/matched_seed_iqm_ci.json"
    output_dir = source.parent
    with source.open() as handle:
        payload = json.load(handle)

    # x = forward transfer, y = average performance.
    fig, ax = plt.subplots(figsize=(11.5, 7.8), dpi=180)
    add_paper(ax, 0, 1)
    add_current(ax, payload, "raw_ft", "ap")
    base_axis(ax, "Forward transfer", "Average performance")
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    finish(
        ax,
        output_dir / "requested_forward_transfer_vs_average_performance.png",
        "lower left",
        xlim=(-1.30, 0.60),
        xticks=[-1.2, -0.8, -0.4, 0.0, 0.4, 0.6],
    )

    # x = average performance, y = forgetting.
    fig, ax = plt.subplots(figsize=(11.5, 7.8), dpi=180)
    add_paper(ax, 1, 2)
    add_current(ax, payload, "ap", "forget")
    base_axis(ax, "Average performance", "Average forgetting (lower is better)")
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.10, 0.80)
    ax.set_yticks([-0.1, 0.0, 0.2, 0.4, 0.6, 0.8])
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    finish(
        ax,
        output_dir / "requested_forgetting_vs_average_performance.png",
        "upper left",
        xlim=(0.0, 1.05),
        xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )


if __name__ == "__main__":
    main()
