#!/usr/bin/env python3
"""Recreate the ClonEx-style CW10 scatter plots with current results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER = {
    "Fine-tuning": ((0.31, 0.27, 0.34), (0.10, 0.10, 0.10), (0.75, 0.73, 0.76)),
    "Fine-tuning, best-return exploration": ((0.30, 0.25, 0.34), (0.10, 0.10, 0.11), (0.73, 0.71, 0.75)),
    "A-GEM": ((0.26, 0.22, 0.29), (0.13, 0.12, 0.14), (0.68, 0.66, 0.70)),
    "CloneX-SAC (paper)": ((0.44, 0.42, 0.46), (0.86, 0.84, 0.87), (0.02, 0.01, 0.04)),
    "Behavioral cloning (paper)": ((0.41, 0.38, 0.43), (0.84, 0.81, 0.86), (0.02, 0.01, 0.03)),
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
    "CloneX-SAC (paper)": "#D4B55B",
    "Behavioral cloning (paper)": "#777777",
    "EWC": "#59A14F",
    "L2": "#F28E2B",
    "MAS": "#C44E52",
    "PackNet": "#CC79A7",
    "Perfect memory": "#8C564B",
    "VCL": "#8172B2",
}

OURS = {
    "Full BC (ours)": ("Full BC", "#2F6FAE"),
    "Frozen Transfer PCGrad (ours)": ("Frozen Transfer PCGrad", "#2E8B57"),
    "Adaptive PCGrad (ours)": ("Adaptive PCGrad", "#A24E9A"),
    "CloneX-SAC (ours)": ("CloneX-SAC", "#B58B16"),
}


def errors(value: float, interval: list[float]) -> tuple[list[float], list[float]]:
    return [[value - interval[0]], [interval[1] - value]]


def draw_point(ax, x, y, x_ci, y_ci, color, label, marker="o", alpha=0.88, size=9, zorder=2):
    ax.errorbar(
        x,
        y,
        xerr=errors(x, x_ci),
        yerr=errors(y, y_ci),
        fmt=marker,
        ms=size,
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


def style_axis(ax, ylabel: str) -> None:
    ax.set_xlabel("Forward transfer", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlim(-1.30, 0.60)
    ax.set_xticks([-1.2, -0.8, -0.4, 0.0, 0.4, 0.6])
    ax.grid(True, color="#D0D0D0", linewidth=0.9, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)


def add_paper_points(ax, y_index: int) -> None:
    for label, (x, y, forget) in PAPER.items():
        values = (x, y, forget)
        v, lo, hi = values[y_index]
        x_value, x_lo, x_hi = values[0]
        draw_point(
            ax,
            x_value,
            v,
            [x_lo, x_hi],
            [lo, hi],
            PAPER_COLORS[label],
            label,
            marker="o",
            alpha=0.58,
            size=8,
            zorder=1,
        )


def add_ours(ax, payload: dict, y_index: int) -> None:
    methods = payload["common_all_methods"]["methods"]
    for display, (method, color) in OURS.items():
        item = methods[method]
        x = item["raw_ft"]["iqm"]
        x_ci = item["raw_ft"]["ci95_iqm"]
        if y_index == 1:
            y = item["ap"]["iqm"]
            y_ci = item["ap"]["ci95_iqm"]
        else:
            y = item["forget"]["iqm"]
            y_ci = item["forget"]["ci95_iqm"]
        draw_point(ax, x, y, x_ci, y_ci, color, display, marker="D", alpha=0.95, size=8, zorder=3)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = root / "outputs/analysis/matched_seed_iqm_ci/matched_seed_iqm_ci.json"
    output_dir = root / "outputs/analysis/matched_seed_iqm_ci"
    with input_path.open() as handle:
        payload = json.load(handle)

    fig, ax = plt.subplots(figsize=(11.5, 7.8), dpi=180)
    add_paper_points(ax, 1)
    add_ours(ax, payload, 1)
    style_axis(ax, "Average performance")
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="lower left", fontsize=9, ncol=2, framealpha=0.92, edgecolor="#D0D0D0")
    fig.tight_layout()
    fig.savefig(output_dir / "paper_style_cw10_forward_transfer_performance.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 7.8), dpi=180)
    add_paper_points(ax, 2)
    add_ours(ax, payload, 2)
    style_axis(ax, "Average forgetting (lower is better)")
    ax.set_ylim(-0.10, 0.80)
    ax.set_yticks([-0.1, 0.0, 0.2, 0.4, 0.6, 0.8])
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.92, edgecolor="#D0D0D0")
    fig.tight_layout()
    fig.savefig(output_dir / "paper_style_cw10_forward_transfer_forgetting.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
