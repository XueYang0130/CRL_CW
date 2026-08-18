#!/usr/bin/env python3
"""Plot IQM forward transfer versus average performance with bootstrap CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "CloneX-SAC": "CloneX-SAC",
    "Adaptive PCGrad": "Adaptive PCGrad",
    "Frozen Transfer PCGrad": "Frozen Transfer PCGrad",
    "Full BC": "Full BC",
}


def plot_scenario(payload: dict, scenario: str, output_path: Path, title: str) -> None:
    methods = payload[scenario]["methods"]
    order = [
        "Full BC",
        "Frozen Transfer PCGrad",
        "Adaptive PCGrad",
        "CloneX-SAC",
    ]
    colors = {
        "Full BC": "#4C78A8",
        "Frozen Transfer PCGrad": "#59A14F",
        "Adaptive PCGrad": "#B279A2",
        "CloneX-SAC": "#D4B55B",
    }

    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=180)
    for label in order:
        if label not in methods:
            continue
        item = methods[label]
        x = item["raw_ft"]["iqm"]
        y = item["ap"]["iqm"]
        x_ci = item["raw_ft"]["ci95_iqm"]
        y_ci = item["ap"]["ci95_iqm"]
        ax.errorbar(
            x,
            y,
            xerr=[[x - x_ci[0]], [x_ci[1] - x]],
            yerr=[[y - y_ci[0]], [y_ci[1] - y]],
            fmt="o",
            ms=12,
            color=colors[label],
            markeredgecolor="#555555",
            markeredgewidth=0.8,
            ecolor=colors[label],
            elinewidth=1.5,
            capsize=5,
            label=METHOD_LABELS[label],
            alpha=0.92,
        )

    ax.set_xlabel("Forward transfer", fontsize=18)
    ax.set_ylabel("Average performance", fontsize=18)
    ax.set_title(title, fontsize=17, pad=14)
    ax.set_xlim(-0.05, 0.32)
    ax.set_ylim(0.70, 1.02)
    ax.set_xticks([-0.05, 0.0, 0.1, 0.2, 0.3])
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.grid(True, color="#D0D0D0", linewidth=1.0, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/analysis/matched_seed_iqm_ci/matched_seed_iqm_ci.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/matched_seed_iqm_ci"),
    )
    args = parser.parse_args()
    with args.input.open() as handle:
        payload = json.load(handle)

    plot_scenario(
        payload,
        "common_all_methods",
        args.output_dir / "iqm_forward_transfer_vs_performance_common4.png",
        "CW10: IQM forward transfer vs. average performance (matched seeds 0–2)",
    )
    plot_scenario(
        payload,
        "common_core_methods",
        args.output_dir / "iqm_forward_transfer_vs_performance_main3.png",
        "CW10: IQM forward transfer vs. average performance (matched seeds 0–5)",
    )


if __name__ == "__main__":
    main()
