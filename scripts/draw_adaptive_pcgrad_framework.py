#!/usr/bin/env python3
"""Draw a clean framework figure for Success Replay Best Adaptive PCGrad."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


BLUE = "#1B5BBE"
GREEN = "#2F6B35"
ORANGE = "#D96400"
PURPLE = "#5A3AA5"
GRAY = "#46515F"


def add_box(ax, x, y, w, h, text, color, face, fontsize=13, dashed=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=2.0,
        edgecolor=color,
        facecolor=face,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold",
        color=color,
        linespacing=1.15,
        zorder=3,
    )
    return patch


def add_arrow(ax, start, end, color, rad=0.0, lw=2.2, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=lw,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=5,
            shrinkB=5,
            zorder=1,
        )
    )


def database_icon(ax, cx, cy, color):
    ax.add_patch(Rectangle((cx - 0.012, cy - 0.020), 0.024, 0.040, color=color, alpha=0.95, zorder=4))
    ax.add_patch(Circle((cx, cy + 0.020), 0.012, color=color, alpha=0.95, zorder=4))
    ax.add_patch(Circle((cx, cy), 0.012, color="white", alpha=0.92, zorder=5))
    ax.add_patch(Circle((cx, cy - 0.020), 0.012, color=color, alpha=0.95, zorder=4))


def rollout_icon(ax, x, y, color):
    pts = [(x, y), (x + 0.020, y + 0.016), (x + 0.045, y + 0.010), (x + 0.070, y + 0.040)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.8, zorder=4)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.007, color=color, zorder=5))
    ax.plot([x + 0.070, x + 0.070], [y + 0.040, y + 0.070], color=color, lw=1.8, zorder=4)
    ax.add_patch(
        FancyArrowPatch(
            (x + 0.070, y + 0.070),
            (x + 0.100, y + 0.070),
            arrowstyle="-|>",
            mutation_scale=11,
            color=color,
            lw=1.8,
            zorder=4,
        )
    )


def network_icon(ax, cx, cy, color):
    pts = [(cx, cy + 0.030), (cx - 0.036, cy), (cx + 0.036, cy), (cx, cy - 0.032)]
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=color, lw=1.6, zorder=4)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.010, facecolor="#E7DDFC", edgecolor=color, lw=1.5, zorder=5))


def heads_icon(ax, x, y, color):
    for i in range(4):
        ax.add_patch(
            Rectangle(
                (x + i * 0.034, y),
                0.020,
                0.020,
                facecolor="#E7DDFC" if i == 0 else "white",
                edgecolor=color,
                lw=1.5,
                linestyle="-" if i == 0 else "--",
                zorder=4,
            )
        )
        if i > 0:
            ax.plot([x + (i - 1) * 0.034 + 0.020, x + i * 0.034], [y + 0.010, y + 0.010], color=color, lw=1.4, zorder=4)


def main() -> None:
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "adaptive_pcgrad_framework.png"
    svg_path = out_dir / "adaptive_pcgrad_framework.svg"

    fig, ax = plt.subplots(figsize=(18.0, 9.0), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.text(0.035, 0.930, "Current-task learning", fontsize=18, weight="bold", color=BLUE)
    ax.text(0.035, 0.500, "Old-task retention", fontsize=18, weight="bold", color=GREEN)

    # Current-task SAC stream.
    add_box(ax, 0.035, 0.760, 0.130, 0.120, "Current Task\n$T_k$", BLUE, "#F4F8FF", fontsize=14)
    add_box(ax, 0.205, 0.760, 0.135, 0.120, "Online\nReplay Buffer", BLUE, "#F4F8FF", fontsize=14)
    add_box(ax, 0.380, 0.760, 0.145, 0.120, "Current-task\nSAC Loss\n$L^k_{SAC}$", BLUE, "#F4F8FF", fontsize=14)
    add_box(ax, 0.575, 0.772, 0.120, 0.096, "Actor SAC\nGradient\n$g_{SAC}$", BLUE, "#F4F8FF", fontsize=12.5)
    add_box(ax, 0.805, 0.782, 0.160, 0.110, "Critic Update\n$\\nabla_Q L^k_{SAC}$", PURPLE, "#FAF8FF", fontsize=12.8)

    add_arrow(ax, (0.165, 0.820), (0.205, 0.820), BLUE)
    add_arrow(ax, (0.340, 0.820), (0.380, 0.820), BLUE)
    add_arrow(ax, (0.525, 0.820), (0.575, 0.820), BLUE)
    add_arrow(ax, (0.525, 0.860), (0.805, 0.840), BLUE)

    # Old-task memory stream.
    add_box(ax, 0.035, 0.345, 0.130, 0.125, "Previous Tasks\n$T_1 \\ldots T_{k-1}$", GREEN, "#F6FBF5", fontsize=13.5)
    ax.text(0.100, 0.362, "▣▣▣", color=GREEN, fontsize=16, ha="center", va="center")
    add_box(ax, 0.205, 0.345, 0.135, 0.125, "Successful\nOld Rollouts", GREEN, "#F6FBF5", fontsize=14)
    add_box(ax, 0.380, 0.332, 0.155, 0.150, "Best-teacher\nRelabeling\nbest actor snapshot", GREEN, "#F6FBF5", fontsize=12.8)
    add_box(
        ax,
        0.575,
        0.332,
        0.170,
        0.150,
        "Mixed Success\nReference Memory\n$(s, \\mu_b, \\log\\sigma_b, task\\_id)$",
        GREEN,
        "#F6FBF5",
        fontsize=11.8,
    )
    add_box(ax, 0.575, 0.205, 0.170, 0.080, "BC KL Loss\n$L^k_{KL}$", GREEN, "#F6FBF5", fontsize=12.5)
    add_box(ax, 0.380, 0.185, 0.155, 0.080, "Optional Semantic /\nMemory Gate", GREEN, "white", fontsize=11.2, dashed=True)

    add_arrow(ax, (0.165, 0.408), (0.205, 0.408), GREEN)
    add_arrow(ax, (0.340, 0.408), (0.380, 0.408), GREEN)
    add_arrow(ax, (0.535, 0.408), (0.575, 0.408), GREEN)
    add_arrow(ax, (0.660, 0.332), (0.660, 0.285), GREEN)
    add_arrow(ax, (0.458, 0.265), (0.575, 0.332), GREEN, dashed=True, rad=-0.15)

    # Gradient conflict control.
    add_box(ax, 0.575, 0.555, 0.230, 0.135, "Adaptive PCGrad\nConflict-gradient projection on BC\nSAC-norm adaptive scaling", ORANGE, "#FFF5EA", fontsize=12.7)
    add_box(ax, 0.840, 0.555, 0.125, 0.135, "Joint Actor\nUpdate\n$g_{SAC}+\\tilde g_{BC}$", ORANGE, "#FFF5EA", fontsize=12.5)
    add_arrow(ax, (0.635, 0.772), (0.650, 0.690), BLUE)
    add_arrow(ax, (0.745, 0.245), (0.690, 0.555), GREEN, rad=-0.20)
    add_arrow(ax, (0.805, 0.622), (0.840, 0.622), ORANGE)

    # Actor outputs.
    add_box(ax, 0.805, 0.330, 0.160, 0.110, "Updated Shared\nActor Backbone", PURPLE, "#FAF8FF", fontsize=12.8)
    add_box(ax, 0.805, 0.135, 0.160, 0.125, "Current-task Head\n+\nRegularized Old Heads", PURPLE, "#FAF8FF", fontsize=12.2)
    add_arrow(ax, (0.900, 0.555), (0.885, 0.440), PURPLE, rad=0.10)
    add_arrow(ax, (0.915, 0.555), (0.885, 0.260), PURPLE, rad=-0.09)

    ax.text(
        0.500,
        0.055,
        "Success Replay Best Adaptive PCGrad: successful replay + best-teacher relabeling + SAC-priority gradient conflict control",
        ha="center",
        va="center",
        color=GRAY,
        fontsize=13.5,
        weight="bold",
    )

    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
