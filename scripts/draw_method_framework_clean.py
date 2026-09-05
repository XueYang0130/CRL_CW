#!/usr/bin/env python3
"""Draw a clean paper-style framework figure for the current method."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


BLUE = "#0E56B3"
GREEN = "#216B35"
ORANGE = "#D85C00"
PURPLE = "#5534A5"
TEXT = "#172033"


def box(ax, x, y, w, h, text, edge, face, fs=14.0, weight="bold", text_y=0.63):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=2.0,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * text_y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        weight=weight,
        color=edge,
        linespacing=1.15,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, color, lw=2.3, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
            zorder=1,
        )
    )


def db_icon(ax, cx, cy, color):
    ax.add_patch(Rectangle((cx - 0.014, cy - 0.020), 0.028, 0.040, color=color, alpha=0.96, zorder=4))
    for off in (0.020, 0.000, -0.020):
        ax.add_patch(Circle((cx, cy + off), 0.014, facecolor=color, edgecolor="white", lw=0.8, zorder=5))


def task_stack_icon(ax, cx, cy, color):
    for i in range(4):
        ax.add_patch(
            Rectangle(
                (cx - 0.030 + i * 0.010, cy - 0.020 + i * 0.010),
                0.040,
                0.040,
                facecolor=color,
                edgecolor="white",
                lw=1.0,
                alpha=0.78,
                zorder=4,
            )
        )


def rollout_icon(ax, x, y, color):
    pts = [(x, y), (x + 0.020, y + 0.016), (x + 0.043, y + 0.010), (x + 0.068, y + 0.040)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.8, zorder=4)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.0065, color=color, zorder=5))
    ax.plot([x + 0.068, x + 0.068], [y + 0.040, y + 0.067], color=color, lw=1.8, zorder=4)
    ax.add_patch(
        FancyArrowPatch(
            (x + 0.068, y + 0.067),
            (x + 0.096, y + 0.067),
            arrowstyle="-|>",
            mutation_scale=11,
            color=color,
            lw=1.8,
            zorder=4,
        )
    )


def gradient_icon(ax, cx, cy, color):
    ax.plot([cx - 0.030, cx - 0.012, cx + 0.005, cx + 0.026], [cy - 0.020, cy + 0.004, cy - 0.008, cy + 0.025], color=color, lw=2.0, zorder=4)
    for px, py in [(cx - 0.030, cy - 0.020), (cx - 0.012, cy + 0.004), (cx + 0.005, cy - 0.008), (cx + 0.026, cy + 0.025)]:
        ax.add_patch(Circle((px, py), 0.0055, facecolor="white", edgecolor=color, lw=1.4, zorder=5))


def network_icon(ax, cx, cy, color, scale=1.0):
    pts = [
        (cx, cy + 0.030 * scale),
        (cx - 0.035 * scale, cy),
        (cx + 0.035 * scale, cy),
        (cx, cy - 0.032 * scale),
    ]
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=color, lw=1.6, zorder=4)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.010 * scale, facecolor="#E8DFFF", edgecolor=color, lw=1.4, zorder=5))


def heads_icon(ax, x, y, color):
    for i in range(4):
        ax.add_patch(
            Rectangle(
                (x + i * 0.031, y),
                0.018,
                0.018,
                facecolor="#E8DFFF" if i == 0 else "white",
                edgecolor=color,
                lw=1.4,
                linestyle="-" if i == 0 else "--",
                zorder=4,
            )
        )


def globe_icon(ax, cx, cy, color):
    ax.add_patch(Circle((cx, cy), 0.025, facecolor="none", edgecolor=color, lw=1.8, zorder=4))
    ax.plot([cx - 0.025, cx + 0.025], [cy, cy], color=color, lw=1.4, zorder=4)
    ax.plot([cx, cx], [cy - 0.025, cy + 0.025], color=color, lw=1.4, zorder=4)
    ax.add_patch(Circle((cx, cy), 0.014, facecolor="none", edgecolor=color, lw=1.2, zorder=4))


def gear_icon(ax, cx, cy, color, scale=1.0):
    ax.add_patch(Circle((cx, cy), 0.022 * scale, facecolor=color, edgecolor=color, lw=1.0, zorder=4))
    ax.add_patch(Circle((cx, cy), 0.010 * scale, facecolor="white", edgecolor="white", lw=0.8, zorder=5))
    for dx, dy in [(0.032, 0), (-0.032, 0), (0, 0.032), (0, -0.032)]:
        ax.plot(
            [cx + dx * 0.62 * scale, cx + dx * scale],
            [cy + dy * 0.62 * scale, cy + dy * scale],
            color=color,
            lw=2.3 * scale,
            zorder=4,
        )


def main() -> None:
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "method_framework_clean.png"
    svg_path = out_dir / "method_framework_clean.svg"

    fig, ax = plt.subplots(figsize=(17.8, 7.6), dpi=240)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Current-task stream.
    box(ax, 0.035, 0.705, 0.135, 0.155, "Current Task\n$T_k$", BLUE, "#F6FAFF", fs=14.2)
    box(ax, 0.205, 0.705, 0.135, 0.155, "Online\nReplay Buffer", BLUE, "#F6FAFF", fs=14.2)
    box(ax, 0.380, 0.705, 0.150, 0.155, "Current-task\nSAC Loss\n$L^k_{SAC}$", BLUE, "#F6FAFF", fs=13.8)
    box(ax, 0.575, 0.715, 0.125, 0.135, "Actor SAC\nGradient\n$g_{SAC}$", BLUE, "#F6FAFF", fs=11.9)
    box(ax, 0.805, 0.730, 0.150, 0.125, "Critic\nUpdate", PURPLE, "#FAF8FF", fs=13.8)

    globe_icon(ax, 0.102, 0.728, BLUE)
    db_icon(ax, 0.272, 0.724, BLUE)
    gradient_icon(ax, 0.640, 0.735, BLUE)
    gradient_icon(ax, 0.880, 0.755, PURPLE)

    arrow(ax, (0.170, 0.782), (0.205, 0.782), BLUE)
    arrow(ax, (0.340, 0.782), (0.380, 0.782), BLUE)
    arrow(ax, (0.530, 0.782), (0.575, 0.782), BLUE)
    arrow(ax, (0.530, 0.820), (0.805, 0.792), BLUE)

    # Old-task stream.
    box(ax, 0.035, 0.335, 0.135, 0.155, "Previous Tasks\n$T_1 \\ldots T_{k-1}$", GREEN, "#F7FBF5", fs=13.6)
    box(ax, 0.205, 0.335, 0.135, 0.155, "Successful\nOld Rollouts", GREEN, "#F7FBF5", fs=13.8)
    box(ax, 0.380, 0.335, 0.155, 0.155, "Reference Memory\n$(s, \\mu_b, \\log\\sigma_b,$\n$task\\_id)$", GREEN, "#F7FBF5", fs=10.4)
    box(ax, 0.575, 0.350, 0.125, 0.125, "BC KL\nLoss\n$L^k_{KL}$", GREEN, "#F7FBF5", fs=12.4)
    box(ax, 0.735, 0.350, 0.130, 0.125, "Actor BC\nGradient\n$g_{BC}$", GREEN, "#F7FBF5", fs=12.0)

    task_stack_icon(ax, 0.102, 0.353, GREEN)
    rollout_icon(ax, 0.242, 0.350, GREEN)
    db_icon(ax, 0.458, 0.348, GREEN)
    gradient_icon(ax, 0.800, 0.370, GREEN)

    arrow(ax, (0.170, 0.412), (0.205, 0.412), GREEN)
    arrow(ax, (0.340, 0.412), (0.380, 0.412), GREEN)
    arrow(ax, (0.535, 0.412), (0.575, 0.412), GREEN)
    arrow(ax, (0.700, 0.412), (0.735, 0.412), GREEN)

    # Central method block.
    box(
        ax,
        0.590,
        0.535,
        0.205,
        0.125,
        "",
        ORANGE,
        "#FFF6EF",
        fs=12.5,
        text_y=0.73,
    )
    gear_icon(ax, 0.620, 0.600, ORANGE, scale=0.46)
    ax.text(
        0.715,
        0.615,
        "SAC-priority\nBC Gradient Balancing",
        ha="center",
        va="center",
        fontsize=12.1,
        weight="bold",
        color=ORANGE,
        linespacing=1.15,
        zorder=3,
    )
    ax.text(
        0.715,
        0.550,
        "project conflicts  ·  adaptive norm cap",
        ha="center",
        va="center",
        fontsize=9.3,
        color=ORANGE,
        zorder=3,
    )

    arrow(ax, (0.638, 0.715), (0.662, 0.660), BLUE)
    arrow(ax, (0.800, 0.475), (0.730, 0.535), GREEN, rad=-0.10)

    # Actor update outputs.
    box(ax, 0.835, 0.535, 0.130, 0.125, "Actor\nUpdate", ORANGE, "#FFF6EF", fs=13.5)
    box(ax, 0.805, 0.235, 0.150, 0.130, "Shared Actor\nBackbone", PURPLE, "#FAF8FF", fs=12.3, text_y=0.70)
    box(ax, 0.805, 0.080, 0.150, 0.115, "Task Heads", PURPLE, "#FAF8FF", fs=13.0)
    network_icon(ax, 0.880, 0.262, PURPLE, scale=0.72)
    heads_icon(ax, 0.835, 0.105, PURPLE)

    arrow(ax, (0.795, 0.598), (0.835, 0.598), ORANGE)
    arrow(ax, (0.900, 0.535), (0.880, 0.365), PURPLE)
    arrow(ax, (0.900, 0.535), (0.880, 0.195), PURPLE)

    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.10)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
