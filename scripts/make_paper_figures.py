#!/usr/bin/env python3
"""Render paper figures from saved tables; never fit or select knots."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOURCES = {
    "2": "33_view_ablation_and_fibers/heldout_view_ablation_n31_summary.csv",
    "3": "31B_corrected_four_view_mirror_withheld_khovanov/four_view_mirror_withheld_khovanov_summary.csv",
    "4": "34_turaev_width_consequence_audit/WKh_ge4_selected_count_audit.csv",
}
NAMES = {"1": "fig1_score_geometry", "2": "fig2_view_ablation",
         "3": "fig3_mirror_assignments", "4": "fig4_width_topology"}


def load_table(root, figure, index, order):
    path = root / SOURCES[figure]
    if not path.is_file():
        raise FileNotFoundError(f"Figure {figure} requires the saved table: {path}")
    frame = pd.read_csv(path)
    if frame[index].duplicated().any():
        raise ValueError(f"Duplicate {index} in {path}")
    missing = set(order) - set(frame[index])
    if missing:
        raise ValueError(f"Missing rows in {path}: {sorted(missing)}")
    return frame.set_index(index).loc[order].reset_index()

def figure_1(root):
    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.patches import FancyBboxPatch, Arc


    # ============================================================
    # GLOBAL STYLE
    # ============================================================

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "figure.titlesize": 16,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(13.8, 4.8))

    gs = fig.add_gridspec(
        1, 3,
        wspace=0.12
    )


    # ============================================================
    # CARD HELPER
    # ============================================================

    def add_card(ax):

        card = FancyBboxPatch(
            (0.01, 0.02),
            0.98,
            0.96,
            transform=ax.transAxes,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.0,
            edgecolor="0.82",
            facecolor="white",
            zorder=-10,
        )

        ax.add_patch(card)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])


    label_box = dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.90,
        pad=1.5
    )


    # ============================================================
    # PANEL A
    # RAW DISTANCE
    # ============================================================

    ax = fig.add_subplot(gs[0, 0])

    add_card(ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


    # title
    ax.text(
        0.25, 0.92,
        "(a) Raw distance",
        fontsize=12.5,
        fontweight="semibold",
        ha="left",
        va="center"
    )


    # equation
    ax.text(
        0.36, 0.82,
        r"$D(x)=r^2\sin^2\theta$",
        fontsize=10,
        ha="left"
    )


    # PCA subspace
    y0 = 0.25

    ax.plot(
        [0.12, 0.88],
        [y0, y0],
        linewidth=2
    )

    ax.text(
        0.50, 0.185,
        "PCA subspace",
        ha="center",
        fontsize=9.5,
        color="0.25"
    )


    # geometry
    origin = np.array([0.15, y0])
    x = np.array([0.72, 0.66])
    proj = np.array([x[0], y0])

    ax.plot(
        [origin[0], x[0]],
        [origin[1], x[1]],
        linewidth=2.3
    )

    ax.plot(
        [proj[0], x[0]],
        [proj[1], x[1]],
        linestyle="--",
        linewidth=1.5
    )

    ax.scatter(
        [origin[0], proj[0], x[0]],
        [origin[1], proj[1], x[1]],
        s=[28, 32, 55],
        zorder=5
    )


    # labels
    ax.text(
        x[0] + 0.02,
        x[1] + 0.01,
        r"$x=ru$",
        fontsize=11
    )

    # radius label moved ABOVE the line
    ax.text(
        0.42, 0.50,
        r"$r=\|x\|$",
        fontsize=10.5,
        bbox=label_box
    )

    # residual label moved right of dashed line
    ax.text(
        0.745, 0.45,
        r"$\|(I-P)x\|$",
        fontsize=10.5,
        ha="left",
        bbox=label_box
    )

    # projection label below line
    ax.text(
        0.75, 0.195,
        r"$Px$",
        fontsize=10.5,
        ha="center"
    )


    # theta
    angle_deg = np.degrees(
        np.arctan2(
            x[1] - origin[1],
            x[0] - origin[0]
        )
    )

    arc = Arc(
        origin,
        0.22,
        0.22,
        theta1=0,
        theta2=angle_deg,
        linewidth=1.2
    )

    ax.add_patch(arc)

    ax.text(
        0.260, 0.295,
        r"$\theta$",
        fontsize=10.5
    )


    # interpretation
    ax.text(
        0.50, 0.075,
        "Depends on both radius and angular deviation.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="0.30"
    )


    # ============================================================
    # PANEL B
    # RELATIVE ANGLE
    # ============================================================

    ax = fig.add_subplot(gs[0, 1])

    add_card(ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


    # title
    ax.text(
        0.25, 0.92,
        "(b) Relative angle",
        fontsize=12.5,
        fontweight="semibold",
        ha="left",
        va="center"
    )


    # equation
    ax.text(
        0.32, 0.82,
        r"$R(x)=\frac{D(x)}{\|x\|^2}=\sin^2\theta$",
        fontsize=10,
        ha="left"
    )


    # PCA subspace
    y0 = 0.25

    ax.plot(
        [0.12, 0.88],
        [y0, y0],
        linewidth=2
    )

    ax.text(
        0.50, 0.185,
        "PCA subspace",
        ha="center",
        fontsize=9.5,
        color="0.25"
    )


    origin = np.array([0.15, y0])

    theta = np.deg2rad(31)

    r1 = 0.38
    r2 = 0.65

    p1 = origin + np.array([
        r1*np.cos(theta),
        r1*np.sin(theta)
    ])

    p2 = origin + np.array([
        r2*np.cos(theta),
        r2*np.sin(theta)
    ])


    for p in [p1, p2]:

        ax.plot(
            [origin[0], p[0]],
            [origin[1], p[1]],
            linewidth=2.2
        )

        ax.plot(
            [p[0], p[0]],
            [y0, p[1]],
            linestyle="--",
            linewidth=1.3
        )

        ax.scatter(
            [p[0]],
            [p[1]],
            s=50,
            zorder=5
        )


    ax.scatter(
        [origin[0]],
        [origin[1]],
        s=28,
        zorder=5
    )


    # r labels placed outside lines
    ax.text(
        p1[0] + 0.018,
        p1[1] + 0.028,
        r"$r_1$",
        fontsize=10.5,
        bbox=label_box
    )

    ax.text(
        p2[0] + 0.018,
        p2[1] + 0.028,
        r"$r_2$",
        fontsize=10.5,
        bbox=label_box
    )


    # theta
    arc = Arc(
        origin,
        0.22,
        0.22,
        theta1=0,
        theta2=np.degrees(theta),
        linewidth=1.2
    )

    ax.add_patch(arc)

    ax.text(
        0.265, 0.285,
        r"$\theta$",
        fontsize=10.5
    )


    # explanation deliberately placed BELOW the vectors
    ax.text(
        0.595, 0.33,
        "same angle",
        ha="center",
        fontsize=8.3,
        bbox=label_box
    )

    ax.text(
        0.595, 0.285,
        "different radii",
        ha="center",
        fontsize=7,
        color="0.35",
        bbox=label_box
    )


    # interpretation
    ax.text(
        0.50, 0.075,
        "Removes radius and keeps angular deviation.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="0.30"
    )


    # ============================================================
    # PANEL C
    # CONDITIONAL EXTREMENESS
    # ============================================================

    ax = fig.add_subplot(gs[0, 2])

    add_card(ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


    # title
    ax.text(
        0.08, 0.92,
        "(c) Conditional extremeness",
        fontsize=12.5,
        fontweight="semibold",
        ha="left",
        va="center"
    )


    # equation
    ax.text(
        0.06, 0.82,
        r"$C(x)=$ local percentile of $D(x)$ among similar $r$",
        fontsize=9,
        ha="left"
    )


    # schematic axes
    x_left = 0.16
    x_right = 0.88
    y_bottom = 0.25
    y_top = 0.70

    ax.plot(
        [x_left, x_right],
        [y_bottom, y_bottom],
        linewidth=1.5,
        color="0.35"
    )

    ax.plot(
        [x_left, x_left],
        [y_bottom, y_top],
        linewidth=1.5,
        color="0.35"
    )


    ax.text(
        0.52,
        0.19,
        "representation radius",
        ha="center",
        fontsize=9.5
    )

    ax.text(
        0.075,
        0.47,
        "reconstruction\nerror",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9.5
    )


    # radius band
    band_left = 0.53
    band_right = 0.68

    rect = FancyBboxPatch(
        (band_left, y_bottom),
        band_right - band_left,
        y_top - y_bottom,
        boxstyle="square,pad=0",
        linewidth=0,
        facecolor="tab:blue",
        alpha=0.09,
        zorder=-1
    )

    ax.add_patch(rect)


    # background points
    points = np.array([
        [0.22, 0.31],
        [0.29, 0.35],
        [0.36, 0.39],
        [0.43, 0.43],
        [0.50, 0.46],

        [0.56, 0.47],
        [0.59, 0.51],
        [0.61, 0.55],
        [0.64, 0.50],
        [0.66, 0.58],

        [0.73, 0.60],
        [0.79, 0.63],
        [0.83, 0.67],
    ])

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=26,
        color="0.65",
        zorder=3
    )


    # target
    target = np.array([
        0.61,
        0.65
    ])

    ax.scatter(
        [target[0]],
        [target[1]],
        s=110,
        marker="*",
        zorder=6
    )


    # clean annotation in free space
    ax.annotate(
        "unusually large error",
        xy=target,
        xytext=(0.35, 0.735),
        fontsize=9.6,
        ha="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.0
        )
    )




    # interpretation
    ax.text(
        0.50, 0.075,
        "Ranks residuals among knots with similar radius.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="0.30"
    )


    # ============================================================
    # GLOBAL TITLE
    # ============================================================

    fig.suptitle(
        "Three geometries of anomaly scoring",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )


    return fig

def figure_2(root):
    df = load_table(root, "2", "selection", [
        "HOMFLY_PT_alone", "Theta_alone", "HOMFLY_PT_plus_Theta_2of2", "four_view_3of4"])
    # ============================================================
    # GLOBAL STYLE
    # ============================================================

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    PALETTE = {
        "HOMFLY_PT_alone": "#B8C4D6",
        "Theta_alone": "#6E97C4",
        "HOMFLY_PT_plus_Theta_2of2": "#4E7E8C",
        "four_view_3of4": "#1F4E79",
    }

    PRETTY = {
        "HOMFLY_PT_alone": "HOMFLY-PT",
        "Theta_alone": r"$\theta$",
        "HOMFLY_PT_plus_Theta_2of2": r"$(P,\theta)$" + "\n2 of 2",
        "four_view_3of4": "Four views\n3 of 4",
    }

    order = [
        "HOMFLY_PT_alone",
        "Theta_alone",
        "HOMFLY_PT_plus_Theta_2of2",
        "four_view_3of4",
    ]

    df = (
        df
        .set_index("selection")
        .loc[order]
        .reset_index()
    )

    colors = [PALETTE[s] for s in df["selection"]]
    labels = [PRETTY[s] for s in df["selection"]]
    x = np.arange(len(df))

    fig, axes = plt.subplots(
        1, 4,
        figsize=(14.2, 4.9)
    )


    def style_ax(ax, title):
        ax.set_title(
            title,
            pad=10,
            fontweight="semibold"
        )

        ax.grid(
            axis="y",
            alpha=0.10,
            linewidth=0.8
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=0,
            ha="center",
            fontsize=9
        )


    # ============================================================
    # (a) Mean stored width
    # ============================================================

    means = df["kh_diagonal_mean"].to_numpy(float)

    lo = (
        means
        - df["kh_diagonal_mean_ci_low"].to_numpy(float)
    )

    hi = (
        df["kh_diagonal_mean_ci_high"].to_numpy(float)
        - means
    )

    axes[0].bar(
        x,
        means,
        color=colors,
        edgecolor="none",
        width=0.78
    )

    axes[0].errorbar(
        x,
        means,
        yerr=np.vstack([lo, hi]),
        fmt="none",
        ecolor="#355C7D",
        elinewidth=1.0,
        capsize=3,
        capthick=1.0
    )

    axes[0].set_ylabel(r"Mean $W_F$")

    style_ax(
        axes[0],
        "(a) Mean stored width"
    )


    # ============================================================
    # (b) Broad signal
    # ============================================================

    axes[1].bar(
        x,
        df["kh_diagonal_ge_3_prop"],
        color=colors,
        edgecolor="none",
        width=0.78
    )

    axes[1].set_ylabel("Proportion")
    axes[1].set_ylim(0, 1.02)

    style_ax(
        axes[1],
        r"(b) Broad signal: $P(W_F\geq3)$"
    )


    # ============================================================
    # (c) Extreme tail
    # ============================================================

    axes[2].bar(
        x,
        df["kh_diagonal_ge_4_prop"],
        color=colors,
        edgecolor="none",
        width=0.78
    )

    axes[2].set_ylabel("Proportion")
    axes[2].set_ylim(0, 0.35)

    style_ax(
        axes[2],
        r"(c) Extreme tail: $P(W_F\geq4)$"
    )


    # ============================================================
    # (d) Alternating knots
    # ============================================================

    axes[3].bar(
        x,
        df["alternating_prop"],
        color=colors,
        edgecolor="none",
        width=0.78
    )

    axes[3].set_ylabel("Proportion")
    axes[3].set_ylim(0, 0.72)

    style_ax(
        axes[3],
        "(d) Alternating knots"
    )


    # ============================================================
    # Global title
    # ============================================================

    fig.suptitle(
        "Selection profiles by view set and voting rule",
        fontweight="bold",
        y=1.03
    )

    plt.tight_layout()
    return fig

def figure_3(root):
    import numpy as np
    import matplotlib.pyplot as plt

    # ============================================================
    # STYLE
    # ============================================================

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Main figure palette
    BLUE = "#2A78B8"
    BLUE_DARK = "#1F4E79"
    BLUE_LIGHT = "#8FB7D9"
    ORANGE = "#F28E2B"
    GRAY = "#666666"

    # ============================================================
    # DATA
    # Replace these arrays with your stage-31B values
    # ============================================================

    order = ["baseline_canonical", "global_all_mirrored"] + [f"random_seed_{i}" for i in range(20260830, 20260835)]
    df = load_table(root, "3", "assignment", order)
    xlabels = ["canonical", "all\nmirrored"] + [f"random\n{i}" for i in range(30, 35)]
    x = np.arange(len(order))
    jaccard = df["jaccard_with_canonical"].to_numpy(float)
    recovered = df["canonical_recovered_prop"].to_numpy(float)
    mean_wf = df["kh_diagonal_mean"].to_numpy(float)
    mean_wf_ref = mean_wf[0]
    broad = df["kh_diagonal_ge_3_prop"].to_numpy(float)
    broad_ref = broad[0]
    extreme = df["kh_diagonal_ge_4_prop"].to_numpy(float)
    extreme_ref = extreme[0]

    # ============================================================
    # FIGURE
    # ============================================================

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0))
    axes = axes.ravel()

    def style_ax(ax, title):
        ax.set_title(title, pad=8, fontweight="semibold")
        ax.grid(axis="y", alpha=0.12, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.tick_params(axis="x", pad=6)

    # ============================================================
    # (a) Membership sensitivity
    # ============================================================

    ax = axes[0]

    ax.plot(
        x, jaccard,
        marker="o",
        linewidth=1.8,
        markersize=6,
        color=BLUE,
        label="Jaccard"
    )

    ax.plot(
        x, recovered,
        marker="s",
        linewidth=1.6,
        markersize=5.5,
        color=ORANGE,
        label="canonical recovered"
    )

    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    style_ax(ax, "(a) Membership sensitivity")
    ax.legend(frameon=False, loc="lower left")

    # ============================================================
    # (b) Mean stored width
    # ============================================================

    ax = axes[1]

    ax.plot(
        x, mean_wf,
        marker="o",
        linewidth=1.8,
        markersize=6,
        color=BLUE
    )

    ax.axhline(
        mean_wf_ref,
        linestyle="--",
        linewidth=1.2,
        color=BLUE,
        alpha=0.85
    )

    ax.set_ylabel(r"Mean withheld $W_F$")
    style_ax(ax, r"(b) Mean stored width")

    # tighten around the action, as in your draft
    pad = 0.004
    ax.set_ylim(mean_wf.min() - pad, mean_wf.max() + pad)

    # ============================================================
    # (c) Broad signal
    # ============================================================

    ax = axes[2]

    ax.plot(
        x, broad,
        marker="o",
        linewidth=1.8,
        markersize=6,
        color=BLUE
    )

    ax.axhline(
        broad_ref,
        linestyle="--",
        linewidth=1.2,
        color=BLUE,
        alpha=0.85
    )

    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.0)
    style_ax(ax, r"(c) Broad signal: $P(W_F\geq 3)$")

    # ============================================================
    # (d) Extreme tail
    # ============================================================

    ax = axes[3]

    ax.plot(
        x, extreme,
        marker="o",
        linewidth=1.8,
        markersize=6,
        color=BLUE
    )

    ax.axhline(
        extreme_ref,
        linestyle="--",
        linewidth=1.2,
        color=BLUE,
        alpha=0.85
    )

    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 0.27)
    style_ax(ax, r"(d) Extreme tail: $P(W_F\geq 4)$")

    # ============================================================
    # Global title
    # ============================================================

    fig.suptitle(
        "Mirror assignments change membership and width profiles",
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()
    return fig

def figure_4(root):
    order = ["canonical_noKh_n220", "heldout_conditional_n31", "heldout_relative_n31"]
    df = load_table(root, "4", "selection", order)
    totals = df["selected_n"].to_numpy(int)
    counts = df["n_WKh_ge_4"].to_numpy(int)
    props = df["prop_WKh_ge_4"].to_numpy(float)
    if (totals <= 0).any() or (counts < 0).any() or (counts > totals).any():
        raise ValueError("Invalid threshold counts")
    if not np.allclose(counts / totals, props, atol=1e-6):
        raise ValueError("Threshold proportions disagree with counts")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(["Canonical\nconditional", "Held-out\nconditional", "Held-out\nrelative"],
                  props, color=["#1F4E79", "#6E97C4", "#4E7E8C"])
    ax.bar_label(bars, labels=[f"{n}/{total}" for n,total in zip(counts,totals)], padding=5)
    ax.set_ylim(0, max(props) * 1.25)
    ax.set_ylabel(r"Proportion with $W_F\geq4$")
    ax.set_title("Selected knots meeting the topological width threshold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--figures", nargs="+", choices=["1", "2", "3", "4"], default=["1", "2", "3", "4"])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype":42, "ps.fonttype":42, "font.size":11})
    records = []
    for number in args.figures:
        fig = globals()[f"figure_{number}"](args.root)
        files = []
        for suffix in ["pdf", "png"]:
            dest = args.out / f"{NAMES[number]}.{suffix}"
            fig.savefig(dest, dpi=300, bbox_inches="tight")
            files.append({"file":dest.name, "sha256":hashlib.sha256(dest.read_bytes()).hexdigest()})
            print(dest)
        plt.close(fig)
        src = args.root / SOURCES[number] if number in SOURCES else None
        record = {"figure":number, "source":str(src) if src else "Illustrative geometry, not empirical data",
                  "source_sha256":hashlib.sha256(src.read_bytes()).hexdigest() if src else None,
                  "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "outputs":files}
        # One record per figure avoids overwriting provenance when rendered separately.
        (args.out / f"{NAMES[number]}_provenance.json").write_text(json.dumps(record,indent=2)+"\n")

if __name__ == "__main__":
    main()
