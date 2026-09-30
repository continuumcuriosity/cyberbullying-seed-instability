#!/usr/bin/env python3
"""Generates the two figures.

  fig1_pipeline.pdf  - system diagram
  fig2_tradeoff.pdf  - left: in-distribution vs cross-domain, one point per run
                       right: the mechanism, recall tracks benign FPR

Data: runs12.csv (4 configurations x 3 training seeds). The split is identical
in every run (SPLIT_SEED=42); only TRAIN_SEED varies within an arm.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

RUNS12 = []
with open(os.path.join(_DATA, "runs12.csv")) as f:
    for r in csv.DictReader(f):
        RUNS12.append(dict(arm=r["arm"], seed=int(r["seed"]),
                           f1=float(r["test_f1"]), rec=float(r["recall"]),
                           fpr=float(r["fpr"])))

ARMS = ["soft_uw1", "soft_off", "main", "no_contrastive"]
MARK = {"soft_uw1": "s", "soft_off": "o", "main": "^", "no_contrastive": "D"}


def fig_tradeoff(path="fig2_tradeoff.pdf"):
    """(a) what a single-seed study would have concluded; (b) the item-level
    picture: how many of the twelve runs flag each comment, by gold label."""
    votes = []
    with open(os.path.join(_DATA, "votes12.csv")) as f:
        for r in csv.DictReader(f):
            votes.append((int(r["k"]), int(r["n_harmful"]), int(r["n_benign"])))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.5))

    for arm in ARMS:
        pts = [r for r in RUNS12 if r["arm"] == arm]
        ax.plot([p["f1"] for p in pts], [p["rec"] for p in pts], MARK[arm],
                ms=4.2, mfc="white", mec="black", mew=0.9, ls="none",
                zorder=3, label=arm)
    xs = np.array([r["f1"] for r in RUNS12]); ys = np.array([r["rec"] for r in RUNS12])
    b_, a_ = np.polyfit(xs, ys, 1)
    gx = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(gx, a_ + b_ * gx, "--", lw=0.8, color="0.6", zorder=2)
    ax.set_xlabel("In-distribution test macro-$F_1$")
    ax.set_ylabel("Cyberbullying recall")
    ax.set_title("(a) the apparent trade-off", fontsize=7.5, pad=4)
    ax.text(0.03, 0.04,
            "$\\rho=-0.74$; $-0.17$ once\nflag count is controlled",
            transform=ax.transAxes, fontsize=6.2, va="bottom", color="0.3")
    ax.legend(loc="upper right", frameon=False, fontsize=6.2, numpoints=1,
              handletextpad=0.25, borderpad=0.1, labelspacing=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(lw=0.3, color="0.93", zorder=0)

    ks = [v[0] for v in votes][1:]
    hh = [v[1] for v in votes][1:]
    bb = [v[2] for v in votes][1:]
    bx.bar(ks, hh, width=0.78, color="0.35", ec="black", lw=0.5,
           label="harmful", zorder=3)
    bx.bar(ks, bb, width=0.78, bottom=hh, color="0.85", ec="black", lw=0.5,
           label="benign", zorder=3)
    bx.set_xlabel("Number of the 12 runs flagging the comment")
    bx.set_ylabel("Comments")
    bx.set_title("(b) item-level disagreement", fontsize=7.5, pad=4)
    bx.set_xticks(range(1, 13))
    bx.legend(loc="upper center", frameon=False, fontsize=6.2,
              handletextpad=0.35, borderpad=0.1, labelspacing=0.25)
    bx.spines[["top", "right"]].set_visible(False)
    bx.grid(axis="y", lw=0.3, color="0.93", zorder=0)

    fig.tight_layout(pad=0.4, w_pad=2.2)
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path); plt.close(fig)


def fig_pipeline(path="fig1_pipeline.pdf"):
    fig, ax = plt.subplots(figsize=(7.16, 1.7))
    ax.set_xlim(-4, 111); ax.set_ylim(-1, 25); ax.axis("off")
    def box(x, w, y, h, text, fc="white", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.35,rounding_size=0.8",
                     fc=fc, ec="black", lw=0.7, ls=ls, zorder=2))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=7, zorder=3, linespacing=1.3)
    def arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=7, lw=0.7, color="black", zorder=2))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2, label, ha="center", va="center",
                    fontsize=6, color="0.3", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
    MID, TOP, BOT, H, HB = 9.0, 16.5, 1.5, 7.0, 6.0
    box(0, 12, MID, H, "input\ncomment")
    box(17, 15, MID, H, "perturbation gate\n$looks\\_perturbed()$", fc="0.93")
    box(37, 14, TOP, HB, "normalize", fc="0.93")
    box(37, 14, BOT, HB, "pass through", fc="0.97")
    box(56, 14, MID, H, "BERT +\ntemperature scaling")
    box(75, 13, MID, H, "confidence\n$< \\tau$ ?", fc="0.93")
    box(93, 14, TOP, HB, "defer to LLM", fc="0.97", ls="--")
    box(93, 14, BOT, HB, "label + occlusion\nrationale")
    arrow(12, MID+H/2, 16.6, MID+H/2)
    arrow(32, MID+H/2+1.0, 36.6, TOP+HB/2, "yes")
    arrow(32, MID+H/2-1.0, 36.6, BOT+HB/2, "no")
    arrow(51, TOP+HB/2, 55.6, MID+H/2+1.0)
    arrow(51, BOT+HB/2, 55.6, MID+H/2-1.0)
    arrow(70, MID+H/2, 74.6, MID+H/2)
    arrow(88, MID+H/2+1.0, 92.6, TOP+HB/2, "yes")
    arrow(88, MID+H/2-1.0, 92.6, BOT+HB/2, "no")
    arrow(100, TOP, 100, BOT+HB+0.6)
    ax.text(44, 24.2, "gate fires on 8.5% of cases", ha="center", fontsize=6, color="0.35")
    ax.text(100, 24.2, "evaluated, not adopted", ha="center", fontsize=6, color="0.35")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path); plt.close(fig)


if __name__ == "__main__":
    fig_tradeoff(); fig_pipeline()
