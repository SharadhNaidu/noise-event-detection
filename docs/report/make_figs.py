"""Figures for the technical report (vector PDF, IEEE column width 3.5 in).

    python docs/report/make_figs.py          # -> docs/report/fig/lb_waterfall.pdf
Numbers are the official leaderboard scores of the five uploaded files (README, Score history)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "fig"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3 (dataviz reference palette)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})


def waterfall():
    lb = [1.5831, 1.5852, 1.5902, 1.5907, 1.5926]
    steps = ["Background\nextension\n+ syn. shift", "Habit\nfull-span", "Type-C\nnested", "TTA\n+ syn. shift 2"]
    fig, ax = plt.subplots(figsize=(3.45, 1.5))
    x = list(range(len(steps) + 2))
    # start and end totals as bars from the axis floor, increments as floating bars
    lo = 1.5800
    ax.bar(0, lb[0] - lo, bottom=lo, width=0.62, color=BLUE, edgecolor="white", linewidth=0.8)
    ax.bar(len(x) - 1, lb[-1] - lo, bottom=lo, width=0.62, color=BLUE, edgecolor="white", linewidth=0.8)
    for i, s in enumerate(steps, start=1):
        d = lb[i] - lb[i - 1]
        ax.bar(i, d, bottom=lb[i - 1], width=0.62, color=AQUA, edgecolor="white", linewidth=0.8)
        ax.text(i, lb[i] + 0.00025, f"+{d:.4f}", ha="center", va="bottom", color=INK, fontsize=7)
        ax.plot([i - 1 + 0.31, i - 0.31], [lb[i - 1]] * 2, color=MUTED, lw=0.5, ls=(0, (2, 1.5)))
    ax.plot([len(steps) + 0.31 - 1, len(steps) + 1 - 0.31], [lb[-1]] * 2, color=MUTED, lw=0.5, ls=(0, (2, 1.5)))
    ax.text(0, lb[0] + 0.00025, f"{lb[0]:.4f}", ha="center", va="bottom", color=INK, fontsize=7)
    ax.text(len(x) - 1, lb[-1] + 0.00025, f"{lb[-1]:.4f}", ha="center", va="bottom", color=INK, fontsize=7, fontweight="bold")
    ax.set_xticks(x, ["Ensemble\n+ readout"] + steps + ["Final\n(rank 6)"], fontsize=6.6)
    ax.set_ylim(lo, 1.5945)
    ax.set_yticks([1.580, 1.584, 1.588, 1.592])
    ax.set_ylabel("LB score (F1 + Dice)", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, width=0.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "lb_waterfall.pdf")
    fig.savefig(OUT / "lb_waterfall.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    waterfall()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
