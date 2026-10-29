"""Figure 2 of the technical report (vector PDF, IEEE column width 3.45 in).

    python docs/report/make_figs.py          # -> docs/report/fig/miss_causes.pdf
Numbers: the DECIDE error anatomy of missed natural reference events (report, Section IV)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "fig"
INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3 (dataviz reference palette)
OTHER = "#b9b7b1"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 7.5, "pdf.fonttype": 42})


def miss_causes():
    """100% bar: why natural reference events are missed on DECIDE."""
    parts = [("one prediction spans\nseveral references", 44, BLUE, "white"),
             ("edge outside\ntolerance", 34, ORANGE, "white"),
             ("not\ndetected", 18, AQUA, INK), ("", 4, OTHER, INK)]
    fig, ax = plt.subplots(figsize=(3.45, 0.5))
    left = 0
    for name, v, c, tc in parts:
        ax.barh(0, v, left=left, height=0.9, color=c, edgecolor="white", linewidth=1.2)
        if name:
            ax.text(left + 1.2, 0.0, f"{v}%", ha="left", va="center", fontsize=8.5, color=tc, fontweight="bold")
            ax.text(left + 10.3, 0.0, name, ha="left", va="center", fontsize=6.8, color=tc, linespacing=0.95)
        left += v
    ax.text(100.8, 0, "other\n4%", ha="left", va="center", fontsize=5.8, color=MUTED, linespacing=0.9)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.46, 0.46)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.93, top=0.99, bottom=0.01)
    fig.savefig(OUT / "miss_causes.pdf")
    fig.savefig(OUT / "miss_causes.png", dpi=220)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    miss_causes()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
