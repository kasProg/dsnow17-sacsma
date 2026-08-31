"""Per-basin held-out NSE, hybrid model vs. the external NeuralHydrology
LSTM (results/README.md's "External reference" comparison), as a bar
chart -- the median-only table already in README.md/results/README.md
compressed two numbers out of ten each; this shows the actual
distribution, including the basins where the LSTM wins by a lot and the
two where the hybrid model doesn't.

Data: results/runs/model_9yrs_spatial/test_predictions.json (this
project's own run) and results/external/neuralhydrology_lstm_pub/
test_metrics.csv (the external run) -- both score the identical 10
held-out basins, verified here by exact gauge_id set match before
plotting anything.

Palette/marks follow this project's dataviz skill: validated
colorblind-safe categorical slots 1 (blue) and 2 (orange) -- adjacent
pair, worst-case CVD Delta E 9.1, clear of the >=8 target -- fixed
mark spec (rounded bar tops, 2px surface gaps, hairline recessive
gridlines, text in ink tokens never the series color, direct value
labels skipped in favor of median reference lines, since ten side-by-side
labels above tightly-gapped bars would collide).

Usage (matplotlib isn't a base dependency -- same optional-extra pattern
as results/compare_runs.py's plot):
    .venv/bin/python -m pip install matplotlib
    .venv/bin/python results/plot_basin_comparison.py
Writes results/basin_nse_comparison.png.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
HYBRID_PREDICTIONS = REPO_ROOT / "results/runs/model_9yrs_spatial/test_predictions.json"
NH_METRICS = REPO_ROOT / "results/external/neuralhydrology_lstm_pub/test_metrics.csv"
OUT_PATH = REPO_ROOT / "results/basin_nse_comparison.png"

# Validated categorical slots (dataviz skill's references/palette.md) --
# slot 1 (blue) / slot 2 (orange), the adjacent pair, light mode.
COLOR_HYBRID = "#2a78d6"
COLOR_NH = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def _load_data() -> tuple[list[str], list[float], list[float]]:
    preds = json.loads(HYBRID_PREDICTIONS.read_text())
    hybrid = {gid: p["nse"] for gid, p in preds["predictions"].items()}

    nh: dict[str, float] = {}
    with NH_METRICS.open() as f:
        for row in csv.DictReader(f):
            nh[row["basin"]] = float(row["NSE"])

    assert set(hybrid) == set(nh), (
        f"basin set mismatch -- hybrid {sorted(hybrid)} vs NH {sorted(nh)}"
    )
    gauge_ids = sorted(hybrid)  # same order as the markdown tables citing these numbers
    return gauge_ids, [hybrid[g] for g in gauge_ids], [nh[g] for g in gauge_ids]


def _rounded_bar(ax, x: float, width: float, height: float, color: str) -> None:
    """A bar with a 4px-equivalent rounded top, square at the baseline --
    matplotlib has no native rounded-bar primitive, so this draws a
    FancyBboxPatch sized to read as a bar with just the top corners eased."""
    if height <= 0:
        return
    pad = 0.0
    rounding = min(width, height) * 0.12
    box = FancyBboxPatch(
        (x - width / 2, 0), width, height,
        boxstyle=f"round,pad={pad},rounding_size={rounding}",
        linewidth=0, facecolor=color, mutation_aspect=1,
    )
    ax.add_patch(box)


def main() -> None:
    gauge_ids, hybrid_nse, nh_nse = _load_data()
    hybrid_med = statistics.median(hybrid_nse)
    nh_med = statistics.median(nh_nse)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    n = len(gauge_ids)
    x = list(range(n))
    bar_w = 0.36  # leaves a visible surface gap between the pair and between basin groups

    for xi, h, m in zip(x, hybrid_nse, nh_nse):
        _rounded_bar(ax, xi - bar_w / 2 - 0.02, bar_w, h, COLOR_HYBRID)
        _rounded_bar(ax, xi + bar_w / 2 + 0.02, bar_w, m, COLOR_NH)

    # Median reference lines instead of a label on every bar (10 basins x
    # 2 series would collide badly at this width) -- ties directly back
    # to the numbers already in the README/results tables. Labels sit in
    # a dedicated right margin OUTSIDE the bars (xlim extended below),
    # not stamped over the data, so they can never collide with a bar --
    # the first attempt at this put them over the tallest bars and one
    # was literally clipped mid-word.
    ax.axhline(hybrid_med, color=COLOR_HYBRID, linewidth=1.25, linestyle=(0, (4, 3)), alpha=0.6, zorder=1)
    ax.axhline(nh_med, color=COLOR_NH, linewidth=1.25, linestyle=(0, (4, 3)), alpha=0.6, zorder=1)
    label_x = n - 0.05
    ax.text(label_x, hybrid_med, f"hybrid\nmedian {hybrid_med:.2f}",
            color=COLOR_HYBRID, fontsize=9, ha="left", va="center", linespacing=1.3)
    ax.text(label_x, nh_med, f"NeuralHydrology\nmedian {nh_med:.2f}",
            color=COLOR_NH, fontsize=9, ha="left", va="center", linespacing=1.3)

    # Hairline recessive gridlines, y-axis only.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.xaxis.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)

    ax.set_xticks(x)
    ax.set_xticklabels(gauge_ids, rotation=30, ha="right", fontsize=9, color=INK_MUTED)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0, labelsize=9, colors=INK_MUTED)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.7, n + 1.6)  # dedicated right margin for the median labels above
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax.set_ylabel("Held-out NSE", fontsize=10, color=INK_SECONDARY)

    # Title above subtitle, both above the axes, legend in its own row
    # between them -- all outside the plot area so nothing can overlap
    # a bar regardless of data heights.
    fig.suptitle(
        "Held-out streamflow skill, basin by basin",
        x=0.02, y=0.995, fontsize=14, color=INK_PRIMARY, fontweight="bold", ha="left",
    )
    fig.text(
        0.02, 0.945,
        "Same 10 held-out basins, same 9-year window -- our hybrid Snow-17+SAC-SMA model vs. a properly-engineered LSTM",
        fontsize=9.5, color=INK_SECONDARY, ha="left", va="top",
    )
    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_HYBRID, edgecolor="none", label="Our hybrid model"),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_NH, edgecolor="none", label="NeuralHydrology LSTM"),
        ],
        loc="upper left", bbox_to_anchor=(0.02, 0.905), frameon=False, fontsize=9.5,
        handlelength=1.2, handleheight=1.2, labelcolor=INK_SECONDARY, ncols=2,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
