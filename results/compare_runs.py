"""Compare saved training runs' history.json files side by side --
results/README.md's hybrid-vs-benchmark_lstm comparison, made scriptable
instead of hand-copied into a table each time a new run needs checking
against the existing baselines (see notes/logs.md).

Usage:
    .venv/bin/python results/compare_runs.py \\
        results/runs/hybrid_spatial_seed0_postvjp \\
        results/runs/benchmark_lstm_spatial_seed0

Any number of run directories works, not just two -- e.g. add the
original results/runs/hybrid_spatial_seed0 as a third argument to see
the pre-change baseline, the post-change rerun, and the benchmark LSTM
all in one table/plot.

Prints a summary table (final train/test NSE, train/test gap, n_epochs,
avg seconds/epoch) and, if matplotlib is installed, saves an overlaid
NSE-vs-epoch plot to <first run dir>/comparison.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# src/train.py switched cross-basin NSE reporting from mean to median
# (median_train_nse/median_test_nse) after these baseline runs were
# saved -- history.json only ever stored the aggregate, not the
# per-basin breakdown, so old runs can't be retroactively recomputed as
# median without a full retrain. Read whichever key a run actually has
# rather than erroring, but track+report which one it was: silently
# comparing a mean-aggregated run against a median-aggregated one in
# the same table would be a real, easy-to-miss apples-to-oranges bug.
def _nse_keys(row: dict) -> tuple[str, str]:
    if "median_train_nse" in row:
        return "median_train_nse", "median_test_nse"
    return "mean_train_nse", "mean_test_nse"


def load_run(run_dir: Path) -> dict:
    return json.loads((run_dir / "history.json").read_text())


def summarize(run_dir: Path, history: dict) -> dict:
    rows = history["history"]
    train_key, test_key = _nse_keys(rows[-1])
    # Test NSE isn't logged every epoch (see train.py's cfg.train.eval_every)
    # -- use the last row that has one, not necessarily the last row.
    last_with_test = next(r for r in reversed(rows) if test_key in r)
    return {
        "run": run_dir.name,
        "model": history["model"],
        "n_epochs": history["n_epochs"],
        "agg": train_key.split("_")[0],  # "mean" or "median" -- see _nse_keys
        "final_train_nse": rows[-1][train_key],
        "final_test_nse": last_with_test[test_key],
        "train_test_gap": rows[-1][train_key] - last_with_test[test_key],
        "avg_seconds_per_epoch": sum(r["seconds"] for r in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="results/runs/<name> directories")
    parser.add_argument("--no-plot", dest="plot", action="store_false", default=True)
    args = parser.parse_args()

    histories = {run_dir: load_run(run_dir) for run_dir in args.run_dirs}
    summaries = [summarize(run_dir, h) for run_dir, h in histories.items()]

    name_w = max(len(s["run"]) for s in summaries)
    header = (
        f"{'run':<{name_w}}  {'model':<14}  {'epochs':>6}  {'agg':>6}  "
        f"{'train_nse':>9}  {'test_nse':>8}  {'gap':>6}  {'s/epoch':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['run']:<{name_w}}  {s['model']:<14}  {s['n_epochs']:>6}  {s['agg']:>6}  "
            f"{s['final_train_nse']:>+9.4f}  {s['final_test_nse']:>+8.4f}  "
            f"{s['train_test_gap']:>6.3f}  {s['avg_seconds_per_epoch']:>8.2f}"
        )
    if len({s["agg"] for s in summaries}) > 1:
        print(
            "\nWARNING: comparing a mean-aggregated run against a median-aggregated "
            "one (see 'agg' column) -- not apples to apples, rerun the mean-aggregated "
            "run's training to get a directly comparable median number."
        )

    if not args.plot:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot; pip install matplotlib to enable)")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10.colors
    for i, (run_dir, history) in enumerate(histories.items()):
        rows = history["history"]
        train_key, test_key = _nse_keys(rows[-1])
        color = colors[i % len(colors)]
        ax.plot(
            [r["epoch"] for r in rows], [r[train_key] for r in rows],
            color=color, linestyle="-", label=f"{run_dir.name} (train)",
        )
        test_rows = [r for r in rows if test_key in r]
        ax.plot(
            [r["epoch"] for r in test_rows], [r[test_key] for r in test_rows],
            color=color, linestyle="--", label=f"{run_dir.name} (test)",
        )

    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("NSE")
    ax.set_title("Train vs held-out NSE across runs")
    ax.legend(fontsize=8)
    out_path = args.run_dirs[0] / "comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot -> {out_path}")


if __name__ == "__main__":
    main()
