# Training results

Saved, reproducible training runs (seeded, `torch.manual_seed(0)`),
produced by the config-driven `src/train.py` CLI (see the main README's
"Reproducing experiments" section). Each run directory under
`results/runs/` contains everything needed to reproduce or reload it:

```
results/runs/<name>/
  config.yaml       # the fully-resolved Hydra config this run used
  checkpoint.pt      # trained model weights -- reloadable via src/infer.py
  history.json       # per-epoch train/test NSE, basin ID lists, timing
```

## Primary comparison: `runs/model_10yrs_spatial/` vs `runs/lstm_10yrs_spatial/`

The headline result. Both models trained on the *same* 35 train / 10
heldout basins (`split=spatial`), the *same* WY1991-1999 (9-year)
window, the *same* masked-NSE loss, and -- unlike the earlier pilot
below -- the *same* 150 epochs each, so neither model is favored by an
epoch-count mismatch. These are `configs/`'s current defaults; regenerate
with:

```bash
.venv/bin/python src/train.py output_dir=results/runs/model_10yrs_spatial
.venv/bin/python src/train.py model=benchmark_lstm train=benchmark_lstm \
    output_dir=results/runs/lstm_10yrs_spatial
```

**`runs/model_10yrs_spatial/`** -- Snow17 + SAC-SMA + `ParamNet` (LSTM
climatology encoder + static attributes -> 27 bounded physical
parameters), trained end-to-end through both Tesseracts via
`src/coupling.py`. ~5.1s/epoch (~45 real Fortran/Tesseract calls per
basin per epoch via finite-difference gradients), ~13 minutes total.

| epoch | median train NSE | median test (heldout) NSE |
|---|---|---|
| 1 | -- | +0.28 |
| 150 (final) | **+0.84** | **+0.70** |

Train/test gap at epoch 150: **~0.14**.

**`runs/lstm_10yrs_spatial/`** -- pure data-driven LSTM
(`src/benchmark_lstm.py`), no physical model, streamflow predicted
directly from forcing + static attributes. ~2.4s/epoch (ordinary batched
backprop, no per-basin Fortran/Tesseract calls), ~6 minutes total.

| epoch | median train NSE | median test (heldout) NSE |
|---|---|---|
| 1 | -0.10 | -0.12 |
| 150 (final) | **+0.78** | **+0.15** |

Train/test gap at epoch 150: **~0.63**.

### Reading these together

The physically-constrained hybrid model wins on both axes here, not
just generalization: higher final training NSE (0.84 vs 0.78) *and* a
dramatically smaller train/test gap (0.14 vs 0.63) on the exact same
basins, window, loss, and epoch budget. The pure LSTM's held-out NSE
(+0.15) is barely better than predicting the mean observed flow, despite
matching the hybrid model's training NSE reasonably closely -- it's
fitting the 35 training basins' idiosyncrasies, not learning
transferable rainfall-runoff behavior, while the physical structure
(Snow17/SAC-SMA's actual water-balance equations, with only their
*parameters* learned per basin) forces the network to hand off to
mechanisms that generalize by construction. This is a substantially
larger effect than the earlier 3-year pilot showed (gap 0.14 vs 0.34
there) -- see `notes/logs.md` for the window-extension rationale and
full discussion.

## Earlier 3-year pilot: `runs/hybrid_spatial_seed0/` vs `runs/benchmark_lstm_spatial_seed0/`

The first version of this comparison, over a shorter WY1991-1993 (3-year)
window, kept here as an earlier, independently-informative data point
(not deleted -- see `notes/logs.md`'s reproducibility check across two
seeded runs). Note the epoch counts are *not* matched here (hybrid: 25,
benchmark: 150) -- the primary comparison above fixes that.

```bash
.venv/bin/python src/train.py split.window.end=1993-09-30 train.n_epochs=25 \
    output_dir=results/runs/hybrid_spatial_seed0
.venv/bin/python src/train.py model=benchmark_lstm train=benchmark_lstm split.window.end=1993-09-30 \
    output_dir=results/runs/benchmark_lstm_spatial_seed0
```

**`runs/hybrid_spatial_seed0/`** -- 25 epochs, ~7s/epoch, ~3 minutes total.

| epoch | train NSE | test (heldout) NSE |
|---|---|---|
| 1 | -0.158 | +0.074 |
| 25 (final) | **+0.66** | **+0.52** |

Train/test gap at epoch 25: **~0.14**.

**`runs/benchmark_lstm_spatial_seed0/`** -- same basins/window/loss, 150
epochs (~10x cheaper per epoch, ~0.7s/epoch), ~2 minutes total.

| epoch | train NSE | test (heldout) NSE |
|---|---|---|
| 1 | -0.337 | -0.063 |
| 150 (final) | **+0.676** | **+0.335** |

Train/test gap at epoch 150: **0.34**.

Both models reach similar *training* NSE (~0.66-0.68) despite the
benchmark getting 6x more epochs -- i.e. it isn't simply undertrained
relative to the hybrid model, yet its held-out NSE still sits far below
its training NSE while the hybrid model's held-out NSE stays close to
its training NSE. Same qualitative finding as the primary comparison
above, on a smaller data regime (35/10 basins, 3-year window) and a less
epoch-matched setup.

Note these two run directories still carry `history.json`'s old
**mean**-aggregated NSE fields (`mean_train_nse`/`mean_test_nse`); the
primary comparison above and any run from here on report **median**
instead -- see `results/compare_runs.py`'s note below.

## Comparing runs

`results/compare_runs.py` turns either comparison above into a
repeatable script instead of a hand-copied table -- run it whenever a
new run needs checking against the existing baselines (e.g. after a
coupling/gradient change, or a different seed):

```bash
.venv/bin/python results/compare_runs.py \
    results/runs/model_10yrs_spatial \
    results/runs/lstm_10yrs_spatial
```

Takes any number of run directories, not just two -- add a third path
(e.g. a rerun after a code change, or the 3-year pilot runs above) to
see it alongside in the same table. Prints final train/test NSE, the
train/test gap, and seconds/epoch for each run; also saves an overlaid
NSE-vs-epoch plot to `<first run dir>/comparison.png` if matplotlib is
installed (skip with `--no-plot`).

**`agg` column:** `src/train.py` reports cross-basin NSE as a **median**
as of the theta_B-VJP change (a few badly-fit basins shouldn't dominate
the headline number the way they would under a mean); the two 3-year
pilot runs predate that and still carry the old **mean**-aggregated
`history.json`. The script reads whichever a given run actually has and
labels it in the `agg` column rather than assuming -- and prints an
explicit warning if you pass runs that mix the two, since that
comparison isn't apples to apples.

## Migration note

`runs/hybrid_spatial_seed0/`/`runs/benchmark_lstm_spatial_seed0/` replace
the old flat `results/hybrid_lstm_hargreaves_history.json` and
`results/benchmark_lstm_history.json` files (pre-Hydra-refactor).
Regenerating both through the new config-driven `src/train.py` CLI with
the same seed reproduced the old numbers epoch-for-epoch (verified
directly, see `notes/logs.md`'s refactor entry) -- this is a format/
infrastructure change, not a re-run with different behavior.

## Predictions from a trained checkpoint

`src/infer.py` loads a saved `checkpoint.pt` and scores it (no
training) against a config-selected split -- useful for checking a
model against a different window or basin set without retraining:

```bash
.venv/bin/python src/infer.py \
    checkpoint=results/runs/model_10yrs_spatial/checkpoint.pt
```

Writes `<output_dir>/predictions.json`: per-basin simulated streamflow
(mm/day) + NSE for every basin in the composed split.
