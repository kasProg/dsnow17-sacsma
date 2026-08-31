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

## `runs/model_10yrs_spatial/` -- primary result

Snow17 + SAC-SMA + `ParamNet` (LSTM climatology encoder + static
attributes -> 27 bounded physical parameters), trained end-to-end
through both Tesseracts via `src/coupling.py`. 35 train / 10 heldout
basins (`split=spatial`), WY1991-1999 (9-year) window, 150 epochs,
~5.1s/epoch (~45 real Fortran/Tesseract calls per basin per epoch via
finite-difference gradients), ~13 minutes total. This is `configs/`'s
current default; regenerate with:

```bash
.venv/bin/python src/train.py output_dir=results/runs/model_10yrs_spatial
```

| epoch | median train NSE | median test (heldout) NSE |
|---|---|---|
| 1 | +0.38 | +0.28 |
| 150 (final) | **+0.84** | **+0.70** |

Train/test gap at epoch 150: **~0.14** -- held-out basins track training
basins closely throughout, no overfitting observed at this scale.

## External reference: a properly-engineered LSTM

[`results/external/neuralhydrology_lstm_pub/`](external/neuralhydrology_lstm_pub/README.md)
-- a [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
LSTM trained and tested on **the exact same 35/10 basin split** as the
run above:

| model | median test NSE |
|---|---|
| NeuralHydrology LSTM | **0.795** |
| our hybrid model (`runs/model_10yrs_spatial/`) | 0.70 |

See that directory's README for details and reproduction steps.

## Comparing runs

`results/compare_runs.py` turns a comparison like the one above into a
repeatable script instead of a hand-copied table -- run it whenever a
new run needs checking against the existing baseline (e.g. after a
coupling/gradient change, or a different seed):

```bash
.venv/bin/python results/compare_runs.py \
    results/runs/model_10yrs_spatial \
    results/runs/hybrid_spatial_seed0
```

Takes any number of run directories, not just two. Prints final
train/test NSE, the train/test gap, and seconds/epoch for each run;
also saves an overlaid NSE-vs-epoch plot to `<first run dir>/comparison.png`
if matplotlib is installed (skip with `--no-plot`).

**`agg` column:** `src/train.py` reports cross-basin NSE as a **median**
as of the theta_B-VJP change (a few badly-fit basins shouldn't dominate
the headline number the way they would under a mean); `runs/hybrid_spatial_seed0/`
predates that and still carries the old **mean**-aggregated
`history.json`. The script reads whichever a given run actually has and
labels it in the `agg` column rather than assuming -- and prints an
explicit warning if you pass runs that mix the two, since that
comparison isn't apples to apples.

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
