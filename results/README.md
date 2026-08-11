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

Regenerate either run with:

```bash
.venv/bin/python src/train.py output_dir=results/runs/hybrid_spatial_seed0
.venv/bin/python src/train.py model=benchmark_lstm train=benchmark_lstm \
    output_dir=results/runs/benchmark_lstm_spatial_seed0
```

Both use the same 35 train / 10 heldout basins, same WY1991-1993 window
(`split=spatial`, the default), same masked-NSE loss -- see each run's
own `config.yaml` and `history.json`'s `train_basin_ids`/`test_basin_ids`
to confirm.

## `runs/hybrid_spatial_seed0/`

Snow17 + SAC-SMA + `ParamNet` (LSTM climatology encoder + static
attributes -> 27 bounded physical parameters), trained end-to-end
through both Tesseracts via `src/coupling.py`. 25 epochs (each epoch
costs ~45 real Fortran/Tesseract calls per basin via finite-difference
gradients, so epochs are expensive relative to the benchmark below --
~7s/epoch, ~3 minutes total).

| epoch | train NSE | test (heldout) NSE |
|---|---|---|
| 1 | -0.158 | +0.074 |
| 25 (final) | **+0.66** | **+0.52** |

Train/test gap at epoch 25: **~0.14** (exact values in
`runs/hybrid_spatial_seed0/history.json`).

## `runs/benchmark_lstm_spatial_seed0/`

Pure data-driven LSTM (`src/benchmark_lstm.py`) -- no physical model,
streamflow predicted directly from forcing + static attributes. Same
basins/window/loss as above, ordinary backprop (no Fortran/Tesseract
calls), so epochs are ~10x cheaper (~0.7s/epoch) -- run for 150 epochs.

| epoch | train NSE | test (heldout) NSE |
|---|---|---|
| 1 | -0.337 | -0.063 |
| 150 (final) | **+0.676** | **+0.335** |

Train/test gap at epoch 150: **0.34**.

## Reading these together

Both models reach similar *training* NSE (~0.66-0.68). The gap is in
generalization: the physically-constrained hybrid model's held-out NSE
sits close to its training NSE, while the pure LSTM's held-out NSE sits
far below its training NSE, despite 6x more epochs. This isn't a
strawman comparison -- both models see identical basins, forcing
variables, time window, and loss function, and the benchmark was given
far more epochs and reached higher *training* NSE, i.e. it isn't simply
undertrained relative to the hybrid model. See `notes/logs.md` for the
full discussion, caveats (35/10 basins and a 3-year window is a small
data regime for any model), and the reproducibility check across two
independently-seeded runs that both showed the same qualitative gap.

## Migration note

These two runs replace the old flat `results/hybrid_lstm_hargreaves_history.json`
and `results/benchmark_lstm_history.json` files (pre-Hydra-refactor).
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
    checkpoint=results/runs/hybrid_spatial_seed0/checkpoint.pt
```

Writes `<output_dir>/predictions.json`: per-basin simulated streamflow
(mm/day) + NSE for every basin in the composed split.
