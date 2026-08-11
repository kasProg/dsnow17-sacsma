# Training results

Saved, reproducible training runs (seeded, `torch.manual_seed(0)`) for
the two models compared in `notes/logs.md`. Regenerate with:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'data')
import train
train.main(n_epochs=25, eval_every=3, lr=3e-3, seed=0,
           save_path='results/hybrid_lstm_hargreaves_history.json')
"
.venv/bin/python -c "
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'data')
import benchmark_lstm as bl
bl.main(n_epochs=150, eval_every=10, lr=1e-3, seed=0,
         save_path='results/benchmark_lstm_history.json')
"
```

Both require CAMELS data downloaded first (see main README's Reproduce
section). Same 35 train / 10 held-out basins, same WY1991-1993 window,
same masked-NSE loss for both files -- see each JSON's own
`train_basin_ids`/`heldout_basin_ids`/`window_start`/`window_end` fields
to confirm.

## `hybrid_lstm_hargreaves_history.json`

Snow17 + SAC-SMA + `ParamNet` (LSTM climatology encoder + static
attributes -> 27 bounded physical parameters), trained end-to-end
through both Tesseracts via `src/coupling.py`. 25 epochs (each epoch
costs ~45 real Fortran/Tesseract calls per basin via finite-difference
gradients, so epochs are expensive relative to the benchmark below).

| epoch | train NSE | held-out NSE |
|---|---|---|
| 1 | -0.158 | +0.074 |
| 25 (final) | **+0.663** | **+0.519** |

Train/held-out gap at epoch 25: **0.14**.

## `benchmark_lstm_history.json`

Pure data-driven LSTM (`src/benchmark_lstm.py`) -- no physical model,
streamflow predicted directly from forcing + static attributes. Same
basins/window/loss as above, ordinary backprop (no Fortran/Tesseract
calls), so epochs are ~10x cheaper -- run for 150 epochs.

| epoch | train NSE | held-out NSE |
|---|---|---|
| 1 | -0.337 | -0.063 |
| 150 (final) | **+0.676** | **+0.335** |

Train/held-out gap at epoch 150: **0.34**.

## Reading these together

Both models reach similar *training* NSE (~0.66-0.68). The gap is in
generalization: the physically-constrained hybrid model's held-out NSE
(0.52) sits close to its training NSE, while the pure LSTM's held-out
NSE (0.34) sits far below its training NSE, despite 6x more epochs.
Rerunning the hybrid model with a different fixed seed than the
unseeded exploratory run first reported in `notes/logs.md` (which got
train/held-out 0.62/0.58, a 0.04 gap) shifted the exact numbers here
(0.66/0.52, a 0.14 gap) but not the conclusion -- the hybrid model's
generalization gap is consistently a fraction of the pure LSTM's across
both the exploratory and this seeded, saved run. See `notes/logs.md`
for the full discussion, caveats, and why this comparison was built to
be controlled/fair in the first place.
