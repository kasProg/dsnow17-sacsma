# External reference: NeuralHydrology LSTM, PUB, our exact split

A properly-configured LSTM (not one we built), trained and tested on
**our exact 35 train / 10 heldout basin split** (see
`data/camels/selected_basins.csv`'s `split` column) over **our exact
WY1991-1999 window** -- an apples-to-apples PUB comparison point for
`results/runs/model_10yrs_spatial/`, produced with
[NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
(Kratzert et al., BSD-3-Clause; this directory contains only a results
CSV and a run config, no source code).

A first attempt used NeuralHydrology's own example basin-list files,
which turned out to assign basins almost inversely to our split (9 of
its 10 "test" basins were in our training set) -- invalid, not a
measure of generalization to our heldout basins. This run fixes that by
matching the basin lists to `data/camels/selected_basins.csv` exactly
(verified by direct set comparison) before retraining. See
`notes/logs.md` for the full investigation.

## Result

| basin | NSE |
|---|---|
| 06622700 | 0.911 |
| 09035800 | 0.661 |
| 09035900 | 0.809 |
| 11230500 | 0.827 |
| 11264500 | 0.960 |
| 12447390 | 0.494 |
| 13023000 | 0.885 |
| 13240000 | 0.782 |
| 13310700 | 0.740 |
| 13313000 | 0.606 |

**Median NSE: 0.795.** (`test_metrics.csv` has the raw per-basin
NSE/KGE/Alpha-NSE/Beta-NSE values.)

| model | median test NSE |
|---|---|
| this NeuralHydrology LSTM | **0.795** |
| our hybrid Snow17+SAC-SMA model (`runs/model_10yrs_spatial/`) | 0.70 |

On this held-out set, this LSTM currently outperforms our hybrid model.
See `results/README.md` for how this fits into the project's results.

## Reproducing this run

Requires a [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
checkout (commit `e4329c3`, see `config.yml`'s `commit_hash`) and CAMELS
Daymet forcing data, neither vendored in this repo. Given those:

1. Write `data/camels/selected_basins.csv`'s `train`/`heldout` gauge IDs
   (one per line) to two basin-list text files.
2. Point `config.yml`'s `train_basin_file`/`test_basin_file`/
   `validation_basin_file` at them (everything else in `config.yml` is
   already the exact config used).
3. `nh-run train --config-file config.yml`, then `nh-run evaluate
   --run-dir <the resulting run dir>`.

~20 minutes on one modern GPU (30 epochs, cudalstm, 35 basins).
