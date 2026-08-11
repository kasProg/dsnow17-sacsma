"""Regression test for src/benchmark_lstm.py -- the pure data-driven
LSTM baseline used to benchmark the hybrid Snow17+SAC-SMA+ParamNet
model. Skips gracefully without CAMELS data, same pattern as
tests/test_train.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))

CAMELS_DIR = REPO_ROOT / "data" / "camels"
_DATA_AVAILABLE = (
    (CAMELS_DIR / "selected_basins.csv").exists()
    and (CAMELS_DIR / "basin_attributes.npz").exists()
    and (CAMELS_DIR / "basin_dataset_public_v1p2").exists()
)


@pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="CAMELS data not downloaded -- see tests/test_train.py's skip reason",
)
def test_benchmark_loop_runs_and_loss_improves():
    import torch

    from benchmark_lstm import LSTMBenchmark, build_normalized_dynamic_arrays
    from data_module import build_split
    from train import run_epoch_benchmark

    cfg = OmegaConf.create({
        "data": {
            "name": "camels_snow35",
            "selected_basins_csv": str(CAMELS_DIR / "selected_basins.csv"),
            "attributes_npz": str(CAMELS_DIR / "basin_attributes.npz"),
            "climatology_npz": str(CAMELS_DIR / "basin_climatology.npz"),
        },
        "split": {
            "mode": "spatial",
            "window": {"start": "1990-10-01", "end": "1993-09-30"},
        },
    })
    split = build_split(cfg)
    train_basins = split.train_examples[:4]
    heldout_basins = split.test_examples[:2]
    train_ids = [b.gauge_id for b in train_basins]

    dynamic_arrays, mean, std = build_normalized_dynamic_arrays(
        train_basins + heldout_basins, train_ids
    )

    net = LSTMBenchmark(n_dynamic=3, n_static=split.X_static[train_ids[0]].shape[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    nse_before = run_epoch_benchmark(net, train_basins, split.X_static, dynamic_arrays, mean, std, optimizer=None)
    for _ in range(20):
        run_epoch_benchmark(net, train_basins, split.X_static, dynamic_arrays, mean, std, optimizer)
    nse_after = run_epoch_benchmark(net, train_basins, split.X_static, dynamic_arrays, mean, std, optimizer=None)

    mean_before = np.mean(list(nse_before.values()))
    mean_after = np.mean(list(nse_after.values()))
    assert mean_after > mean_before, (
        f"mean train NSE did not improve over 20 epochs: {mean_before:.4f} -> {mean_after:.4f}"
    )

    heldout_nses = run_epoch_benchmark(net, heldout_basins, split.X_static, dynamic_arrays, mean, std, optimizer=None)
    assert all(np.isfinite(v) for v in heldout_nses.values())
