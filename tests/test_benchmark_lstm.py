"""Regression test for src/benchmark_lstm.py -- the pure data-driven
LSTM baseline used to benchmark the hybrid Snow17+SAC-SMA+ParamNet
model. Skips gracefully without CAMELS data, same pattern as
tests/test_train.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

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

    import benchmark_lstm as bl
    import train

    selected, X_static, _X_climate = train.load_basins()
    train_ids = selected.loc[selected["split"] == "train", "gauge_id"].tolist()[:4]
    heldout_ids = selected.loc[selected["split"] == "heldout", "gauge_id"].tolist()[:2]

    train_basins = [train.BasinExample(g) for g in train_ids]
    heldout_basins = [train.BasinExample(g) for g in heldout_ids]

    dynamic_arrays = {ex.gauge_id: bl.build_dynamic_array(ex) for ex in train_basins + heldout_basins}
    stack = np.concatenate([dynamic_arrays[g] for g in train_ids], axis=0)
    mean, std = stack.mean(axis=0), stack.std(axis=0)

    net = bl.LSTMBenchmark(n_dynamic=3, n_static=X_static[train_ids[0]].shape[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    nse_before = bl.run_epoch(net, train_basins, X_static, dynamic_arrays, mean, std, optimizer=None)
    for _ in range(20):
        bl.run_epoch(net, train_basins, X_static, dynamic_arrays, mean, std, optimizer)
    nse_after = bl.run_epoch(net, train_basins, X_static, dynamic_arrays, mean, std, optimizer=None)

    mean_before = np.mean(list(nse_before.values()))
    mean_after = np.mean(list(nse_after.values()))
    assert mean_after > mean_before, (
        f"mean train NSE did not improve over 20 epochs: {mean_before:.4f} -> {mean_after:.4f}"
    )

    heldout_nses = bl.run_epoch(net, heldout_basins, X_static, dynamic_arrays, mean, std, optimizer=None)
    assert all(np.isfinite(v) for v in heldout_nses.values())
