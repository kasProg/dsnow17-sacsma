"""Regression test for src/train.py's multi-basin training loop.

Skips gracefully (not part of the fast local suite's assumptions) when
CAMELS data hasn't been downloaded -- data/download_camels.sh is a
separate, ~3.4GB, one-time step, deliberately not run by `make test`.
Same pattern as tests/test_tesseract_build.py skipping when Docker/the
built images aren't available.

Small and fast on purpose: 3 train + 2 heldout basins, 3 epochs -- this
is a smoke test that the training loop runs and produces a real
gradient signal (loss moves, held-out eval doesn't crash), not a
reproduction of the full 45-basin, 25-epoch training run documented in
notes/logs.md (train NSE -1.62 -> +0.50, heldout -0.46 -> +0.54).
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
    and (CAMELS_DIR / "basin_climatology.npz").exists()
    and (CAMELS_DIR / "basin_dataset_public_v1p2").exists()
)


@pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="CAMELS data not downloaded -- run data/download_camels.sh, "
    "data/select_basins.py, data/build_attributes.py, data/build_climatology.py first",
)
def test_training_loop_runs_and_loss_improves():
    import torch

    import train

    selected, X_static, X_climate = train.load_basins()
    train_ids = selected.loc[selected["split"] == "train", "gauge_id"].tolist()[:3]
    heldout_ids = selected.loc[selected["split"] == "heldout", "gauge_id"].tolist()[:2]

    train_basins = [train.BasinExample(g) for g in train_ids]
    heldout_basins = [train.BasinExample(g) for g in heldout_ids]

    net = train.ParamNet(
        n_static_features=X_static[train_ids[0]].shape[0],
        n_climate_features=X_climate[train_ids[0]].shape[1],
    )
    stack = train.CoupledNWSStack()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)

    nse_before = train.run_epoch(net, stack, train_basins, X_static, X_climate, optimizer=None)
    for _ in range(3):
        train.run_epoch(net, stack, train_basins, X_static, X_climate, optimizer)
    nse_after = train.run_epoch(net, stack, train_basins, X_static, X_climate, optimizer=None)

    mean_before = np.mean(list(nse_before.values()))
    mean_after = np.mean(list(nse_after.values()))
    assert mean_after > mean_before, (
        f"mean train NSE did not improve over 3 epochs: {mean_before:.4f} -> {mean_after:.4f}"
    )

    heldout_nses = train.run_epoch(net, stack, heldout_basins, X_static, X_climate, optimizer=None)
    assert all(np.isfinite(v) for v in heldout_nses.values())
