"""Regression test for src/train.py's multi-basin hybrid training loop
and its Hydra-config plumbing (src/data_module.py's build_split()).

Skips gracefully (not part of the fast local suite's assumptions) when
CAMELS data hasn't been downloaded -- data/download_camels.sh is a
separate, ~3.4GB, one-time step, deliberately not run by `make test`.
Same pattern as tests/test_tesseract_build.py skipping when Docker/the
built images aren't available.

Small and fast on purpose: 3 train + 2 test basins, 3 epochs, a fixed
3-year window (hardcoded below, independent of configs/split/spatial.yaml's
current default) -- this is a smoke test that the training loop runs and
produces a real gradient signal (loss moves, held-out eval doesn't
crash), not a reproduction of the full 45-basin run in
results/runs/model_9yrs_spatial/ (see results/README.md).

Builds a config with OmegaConf.create() directly rather than going
through Hydra's compose()/initialize() -- avoids that API's global-state
gotchas across repeated calls in one pytest process, and keeps this test
independent of configs/*.yaml's exact file layout (still exercises the
same build_split()/run_epoch_hybrid() code path run_training() uses).
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
    and (CAMELS_DIR / "basin_climatology.npz").exists()
    and (CAMELS_DIR / "basin_dataset_public_v1p2").exists()
)


def _spatial_cfg() -> OmegaConf:
    return OmegaConf.create({
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


@pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="CAMELS data not downloaded -- run data/download_camels.sh, "
    "data/select_basins.py, data/build_attributes.py, data/build_climatology.py first",
)
def test_training_loop_runs_and_loss_improves():
    import torch

    from data_module import build_split
    from paramnet import ParamNet
    from pipeline import CoupledNWSStack
    from train import run_epoch_hybrid

    split = build_split(_spatial_cfg())
    train_basins = split.train_examples[:3]
    heldout_basins = split.test_examples[:2]

    net = ParamNet(
        n_static_features=split.X_static[train_basins[0].gauge_id].shape[0],
        n_climate_features=split.X_climate[train_basins[0].gauge_id].shape[1],
    )
    stack = CoupledNWSStack()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)

    nse_before = run_epoch_hybrid(net, stack, train_basins, split.X_static, split.X_climate, optimizer=None)
    for _ in range(3):
        run_epoch_hybrid(net, stack, train_basins, split.X_static, split.X_climate, optimizer)
    nse_after = run_epoch_hybrid(net, stack, train_basins, split.X_static, split.X_climate, optimizer=None)

    mean_before = np.mean(list(nse_before.values()))
    mean_after = np.mean(list(nse_after.values()))
    assert mean_after > mean_before, (
        f"mean train NSE did not improve over 3 epochs: {mean_before:.4f} -> {mean_after:.4f}"
    )

    heldout_nses = run_epoch_hybrid(net, stack, heldout_basins, split.X_static, split.X_climate, optimizer=None)
    assert all(np.isfinite(v) for v in heldout_nses.values())


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_build_split_spatial_matches_selected_basins_csv():
    """The config-driven split must reproduce exactly what
    data/select_basins.py wrote -- this is the thing the whole Hydra
    refactor has to get right, so it gets its own direct check rather
    than only being exercised incidentally by the training smoke test."""
    import pandas as pd

    from data_module import build_split

    split = build_split(_spatial_cfg())
    selected = pd.read_csv(CAMELS_DIR / "selected_basins.csv", dtype={"gauge_id": str})

    expected_train = set(selected.loc[selected["split"] == "train", "gauge_id"])
    expected_heldout = set(selected.loc[selected["split"] == "heldout", "gauge_id"])
    assert set(split.train_ids) == expected_train
    assert set(split.test_ids) == expected_heldout
    assert set(split.train_ids).isdisjoint(split.test_ids)


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_build_split_temporal_uses_same_basins_different_windows():
    """temporal split's defining property: train_ids == test_ids (same
    basins), but each BasinExample's own window differs."""
    from data_module import build_split

    cfg = OmegaConf.create({
        "data": {
            "name": "camels_snow35",
            "selected_basins_csv": str(CAMELS_DIR / "selected_basins.csv"),
            "attributes_npz": str(CAMELS_DIR / "basin_attributes.npz"),
            "climatology_npz": str(CAMELS_DIR / "basin_climatology.npz"),
        },
        "split": {
            "mode": "temporal",
            "train_window": {"start": "1990-10-01", "end": "1991-09-30"},
            "test_window": {"start": "1991-10-01", "end": "1992-09-30"},
        },
    })
    split = build_split(cfg)
    assert split.train_ids == split.test_ids
    assert len(split.train_examples) == len(split.test_examples) > 0
    for train_ex, test_ex in zip(split.train_examples[:3], split.test_examples[:3]):
        assert train_ex.gauge_id == test_ex.gauge_id
        assert train_ex.window_start != test_ex.window_start
