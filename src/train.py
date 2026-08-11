"""Multi-basin training: ParamNet -> Snow17 + SAC-SMA -> NSE loss, across
35 snow-dominated CAMELS training basins, with periodic held-out
evaluation on 10 reserved basins. CLAUDE.md's Day 7-10 milestone.

Fixed training window: WY1991-1993 (1990-10-01 .. 1993-09-30, 3 years).
Verified (not assumed) that all 45 selected basins have full daily
coverage and <=5% missing streamflow over this window before picking it
-- several earlier candidate windows (WY1981-83, WY1986-88) failed for
5-6 basins each (see notes/logs.md for the exact search).

One gradient step per epoch, not per basin: theta_A/theta_B are computed
for all 35 training basins in a single batched ParamNet forward pass,
then each basin's own (expensive, Fortran-backed) CoupledTwoStageFunction
call runs individually -- that part isn't batchable, SAC-SMA/Snow17 are
single-HRU by construction -- and their losses are averaged into ONE
scalar before a single .backward()/optimizer.step() call. This is
standard full-batch gradient descent over basins, not per-basin SGD:
each step reflects the whole training set's gradient, not one basin's
noisy estimate of it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from camels_loader import load_basin_timeseries  # noqa: E402
from paramnet import ParamNet  # noqa: E402
from pipeline import CoupledNWSStack, SacSmaForcing, Snow17Forcing  # noqa: E402

CAMELS_DIR = REPO_ROOT / "data" / "camels"
WINDOW_START = pd.Timestamp("1990-10-01")
WINDOW_END = pd.Timestamp("1993-09-30")

# Fixed, not learnable -- Snow17's areal depletion curve. A standard
# generic curve shape (matches what's used throughout this project's
# earlier single-basin work), since CLAUDE.md's own plan keeps ADC out
# of scope for the differentiable parameters (see notes/logs.md).
DEFAULT_ADC = np.array(
    [0.05, 0.09, 0.16, 0.31, 0.54, 0.74, 0.84, 0.89, 0.93, 0.97, 1.0], dtype=np.float64
)


class BasinExample:
    """Everything needed to run + score one basin, precomputed once and
    reused across every epoch (forcing/observations don't change)."""

    def __init__(self, gauge_id: str):
        ts = load_basin_timeseries(gauge_id)
        mask = (ts.dates >= WINDOW_START) & (ts.dates <= WINDOW_END)
        dates = ts.dates[mask]

        self.gauge_id = gauge_id
        self.snow17_forcing = Snow17Forcing(
            idt=24, idts=86400,
            iyr=dates.year.to_numpy().astype(np.int32),
            imn=dates.month.to_numpy().astype(np.int32),
            ida=dates.day.to_numpy().astype(np.int32),
            pcp=ts.prcp[mask], tmp=ts.tmean[mask],
            alat=ts.lat, elev=ts.elev,
            adc=DEFAULT_ADC,
            cs0=np.zeros(19, dtype=np.float64), tprev0=0.0,
        )
        self.sacsma_forcing = SacSmaForcing(
            dtm=86400.0, tmp=ts.tmean[mask], etp=ts.pet[mask],
            state0=np.zeros(6, dtype=np.float64),
        )
        q_obs = ts.q_obs[mask]
        self.observed = torch.tensor(q_obs, dtype=torch.float64)
        self.valid_mask = torch.tensor(~np.isnan(q_obs))
        assert self.valid_mask.sum() > 0, f"{gauge_id}: no valid observed days in window"


def masked_nse_loss(sim: torch.Tensor, example: BasinExample) -> torch.Tensor:
    """1 - NSE, computed only over non-missing observed days."""
    obs = example.observed[example.valid_mask]
    sim_valid = sim[example.valid_mask]
    denom = torch.clamp(torch.sum((obs - obs.mean()) ** 2), min=1e-6)
    nse = 1.0 - torch.sum((obs - sim_valid) ** 2) / denom
    return 1.0 - nse


def nse_value(sim: torch.Tensor, example: BasinExample) -> float:
    with torch.no_grad():
        return float(1.0 - masked_nse_loss(sim, example).item())


def load_basins() -> tuple[pd.DataFrame, dict, dict]:
    """Returns (selected_basins_df, static_attrs_by_gauge_id,
    climatology_by_gauge_id)."""
    selected = pd.read_csv(CAMELS_DIR / "selected_basins.csv", dtype={"gauge_id": str})
    attrs = np.load(CAMELS_DIR / "basin_attributes.npz", allow_pickle=True)
    gauge_ids = list(attrs["gauge_ids"])
    X_static = {gid: attrs["X"][i] for i, gid in enumerate(gauge_ids)}

    clim = np.load(CAMELS_DIR / "basin_climatology.npz", allow_pickle=True)
    clim_ids = list(clim["gauge_ids"])
    X_climate = {gid: clim["X"][i] for i, gid in enumerate(clim_ids)}
    assert set(X_static) == set(X_climate), "static/climatology basin sets don't match"

    return selected, X_static, X_climate


def run_epoch(
    net: ParamNet,
    stack: CoupledNWSStack,
    basins: list[BasinExample],
    X_static: dict,
    X_climate: dict,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """optimizer=None -> eval mode, no gradient step (used for held-out
    basins). Returns {gauge_id: nse}."""
    x_static_batch = torch.tensor(
        np.stack([X_static[b.gauge_id] for b in basins]), dtype=torch.float64
    )
    x_climate_batch = torch.tensor(
        np.stack([X_climate[b.gauge_id] for b in basins]), dtype=torch.float64
    )
    if optimizer is not None:
        net.train()
        theta_A_batch, theta_B_batch = net(x_static_batch, x_climate_batch)
    else:
        net.eval()
        with torch.no_grad():
            theta_A_batch, theta_B_batch = net(x_static_batch, x_climate_batch)
        theta_A_batch = theta_A_batch.detach().requires_grad_(False)
        theta_B_batch = theta_B_batch.detach().requires_grad_(False)

    losses = []
    nses = {}
    for i, ex in enumerate(basins):
        sim = stack.run(theta_A_batch[i], theta_B_batch[i], ex.snow17_forcing, ex.sacsma_forcing)
        loss = masked_nse_loss(sim, ex)
        losses.append(loss)
        nses[ex.gauge_id] = nse_value(sim, ex)

    if optimizer is not None:
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

    return nses


def main(n_epochs: int = 15, eval_every: int = 3, lr: float = 3e-3) -> None:
    selected, X_static, X_climate = load_basins()
    train_ids = selected.loc[selected["split"] == "train", "gauge_id"].tolist()
    heldout_ids = selected.loc[selected["split"] == "heldout", "gauge_id"].tolist()

    print(f"Loading {len(train_ids)} train + {len(heldout_ids)} heldout basins' timeseries...")
    t0 = time.time()
    train_basins = [BasinExample(gid) for gid in train_ids]
    heldout_basins = [BasinExample(gid) for gid in heldout_ids]
    print(f"  done in {time.time()-t0:.1f}s")

    net = ParamNet(
        n_static_features=X_static[train_ids[0]].shape[0],
        n_climate_features=X_climate[train_ids[0]].shape[1],
    )
    stack = CoupledNWSStack()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    history = []
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_nses = run_epoch(net, stack, train_basins, X_static, X_climate, optimizer)
        mean_train_nse = float(np.mean(list(train_nses.values())))
        dt = time.time() - t0

        row = {"epoch": epoch, "mean_train_nse": mean_train_nse, "seconds": dt}
        if epoch == 1 or epoch % eval_every == 0 or epoch == n_epochs:
            heldout_nses = run_epoch(net, stack, heldout_basins, X_static, X_climate, optimizer=None)
            row["mean_heldout_nse"] = float(np.mean(list(heldout_nses.values())))
        history.append(row)
        print(
            f"epoch {epoch:3d}  train_nse={mean_train_nse:+.4f}"
            + (f"  heldout_nse={row.get('mean_heldout_nse'):+.4f}" if "mean_heldout_nse" in row else "")
            + f"  ({dt:.1f}s)"
        )

    return net, history, train_basins, heldout_basins


if __name__ == "__main__":
    main()
