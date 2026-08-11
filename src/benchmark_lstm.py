"""Pure data-driven LSTM benchmark for src/train.py's hybrid
Snow17+SAC-SMA+ParamNet model -- NOT part of the submission's core
pipeline, built specifically to answer "does the physically-constrained
model actually hold up against a black-box baseline on the same data?"

Structurally similar to the general LSTM-rainfall-runoff pattern
NeuralHydrology-style models use (daily forcing sequence + static
attributes -> streamflow directly, no physical model in the loop) --
see src/paramnet.py's docstring for why NeuralHydrology (BSD-3-Clause)
was the studied reference and MHPI's dPLHBVrelease (PSU Non-Commercial)
was deliberately not read. Original implementation of a well-documented,
published pattern (Kratzert et al.'s LSTM rainfall-runoff line of work),
no code copied from either source.

Deliberately controlled for a fair comparison: same 35 train / 10
held-out basins, same WY1991-1993 window, same masked-NSE loss and
held-out evaluation protocol, same 3 forcing variables (prcp, tmean,
PET) as what actually drives Snow17/SAC-SMA -- any performance
difference between this and the hybrid model is attributable to the
modeling approach, not to more data or more information. Unlike the
hybrid model, this one has no per-basin Fortran/Tesseract cost, so all
35 basins batch through the LSTM in a single forward pass per epoch --
cheap enough to run far more epochs than the hybrid model's per-epoch
cost allows.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from train import BasinExample, load_basins, masked_nse_loss  # noqa: E402


class LSTMBenchmark(nn.Module):
    """x_dynamic: (batch, T, n_dynamic), x_static: (batch, n_static) ->
    q_hat: (batch, T), mm/day. Static attributes are concatenated to the
    dynamic input at every timestep -- standard practice in the
    LSTM-rainfall-runoff literature (e.g. Kratzert et al. 2019), not
    specific to any one implementation.
    """

    def __init__(self, n_dynamic: int, n_static: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_dynamic + n_static, hidden_size=hidden, batch_first=True
        ).double()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1).double()

    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        t = x_dynamic.shape[1]
        static_rep = x_static.unsqueeze(1).expand(-1, t, -1)
        x = torch.cat([x_dynamic.to(torch.float64), static_rep.to(torch.float64)], dim=-1)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        q_hat = self.head(out).squeeze(-1)
        # Streamflow can't be negative -- softplus keeps this smooth/
        # differentiable everywhere, unlike a hard clamp at 0.
        return torch.nn.functional.softplus(q_hat)


def build_dynamic_array(ex: BasinExample) -> np.ndarray:
    """(T, 3): prcp, tmean, PET -- exactly the 3 forcing variables that
    actually drive Snow17/SAC-SMA in the hybrid model (see
    src/pipeline.py's Snow17Forcing/SacSmaForcing) -- not a richer
    feature set, so the comparison isn't stacked in either direction."""
    f = ex.snow17_forcing
    return np.stack([f.pcp, f.tmp, ex.sacsma_forcing.etp], axis=-1)


def run_epoch(
    net: LSTMBenchmark,
    basins: list[BasinExample],
    X_static: dict,
    dynamic_arrays: dict,
    dyn_mean: np.ndarray,
    dyn_std: np.ndarray,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    x_dynamic = torch.tensor(
        np.stack([(dynamic_arrays[b.gauge_id] - dyn_mean) / dyn_std for b in basins]),
        dtype=torch.float64,
    )
    x_static = torch.tensor(np.stack([X_static[b.gauge_id] for b in basins]), dtype=torch.float64)

    if optimizer is not None:
        net.train()
        q_hat = net(x_dynamic, x_static)
    else:
        net.eval()
        with torch.no_grad():
            q_hat = net(x_dynamic, x_static)

    losses = []
    nses = {}
    for i, ex in enumerate(basins):
        sim = q_hat[i]
        loss = masked_nse_loss(sim, ex)
        losses.append(loss)
        with torch.no_grad():
            nses[ex.gauge_id] = float(1.0 - loss.item())

    if optimizer is not None:
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

    return nses


def main(n_epochs: int = 150, eval_every: int = 10, lr: float = 1e-3) -> None:
    selected, X_static, _X_climate = load_basins()
    train_ids = selected.loc[selected["split"] == "train", "gauge_id"].tolist()
    heldout_ids = selected.loc[selected["split"] == "heldout", "gauge_id"].tolist()

    print(f"Loading {len(train_ids)} train + {len(heldout_ids)} heldout basins' timeseries...")
    t0 = time.time()
    train_basins = [BasinExample(gid) for gid in train_ids]
    heldout_basins = [BasinExample(gid) for gid in heldout_ids]
    print(f"  done in {time.time()-t0:.1f}s")

    dynamic_arrays = {ex.gauge_id: build_dynamic_array(ex) for ex in train_basins + heldout_basins}
    # Normalization stats from TRAIN basins only -- avoids held-out-set
    # leakage into the normalization, standard ML practice.
    train_dynamic_stack = np.concatenate([dynamic_arrays[gid] for gid in train_ids], axis=0)
    dyn_mean = train_dynamic_stack.mean(axis=0)
    dyn_std = train_dynamic_stack.std(axis=0)
    dyn_std[dyn_std == 0] = 1.0

    net = LSTMBenchmark(n_dynamic=3, n_static=X_static[train_ids[0]].shape[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    history = []
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_nses = run_epoch(net, train_basins, X_static, dynamic_arrays, dyn_mean, dyn_std, optimizer)
        mean_train_nse = float(np.mean(list(train_nses.values())))
        dt = time.time() - t0

        row = {"epoch": epoch, "mean_train_nse": mean_train_nse, "seconds": dt}
        if epoch == 1 or epoch % eval_every == 0 or epoch == n_epochs:
            heldout_nses = run_epoch(
                net, heldout_basins, X_static, dynamic_arrays, dyn_mean, dyn_std, optimizer=None
            )
            row["mean_heldout_nse"] = float(np.mean(list(heldout_nses.values())))
        history.append(row)
        print(
            f"epoch {epoch:3d}  train_nse={mean_train_nse:+.4f}"
            + (f"  heldout_nse={row.get('mean_heldout_nse'):+.4f}" if "mean_heldout_nse" in row else "")
            + f"  ({dt:.2f}s)"
        )

    return net, history, train_basins, heldout_basins


if __name__ == "__main__":
    main()
