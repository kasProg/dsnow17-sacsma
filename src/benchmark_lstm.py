"""Pure data-driven LSTM benchmark model -- NOT part of the submission's
core pipeline, built specifically to answer "does the physically-
constrained hybrid model (src/paramnet.py + src/pipeline.py) actually
hold up against a black-box baseline on the same data?" See
results/README.md for the comparison and src/train.py for the shared
training loop (both models train through the same config-driven CLI as
of the Hydra refactor -- this file now holds only the model definition,
not a training loop of its own).

Structurally similar to the general LSTM-rainfall-runoff pattern
NeuralHydrology-style models use (daily forcing sequence + static
attributes -> streamflow directly, no physical model in the loop) --
see src/paramnet.py's docstring for why NeuralHydrology (BSD-3-Clause)
was the studied reference and MHPI's dPLHBVrelease (PSU Non-Commercial)
was deliberately not read. Original implementation of a well-documented,
published pattern (Kratzert et al.'s LSTM rainfall-runoff line of work),
no code copied from either source.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from data_module import BasinExample


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


def build_normalized_dynamic_arrays(
    all_examples: list[BasinExample], train_ids: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """dynamic_arrays keyed by gauge_id, plus (mean, std) computed from
    TRAIN basins only -- avoids test-set leakage into the normalization,
    standard ML practice. Shared by src/train.py and src/infer.py so
    both compute normalization the same way."""
    dynamic_arrays = {ex.gauge_id: build_dynamic_array(ex) for ex in all_examples}
    train_stack = np.concatenate([dynamic_arrays[gid] for gid in train_ids], axis=0)
    mean = train_stack.mean(axis=0)
    std = train_stack.std(axis=0)
    std[std == 0] = 1.0
    return dynamic_arrays, mean, std
