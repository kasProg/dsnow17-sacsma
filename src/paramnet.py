"""LSTM-encoder + MLP head predicting Snow17 + SAC-SMA parameters from
basin static attributes AND a monthly climatology sequence.
CLAUDE.md's Day 7-10 milestone.

Two inputs:
  x_static:  (batch, 39) z-score-normalized CAMELS static attributes
             (data/build_attributes.py) -- climate indices, topography,
             soil, vegetation, geology.
  x_climate: (batch, 12, 3) z-score-normalized monthly climatology
             (data/build_climatology.py) -- mean [prcp, tmean, pet] per
             calendar month, from each basin's full available record.

An LSTM encodes x_climate into a learned seasonal-pattern embedding
(its final hidden state), concatenated with x_static, then a small MLP
head maps the combination to all 27 parameters. Parameters stay STATIC
per rollout (one value per basin per training window) -- a hard
constraint from this project's FD-gradient-cost design (a genuinely
time-varying-parameter output would reintroduce the "one FD run per
RAIM timestep" cost src/coupling.py exists specifically to avoid, see
notes/NOTES.md). What the LSTM buys instead: richer input signal than
static summary attributes alone -- actual monthly seasonal patterns in
precipitation/temperature/PET -- while the output stays a compact,
FD-tractable parameter vector.

Output: each of the 27 parameters is mapped into its own physically
valid range via a bounded sigmoid (see *_BOUNDS below) -- never left as
raw unconstrained network output. tests/test_pipeline_hhwm8.py already
demonstrated what unconstrained direct optimization does to parameters
spanning wildly different scales (SI ~1500 next to MBASE ~0): one bad
step and the Fortran call gets fed a value it can't handle. Bounding at
the network's own output layer prevents that structurally, for every
training step, not dependent on a manually-tuned learning rate.

Methodological lineage, not code: this LSTM-encoder-plus-physical-model
pattern follows the general differentiable-parameter-learning approach
described in Feng et al. (2022, WRR) and the broader MHPI dPL line of
work, and the architecture patterns here were informed by studying
NeuralHydrology (Kratzert et al., github.com/neuralhydrology/neuralhydrology,
BSD-3-Clause) -- NOT MHPI's own dPLHBVrelease/generic_deltamodel/hydrodl2,
which are PSU Non-Commercial licensed and incompatible with this
project's Apache-2.0 submission (CLAUDE.md's existing hard constraint;
their source was deliberately not read while building this file, only
cited as prior work). No code from either source is copied here; this
is an original implementation of a well-documented, published modeling
pattern.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pipeline import SACSMA_PARAMS, SNOW17_PARAMS

# (low, high) physically valid range for each parameter.
#
# Snow17 ranges: informed by the state-contract/units notes in
# CLAUDE.md and the real calibrated HHWM8 values seen throughout this
# project (external/snow17/test_cases/ex1/input/params/snow17_params.HHWM8.txt),
# widened to plausible operational bounds rather than tightly bracketing
# just that one basin.
#
# SAC-SMA ranges: standard NWS/SCE-UA calibration bounds widely used in
# the hydrology literature (e.g. Duan, Sorooshian & Gupta 1992) -- not
# tuned specifically for this project. A reasonable-effort default given
# hackathon scope; revisit if training pushes many basins to a bound.
SNOW17_BOUNDS: dict[str, tuple[float, float]] = {
    "scf": (0.7, 2.0),
    "mfmax": (0.5, 2.0),
    "mfmin": (0.05, 0.6),
    "uadj": (0.02, 0.5),
    "si": (0.0, 2000.0),
    "nmf": (0.05, 0.5),
    "tipm": (0.01, 1.0),
    "mbase": (-1.0, 1.0),
    "pxtemp": (-2.0, 2.0),
    "plwhc": (0.02, 0.3),
    "daygm": (0.0, 0.3),
}

SACSMA_BOUNDS: dict[str, tuple[float, float]] = {
    "uztwm": (1.0, 150.0),
    "uzfwm": (1.0, 150.0),
    "uzk": (0.1, 0.5),
    "pctim": (0.0, 0.1),
    "adimp": (0.0, 0.4),
    "riva": (0.0, 0.3),
    "zperc": (1.0, 250.0),
    "rexp": (1.0, 5.0),
    "lztwm": (1.0, 500.0),
    "lzfsm": (1.0, 400.0),
    "lzfpm": (1.0, 1000.0),
    "lzsk": (0.01, 0.35),
    "lzpk": (0.0001, 0.025),
    "pfree": (0.0, 0.6),
    "side": (0.0, 0.3),
    "rserv": (0.0, 0.4),
}


class ParamNet(nn.Module):
    """forward(x_static, x_climate) -> (theta_A, theta_B):
    (batch, 11) float32, (batch, 16) float64 -- dtypes match what
    src/pipeline.py's CoupledNWSStack.run() (and src/coupling.py's
    per-leaf dtype handling) expect for Snow17 vs. SAC-SMA respectively.
    """

    def __init__(
        self,
        n_static_features: int,
        n_climate_features: int = 3,
        lstm_hidden: int = 32,
        mlp_hidden: int = 64,
        n_hidden_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.snow17_names = SNOW17_PARAMS
        self.sacsma_names = SACSMA_PARAMS

        snow17_bounds = torch.tensor(
            [SNOW17_BOUNDS[n] for n in self.snow17_names], dtype=torch.float64
        )
        sacsma_bounds = torch.tensor(
            [SACSMA_BOUNDS[n] for n in self.sacsma_names], dtype=torch.float64
        )
        # Registered as buffers (not parameters) so they move with
        # .to(device)/.double() etc. and are saved/loaded with the model,
        # without being trained.
        self.register_buffer("snow17_lo", snow17_bounds[:, 0])
        self.register_buffer("snow17_hi", snow17_bounds[:, 1])
        self.register_buffer("sacsma_lo", sacsma_bounds[:, 0])
        self.register_buffer("sacsma_hi", sacsma_bounds[:, 1])

        self.lstm = nn.LSTM(
            input_size=n_climate_features, hidden_size=lstm_hidden, batch_first=True
        ).double()

        n_out = len(self.snow17_names) + len(self.sacsma_names)
        layers: list[nn.Module] = []
        in_dim = n_static_features + lstm_hidden
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = mlp_hidden
        layers += [nn.Linear(in_dim, n_out)]
        self.head = nn.Sequential(*layers).double()

    def forward(
        self, x_static: torch.Tensor, x_climate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _c_n) = self.lstm(x_climate.to(torch.float64))
        climate_embed = h_n[-1]  # (batch, lstm_hidden) -- final layer's final hidden state

        combined = torch.cat([x_static.to(torch.float64), climate_embed], dim=-1)
        raw = self.head(combined)

        n_a = len(self.snow17_names)
        raw_a, raw_b = raw[..., :n_a], raw[..., n_a:]
        frac_a = torch.sigmoid(raw_a)
        frac_b = torch.sigmoid(raw_b)
        theta_A = (self.snow17_lo + frac_a * (self.snow17_hi - self.snow17_lo)).to(torch.float32)
        theta_B = self.sacsma_lo + frac_b * (self.sacsma_hi - self.sacsma_lo)
        return theta_A, theta_B
