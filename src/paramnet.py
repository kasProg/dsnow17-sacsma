"""MLP predicting Snow17 + SAC-SMA parameters from static basin
attributes -- CLAUDE.md's Day 7-10 milestone.

Input: 39 z-score-normalized CAMELS attributes (data/build_attributes.py).
Output: 11 Snow17 parameters + 16 SAC-SMA parameters, each mapped into
its own physically valid range via a per-parameter bounded sigmoid (see
PARAM_BOUNDS below) -- never left as raw unconstrained network output.
tests/test_pipeline_hhwm8.py already demonstrated what unconstrained
direct optimization does to parameters spanning wildly different scales
(SI ~1500 next to MBASE ~0): one bad step and the Fortran call gets fed
a value it can't handle. Bounding at the network's own output layer
prevents that structurally, for every training step, not just for a
manually-tuned learning rate on one basin.

MLP, not LSTM: CLAUDE.md's own pipeline diagram labels this
"[ LSTM / MLP ]" -- undecided in the planning doc. The input here is
purely static per-basin attributes with no time dimension, so there's
nothing for an LSTM to be recurrent over. MLP is the architecturally
correct choice given the actual input/output shapes, not a stylistic
pick between two equally-valid options.
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
    """x: (batch, n_features) z-score-normalized basin attributes.
    forward(x) -> (theta_A, theta_B): (batch, 11) float32, (batch, 16)
    float64 -- dtypes match what src/pipeline.py's CoupledNWSStack.run()
    (and src/coupling.py's per-leaf dtype handling) expect for Snow17
    vs. SAC-SMA respectively.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
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

        n_out = len(self.snow17_names) + len(self.sacsma_names)
        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, n_out)]
        self.net = nn.Sequential(*layers).double()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(x.to(torch.float64))
        n_a = len(self.snow17_names)
        raw_a, raw_b = raw[..., :n_a], raw[..., n_a:]

        frac_a = torch.sigmoid(raw_a)
        frac_b = torch.sigmoid(raw_b)
        theta_A = (self.snow17_lo + frac_a * (self.snow17_hi - self.snow17_lo)).to(torch.float32)
        theta_B = self.sacsma_lo + frac_b * (self.sacsma_hi - self.sacsma_lo)
        return theta_A, theta_B
