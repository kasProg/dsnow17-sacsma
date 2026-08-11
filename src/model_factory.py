"""Model dispatch by cfg.model.name -- the one place src/train.py and
src/infer.py decide which nn.Module to build. Keeping this as a small
separate module (rather than an if/elif inline in train.py) means
infer.py can build the exact same architecture a checkpoint was trained
with by reading the same config group, without duplicating the
construction logic.
"""

from __future__ import annotations

import torch.nn as nn


def build_model(cfg, n_static: int, n_climate_or_dynamic: int) -> nn.Module:
    """n_climate_or_dynamic: n_climate_features for "hybrid" (ParamNet's
    climatology-sequence width, currently 3: prcp/tmean/pet), n_dynamic
    for "benchmark_lstm" (LSTMBenchmark's per-timestep forcing width,
    also currently 3 -- see src/benchmark_lstm.py's build_dynamic_array).
    Same number in this project's current setup, but kept as separate
    call sites' business, not assumed equal here.
    """
    name = cfg.model.name

    if name == "hybrid":
        from paramnet import ParamNet

        return ParamNet(
            n_static_features=n_static,
            n_climate_features=n_climate_or_dynamic,
            lstm_hidden=cfg.model.lstm_hidden,
            mlp_hidden=cfg.model.mlp_hidden,
            n_hidden_layers=cfg.model.n_hidden_layers,
            dropout=cfg.model.dropout,
        )

    if name == "benchmark_lstm":
        from benchmark_lstm import LSTMBenchmark

        return LSTMBenchmark(
            n_dynamic=n_climate_or_dynamic,
            n_static=n_static,
            hidden=cfg.model.hidden,
            dropout=cfg.model.dropout,
        )

    raise ValueError(f"unknown model.name: {name!r} (expected 'hybrid' or 'benchmark_lstm')")
