"""Tests for src/paramnet.py -- pure PyTorch, no Fortran/Tesseract
involved, so these stay fast regardless of what the physics-backed
suite costs.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from paramnet import SACSMA_BOUNDS, SNOW17_BOUNDS, ParamNet  # noqa: E402

N_FEATURES = 39


def test_output_shapes_and_dtypes():
    net = ParamNet(n_features=N_FEATURES)
    x = torch.randn(5, N_FEATURES, dtype=torch.float64)
    theta_A, theta_B = net(x)

    assert theta_A.shape == (5, 11)
    assert theta_B.shape == (5, 16)
    # Must match src/pipeline.py's CoupledNWSStack.run() expectations --
    # Snow17 float32, SAC-SMA float64 (src/coupling.py tracks each
    # leaf's dtype independently; getting this wrong here would silently
    # break gradient dtype matching downstream).
    assert theta_A.dtype == torch.float32
    assert theta_B.dtype == torch.float64


def test_outputs_always_respect_parameter_bounds():
    """Structural guarantee, not a training outcome: the bounded-sigmoid
    output layer must keep every parameter inside its physical range
    regardless of what the (untrained, random-init) network weights are
    -- this is what prevents the NaN-blowup failure mode
    tests/test_pipeline_hhwm8.py hit with unconstrained direct
    optimization (see that file's docstring)."""
    net = ParamNet(n_features=N_FEATURES)
    # A wide spread of inputs, including extreme values -- the bound
    # guarantee must hold regardless of how extreme the raw network
    # output gets, not just for "reasonable" inputs.
    x = torch.tensor(
        np.concatenate([
            np.random.default_rng(0).normal(0, 1, (20, N_FEATURES)),
            np.random.default_rng(1).normal(0, 50, (20, N_FEATURES)),
        ]),
        dtype=torch.float64,
    )
    theta_A, theta_B = net(x)

    for i, name in enumerate(net.snow17_names):
        lo, hi = SNOW17_BOUNDS[name]
        vals = theta_A[:, i].detach().numpy()
        assert np.all(vals >= lo) and np.all(vals <= hi), (name, vals.min(), vals.max(), lo, hi)

    for i, name in enumerate(net.sacsma_names):
        lo, hi = SACSMA_BOUNDS[name]
        vals = theta_B[:, i].detach().numpy()
        assert np.all(vals >= lo) and np.all(vals <= hi), (name, vals.min(), vals.max(), lo, hi)


def test_gradient_flows_to_every_network_parameter():
    net = ParamNet(n_features=N_FEATURES)
    x = torch.randn(3, N_FEATURES, dtype=torch.float64)
    theta_A, theta_B = net(x)
    loss = theta_A.sum() + theta_B.sum()
    loss.backward()

    params_with_grad = [p for p in net.parameters() if p.grad is not None]
    assert len(params_with_grad) == len(list(net.parameters())), (
        "some network parameter never received a gradient -- a layer is "
        "disconnected from the output"
    )
    assert all(torch.all(torch.isfinite(p.grad)) for p in params_with_grad)
    assert any(p.grad.abs().sum() > 0 for p in params_with_grad), (
        "all gradients are exactly zero -- same silent-zero-gradient "
        "concern flagged elsewhere in this project (CudnnLstmModel, see "
        "CLAUDE.md)"
    )
