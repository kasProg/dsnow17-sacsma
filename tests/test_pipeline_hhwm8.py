"""Single-basin, direct-parameter-optimization proof: Snow17 -> SAC-SMA ->
NSE-style loss -> backward(), chained via src/pipeline.py's
CoupledNWSStack (built on src/coupling.py's CoupledTwoStageFunction).
The actual milestone check: loss goes down.

No real observed streamflow ships with either vendored ex1 test case --
checked directly (grep across both test_cases/ trees for obs/flow/
discharge/streamflow file names came up empty), not assumed. So this
uses a synthetic-target parameter-recovery setup instead: run the real
HHWM8 chain once at the basin's actual calibrated parameters to generate
a synthetic "observed" runoff series, then optimize FROM a perturbed
initial guess back toward it. This proves gradients flow end-to-end
through both Fortran models via real autograd and that an optimizer can
actually use them to reduce a loss -- not a claim about real calibration
skill against observed streamflow, which needs real CAMELS data (see
src/train.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from pipeline import (  # noqa: E402
    SACSMA_PARAMS,
    SNOW17_PARAMS,
    CoupledNWSStack,
    SacSmaForcing,
    Snow17Forcing,
)
from test_sacsma_shim import _load_ex1_forcing as _load_sacsma_forcing  # noqa: E402
from test_sacsma_shim import _load_ex1_params as _load_sacsma_params  # noqa: E402
from test_snow17_shim import _load_ex1_forcing as _load_snow17_forcing  # noqa: E402
from test_snow17_shim import _load_ex1_params as _load_snow17_params  # noqa: E402

HRU_ID = "HHWM8IL"  # same basin/HRU/period in both vendored ex1 test cases


def _theta_vector(obj, names) -> np.ndarray:
    return np.array([getattr(obj, n) for n in names], dtype=np.float64)


@pytest.fixture(scope="module")
def hhwm8_setup():
    snow_params = _load_snow17_params(HRU_ID)
    dates, pcp, tmp = _load_snow17_forcing(HRU_ID)
    mask = (dates >= "1970-10-01") & (dates <= "1971-09-30")
    dates_wy, pcp_wy, tmp_wy = dates[mask], pcp[mask], tmp[mask]
    assert len(dates_wy) == 365

    sac_params = _load_sacsma_params(HRU_ID)
    sac_dates, _sac_pcp, _sac_tmp, sac_etp = _load_sacsma_forcing(HRU_ID)
    sac_mask = (sac_dates >= "1970-10-01") & (sac_dates <= "1971-09-30")
    etp_wy = sac_etp[sac_mask]
    assert len(etp_wy) == 365

    snow17_forcing = Snow17Forcing(
        idt=24, idts=86400,
        iyr=dates_wy.year.to_numpy().astype(np.int32),
        imn=dates_wy.month.to_numpy().astype(np.int32),
        ida=dates_wy.day.to_numpy().astype(np.int32),
        pcp=pcp_wy.astype(np.float64), tmp=tmp_wy.astype(np.float64),
        alat=snow_params.alat, elev=snow_params.elev,
        adc=snow_params.adc.astype(np.float64),
        cs0=np.zeros(19, dtype=np.float64), tprev0=0.0,
    )
    sacsma_forcing = SacSmaForcing(
        dtm=86400.0, tmp=tmp_wy.astype(np.float64), etp=etp_wy.astype(np.float64),
        state0=np.zeros(6, dtype=np.float64),
    )

    stack = CoupledNWSStack()

    theta_A_true = _theta_vector(snow_params, SNOW17_PARAMS)
    theta_B_true = _theta_vector(sac_params, SACSMA_PARAMS)

    return stack, snow17_forcing, sacsma_forcing, theta_A_true, theta_B_true


def test_loss_decreases_with_synthetic_target_recovery(hhwm8_setup):
    """The actual milestone check: gradients flowing through both real
    Fortran models via CoupledNWSStack let an optimizer reduce a
    streamflow loss over a real HHWM8 water year, starting from a
    perturbed initial guess."""
    stack, snow17_forcing, sacsma_forcing, theta_A_true, theta_B_true = hhwm8_setup

    with torch.no_grad():
        theta_A_ref = torch.tensor(theta_A_true, dtype=torch.float32)
        theta_B_ref = torch.tensor(theta_B_true, dtype=torch.float64)
        observed = stack.run(theta_A_ref, theta_B_ref, snow17_forcing, sacsma_forcing).clone()
    assert torch.all(torch.isfinite(observed)) and float(observed.sum()) > 0.0

    # Perturbed initial guess: 15% off truth, multiplicative (keeps sign
    # and legitimate zeros -- e.g. PCTIM/ADIMP are 0 for this basin --
    # unperturbed rather than accidentally flipping sign).
    rng = np.random.default_rng(0)
    theta_A0 = theta_A_true * (1.0 + 0.15 * rng.uniform(-1, 1, len(theta_A_true)))
    theta_B0 = theta_B_true * (1.0 + 0.15 * rng.uniform(-1, 1, len(theta_B_true)))

    # Reparametrize: optimize a dimensionless z = theta / scale (scale =
    # each parameter's own natural magnitude, floored away from 0) rather
    # than raw physical units directly. Necessary, not cosmetic -- a first
    # attempt with plain Adam directly on raw theta (uniform lr=0.02)
    # produced NaN gradients within one step: Adam's per-step update is
    # ~lr in the PARAMETER'S OWN units regardless of that parameter's
    # sensible range, so the same lr that barely nudges SI (~1500) blew
    # MBASE (~0) and PXTEMP (~0.7, must stay near [0,1]-ish) into a
    # physically invalid region the Fortran call couldn't handle. In
    # z-space every parameter starts at the same O(1) scale, so one
    # global lr is actually meaningful for all of them -- this is the
    # standard fix for optimizing heterogeneous-scale physical parameters
    # directly, not a workaround specific to this pipeline.
    scale_A = torch.tensor(np.maximum(np.abs(theta_A_true), 1e-3), dtype=torch.float32)
    scale_B = torch.tensor(np.maximum(np.abs(theta_B_true), 1e-3), dtype=torch.float64)
    z_A = (torch.tensor(theta_A0, dtype=torch.float32) / scale_A).clone().requires_grad_(True)
    z_B = (torch.tensor(theta_B0, dtype=torch.float64) / scale_B).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z_A, z_B], lr=0.05)

    losses = []
    n_steps = 12
    for _ in range(n_steps):
        optimizer.zero_grad()
        theta_A = z_A * scale_A
        theta_B = z_B * scale_B
        sim = stack.run(theta_A, theta_B, snow17_forcing, sacsma_forcing)
        denom = torch.sum((observed - observed.mean()) ** 2)
        nse = 1.0 - torch.sum((observed - sim) ** 2) / denom
        loss = 1.0 - nse
        loss.backward()

        assert z_A.grad is not None and torch.all(torch.isfinite(z_A.grad)), (
            f"z_A.grad is None or non-finite: {z_A.grad}"
        )
        assert z_B.grad is not None and torch.all(torch.isfinite(z_B.grad)), (
            f"z_B.grad is None or non-finite: {z_B.grad}"
        )

        optimizer.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not drop substantially over {n_steps} steps: "
        f"{losses[0]:.4f} -> {losses[-1]:.4f} (full trace: {losses})"
    )


def test_sacsma_vjp_matches_fd_fallback(hhwm8_setup):
    """theta_B's gradient now routes through SAC-SMA's own Tesseract
    vector_jacobian_product() endpoint by default (CoupledNWSStack's
    use_sacsma_vjp=True) instead of coupling.py's hand-rolled central-FD
    sweep (use_sacsma_vjp=False, the previous behavior, still available
    as a fallback). This is exactly the "two different gradient-
    computation machineries feeding the same upstream network" situation
    coupling.py's module docstring flags as a double-counting/sign-error
    risk -- so check the two paths actually agree on real HHWM8 data,
    not just trust the wiring by inspection.

    theta_A's path is untouched by use_sacsma_vjp either way (only
    theta_B's block branches on it), so its gradient should match
    bitwise across both stacks -- checked as a control, not the point of
    this test."""
    _, snow17_forcing, sacsma_forcing, theta_A_true, theta_B_true = hhwm8_setup

    stack_vjp = CoupledNWSStack()  # use_sacsma_vjp=True, the new default
    stack_fd = CoupledNWSStack(use_sacsma_vjp=False)  # old hand-rolled sweep

    def _grads(stack):
        theta_A = torch.tensor(theta_A_true, dtype=torch.float32, requires_grad=True)
        theta_B = torch.tensor(theta_B_true, dtype=torch.float64, requires_grad=True)
        sim = stack.run(theta_A, theta_B, snow17_forcing, sacsma_forcing)
        sim.sum().backward()  # simplest cotangent: all-ones
        return theta_A.grad.clone(), theta_B.grad.clone()

    grad_A_vjp, grad_B_vjp = _grads(stack_vjp)
    grad_A_fd, grad_B_fd = _grads(stack_fd)

    torch.testing.assert_close(grad_A_vjp, grad_A_fd, rtol=0, atol=0)  # unaffected path -- exact

    # theta_B: SAC-SMA's own VJP is forward-difference; the fallback sweep
    # is central-difference. Different formulas at the same eps -- expect
    # first-order agreement, not bitwise, but a sign flip, wrong param
    # order, or wrong output name would blow well past this.
    torch.testing.assert_close(grad_B_vjp, grad_B_fd, rtol=0.05, atol=1e-2)
