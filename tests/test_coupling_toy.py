"""Prototype/validation for src/coupling.py's CoupledTwoStageFunction,
using cheap stand-ins for Snow17/SAC-SMA instead of the real Fortran --
validates the cross-container gradient-coupling design before wiring it
to the real models (see notes/NOTES.md for the full argument this
validates).

Three-way check, matching the discipline already used for the individual
shims (verify a VJP against manual perturbation, don't just trust it):

1. CoupledTwoStageFunction's gradients (FD-based, the actual mechanism)
2. autograd ground truth: the exact same math, written torch-native, so
   .backward() gives an exact gradient to compare against
3. an INDEPENDENTLY implemented brute-force finite difference of the
   whole pipeline's loss, at a DIFFERENT step size than what
   CoupledTwoStageFunction itself uses in production. This is the point:
   comparing FD to itself at the same eps proves nothing (a wiring bug --
   wrong sign, wrong parameter, wrong output -- doesn't shrink as eps
   shrinks, but doesn't grow either if it happens to be eps-independent;
   two INDEPENDENT computations at two DIFFERENT step sizes converging to
   the same answer is what actually rules that out).

Also structurally verifies RAIM (stage_a's output) never requires grad on
the theta_B path, and that stage_a is never even called while computing
theta_B's gradient block -- the entire point of option 1.5.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from coupling import CoupledTwoStageFunction, FDConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Toy stand-ins. Same shape as the real problem on purpose:
#   - stage A (toy "Snow17"): a temperature-threshold rain/snow partition
#     (sigmoid-relaxed, mirroring Snow17's planned PXTEMP relaxation),
#     3 parameters at different scales.
#   - stage B (toy "SAC-SMA"): a leaky-reservoir recursion with real
#     memory/lag (state carries across timesteps), 3 parameters including
#     one deliberately near zero to exercise the FD step floor.
# Each has a numpy version (the "black-box FD-only" stand-in passed to
# CoupledTwoStageFunction) and a torch-native version (autograd ground
# truth) implementing the identical math.
# ---------------------------------------------------------------------------

T = 60  # timesteps -- cheap, but long enough for the recursion to matter


def toy_snow_numpy(theta_a: np.ndarray, forcings) -> np.ndarray:
    scale, sharpness, threshold = theta_a
    precip, temp = forcings
    frac = 1.0 / (1.0 + np.exp(-sharpness * (temp - threshold)))
    return scale * precip * frac


def toy_snow_torch(theta_a: torch.Tensor, forcings) -> torch.Tensor:
    scale, sharpness, threshold = theta_a[0], theta_a[1], theta_a[2]
    precip, temp = forcings
    frac = torch.sigmoid(sharpness * (temp - threshold))
    return scale * precip * frac


def toy_sac_numpy(theta_b: np.ndarray, raim: np.ndarray) -> np.ndarray:
    k, c, q0 = theta_b
    runoff = np.empty_like(raim)
    state = 0.0
    for t in range(len(raim)):
        state = k * state + c * raim[t]
        runoff[t] = state + q0
    return runoff


def toy_sac_torch(theta_b: torch.Tensor, raim: torch.Tensor) -> torch.Tensor:
    k, c, q0 = theta_b[0], theta_b[1], theta_b[2]
    state = torch.zeros((), dtype=raim.dtype)
    out = []
    for t in range(raim.shape[0]):
        state = k * state + c * raim[t]
        out.append(state + q0)
    return torch.stack(out)


def nse_loss_torch(sim: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """1 - NSE, so minimizing this maximizes NSE -- the actual loss shape
    the real project uses. Nonlinear (mean-subtraction + ratio), not just
    a sum -- exercises the "differentiates for free downstream of runoff"
    claim in notes/logs.md."""
    denom = torch.sum((obs - obs.mean()) ** 2)
    nse = 1.0 - torch.sum((obs - sim) ** 2) / denom
    return 1.0 - nse


def nse_loss_numpy(sim: np.ndarray, obs: np.ndarray) -> float:
    denom = np.sum((obs - obs.mean()) ** 2)
    nse = 1.0 - np.sum((obs - sim) ** 2) / denom
    return 1.0 - nse


@pytest.fixture(scope="module")
def forcings():
    rng = np.random.default_rng(0)
    precip = rng.gamma(1.0, 3.0, T)
    temp = rng.normal(2.0, 4.0, T)  # spans well above/below the toy threshold
    return precip, temp


@pytest.fixture(scope="module")
def observed(forcings):
    """A fixed, arbitrary 'observed' target -- just needs to make the
    loss nontrivial and not identically zero at the test parameters."""
    precip, temp = forcings
    rng = np.random.default_rng(1)
    return np.abs(rng.normal(1.0, 0.5, T)) + 0.1 * precip


THETA_A0 = np.array([1.5, 3.0, 2.0])   # scale, sharpness, threshold
THETA_B0 = np.array([0.7, 0.5, 0.02])  # k, c, q0 (q0 deliberately near zero)


# ---------------------------------------------------------------------------
# Call-counting wrapper -- proves stage_a is never invoked while computing
# theta_B's gradient block (the structural claim option 1.5 depends on).
# ---------------------------------------------------------------------------

class _CountingStage:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.fn(*args)


def _run_coupled(forcings, fd_a: FDConfig, fd_b: FDConfig):
    theta_A = torch.tensor(THETA_A0, dtype=torch.float64, requires_grad=True)
    theta_B = torch.tensor(THETA_B0, dtype=torch.float64, requires_grad=True)

    stage_a = _CountingStage(toy_snow_numpy)
    stage_b = _CountingStage(toy_sac_numpy)

    # vjp_b=None: toy stand-ins have no Tesseract endpoint to route
    # through, so theta_B falls back to the same hand-rolled FD sweep as
    # theta_A -- see coupling.py's module docstring.
    runoff = CoupledTwoStageFunction.apply(theta_A, theta_B, forcings, stage_a, stage_b, fd_a, fd_b, None)
    assert not runoff.requires_grad or runoff.grad_fn is not None  # sanity: real graph node
    return theta_A, theta_B, runoff, stage_a, stage_b


@pytest.mark.parametrize("rel_step", [1e-3, 4e-3])
def test_coupled_gradients_match_autograd_and_independent_fd(forcings, observed, rel_step):
    fd = FDConfig(rel=rel_step, floor=1e-5, central=True)

    # ---- 1. CoupledTwoStageFunction (the mechanism under test) ----
    theta_A, theta_B, runoff, stage_a, stage_b = _run_coupled(forcings, fd, fd)
    obs_t = torch.tensor(observed, dtype=torch.float64)
    loss = nse_loss_torch(runoff, obs_t)
    loss.backward()
    grad_A_coupled = theta_A.grad.numpy().copy()
    grad_B_coupled = theta_B.grad.numpy().copy()

    n_a = len(THETA_A0)
    n_b = len(THETA_B0)
    # forward: 1 call. backward theta_A block: central diff -> 2 calls/param.
    assert stage_a.calls == 1 + 2 * n_a, (
        f"stage_a called {stage_a.calls} times, expected {1 + 2 * n_a} -- "
        "either the theta_A sweep isn't central, or stage_a leaked into "
        "the theta_B block (see next assertion)."
    )
    # theta_B block must NEVER call stage_a -- RAIM stays at the cached base.
    # forward: 1 call. theta_A block: 1 base-B call per A-perturbation (2*n_a
    # since central). theta_B block: 2 calls/param (central).
    assert stage_b.calls == 1 + 2 * n_a + 2 * n_b

    # ---- 2. autograd ground truth (same math, torch-native) ----
    theta_A_ref = torch.tensor(THETA_A0, dtype=torch.float64, requires_grad=True)
    theta_B_ref = torch.tensor(THETA_B0, dtype=torch.float64, requires_grad=True)
    precip, temp = forcings
    precip_t = torch.tensor(precip, dtype=torch.float64)
    temp_t = torch.tensor(temp, dtype=torch.float64)
    raim_ref = toy_snow_torch(theta_A_ref, (precip_t, temp_t))
    runoff_ref = toy_sac_torch(theta_B_ref, raim_ref)
    loss_ref = nse_loss_torch(runoff_ref, obs_t)
    loss_ref.backward()
    grad_A_autograd = theta_A_ref.grad.numpy()
    grad_B_autograd = theta_B_ref.grad.numpy()

    np.testing.assert_allclose(grad_A_coupled, grad_A_autograd, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(grad_B_coupled, grad_B_autograd, rtol=1e-3, atol=1e-6)

    # ---- 3. independent brute-force FD of the LOSS, at a DIFFERENT eps ----
    # Deliberately not reusing FDConfig/CoupledTwoStageFunction's own
    # machinery -- a hand-rolled central difference at eps/4, straight
    # through the numpy toy functions and numpy loss.
    def loss_at(theta_a, theta_b):
        raim = toy_snow_numpy(theta_a, forcings)
        runoff = toy_sac_numpy(theta_b, raim)
        return nse_loss_numpy(runoff, observed)

    def brute_force_grad(theta, other, theta_is_a: bool):
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            eps = max(abs(theta[i]) * rel_step, 1e-5) / 4.0
            plus = theta.copy(); plus[i] += eps
            minus = theta.copy(); minus[i] -= eps
            if theta_is_a:
                l_plus = loss_at(plus, other)
                l_minus = loss_at(minus, other)
            else:
                l_plus = loss_at(other, plus)
                l_minus = loss_at(other, minus)
            grad[i] = (l_plus - l_minus) / (2.0 * eps)
        return grad

    grad_A_brute = brute_force_grad(THETA_A0, THETA_B0, theta_is_a=True)
    grad_B_brute = brute_force_grad(THETA_B0, THETA_A0, theta_is_a=False)

    np.testing.assert_allclose(grad_A_coupled, grad_A_brute, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(grad_B_coupled, grad_B_brute, rtol=1e-3, atol=1e-6)


def test_raim_never_requires_grad_on_theta_b_path(forcings, observed):
    """Structural guarantee, not just a numerical one: stage_a's output
    (RAIM) must be a plain array (never a torch tensor / never part of an
    autograd graph) at the point it's fed into stage_b for the theta_B
    gradient block -- that's what keeps theta_B's gradient cheap. Checked
    directly inside CoupledTwoStageFunction.forward() via assertion; this
    test additionally confirms stage_a's call count is exactly 1 (the
    base forward pass) plus the theta_A sweep, and zero during the theta_B
    sweep specifically -- see the call-count assertions in the test above,
    reproduced here in isolation for a targeted failure message.
    """
    fd = FDConfig(rel=1e-3, floor=1e-5, central=True)
    _, _, runoff, stage_a, stage_b = _run_coupled(forcings, fd, fd)
    calls_after_forward = stage_a.calls
    assert calls_after_forward == 1

    obs_t = torch.tensor(observed, dtype=torch.float64)
    loss = nse_loss_torch(runoff, obs_t)
    loss.backward()

    n_a = len(THETA_A0)
    calls_after_backward = stage_a.calls
    assert calls_after_backward == 1 + 2 * n_a, (
        "stage_a's call count changed during what should be theta_B's "
        "isolated sweep -- RAIM leaked into a path that reruns Snow17."
    )


def test_mixed_dtype_thetas_get_matching_gradient_dtypes(forcings, observed):
    """Regression test for a real bug caught only once the real Tesseracts
    were wired in: Snow17 predicts/consumes float32, SAC-SMA float64 (see
    notes/NOTES.md). Every other test in this file uses float64 for BOTH
    theta_A and theta_B, which silently masked forward()/backward()
    originally casting both gradients (and the output) to theta_A's
    dtype regardless of theta_B's own -- wrong whenever the two blocks
    don't share a dtype, which is exactly the real integration's case.
    Fixed in src/coupling.py to track each leaf's dtype/device
    separately; this test exists so a regression shows up here, in a
    cheap toy run, rather than being caught only when the real,
    much-slower Fortran-backed pipeline is exercised.
    """
    theta_A = torch.tensor(THETA_A0, dtype=torch.float32, requires_grad=True)
    theta_B = torch.tensor(THETA_B0, dtype=torch.float64, requires_grad=True)
    fd = FDConfig(rel=1e-3, floor=1e-5, central=True)

    stage_a = _CountingStage(toy_snow_numpy)
    stage_b = _CountingStage(toy_sac_numpy)
    runoff = CoupledTwoStageFunction.apply(theta_A, theta_B, forcings, stage_a, stage_b, fd, fd, None)
    assert runoff.dtype == torch.float64

    obs_t = torch.tensor(observed, dtype=torch.float64)
    loss = nse_loss_torch(runoff, obs_t)
    loss.backward()

    assert theta_A.grad.dtype == torch.float32, (
        f"theta_A.grad has dtype {theta_A.grad.dtype}, expected float32 -- "
        "gradients must come back in each leaf's OWN dtype, not the other block's."
    )
    assert theta_B.grad.dtype == torch.float64, (
        f"theta_B.grad has dtype {theta_B.grad.dtype}, expected float64."
    )
    assert torch.all(torch.isfinite(theta_A.grad))
    assert torch.all(torch.isfinite(theta_B.grad))
