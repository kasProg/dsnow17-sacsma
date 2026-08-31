"""Cross-container gradient coupling for the Snow17 -> SAC-SMA chain.

Implemented as a single custom `torch.autograd.Function` rather than two
separately-composed Tesseract VJPs. See notes/NOTES.md for the full cost
argument; short version:

Standard composition (`apply_tesseract(A)` piped into `apply_tesseract(B)`,
then `.backward()`) asks Tesseract B for `d(runoff)/d(RAIM)` -- a
Jacobian-vector product with respect to a several-thousand-element daily
time series. Our VJPs are finite-difference, and FD cost scales with the
dimensionality of what's being differentiated, not the output -- so that
VJP alone would cost one perturbed SAC-SMA run per RAIM timestep
(thousands of runs). Intractable.

This avoids ever materializing that object. theta_A's (Snow17's)
gradient is computed by perturbing each Snow17 parameter, running BOTH
stages, and reading how the resulting *runoff* series moved -- an
end-to-end VJP that only costs one A+B run pair per parameter. theta_B's
(SAC-SMA's) gradient is cheap and ordinary: RAIM is held fixed at its
cached base value and only stage B reruns.

Both blocks are computed inside ONE Function's backward(), sharing one
ctx cache of the base forward pass. theta_A's block has no honest
alternative to the hand-rolled FD sweep below -- it needs stage_a rerun,
which no single Tesseract's own vector_jacobian_product() can do (that
endpoint only differentiates ITS OWN apply(), not a downstream
container). theta_B's block is different: RAIM is held fixed and never
recomputed, so it needs nothing stage_b's own Tesseract doesn't already
offer through its own vector_jacobian_product() endpoint (FD-based,
identically cheap -- one perturbed rollout per parameter, no per-RAIM-
timestep cost). CoupledNWSStack (pipeline.py) passes an optional `vjp_b`
callable wired to that real endpoint; when present, backward() calls it
instead of reimplementing the same sweep here. `vjp_b=None` (the toy
tests, and any caller without a real Tesseract handy) falls back to the
hand-rolled block, so both paths are exercised and kept honest against
each other -- see tests/test_pipeline_hhwm8.py's
test_sacsma_vjp_matches_fd_fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

# A "stage" is a black-box, non-torch-differentiable forward call -- in
# production this is a Tesseract's apply(), in tests/test_coupling_toy.py
# it's a cheap numpy stand-in. Signature: stage(theta_np, upstream) -> output_np.
Stage = Callable[[np.ndarray, object], np.ndarray]


@dataclass
class FDConfig:
    """Per-parameter finite-difference step: eps_i = max(rel * |theta_i|, floor).

    rel/floor may be scalars (applied uniformly) or per-parameter arrays --
    real Snow17/SAC-SMA parameters span very different scales (e.g. UZTWM
    ~O(100) mm vs UZK ~O(0.1)), so a single fixed eps is simultaneously too
    coarse for small parameters and too fine (noise-dominated) for large
    ones. `central=True` uses (f(x+eps)-f(x-eps))/2eps, which kills the
    leading (O(eps)) truncation term that a forward difference carries --
    worth the extra evaluation per parameter given float32 (Snow17) noise.
    """

    rel: float | np.ndarray = 1e-3
    floor: float | np.ndarray = 1e-4
    central: bool = True

    def steps(self, theta: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(theta) * self.rel, self.floor)


class CoupledTwoStageFunction(torch.autograd.Function):
    """Couples stage_a -> stage_b (A's output feeds B) as one autograd node.

    forward(theta_A, theta_B, forcings, stage_a, stage_b, fd_a, fd_b, vjp_b) -> output_b

    theta_A, theta_B: torch tensors (leaves predicted by the upstream
    network), differentiated by this Function.
    forcings: passed through to stage_a unchanged; never differentiated.
    stage_a, stage_b, fd_a, fd_b, vjp_b: plain Python objects
    (callables/config), not tensors -- torch.autograd.Function accepts
    non-tensor forward args positionally; backward must return None for
    each of their slots. vjp_b is REQUIRED at every call site (pass
    `None` explicitly to use the FD fallback) -- it is not given a
    Python-level default, because torch.autograd.Function's backward
    must return exactly as many values as forward received positional
    args at that specific .apply() call, and a default silently omitted
    at one call site but not another would desync that count.
    """

    @staticmethod
    def forward(ctx, theta_A, theta_B, forcings, stage_a, stage_b, fd_a, fd_b, vjp_b):
        theta_A_np = theta_A.detach().cpu().numpy().astype(np.float64)
        theta_B_np = theta_B.detach().cpu().numpy().astype(np.float64)

        output_a = stage_a(theta_A_np, forcings)
        assert not torch.is_tensor(output_a), (
            "stage_a must return a plain array, not a torch tensor -- "
            "if this fires, RAIM would carry an autograd graph edge and "
            "the whole point of this Function (avoiding the expensive "
            "standard composition) is defeated."
        )
        output_b = stage_b(theta_B_np, output_a)
        assert not torch.is_tensor(output_b)

        # Cached for backward(). Scoped to exactly one forward/backward
        # pair -- this IS the object's lifetime, no separate cache needed.
        ctx.stage_a = stage_a
        ctx.stage_b = stage_b
        ctx.fd_a = fd_a
        ctx.fd_b = fd_b
        ctx.vjp_b = vjp_b
        ctx.theta_A_np = theta_A_np
        ctx.theta_B_np = theta_B_np
        ctx.forcings = forcings
        ctx.output_a_base = output_a  # RAIM, held fixed for the theta_B block
        ctx.output_b_base = output_b  # runoff, reused by forward-difference mode
        # theta_A and theta_B are NOT assumed to share a dtype/device --
        # in the real integration Snow17 is float32 and SAC-SMA is
        # float64 (see notes/NOTES.md). Each gradient must be cast back
        # to its OWN leaf's original dtype/device, not the other's --
        # tests/test_coupling_toy.py used float64 for both blocks, which
        # silently masked this until the real wrappers were wired in
        # (see notes/logs.md).
        ctx.theta_A_dtype, ctx.theta_A_device = theta_A.dtype, theta_A.device
        ctx.theta_B_dtype, ctx.theta_B_device = theta_B.dtype, theta_B.device

        # Output dtype: float64 always, matching the last stage's (SAC-SMA)
        # true computational precision, independent of either theta's
        # dtype -- not tied to theta_A's dtype, which would silently
        # truncate the forward pass's own fidelity whenever theta_A
        # happens to be float32. np.array(..., copy=True) rather than
        # torch.as_tensor directly on output_b: some real stage_b
        # implementations (e.g. ctypes-backed ones) can return read-only
        # or externally-owned buffers, which torch.as_tensor would wrap
        # without copying -- undefined behavior if anything downstream
        # writes into it. Cheap insurance, not a hot path.
        return torch.as_tensor(np.array(output_b, copy=True), dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_np = grad_output.detach().cpu().numpy().astype(np.float64)

        grad_theta_A = _fd_vjp_block(
            theta=ctx.theta_A_np,
            fd=ctx.fd_a,
            grad_output=grad_output_np,
            run=lambda theta_perturbed: ctx.stage_b(
                ctx.theta_B_np, ctx.stage_a(theta_perturbed, ctx.forcings)
            ),
            base_output=ctx.output_b_base,
        )

        if ctx.vjp_b is not None:
            # theta_B never needs stage_a rerun (RAIM held at the cached
            # base) -- so, unlike theta_A, there is no FD-cost reason to
            # avoid stage_b's own Tesseract vector_jacobian_product()
            # endpoint. vjp_b (built by pipeline.py's CoupledNWSStack)
            # wraps exactly that call.
            grad_theta_B = ctx.vjp_b(ctx.theta_B_np, ctx.output_a_base, grad_output_np)
        else:
            grad_theta_B = _fd_vjp_block(
                theta=ctx.theta_B_np,
                fd=ctx.fd_b,
                grad_output=grad_output_np,
                # RAIM held at the cached base -- stage_a is NEVER called here.
                run=lambda theta_perturbed: ctx.stage_b(theta_perturbed, ctx.output_a_base),
                base_output=ctx.output_b_base,
            )

        grad_theta_A_t = torch.as_tensor(grad_theta_A, dtype=ctx.theta_A_dtype, device=ctx.theta_A_device)
        grad_theta_B_t = torch.as_tensor(grad_theta_B, dtype=ctx.theta_B_dtype, device=ctx.theta_B_device)
        return grad_theta_A_t, grad_theta_B_t, None, None, None, None, None, None


def _fd_vjp_block(
    theta: np.ndarray,
    fd: FDConfig,
    grad_output: np.ndarray,
    run: Callable[[np.ndarray], np.ndarray],
    base_output: np.ndarray,
) -> np.ndarray:
    """One parameter block's VJP: for each theta[i], perturb, rerun the
    (possibly multi-stage) pipeline via `run`, and dot the resulting
    change in output against the incoming cotangent. This is what makes
    it an honest VJP rather than a loss-specific gradient: `run` returns
    the full output series, and the cotangent contraction happens here,
    not baked into what gets finite-differenced.

    `base_output` is the pipeline's cached unperturbed output -- reused
    directly by forward-difference mode instead of recomputing it once
    per parameter."""
    steps = fd.steps(theta)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        eps = steps[i]
        theta_plus = theta.copy()
        theta_plus[i] += eps
        out_plus = run(theta_plus)

        if fd.central:
            theta_minus = theta.copy()
            theta_minus[i] -= eps
            out_minus = run(theta_minus)
            d_output = (out_plus - out_minus) / (2.0 * eps)
        else:
            d_output = (out_plus - base_output) / eps

        grad[i] = np.dot(grad_output, d_output)
    return grad
