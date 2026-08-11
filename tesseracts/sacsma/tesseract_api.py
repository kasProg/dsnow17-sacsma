# SPDX-License-Identifier: Apache-2.0

"""Tesseract API module for SAC-SMA.

Wraps fortran/sacsma_shim.f90 (via src/sacsma.py's ctypes binding) as a
Tesseract: a full-rollout apply() plus a finite-difference
vector_jacobian_product(). Mirrors tesseracts/snow17/tesseract_api.py's
design and structure closely -- see that file's docstring for the shared
rationale (why full-rollout granularity, why FD is explicitly permitted
by the hackathon rules) -- adapted for SAC-SMA's specifics below.

Differentiable inputs: all 16 named scalar SAC-SMA parameters (UZTWM,
UZFWM, UZK, PCTIM, ADIMP, RIVA, ZPERC, REXP, LZTWM, LZFSM, LZFPM, LZSK,
LZPK, PFREE, SIDE, RSERV) -- unlike Snow17, there's no extra curve
parameter to exclude; CLAUDE.md's named list and the actual ex1 parameter
file agree exactly on 16.

Real kind: EXSAC/SAC1 use explicit DOUBLE PRECISION (confirmed by reading
the source -- see notes/NOTES.md), not Snow17's default 4-byte REAL. This
schema uses Float64 throughout; do not copy Snow17's Float32 convention
here.

VJP method: forward-difference finite differences (relative step with a
floor), same convention as tesseracts/snow17/tesseract_api.py. This is a
DIFFERENT, independent mechanism from src/coupling.py's internal FD sweep
-- this file's vector_jacobian_product exists so the SAC-SMA Tesseract is
independently testable/reusable on its own (matching the "a standalone
Tesseract works with any downstream model" reusability argument for
having two Tesseracts in the first place), and is what a caller doing
plain single-Tesseract composition (e.g. training SAC-SMA parameters
alone against observed RAIM, or any future reuse outside this pipeline)
would go through. The actual Snow17->SAC-SMA coupled training path in
this project does NOT call this endpoint -- src/coupling.py calls apply()
directly (see that module's docstring for why: an honest VJP through
SAC-SMA's own d(runoff)/d(RAIM) would need one perturbed run per RAIM
timestep, thousands of runs, which this endpoint would produce correctly
but far too slowly for that use).
"""

import os
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

# Local dev: tesseract_api.py -> tesseracts/sacsma -> tesseracts -> repo root
# (3 parents up). Inside a built container this doesn't hold -- see the
# matching comment in tesseracts/snow17/tesseract_api.py for why, and
# TESSERACT_PROJECT_ROOT's role.
_REPO_ROOT = Path(os.environ.get("TESSERACT_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sacsma import STATE_SIZE, SacSmaParams, run_sacsma  # noqa: E402

#
# Schemas
#

DIFFERENTIABLE_PARAMS = (
    "uztwm", "uzfwm", "uzk", "pctim", "adimp", "riva", "zperc", "rexp",
    "lztwm", "lzfsm", "lzfpm", "lzsk", "lzpk", "pfree", "side", "rserv",
)


class InputSchema(BaseModel):
    # Fixed timestep config -- seconds, matches EXSAC's own DTM
    # convention (it divides internally by 86400 to get days).
    dtm: Float64

    # Forcing, mm/day, deg C, mm/day. Not differentiated: same reasoning
    # as Snow17's pcp/tmp -- the LSTM predicts SAC-SMA *parameters*, not
    # forcing perturbations. tmp currently has zero effect on output
    # (only read behind the disabled IFRZE frozen-ground flag -- see
    # notes/logs.md) but is kept for interface parity with EXSAC.
    pcp: Array[(None,), Float64]
    tmp: Array[(None,), Float64]
    etp: Array[(None,), Float64]

    # Learnable scalar parameters -- differentiable via the FD VJP below.
    uztwm: Differentiable[Float64]
    uzfwm: Differentiable[Float64]
    uzk: Differentiable[Float64]
    pctim: Differentiable[Float64]
    adimp: Differentiable[Float64]
    riva: Differentiable[Float64]
    zperc: Differentiable[Float64]
    rexp: Differentiable[Float64]
    lztwm: Differentiable[Float64]
    lzfsm: Differentiable[Float64]
    lzfpm: Differentiable[Float64]
    lzsk: Differentiable[Float64]
    lzpk: Differentiable[Float64]
    pfree: Differentiable[Float64]
    side: Differentiable[Float64]
    rserv: Differentiable[Float64]

    # Initial carryover state: UZTWC, UZFWC, LZTWC, LZFSC, LZFPC, ADIMC.
    # Cold start is all-zero -- see src/sacsma.py's run_sacsma docstring.
    state0: Array[(STATE_SIZE,), Float64]


class OutputSchema(BaseModel):
    q: Differentiable[Array[(None,), Float64]]      # mm/day, TCI -- "runoff" for the NSE loss
    eta: Differentiable[Array[(None,), Float64]]     # mm/day, actual ET
    qs: Array[(None,), Float64]      # diagnostic: surface flow (not differentiated)
    qg: Array[(None,), Float64]      # diagnostic: groundwater flow
    roimp: Array[(None,), Float64]
    sdro: Array[(None,), Float64]
    ssur: Array[(None,), Float64]
    sif: Array[(None,), Float64]
    bfs: Array[(None,), Float64]
    bfp: Array[(None,), Float64]
    bfncc: Array[(None,), Float64]   # needed for the mass-balance check, see tests/test_sacsma_shim.py
    state_final: Array[(STATE_SIZE,), Float64]


#
# Shared rollout call -- used by apply() and, repeatedly, by
# vector_jacobian_product() for the base + perturbed evaluations.
#


def _params_from_inputs(inputs: InputSchema) -> SacSmaParams:
    return SacSmaParams(
        uztwm=float(inputs.uztwm), uzfwm=float(inputs.uzfwm), uzk=float(inputs.uzk),
        pctim=float(inputs.pctim), adimp=float(inputs.adimp), riva=float(inputs.riva),
        zperc=float(inputs.zperc), rexp=float(inputs.rexp), lztwm=float(inputs.lztwm),
        lzfsm=float(inputs.lzfsm), lzfpm=float(inputs.lzfpm), lzsk=float(inputs.lzsk),
        lzpk=float(inputs.lzpk), pfree=float(inputs.pfree), side=float(inputs.side),
        rserv=float(inputs.rserv),
    )


def _rollout(inputs: InputSchema) -> dict[str, np.ndarray]:
    """One full call to the shim: forcings + parameters + initial state ->
    q, eta, + diagnostics, final state. This is the single unit apply()
    and vector_jacobian_product()'s finite-difference evaluations both
    run."""
    out = run_sacsma(
        np.asarray(inputs.pcp, dtype=np.float64),
        np.asarray(inputs.tmp, dtype=np.float64),
        np.asarray(inputs.etp, dtype=np.float64),
        _params_from_inputs(inputs),
        state0=np.asarray(inputs.state0, dtype=np.float64),
        dtm=float(inputs.dtm),
    )
    return {
        "q": out.q,
        "eta": out.eta,
        "qs": out.qs,
        "qg": out.qg,
        "roimp": out.roimp,
        "sdro": out.sdro,
        "ssur": out.ssur,
        "sif": out.sif,
        "bfs": out.bfs,
        "bfp": out.bfp,
        "bfncc": out.bfncc,
        "state_final": out.state,
    }


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**_rollout(inputs))


#
# Optional endpoints
#

# Forward-difference step size: relative to the parameter's own
# magnitude, with a floor so a near-zero parameter (e.g. PCTIM/ADIMP,
# often 0 in practice) doesn't collapse the step to zero. Same convention
# as tesseracts/snow17/tesseract_api.py's _fd_step -- see that file's
# comment for the reasoning; SAC-SMA's float64 precision could tolerate a
# smaller floor, but there's no benefit to diverging from the sibling
# wrapper's already-validated values without a concrete reason to.
_FD_REL_STEP = 1e-3
_FD_MIN_STEP = 1e-4


def _fd_step(value: float) -> float:
    return max(abs(value) * _FD_REL_STEP, _FD_MIN_STEP)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, np.typing.ArrayLike],
) -> dict[str, np.typing.ArrayLike]:
    unsupported = set(vjp_inputs) - set(DIFFERENTIABLE_PARAMS)
    if unsupported:
        raise ValueError(
            f"vector_jacobian_product only supports {DIFFERENTIABLE_PARAMS}, "
            f"got unsupported input(s): {sorted(unsupported)}"
        )

    base = _rollout(inputs)

    vjp: dict[str, np.typing.ArrayLike] = {}
    for name in vjp_inputs:
        base_value = float(getattr(inputs, name))
        eps = _fd_step(base_value)
        perturbed_inputs = inputs.model_copy(update={name: base_value + eps})
        perturbed = _rollout(perturbed_inputs)

        grad = 0.0
        for out_name in vjp_outputs:
            d_output = (
                np.asarray(perturbed[out_name], dtype=np.float64)
                - np.asarray(base[out_name], dtype=np.float64)
            ) / eps
            cotangent = np.asarray(cotangent_vector[out_name], dtype=np.float64)
            grad += float(np.dot(cotangent.ravel(), d_output.ravel()))
        vjp[name] = np.float64(grad)

    return vjp


def abstract_eval(abstract_inputs) -> dict:
    """Shapes only, no computation. `abstract_inputs` is a
    Tesseract-generated model with ShapeDType values in place of arrays
    (attribute access, not dict-style) -- see the equivalent snow17
    docstring for the same caveat."""
    n = abstract_inputs.pcp.shape[0]
    return {
        "q": ShapeDType(shape=(n,), dtype="float64"),
        "eta": ShapeDType(shape=(n,), dtype="float64"),
        "qs": ShapeDType(shape=(n,), dtype="float64"),
        "qg": ShapeDType(shape=(n,), dtype="float64"),
        "roimp": ShapeDType(shape=(n,), dtype="float64"),
        "sdro": ShapeDType(shape=(n,), dtype="float64"),
        "ssur": ShapeDType(shape=(n,), dtype="float64"),
        "sif": ShapeDType(shape=(n,), dtype="float64"),
        "bfs": ShapeDType(shape=(n,), dtype="float64"),
        "bfp": ShapeDType(shape=(n,), dtype="float64"),
        "bfncc": ShapeDType(shape=(n,), dtype="float64"),
        "state_final": ShapeDType(shape=(STATE_SIZE,), dtype="float64"),
    }
