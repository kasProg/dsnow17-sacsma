# SPDX-License-Identifier: Apache-2.0

"""Tesseract API module for Snow-17.

Wraps fortran/snow17_shim.f90 (via src/snow17.py's ctypes binding) as a
Tesseract: a full-rollout apply() plus a finite-difference
vector_jacobian_product(), so PyTorch's autograd can cross the Fortran
boundary via tesseract-torch's apply_tesseract(). See CLAUDE.md for why
this composition is the point of the project, not incidental to it.

Differentiable inputs: the 11 named scalar snow17 parameters (SCF, MFMAX,
MFMIN, UADJ, SI, NMF, TIPM, MBASE, PXTEMP, PLWHC, DAYGM). The 11-point ADC
(areal depletion curve) is intentionally NOT differentiable in this
version -- see notes/logs.md for why, and CLAUDE.md's Tesseract-specifics
section, which scopes the VJP to "the ~13 scalar parameters" (13 was an
approximate count in that doc; the actual named list is 11).

VJP method: forward-difference finite differences, one perturbed rollout
per differentiable input requested plus one shared base rollout. Wrapped
at full-rollout granularity (this whole module runs one call to
EXSNOW19 per timestep, not per Tesseract call), per Tesseract's own
guidance that it targets kernels running at least several seconds --
finite-differencing per-timestep would be both wrong (state carries
across timesteps within a rollout) and far too fine-grained.
"""

import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float32, Int32, ShapeDType

# tesseracts/snow17/tesseract_api.py -> tesseracts/snow17 -> tesseracts -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from snow17 import CS_SIZE, Snow17Params, run_snow17  # noqa: E402

#
# Schemas
#

# The 11 named scalar parameters snow17_shim.f90 exposes as learnable
# (CLAUDE.md's "Learnable parameters" list, minus ADC -- see module
# docstring). Order matters only for iteration below, not for correctness.
DIFFERENTIABLE_PARAMS = (
    "scf", "mfmax", "mfmin", "uadj", "si",
    "nmf", "tipm", "mbase", "pxtemp", "plwhc", "daygm",
)


class InputSchema(BaseModel):
    # Fixed timestep config -- must match what the shim was built/tested
    # against (fortran/snow17_shim.f90, tests/test_snow17_shim.py use idt=24,
    # idts=86400 throughout, i.e. daily).
    idt: Int32
    idts: Int32

    # Per-timestep calendar (avoids Gregorian/leap-year arithmetic
    # anywhere in this stack -- same reasoning as the shim itself).
    iyr: Array[(None,), Int32]
    imn: Array[(None,), Int32]
    ida: Array[(None,), Int32]

    # Forcing, mm/day and deg C. Not differentiated here: the LSTM this
    # Tesseract feeds into predicts snow17 *parameters* from basin
    # attributes, not forcing perturbations.
    pcp: Array[(None,), Float32]
    tmp: Array[(None,), Float32]

    # Basin attributes -- not learnable.
    alat: Float32
    elev: Float32

    # Learnable scalar parameters -- differentiable via the FD VJP below.
    scf: Differentiable[Float32]
    mfmax: Differentiable[Float32]
    mfmin: Differentiable[Float32]
    uadj: Differentiable[Float32]
    si: Differentiable[Float32]
    nmf: Differentiable[Float32]
    tipm: Differentiable[Float32]
    mbase: Differentiable[Float32]
    pxtemp: Differentiable[Float32]
    plwhc: Differentiable[Float32]
    daygm: Differentiable[Float32]

    # Areal depletion curve -- fixed for v1, see module docstring.
    adc: Array[(11,), Float32]

    # Initial carryover state. Cold start is CS=0, TPREV=0 -- EXSNOW19's
    # own documented convention, see CLAUDE.md's state-contract section.
    cs0: Array[(CS_SIZE,), Float32]
    tprev0: Float32


class OutputSchema(BaseModel):
    raim: Differentiable[Array[(None,), Float32]]   # mm/day, rain-plus-melt -> feeds SAC-SMA (or HBV, fallback-only per CLAUDE.md)
    sneqv: Differentiable[Array[(None,), Float32]]   # m, SWE
    snowh: Array[(None,), Float32]                   # m, snow depth (diagnostic; not differentiated)
    cs_final: Array[(CS_SIZE,), Float32]
    tprev_final: Float32


#
# Shared rollout call -- used by apply() and, repeatedly, by
# vector_jacobian_product() for the base + perturbed evaluations.
#


class _ArrayDates:
    """Adapts plain iyr/imn/ida int arrays to the .year/.month/.day
    attribute interface run_snow17() expects (normally a pandas
    DatetimeIndex) -- keeps this module free of a pandas dependency."""

    def __init__(self, iyr: np.ndarray, imn: np.ndarray, ida: np.ndarray) -> None:
        self.year, self.month, self.day = iyr, imn, ida


def _params_from_inputs(inputs: InputSchema) -> Snow17Params:
    return Snow17Params(
        alat=float(inputs.alat), elev=float(inputs.elev),
        scf=float(inputs.scf), mfmax=float(inputs.mfmax), mfmin=float(inputs.mfmin),
        uadj=float(inputs.uadj), si=float(inputs.si), nmf=float(inputs.nmf),
        tipm=float(inputs.tipm), mbase=float(inputs.mbase), pxtemp=float(inputs.pxtemp),
        plwhc=float(inputs.plwhc), daygm=float(inputs.daygm),
        adc=np.asarray(inputs.adc, dtype=np.float32),
    )


def _rollout(inputs: InputSchema) -> dict[str, np.ndarray]:
    """One full call to the shim: forcings + parameters + initial state ->
    raim, sneqv, snowh, final state. This is the single unit apply() and
    vector_jacobian_product()'s finite-difference evaluations both run."""
    dates = _ArrayDates(
        np.asarray(inputs.iyr), np.asarray(inputs.imn), np.asarray(inputs.ida)
    )
    out = run_snow17(
        dates,
        np.asarray(inputs.pcp, dtype=np.float32),
        np.asarray(inputs.tmp, dtype=np.float32),
        _params_from_inputs(inputs),
        cs0=np.asarray(inputs.cs0, dtype=np.float32),
        tprev0=float(inputs.tprev0),
        idt=int(inputs.idt),
        idts=int(inputs.idts),
    )
    return {
        "raim": out.raim,
        "sneqv": out.sneqv,
        "snowh": out.snowh,
        "cs_final": out.cs,
        "tprev_final": np.float32(out.tprev),
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
# magnitude, with a floor so a near-zero parameter (e.g. MBASE=0) doesn't
# collapse the step to something swamped by float32 rounding in the
# Fortran call. 1e-3 relative is conservative for float32 (~7 significant
# digits) -- small enough that a first-order forward difference is a
# reasonable local slope estimate, large enough that the perturbed run's
# output actually differs from the base run's at float32 precision.
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
        vjp[name] = np.float32(grad)

    return vjp


def abstract_eval(abstract_inputs) -> dict:
    """Shapes only, no computation -- output array lengths are exactly
    the input series length; state arrays are the fixed CS_SIZE/scalar
    shapes regardless of series length.

    `abstract_inputs` is an instance of a Tesseract-generated model with
    the same fields as InputSchema but ShapeDType values in place of
    arrays (attribute access, not dict-style -- it is not a plain dict
    despite the type hint convention used elsewhere in this file)."""
    n = abstract_inputs.pcp.shape[0]
    return {
        "raim": ShapeDType(shape=(n,), dtype="float32"),
        "sneqv": ShapeDType(shape=(n,), dtype="float32"),
        "snowh": ShapeDType(shape=(n,), dtype="float32"),
        "cs_final": ShapeDType(shape=(CS_SIZE,), dtype="float32"),
        "tprev_final": ShapeDType(shape=(), dtype="float32"),
    }
