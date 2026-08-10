"""Tests for tesseracts/snow17/tesseract_api.py.

Two things this file exists to prove, per CLAUDE.md's Day 3-4 milestone:

1. apply() runs and its outputs match the same-inputs shim call directly
   (tests/test_shim.py already exhaustively validates the shim itself --
   this just proves the Tesseract wrapper doesn't corrupt anything in
   translation).
2. vector_jacobian_product() matches an independent manual perturbation
   done through the same apply() endpoint a caller would actually use --
   not by re-deriving the same formula internally, but by calling apply()
   twice and finite-differencing the result exactly the way an outside
   caller with no knowledge of the VJP implementation would.

Plus one more thing that isn't optional for this project's actual claim:
CLAUDE.md's whole thesis is that Tesseract is *load-bearing* -- gradients
have to actually cross the Fortran boundary via autograd, not just be
computable if you call vector_jacobian_product by hand.
test_backward_through_apply_tesseract proves that end to end: build a
torch graph with tesseract_torch.apply_tesseract(), call .backward(),
and confirm the resulting .grad matches vector_jacobian_product.

Runs entirely through tesseract_core.Tesseract.from_tesseract_api(),
which imports the API module directly -- no Docker required. Docker is
only needed for `tesseract build` (containerizing for submission), which
is blocked on this machine (see notes/logs.md) and is a separate concern
from whether apply()/vector_jacobian_product() are correct.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tesseract_core import Tesseract  # noqa: E402

from snow17 import CS_SIZE, Snow17Params, run_snow17  # noqa: E402

TESSERACT_DIR = REPO_ROOT / "tesseracts" / "snow17"

DIFFERENTIABLE_PARAMS = (
    "scf", "mfmax", "mfmin", "uadj", "si",
    "nmf", "tipm", "mbase", "pxtemp", "plwhc", "daygm",
)

# Same forward-difference convention as tesseract_api.py's own _fd_step --
# see that module's comment for why (float32-safe relative step, floored).
_FD_REL_STEP = 1e-3
_FD_MIN_STEP = 1e-4


def _fd_step(value: float) -> float:
    return max(abs(value) * _FD_REL_STEP, _FD_MIN_STEP)


@pytest.fixture(scope="module")
def tess():
    return Tesseract.from_tesseract_api(str(TESSERACT_DIR / "tesseract_api.py"))


def _synthetic_inputs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    doy = np.arange(n)
    return {
        "idt": 24, "idts": 86400,
        "iyr": np.full(n, 2001, dtype=np.int32),
        "imn": np.full(n, 1, dtype=np.int32),
        "ida": (np.arange(n, dtype=np.int32) % 28) + 1,
        "pcp": rng.gamma(1.0, 3.0, n).astype(np.float32),
        "tmp": (10 * np.sin(2 * np.pi * doy / 365 - np.pi / 2) + rng.normal(0, 3, n)).astype(np.float32),
        "alat": 47.78, "elev": 1612.5,
        "scf": 1.15, "mfmax": 1.05, "mfmin": 0.15, "uadj": 0.04, "si": 500.0,
        "nmf": 0.15, "tipm": 0.2, "mbase": 0.0, "pxtemp": 1.0, "plwhc": 0.03, "daygm": 0.0,
        "adc": np.array(
            [0.05, 0.09, 0.16, 0.31, 0.54, 0.74, 0.84, 0.89, 0.93, 0.97, 1.0], dtype=np.float32
        ),
        "cs0": np.zeros(CS_SIZE, dtype=np.float32),
        "tprev0": 0.0,
    }


# ---------------------------------------------------------------------------
# apply() matches the shim directly (translation layer sanity)
# ---------------------------------------------------------------------------

def test_apply_matches_shim_directly(tess):
    inputs = _synthetic_inputs()
    out = tess.apply(inputs)

    class _Dates:
        def __init__(self, iyr, imn, ida):
            self.year, self.month, self.day = iyr, imn, ida

    params = Snow17Params(
        alat=inputs["alat"], elev=inputs["elev"], scf=inputs["scf"], mfmax=inputs["mfmax"],
        mfmin=inputs["mfmin"], uadj=inputs["uadj"], si=inputs["si"], nmf=inputs["nmf"],
        tipm=inputs["tipm"], mbase=inputs["mbase"], pxtemp=inputs["pxtemp"],
        plwhc=inputs["plwhc"], daygm=inputs["daygm"], adc=inputs["adc"],
    )
    direct = run_snow17(
        _Dates(inputs["iyr"], inputs["imn"], inputs["ida"]), inputs["pcp"], inputs["tmp"], params,
        cs0=inputs["cs0"], tprev0=inputs["tprev0"], idt=inputs["idt"], idts=inputs["idts"],
    )

    np.testing.assert_array_equal(out["raim"], direct.raim)
    np.testing.assert_array_equal(out["sneqv"], direct.sneqv)
    np.testing.assert_array_equal(out["snowh"], direct.snowh)
    np.testing.assert_array_equal(out["cs_final"], direct.cs)
    assert float(out["tprev_final"]) == direct.tprev


def _to_shapedtype(value) -> dict:
    arr = np.asarray(value)
    dtype = str(arr.dtype)
    # numpy defaults to int64/float64 for plain Python scalars; the schema
    # is float32/int32 throughout, and abstract_eval only reads .shape,
    # but pass the declared dtype anyway so this stays a faithful stand-in
    # for what the Tesseract runtime actually constructs.
    if dtype == "float64":
        dtype = "float32"
    elif dtype == "int64":
        dtype = "int32"
    return {"shape": list(arr.shape), "dtype": dtype}


def test_abstract_eval_matches_apply_shapes(tess):
    inputs = _synthetic_inputs(n=45)
    out = tess.apply(inputs)
    abstract_inputs = {k: _to_shapedtype(v) for k, v in inputs.items()}
    abstract_out = tess.abstract_eval(abstract_inputs)
    for key in ("raim", "sneqv", "snowh", "cs_final", "tprev_final"):
        assert tuple(abstract_out[key]["shape"]) == np.asarray(out[key]).shape


# ---------------------------------------------------------------------------
# vector_jacobian_product vs. manual perturbation through apply()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param", DIFFERENTIABLE_PARAMS)
def test_vjp_matches_manual_perturbation(tess, param):
    """dL/d(param) for L = sum(raim) + sum(sneqv), computed two ways:

    1. vector_jacobian_product with cotangent = all-ones on both outputs
       (the sum-reduction loss this represents).
    2. Manually: call apply() at the base and perturbed parameter values
       (same eps, but via the public apply() endpoint, not any internal
       function -- this is what an outside caller with no knowledge of
       vector_jacobian_product's implementation would do), then
       finite-difference sum(raim)+sum(sneqv) directly.

    These use the same finite-difference formula on both sides by
    construction -- this test is not validating that FD is an accurate
    gradient estimator (it structurally can't be, at hard PXTEMP/ADC
    thresholds -- see notes/NOTES.md and the flat-gradient test below).
    It validates that vector_jacobian_product's *wiring* -- which
    parameter got perturbed, which outputs got read, the cotangent dot
    product, accumulation across outputs -- is correct. A bug in any of
    those (wrong param mutated, sign flipped, output name typo'd,
    cotangent applied to the wrong output) would show up as a mismatch
    here even though the underlying FD math is identical.
    """
    import tesseract_api as api

    inputs = _synthetic_inputs()
    base = tess.apply(inputs)
    n = len(base["raim"])

    cotangent = {"raim": np.ones(n, dtype=np.float32), "sneqv": np.ones(n, dtype=np.float32)}
    vjp = tess.vector_jacobian_product(
        inputs, vjp_inputs=[param], vjp_outputs=["raim", "sneqv"], cotangent_vector=cotangent
    )

    # Use the same float32-coerced base value vector_jacobian_product itself
    # reads (via the validated InputSchema), not the raw Python float from
    # the input dict -- pydantic coerces e.g. 1.05 -> np.float32(1.05) ==
    # 1.0499999523..., and computing eps from a slightly different base
    # value than the one actually perturbed produces a spurious mismatch
    # for threshold/lookup-heavy melt physics that has nothing to do with
    # whether the VJP's wiring is correct.
    validated = api.InputSchema(**inputs)
    base_value = float(getattr(validated, param))
    eps = _fd_step(base_value)
    perturbed_inputs = dict(inputs)
    perturbed_inputs[param] = base_value + eps
    perturbed = tess.apply(perturbed_inputs)

    # Cast to float64 and subtract element-wise BEFORE summing, matching
    # what vector_jacobian_product itself does (tesseract_api.py). Summing
    # the float32 arrays first and subtracting the two sums (as an earlier
    # version of this test did) hits catastrophic cancellation: perturbed
    # and base are nearly-identical float32 arrays, so their sums are two
    # close, largeish float32 numbers, and subtracting them loses far more
    # precision than subtracting the (small) per-element differences and
    # summing those in float64. That earlier version disagreed with the
    # correct vjp by ~0.3-0.5% for several parameters -- confirmed by
    # reproducing vector_jacobian_product's internal arithmetic line by
    # line, which matched vjp bit-for-bit and made clear the mismatch was
    # this test's bug, not the wrapper's.
    raim_diff = (
        np.asarray(perturbed["raim"], dtype=np.float64) - np.asarray(base["raim"], dtype=np.float64)
    ).sum()
    sneqv_diff = (
        np.asarray(perturbed["sneqv"], dtype=np.float64) - np.asarray(base["sneqv"], dtype=np.float64)
    ).sum()
    manual_grad = (raim_diff + sneqv_diff) / eps

    np.testing.assert_allclose(float(vjp[param]), manual_grad, rtol=1e-4, atol=1e-6)


def test_vjp_linear_in_cotangent(tess):
    """VJP(cotangent_raim_only) + VJP(cotangent_sneqv_only) ==
    VJP(both cotangents combined), for every differentiable param at once.
    Validates the per-output accumulation loop independent of any
    external reference -- VJPs are linear in the cotangent by
    definition, so this must hold regardless of what the underlying
    physics does."""
    inputs = _synthetic_inputs()
    n = len(tess.apply(inputs)["raim"])
    ones = np.ones(n, dtype=np.float32)
    zeros = np.zeros(n, dtype=np.float32)

    vjp_raim = tess.vector_jacobian_product(
        inputs, vjp_inputs=list(DIFFERENTIABLE_PARAMS), vjp_outputs=["raim", "sneqv"],
        cotangent_vector={"raim": ones, "sneqv": zeros},
    )
    vjp_sneqv = tess.vector_jacobian_product(
        inputs, vjp_inputs=list(DIFFERENTIABLE_PARAMS), vjp_outputs=["raim", "sneqv"],
        cotangent_vector={"raim": zeros, "sneqv": ones},
    )
    vjp_both = tess.vector_jacobian_product(
        inputs, vjp_inputs=list(DIFFERENTIABLE_PARAMS), vjp_outputs=["raim", "sneqv"],
        cotangent_vector={"raim": ones, "sneqv": ones},
    )

    for p in DIFFERENTIABLE_PARAMS:
        np.testing.assert_allclose(
            float(vjp_raim[p]) + float(vjp_sneqv[p]), float(vjp_both[p]), rtol=1e-5, atol=1e-8
        )


def test_vjp_rejects_adc():
    """ADC is deliberately not differentiable in v1 (see module docstring
    in tesseract_api.py and notes/logs.md) -- requesting it should fail
    loudly, not silently return a meaningless/zero gradient."""
    import tesseract_api

    with pytest.raises(ValueError, match="adc"):
        inputs_model = tesseract_api.InputSchema(**_synthetic_inputs())
        tesseract_api.vector_jacobian_product(
            inputs_model, vjp_inputs={"adc"}, vjp_outputs={"raim"},
            cotangent_vector={"raim": np.ones(60, dtype=np.float32)},
        )


def test_pxtemp_gradient_is_structurally_near_zero(tess):
    """Documents, rather than silently accepts, a known limitation: with
    the hard TA-vs-PXTEMP threshold (unrelaxed -- see CLAUDE.md's
    "sigmoid relaxation" note and PACK19.f:104), a small perturbation to
    PXTEMP flips the rain/snow classification for at most a handful of
    timesteps whose temperature happens to sit within `eps` of the
    threshold -- so d(loss)/d(PXTEMP) is zero almost everywhere and only
    intermittently nonzero, unlike the other 10 parameters which affect
    every timestep continuously. This test would need updating (not
    deleting) once the sigmoid relaxation lands.
    """
    inputs = _synthetic_inputs(n=200)  # longer series -> more chances to catch a threshold-crossing day
    n = 200
    cotangent = {"raim": np.ones(n, dtype=np.float32), "sneqv": np.ones(n, dtype=np.float32)}
    vjp = tess.vector_jacobian_product(
        inputs, vjp_inputs=list(DIFFERENTIABLE_PARAMS), vjp_outputs=["raim", "sneqv"],
        cotangent_vector=cotangent,
    )

    # PXTEMP's gradient should be small relative to a parameter known to
    # affect every timestep continuously (MFMAX drives melt whenever
    # there's snow on the ground and T>0, which synthetic winter data
    # guarantees plenty of).
    assert abs(vjp["pxtemp"]) < 0.1 * abs(vjp["mfmax"]) + 1e-6, (
        f"pxtemp gradient ({vjp['pxtemp']}) is no longer near-zero relative to "
        f"mfmax ({vjp['mfmax']}) -- if the sigmoid relaxation landed, update/remove "
        "this test rather than leaving it passing by coincidence."
    )


# ---------------------------------------------------------------------------
# The actual claim: gradients cross the Fortran boundary via torch autograd,
# not just via a direct vector_jacobian_product call.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param", ("scf", "mfmax"))
def test_backward_through_apply_tesseract(tess, param):
    """d(loss)/d(param) via torch.Tensor(requires_grad=True) ->
    apply_tesseract() -> loss.backward() -> param.grad, compared against
    vector_jacobian_product() called directly with the same cotangent this
    loss implies (all-ones on both outputs, since loss = sum(raim) +
    sum(sneqv)). This is the composition CLAUDE.md's whole argument rests
    on -- Tesseract splicing EXSNOW19 into PyTorch's autograd graph as a
    normal differentiable layer, not just the VJP existing in isolation.
    """
    torch = pytest.importorskip("torch")
    from tesseract_torch import apply_tesseract

    inputs = _synthetic_inputs()
    inputs = dict(inputs)
    inputs[param] = torch.tensor(inputs[param], dtype=torch.float32, requires_grad=True)

    out = apply_tesseract(tess, inputs)
    loss = out["raim"].sum() + out["sneqv"].sum()
    loss.backward()

    autograd_grad = inputs[param].grad.item()

    static_inputs = dict(inputs)
    static_inputs[param] = float(static_inputs[param].detach())
    n = len(tess.apply(static_inputs)["raim"])
    cotangent = {"raim": np.ones(n, dtype=np.float32), "sneqv": np.ones(n, dtype=np.float32)}
    vjp = tess.vector_jacobian_product(
        static_inputs, vjp_inputs=[param], vjp_outputs=["raim", "sneqv"], cotangent_vector=cotangent
    )

    np.testing.assert_allclose(autograd_grad, float(vjp[param]), rtol=1e-4, atol=1e-6)
    assert autograd_grad != 0.0, (
        f"d(loss)/d({param}) came back exactly 0.0 through autograd -- gradients aren't "
        "actually flowing, which is the load-bearing claim this whole project rests on."
    )
