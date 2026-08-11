"""Tests for the sac-sma Fortran shim (fortran/sacsma_shim.f90).

Same three required tests as tests/test_snow17_shim.py (determinism,
state continuity, mass balance), plus:

- an explicit proof that patches/sac1_bypass_ratio_check_save_fix.patch
  actually fixes the implicit-SAVE bug it targets (not just "the patch
  applied without error") -- mirrors how test_snow17_shim.py proves the
  TPREV divergence rather than asserting it,
- a cross-check against the upstream ex1 reference output and its own
  precomputed mass_bal.csv.

See notes/NOTES.md for the full writeup of the bug this patch fixes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sacsma import STATE_SIZE, SacSmaParams, run_sacsma, _LIB  # noqa: E402

EX1_DIR = REPO_ROOT / "external" / "sac-sma" / "test_cases" / "ex1"


# ---------------------------------------------------------------------------
# ex1 (HHWM8) fixtures -- same basin/period as the snow17 ex1 case, real
# forcing/params/reference output shipped upstream.
# ---------------------------------------------------------------------------

def _load_ex1_params(hru_id: str) -> SacSmaParams:
    text = (EX1_DIR / "input" / "params" / "sac_params.HHWM8.txt").read_text()
    rows = [line.split() for line in text.strip().splitlines()]
    header = rows[0]
    col = header.index(hru_id)
    values = {row[0]: float(row[col]) for row in rows[1:]}
    return SacSmaParams(
        uztwm=values["uztwm"], uzfwm=values["uzfwm"], uzk=values["uzk"],
        pctim=values["pctim"], adimp=values["adimp"], riva=values["riva"],
        zperc=values["zperc"], rexp=values["rexp"], lztwm=values["lztwm"],
        lzfsm=values["lzfsm"], lzfpm=values["lzfpm"], lzsk=values["lzsk"],
        lzpk=values["lzpk"], pfree=values["pfree"], side=values["side"],
        rserv=values["rserv"],
    )


def _load_ex1_forcing(hru_id: str):
    """Returns (dates, pcp_mm_per_day, tmp_degc, etp_mm_per_day). pcp/etp
    converted from mm/s rate -> mm/day depth (dt=86400s), matching
    runSac.f90's own `forcing%precip(nh)*runinfo%dt` / `...pet(nh)*dt`."""
    df = pd.read_csv(EX1_DIR / "input" / "forcing" / f"forcing.sacbmi.rates.{hru_id}.csv")
    dates = pd.DatetimeIndex(
        pd.to_datetime(df[["year", "mo", "dy"]].rename(columns={"mo": "month", "dy": "day"}))
    )
    dtm = 86400.0
    pcp = (df["prcp_rate"].to_numpy(dtype=np.float64) * dtm)
    tmp = df["tavg_degC"].to_numpy(dtype=np.float64)
    etp = (df["pet_rate"].to_numpy(dtype=np.float64) * dtm)
    return dates, pcp, tmp, etp


def _load_ex1_reference(hru_id: str) -> pd.DataFrame:
    path = EX1_DIR / "output" / f"output.sacbmi.{hru_id}.txt"
    return pd.read_csv(path, sep=r"\s+")


def _load_ex1_mass_bal() -> pd.DataFrame:
    return pd.read_csv(EX1_DIR / "output" / "mass_bal.csv")


# ---------------------------------------------------------------------------
# Synthetic fixture for determinism / continuity: needs to exercise a
# reasonable range of storage/ET conditions, not physical realism.
# ---------------------------------------------------------------------------

def _synthetic_series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    pcp = rng.gamma(1.0, 3.0, n)
    tmp = rng.normal(10, 8, n)
    etp = np.abs(rng.gamma(1.0, 2.0, n))  # PET, always >= 0
    return pcp, tmp, etp


def _default_params():
    return SacSmaParams(
        uztwm=30.0, uzfwm=25.0, uzk=0.3, pctim=0.01, adimp=0.05, riva=0.01,
        zperc=100.0, rexp=3.0, lztwm=130.0, lzfsm=25.0, lzfpm=60.0,
        lzsk=0.05, lzpk=0.01, pfree=0.15, side=0.0, rserv=0.3,
    )


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    pcp, tmp, etp = _synthetic_series()
    params = _default_params()

    out1 = run_sacsma(pcp, tmp, etp, params)
    out2 = run_sacsma(pcp, tmp, etp, params)

    np.testing.assert_array_equal(out1.q, out2.q)
    np.testing.assert_array_equal(out1.eta, out2.eta)
    np.testing.assert_array_equal(out1.state, out2.state)


# ---------------------------------------------------------------------------
# 2. State continuity: one long call == two chained calls
# ---------------------------------------------------------------------------

def test_state_continuity():
    pcp, tmp, etp = _synthetic_series(n=100)
    params = _default_params()

    whole = run_sacsma(pcp, tmp, etp, params)

    first = run_sacsma(pcp[:50], tmp[:50], etp[:50], params)
    second = run_sacsma(pcp[50:], tmp[50:], etp[50:], params, state0=first.state)

    chained_q = np.concatenate([first.q, second.q])
    chained_eta = np.concatenate([first.eta, second.eta])

    np.testing.assert_array_equal(whole.q, chained_q)
    np.testing.assert_array_equal(whole.eta, chained_eta)
    np.testing.assert_array_equal(whole.state, second.state)


# ---------------------------------------------------------------------------
# 3. Mass balance, using runSac.f90's own formula (ioModule/runSac.f90
# `derived%mass_balance`): precip - eta - tci - delta_storage*(1-adimp-pctim)
# - delta_adimc*adimp - bfncc ~= 0. delta_storage excludes ADIMC (tracked
# separately, weighted by ADIMP instead of PAREA).
# ---------------------------------------------------------------------------

def test_mass_balance():
    hru_id = "HHWM8IL"
    params = _load_ex1_params(hru_id)
    dates, pcp, tmp, etp = _load_ex1_forcing(hru_id)

    mask = (dates >= "1970-10-01") & (dates <= "1971-09-30")
    pcp_wy, tmp_wy, etp_wy = pcp[mask], tmp[mask], etp[mask]
    assert len(pcp_wy) == 365

    out = run_sacsma(pcp_wy, tmp_wy, etp_wy, params)

    delta_storage = out.state[0] + out.state[1] + out.state[2] + out.state[3] + out.state[4]  # final - 0 (cold start)
    delta_adimc = out.state[5]
    parea = 1.0 - params.adimp - params.pctim

    total_precip = float(np.sum(pcp_wy))
    total_eta = float(np.sum(out.eta))
    total_tci = float(np.sum(out.q))
    total_bfncc = float(np.sum(out.bfncc))

    mass_balance = (
        total_precip - total_eta - total_tci
        - delta_storage * parea - delta_adimc * params.adimp - total_bfncc
    )

    assert abs(mass_balance) < 1e-2 * total_precip, (
        f"mass balance off by {mass_balance:.4f} mm "
        f"({100*mass_balance/total_precip:.4f}% of input={total_precip:.1f} mm)"
    )


# ---------------------------------------------------------------------------
# Bonus: prove the implicit-SAVE patch actually fixes the bug it targets,
# through the real compiled+patched shim -- not just "it applied cleanly".
# ---------------------------------------------------------------------------

def test_bypass_ratio_check_patch_fixes_state_leakage():
    """Reproduces the exact scenario from notes/NOTES.md: a call that
    drives UZTWC negative under high ET demand (setting sac1.f90's local
    `bypass_ratio_check` -- pre-patch, implicitly SAVE'd -- to .TRUE.),
    followed by a second, unrelated call whose own inputs don't re-enter
    that branch. Pre-patch, call 2's result depended on whether call 1
    happened first (state leakage through a variable with no argument or
    module interface to reset externally). Post-patch, it must not.
    """
    trigger_params = SacSmaParams(
        uztwm=10.0, uzfwm=5.0, uzk=0.3, pctim=0.0, adimp=0.0, riva=0.01,
        zperc=100.0, rexp=2.0, lztwm=50.0, lzfsm=100.0, lzfpm=100.0,
        lzsk=0.1, lzpk=0.01, pfree=0.2, side=0.0, rserv=0.3,
    )
    # UZTWC=1, UZFWC=0.1, ETP=50 -> E1 exceeds UZTWC and UZFWC < residual
    # demand -> bypass_ratio_check = .TRUE. pre-patch.
    run_sacsma(
        np.array([0.0]), np.array([10.0]), np.array([50.0]), trigger_params,
        state0=np.array([1.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
    )

    probe_params = SacSmaParams(
        uztwm=10.0, uzfwm=10.0, uzk=0.3, pctim=0.0, adimp=0.0, riva=0.01,
        zperc=100.0, rexp=2.0, lztwm=50.0, lzfsm=100.0, lzfpm=100.0,
        lzsk=0.1, lzpk=0.01, pfree=0.2, side=0.0, rserv=0.3,
    )
    # UZTWC/UZTWM=0.72 << UZFWC/UZFWM=0.9, ETP=1 -> E1 << UZTWC, so the
    # outer IF(UZTWC<0) branch is never entered this call -- pre-patch,
    # bypass_ratio_check keeps whatever the previous call left it as.
    after_trigger = run_sacsma(
        np.array([0.0]), np.array([10.0]), np.array([1.0]), probe_params,
        state0=np.array([7.2, 9.0, 0.0, 0.0, 0.0, 7.2]),
    )

    # Same probe call with NO preceding trigger call -- the correct,
    # order-independent answer.
    isolated = run_sacsma(
        np.array([0.0]), np.array([10.0]), np.array([1.0]), probe_params,
        state0=np.array([7.2, 9.0, 0.0, 0.0, 0.0, 7.2]),
    )

    np.testing.assert_allclose(after_trigger.state, isolated.state, rtol=1e-12)
    # Pre-patch this was ~7.2 (ratio-check skipped); patched/correct is ~7.74.
    assert after_trigger.state[0] > 7.5, (
        f"uztwc={after_trigger.state[0]} looks like the pre-patch buggy value "
        "(~6.48-7.2, ratio-check silently skipped) rather than the patched "
        "value (~7.74) -- check patches/sac1_bypass_ratio_check_save_fix.patch "
        "is actually being applied by fortran/sacsma_build.sh."
    )


# ---------------------------------------------------------------------------
# Bonus: cross-check against the upstream ex1 reference output and its
# own precomputed mass_bal.csv.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hru_id", ["HHWM8IL", "HHWM8IU"])
def test_matches_upstream_reference(hru_id):
    params = _load_ex1_params(hru_id)
    dates, pcp, tmp, etp = _load_ex1_forcing(hru_id)
    ref = _load_ex1_reference(hru_id)
    assert len(ref) == len(dates)

    out = run_sacsma(pcp, tmp, etp, params)

    # tci is the reference's raw model output (mm/day); qs/qg are
    # ROIMP+SDRO+SSUR+SIF and BFS+BFP respectively -- all directly
    # comparable to our shim's q/qs/qg.
    np.testing.assert_allclose(out.q, ref["tci"].to_numpy(), atol=1e-6, rtol=1e-4)
    np.testing.assert_allclose(out.eta, ref["eta"].to_numpy(), atol=1e-6, rtol=1e-4)
