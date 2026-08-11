"""ctypes wrapper around fortran/libsacsmashim.so.

Loops EXSAC (NOAA-OWP sac-sma, vendored under external/sac-sma/,
Apache-2.0, patched at build time -- see
patches/sac1_bypass_ratio_check_save_fix.patch and notes/NOTES.md) over a
daily time series for a single HRU, threading the 6-element carryover
state explicitly. See CLAUDE.md and notes/NOTES.md for the state contract
and unit conventions. Mirrors src/snow17.py's design.

Units: mm/day throughout. Deliberately skips the mm/s <-> mm/step
conversions the original driver does -- caller must pre-convert rate
forcing (precip, PET) to mm/day depths.

Real kind: EXSAC/SAC1 use explicit DOUBLE PRECISION -- this wrapper uses
float64 throughout, unlike src/snow17.py's float32 (Snow17 uses gfortran's
default 4-byte REAL). Do not conflate the two shims' array dtypes.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parent.parent / "fortran" / "libsacsmashim.so"

STATE_SIZE = 6  # UZTWC, UZFWC, LZTWC, LZFSC, LZFPC, ADIMC, in that order


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_LIB_PATH))

    f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")

    lib.sacsma_run.argtypes = [
        ctypes.c_int,  # n
        ctypes.c_double,  # dtm, timestep in seconds
        f64,  # pcp
        f64,  # tmp
        f64,  # etp
        ctypes.c_double,  # uztwm
        ctypes.c_double,  # uzfwm
        ctypes.c_double,  # uzk
        ctypes.c_double,  # pctim
        ctypes.c_double,  # adimp
        ctypes.c_double,  # riva
        ctypes.c_double,  # zperc
        ctypes.c_double,  # rexp
        ctypes.c_double,  # lztwm
        ctypes.c_double,  # lzfsm
        ctypes.c_double,  # lzfpm
        ctypes.c_double,  # lzsk
        ctypes.c_double,  # lzpk
        ctypes.c_double,  # pfree
        ctypes.c_double,  # side
        ctypes.c_double,  # rserv
        f64,  # state_io, inout, length STATE_SIZE
        f64,  # qs, out, length n
        f64,  # qg
        f64,  # q      -- TCI, "runoff" for the NSE loss
        f64,  # eta
        f64,  # roimp
        f64,  # sdro
        f64,  # ssur
        f64,  # sif
        f64,  # bfs
        f64,  # bfp
        f64,  # bfncc
    ]
    lib.sacsma_run.restype = None
    return lib


_LIB = _load_lib()


@dataclass
class SacSmaParams:
    uztwm: float
    uzfwm: float
    uzk: float
    pctim: float
    adimp: float
    riva: float
    zperc: float
    rexp: float
    lztwm: float
    lzfsm: float
    lzfpm: float
    lzsk: float
    lzpk: float
    pfree: float
    side: float
    rserv: float


@dataclass
class SacSmaOutput:
    qs: np.ndarray      # mm/day, surface flow (ROIMP+SDRO+SSUR+SIF)
    qg: np.ndarray      # mm/day, groundwater flow (BFS+BFP)
    q: np.ndarray        # mm/day, TCI -- total channel inflow, the coupling flux to the NSE loss
    eta: np.ndarray      # mm/day, actual evapotranspiration
    roimp: np.ndarray
    sdro: np.ndarray
    ssur: np.ndarray
    sif: np.ndarray
    bfs: np.ndarray
    bfp: np.ndarray
    bfncc: np.ndarray    # mm/day, baseflow non-channel component -- a real loss term, needed for mass balance
    state: np.ndarray    # final carryover state, shape (STATE_SIZE,)


def run_sacsma(
    pcp: np.ndarray,
    tmp: np.ndarray,
    etp: np.ndarray,
    params: SacSmaParams,
    state0: np.ndarray | None = None,
    dtm: float = 86400.0,
) -> SacSmaOutput:
    """Run EXSAC over `pcp`/`tmp`/`etp` for one HRU.

    `state0` defaults to all-zero (cold start). Unlike snow17, EXSAC has
    no documented "official" cold-start convention in the source itself,
    but zero storage in every zone is the physically sensible starting
    point and matches what the ex1 reference run effectively does with
    the basin at/near dry conditions in October (see notes/NOTES.md).
    """
    n = len(pcp)
    pcp = np.ascontiguousarray(pcp, dtype=np.float64)
    tmp = np.ascontiguousarray(tmp, dtype=np.float64)
    etp = np.ascontiguousarray(etp, dtype=np.float64)

    state_io = (
        np.zeros(STATE_SIZE, dtype=np.float64)
        if state0 is None
        else np.ascontiguousarray(state0, dtype=np.float64).copy()
    )
    assert state_io.shape == (STATE_SIZE,), f"state must have {STATE_SIZE} elements"

    qs = np.empty(n, dtype=np.float64)
    qg = np.empty(n, dtype=np.float64)
    q = np.empty(n, dtype=np.float64)
    eta = np.empty(n, dtype=np.float64)
    roimp = np.empty(n, dtype=np.float64)
    sdro = np.empty(n, dtype=np.float64)
    ssur = np.empty(n, dtype=np.float64)
    sif = np.empty(n, dtype=np.float64)
    bfs = np.empty(n, dtype=np.float64)
    bfp = np.empty(n, dtype=np.float64)
    bfncc = np.empty(n, dtype=np.float64)

    _LIB.sacsma_run(
        n, dtm,
        pcp, tmp, etp,
        params.uztwm, params.uzfwm, params.uzk, params.pctim, params.adimp,
        params.riva, params.zperc, params.rexp, params.lztwm, params.lzfsm,
        params.lzfpm, params.lzsk, params.lzpk, params.pfree, params.side, params.rserv,
        state_io,
        qs, qg, q, eta, roimp, sdro, ssur, sif, bfs, bfp, bfncc,
    )

    return SacSmaOutput(
        qs=qs, qg=qg, q=q, eta=eta, roimp=roimp, sdro=sdro, ssur=ssur, sif=sif,
        bfs=bfs, bfp=bfp, bfncc=bfncc, state=state_io,
    )
