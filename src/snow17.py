"""ctypes wrapper around fortran/libsnow17shim.so.

Loops EXSNOW19 (NOAA-OWP snow17, vendored under external/snow17/,
Apache-2.0) over a daily time series for a single HRU, threading the
19-element carryover state (CS) and TPREV explicitly. See CLAUDE.md for
the full state contract and unit conventions.

Units: mm/day throughout. Deliberately skips the mm/s <-> mm/step
conversions the original driver does.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parent.parent / "fortran" / "libsnow17shim.so"

CS_SIZE = 19
ADC_SIZE = 11


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_LIB_PATH))

    f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

    lib.snow17_run.argtypes = [
        ctypes.c_int,  # n
        ctypes.c_int,  # idt
        ctypes.c_int,  # idts
        i32,  # iyr
        i32,  # imn
        i32,  # ida
        f32,  # pcp
        f32,  # tmp
        ctypes.c_float,  # alat
        ctypes.c_float,  # elev
        ctypes.c_float,  # scf
        ctypes.c_float,  # mfmax
        ctypes.c_float,  # mfmin
        ctypes.c_float,  # uadj
        ctypes.c_float,  # si
        ctypes.c_float,  # nmf
        ctypes.c_float,  # tipm
        ctypes.c_float,  # mbase
        ctypes.c_float,  # pxtemp
        ctypes.c_float,  # plwhc
        ctypes.c_float,  # daygm
        f32,  # adc11
        f32,  # cs_io, inout, length CS_SIZE
        ctypes.POINTER(ctypes.c_float),  # tprev_io, inout scalar
        f32,  # raim, out, length n
        f32,  # sneqv, out, length n
        f32,  # snowh, out, length n
    ]
    lib.snow17_run.restype = None
    return lib


_LIB = _load_lib()


@dataclass
class Snow17Params:
    alat: float
    elev: float
    scf: float
    mfmax: float
    mfmin: float
    uadj: float
    si: float
    nmf: float
    tipm: float
    mbase: float
    pxtemp: float
    plwhc: float
    daygm: float
    adc: np.ndarray  # shape (11,)


@dataclass
class Snow17Output:
    raim: np.ndarray   # mm/day, rain-plus-melt -> feeds SAC-SMA (or HBV, fallback-only per CLAUDE.md)
    sneqv: np.ndarray  # m, SWE
    snowh: np.ndarray  # m, snow depth
    cs: np.ndarray     # final carryover state, shape (CS_SIZE,)
    tprev: float       # final previous-timestep temperature


def run_snow17(
    dates,  # anything with .year, .month, .day arrays, e.g. pandas DatetimeIndex
    pcp: np.ndarray,
    tmp: np.ndarray,
    params: Snow17Params,
    cs0: np.ndarray | None = None,
    tprev0: float = 0.0,
    idt: int = 24,
    idts: int = 86400,
) -> Snow17Output:
    """Run EXSNOW19 over `dates`/`pcp`/`tmp` for one HRU.

    `cs0` defaults to all-zero (cold start), per EXSNOW19's own convention.
    """
    n = len(pcp)
    pcp = np.ascontiguousarray(pcp, dtype=np.float32)
    tmp = np.ascontiguousarray(tmp, dtype=np.float32)
    iyr = np.ascontiguousarray(np.asarray(dates.year), dtype=np.int32)
    imn = np.ascontiguousarray(np.asarray(dates.month), dtype=np.int32)
    ida = np.ascontiguousarray(np.asarray(dates.day), dtype=np.int32)

    adc11 = np.ascontiguousarray(params.adc, dtype=np.float32)
    assert adc11.shape == (ADC_SIZE,), f"adc must have {ADC_SIZE} elements"

    cs_io = np.zeros(CS_SIZE, dtype=np.float32) if cs0 is None else np.ascontiguousarray(cs0, dtype=np.float32).copy()
    assert cs_io.shape == (CS_SIZE,), f"cs must have {CS_SIZE} elements"
    tprev_io = ctypes.c_float(tprev0)

    raim = np.empty(n, dtype=np.float32)
    sneqv = np.empty(n, dtype=np.float32)
    snowh = np.empty(n, dtype=np.float32)

    _LIB.snow17_run(
        n, idt, idts,
        iyr, imn, ida,
        pcp, tmp,
        params.alat, params.elev, params.scf, params.mfmax, params.mfmin,
        params.uadj, params.si, params.nmf, params.tipm, params.mbase,
        params.pxtemp, params.plwhc, params.daygm,
        adc11,
        cs_io,
        ctypes.byref(tprev_io),
        raim, sneqv, snowh,
    )

    return Snow17Output(raim=raim, sneqv=sneqv, snowh=snowh, cs=cs_io, tprev=tprev_io.value)
