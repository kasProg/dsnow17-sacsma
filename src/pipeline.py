"""Wires the real Snow17 and SAC-SMA Tesseracts into
src/coupling.py's CoupledTwoStageFunction for a single HRU.

This is the "real" stage_a/stage_b CLAUDE.md's coupling design question
was resolved in anticipation of -- tests/test_coupling_toy.py validated
the mechanism against cheap stand-ins first; this module is the drop-in
swap to the actual Fortran-backed Tesseracts, per notes/logs.md's
"what's still open going into the real wrapper integration" note.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tesseract_core import Tesseract

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from coupling import CoupledTwoStageFunction, FDConfig  # noqa: E402

SNOW17_TESSERACT_DIR = _REPO_ROOT / "tesseracts" / "snow17"
SACSMA_TESSERACT_DIR = _REPO_ROOT / "tesseracts" / "sacsma"

# Order theta_A / theta_B vectors are expected in -- matches each
# Tesseract's own DIFFERENTIABLE_PARAMS declaration order.
SNOW17_PARAMS = (
    "scf", "mfmax", "mfmin", "uadj", "si", "nmf", "tipm", "mbase", "pxtemp", "plwhc", "daygm",
)
SACSMA_PARAMS = (
    "uztwm", "uzfwm", "uzk", "pctim", "adimp", "riva", "zperc", "rexp",
    "lztwm", "lzfsm", "lzfpm", "lzsk", "lzpk", "pfree", "side", "rserv",
)


@dataclass
class Snow17Forcing:
    """Everything Snow17's Tesseract needs besides the 11 learnable
    parameters -- static for a given basin/period, passed through
    CoupledTwoStageFunction's `forcings` argument unchanged."""

    idt: int
    idts: int
    iyr: np.ndarray
    imn: np.ndarray
    ida: np.ndarray
    pcp: np.ndarray
    tmp: np.ndarray
    alat: float
    elev: float
    adc: np.ndarray  # fixed, not learnable -- see tesseracts/snow17's docstring
    cs0: np.ndarray
    tprev0: float


@dataclass
class SacSmaForcing:
    """Everything SAC-SMA's Tesseract needs besides the 16 learnable
    parameters and RAIM (which comes from Snow17's stage). Closed over
    by CoupledNWSStack's stage_b rather than threaded through
    CoupledTwoStageFunction's signature -- that Function only has a
    single `forcings` slot, for stage_a; see notes/logs.md."""

    dtm: float
    tmp: np.ndarray
    etp: np.ndarray
    state0: np.ndarray


class CoupledNWSStack:
    """Loads both Tesseracts once (local, no Docker -- see
    notes/logs.md); `.run(theta_A, theta_B)` chains them via
    CoupledTwoStageFunction for the forcing fixed at construction.
    """

    def __init__(
        self,
        snow17_forcing: Snow17Forcing,
        sacsma_forcing: SacSmaForcing,
        fd_a: FDConfig | None = None,
        fd_b: FDConfig | None = None,
    ) -> None:
        self._snow17 = Tesseract.from_tesseract_api(str(SNOW17_TESSERACT_DIR / "tesseract_api.py"))
        self._sacsma = Tesseract.from_tesseract_api(str(SACSMA_TESSERACT_DIR / "tesseract_api.py"))
        self._snow17_forcing = snow17_forcing
        self._sacsma_forcing = sacsma_forcing
        self.fd_a = fd_a or FDConfig(rel=1e-3, floor=1e-4, central=True)
        self.fd_b = fd_b or FDConfig(rel=1e-3, floor=1e-4, central=True)

    def _stage_snow17(self, theta_a_np: np.ndarray, forcing: Snow17Forcing) -> np.ndarray:
        f = forcing
        inputs = dict(
            idt=f.idt, idts=f.idts, iyr=f.iyr, imn=f.imn, ida=f.ida,
            pcp=f.pcp.astype(np.float32), tmp=f.tmp.astype(np.float32),
            alat=f.alat, elev=f.elev,
            adc=f.adc.astype(np.float32), cs0=f.cs0.astype(np.float32), tprev0=f.tprev0,
        )
        for name, value in zip(SNOW17_PARAMS, theta_a_np):
            inputs[name] = float(value)
        out = self._snow17.apply(inputs)
        return np.asarray(out["raim"], dtype=np.float64)

    def _stage_sacsma(self, theta_b_np: np.ndarray, raim: np.ndarray) -> np.ndarray:
        f = self._sacsma_forcing
        inputs = dict(
            dtm=f.dtm, pcp=np.asarray(raim, dtype=np.float64),
            tmp=f.tmp.astype(np.float64), etp=f.etp.astype(np.float64),
            state0=f.state0.astype(np.float64),
        )
        for name, value in zip(SACSMA_PARAMS, theta_b_np):
            inputs[name] = float(value)
        out = self._sacsma.apply(inputs)
        return np.asarray(out["q"], dtype=np.float64)

    def run(self, theta_A: torch.Tensor, theta_B: torch.Tensor) -> torch.Tensor:
        """theta_A: 11 Snow17 params, in SNOW17_PARAMS order.
        theta_B: 16 SAC-SMA params, in SACSMA_PARAMS order.
        Returns: runoff (TCI), float64 torch.Tensor, differentiable
        w.r.t. both theta_A and theta_B."""
        return CoupledTwoStageFunction.apply(
            theta_A, theta_B, self._snow17_forcing,
            self._stage_snow17, self._stage_sacsma,
            self.fd_a, self.fd_b,
        )
