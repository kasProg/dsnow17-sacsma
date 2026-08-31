"""Wires the real Snow17 and SAC-SMA Tesseracts into
src/coupling.py's CoupledTwoStageFunction for a single HRU.

This is the real stage_a/stage_b for the coupling design in
src/coupling.py -- tests/test_coupling_toy.py validated the mechanism
against cheap stand-ins first; this module is the drop-in swap to the
actual Fortran-backed Tesseracts.
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
    """Loads both Tesseracts ONCE (local, no Docker -- see notes/logs.md).
    `.run(theta_A, theta_B, snow17_forcing, sacsma_forcing)` takes forcing
    per call, not per instance -- so one CoupledNWSStack is reused across
    every basin in multi-basin training instead of reloading the
    Tesseract clients (an expensive, basin-independent step) 35+ times
    per epoch. Originally forcing was fixed at construction (fine for
    the single-basin proof in tests/test_pipeline_hhwm8.py); refactored
    once multi-basin training made that assumption wrong -- see
    notes/logs.md.
    """

    def __init__(
        self,
        fd_a: FDConfig | None = None,
        fd_b: FDConfig | None = None,
        use_sacsma_vjp: bool = True,
    ) -> None:
        self._snow17 = Tesseract.from_tesseract_api(str(SNOW17_TESSERACT_DIR / "tesseract_api.py"))
        self._sacsma = Tesseract.from_tesseract_api(str(SACSMA_TESSERACT_DIR / "tesseract_api.py"))
        self.fd_a = fd_a or FDConfig(rel=1e-3, floor=1e-4, central=True)
        # fd_b only matters when use_sacsma_vjp=False -- see .run()'s and
        # coupling.py's docstrings for why theta_B's gradient otherwise
        # routes through SAC-SMA's own Tesseract vector_jacobian_product()
        # endpoint instead of this module's FD sweep.
        self.fd_b = fd_b or FDConfig(rel=1e-3, floor=1e-4, central=True)
        self.use_sacsma_vjp = use_sacsma_vjp

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

    def _make_stage_sacsma(self, sacsma_forcing: SacSmaForcing):
        """A fresh closure per .run() call, capturing THAT call's
        forcing -- not instance state -- so concurrent/interleaved calls
        for different basins can never cross-contaminate each other."""

        def stage_sacsma(theta_b_np: np.ndarray, raim: np.ndarray) -> np.ndarray:
            f = sacsma_forcing
            inputs = dict(
                dtm=f.dtm, pcp=np.asarray(raim, dtype=np.float64),
                tmp=f.tmp.astype(np.float64), etp=f.etp.astype(np.float64),
                state0=f.state0.astype(np.float64),
            )
            for name, value in zip(SACSMA_PARAMS, theta_b_np):
                inputs[name] = float(value)
            out = self._sacsma.apply(inputs)
            return np.asarray(out["q"], dtype=np.float64)

        return stage_sacsma

    def _make_vjp_sacsma(self, sacsma_forcing: SacSmaForcing):
        """Mirrors _make_stage_sacsma's per-call closure, but wraps SAC-
        SMA's own Tesseract vector_jacobian_product() endpoint instead of
        apply() -- used for theta_B's gradient block (coupling.py). theta_B
        never needs stage_a rerun (RAIM is held fixed), so unlike theta_A
        there's no FD-cost reason to bypass the real endpoint here; see
        tesseracts/sacsma/tesseract_api.py's module docstring, which this
        closure is the first real (non-test) caller of."""

        def vjp_sacsma(theta_b_np: np.ndarray, raim: np.ndarray, grad_output_np: np.ndarray) -> np.ndarray:
            f = sacsma_forcing
            inputs = dict(
                dtm=f.dtm, pcp=np.asarray(raim, dtype=np.float64),
                tmp=f.tmp.astype(np.float64), etp=f.etp.astype(np.float64),
                state0=f.state0.astype(np.float64),
            )
            for name, value in zip(SACSMA_PARAMS, theta_b_np):
                inputs[name] = float(value)
            vjp = self._sacsma.vector_jacobian_product(
                inputs,
                vjp_inputs=list(SACSMA_PARAMS),
                vjp_outputs=["q"],
                cotangent_vector={"q": grad_output_np},
            )
            return np.array([vjp[name] for name in SACSMA_PARAMS], dtype=np.float64)

        return vjp_sacsma

    def run(
        self,
        theta_A: torch.Tensor,
        theta_B: torch.Tensor,
        snow17_forcing: Snow17Forcing,
        sacsma_forcing: SacSmaForcing,
    ) -> torch.Tensor:
        """theta_A: 11 Snow17 params, in SNOW17_PARAMS order.
        theta_B: 16 SAC-SMA params, in SACSMA_PARAMS order.
        Returns: runoff (TCI), float64 torch.Tensor, differentiable
        w.r.t. both theta_A and theta_B. theta_A's gradient is always
        coupling.py's hand-rolled FD sweep through both stages (see that
        module's docstring for why); theta_B's gradient goes through
        SAC-SMA's own Tesseract vector_jacobian_product() endpoint unless
        use_sacsma_vjp=False, in which case it falls back to the same
        hand-rolled sweep, RAIM-fixed."""
        vjp_b = self._make_vjp_sacsma(sacsma_forcing) if self.use_sacsma_vjp else None
        return CoupledTwoStageFunction.apply(
            theta_A, theta_B, snow17_forcing,
            self._stage_snow17, self._make_stage_sacsma(sacsma_forcing),
            self.fd_a, self.fd_b, vjp_b,
        )
