# dsnow17-hbv

Differentiable NWS operational forecasting stack — Snow-17 feeding SAC-SMA,
both legacy Fortran — wrapped in two composed
[Tesseract](https://github.com/pasteurlabs/tesseract) containers, built for
the Pasteur Labs Tesseract Hackathon 2026 (Track 03: Hybrid ML + mechanistic
models). See [CLAUDE.md](CLAUDE.md) for the full design writeup and status
log. (Repo name predates the current plan and hasn't been settled yet — see
CLAUDE.md's repo layout note.)

**Status:** Both Fortran shims built and tested — Snow-17
(`fortran/snow17_shim.f90`) and SAC-SMA (`fortran/sacsma_shim.f90`).
Snow-17's Tesseract wrapper is built and gradient-checked end-to-end,
including through a real `torch` graph via `tesseract-torch`'s
`apply_tesseract()`. SAC-SMA's Tesseract wrapper and the cross-model
gradient-coupling logic (`src/coupling.py`, "option 1.5" in CLAUDE.md) are
not yet built. Native-PyTorch HBV is **not** the current plan — it's the
documented Aug 20 fallback only, see CLAUDE.md. `tesseract build`
(containerizing for submission) is blocked on this development machine —
see [Reproduce](#reproduce) and [notes/logs.md](notes/logs.md).

**Found and fixed one real, live upstream bug along the way:** a Fortran
implicit-`SAVE` state-leakage bug in SAC-SMA's `sac1.f90` (not gated behind
any disabled flag, unlike the Snow-17 issues below) — full writeup with an
empirical before/after proof in [notes/NOTES.md](notes/NOTES.md), fixed via
a disclosed, minimal, build-time-only patch (see
[patches/](patches/sac1_bypass_ratio_check_save_fix.patch)); the vendored
submodule itself is never modified.

## What this is

An LSTM/MLP predicts parameters for both Snow-17 and SAC-SMA from CAMELS
basin attributes. Snow-17 produces RAIM (rain-plus-melt), the same coupling
flux NOAA runs operationally into SAC-SMA. Both are compiled Fortran,
each wrapped as its own Tesseract (`apply` + finite-difference
`vector_jacobian_product`), composed so a streamflow NSE loss backpropagates
through both containers to the parameter network. See CLAUDE.md for why two
Tesseracts (not one merged container, not a manufactured boundary) and for
the finite-difference cost analysis behind the gradient-coupling design.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and `gfortran`.

```bash
git submodule update --init --recursive   # vendors NOAA-OWP/snow17 + sac-sma, pinned commits
make test                                  # creates .venv via uv, builds shims, runs pytest
```

`make test` builds the shims without `-fcheck=bounds` for speed. Use
`make build-checked` during development to catch out-of-bounds Fortran
array access (relevant to the `ADC(11)` vs `ADC(12)` issue noted in
[CLAUDE.md](CLAUDE.md) and [notes/NOTES.md](notes/NOTES.md)).

Tests extract `external/snow17/test_cases/ex1.tgz` on first run to get
reference forcing/parameters/output for validation — no manual step needed.
SAC-SMA's ex1 reference (same HHWM8 basin/period) ships unpacked already.

**Docker note:** `tesseract build` (packaging a Tesseract as an OCI
container, needed for the final submission) requires Docker, which this
development machine's account doesn't have permission to use. All Tesseract
development/testing so far runs through
`tesseract_core.Tesseract.from_tesseract_api()`, which imports
`tesseract_api.py` directly and needs no container — see
`tests/test_gradients.py`. Containerizing is unresolved; see
[notes/logs.md](notes/logs.md).

## Layout

```
external/snow17/                 git submodule, pinned commit (Apache-2.0)
external/sac-sma/                 git submodule, pinned commit (Apache-2.0)
patches/                          disclosed, minimal, build-time-only patch(es) to vendored source
fortran/snow17_shim.f90           bind(C) loop around EXSNOW19, state threaded explicitly
fortran/sacsma_shim.f90           same pattern for EXSAC (SAC-SMA)
fortran/sacsma_build.sh           stages + patches a build-time copy of sac1.f90, then compiles
src/snow17.py                     ctypes wrapper around the snow17 shim (float32)
src/sacsma.py                     ctypes wrapper around the sacsma shim (float64)
src/coupling.py                   cross-model gradient orchestration (option 1.5, see CLAUDE.md) -- not yet built
tesseracts/snow17/                 Tesseract wrapper: apply() + finite-difference vector_jacobian_product()
tesseracts/sacsma/                 same, for SAC-SMA -- not yet built
tests/test_snow17_shim.py          determinism, state continuity, mass balance + reference cross-checks
tests/test_sacsma_shim.py          same, for the SAC-SMA shim, + a dedicated implicit-SAVE-patch proof
tests/test_gradients.py            VJP vs. manual perturbation + torch autograd integration, per Tesseract
notes/NOTES.md                     upstream findings (TPREV, SCF, ADC, bypass_ratio_check, ...) -- writeup material
notes/logs.md                      rationale log for our own code/design decisions, kept live
```

Items not yet built (`src/coupling.py`, `tesseracts/sacsma/`) are listed
for the target layout; see CLAUDE.md's timeline for sequencing.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Snow-17 is vendored
and linked against **unmodified**. SAC-SMA is vendored unmodified as a
pinned submodule; one disclosed, minimal patch is applied to a **build-time
copy only** to fix a confirmed upstream defect (see
[notes/NOTES.md](notes/NOTES.md) and
[patches/](patches/sac1_bypass_ratio_check_save_fix.patch)) —
`external/sac-sma` itself is never modified. "Original work" applies to
this submission, not its dependency tree: the shims, patches, Tesseract
wrappers, gradient endpoints, and training pipeline are original work
written during the hackathon period (Aug 3-31, 2026).
