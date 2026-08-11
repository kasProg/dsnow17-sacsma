# dsnow17-hbv

Differentiable NWS operational forecasting stack — Snow-17 feeding SAC-SMA,
both legacy Fortran — wrapped in two composed
[Tesseract](https://github.com/pasteurlabs/tesseract) containers, built for
the Pasteur Labs Tesseract Hackathon 2026 (Track 03: Hybrid ML + mechanistic
models). See [CLAUDE.md](CLAUDE.md) for the full design writeup and status
log. (Repo name predates the current plan and hasn't been settled yet — see
CLAUDE.md's repo layout note.)

**Status:** Both Fortran shims built and tested. Both Tesseract wrappers
(`tesseracts/snow17/`, `tesseracts/sacsma/`) built and gradient-checked
end-to-end, including through real `torch` graphs via `tesseract-torch`'s
`apply_tesseract()`. The cross-model gradient-coupling design
(`src/coupling.py`, "option 1.5" in CLAUDE.md) is built, validated against
autograd ground truth + an independent brute-force check on cheap
stand-ins first (`tests/test_coupling_toy.py`), then wired to the real
Tesseracts (`src/pipeline.py`). `tesseract build` can't run on this
development machine (no Docker access — see [Reproduce](#reproduce)) but
runs in CI on every push, building both containers from scratch and
smoke-testing `apply()` against the actual built images — see
[.github/workflows/ci.yml](.github/workflows/ci.yml).

**CLAUDE.md's Day 5-10 checkpoints are both met.** Day 5-6 (single
basin): chaining Snow17 -> SAC-SMA -> an NSE-style loss -> `.backward()`
on real HHWM8 data and optimizing from a perturbed initial guess drives
the loss from 0.020 to ~0.0003 within a couple of gradient steps
(`tests/test_pipeline_hhwm8.py`). Day 7-10 (parameter network,
multi-basin, held-out): `src/paramnet.py` predicts all 27 learnable
parameters (11 Snow-17 + 16 SAC-SMA) from each basin's 39 static CAMELS
attributes plus an LSTM-encoded 12-month climatology sequence (mean
monthly precip/temp/PET), trained end-to-end (`src/train.py`) across 35
snow-dominated CAMELS basins with 10 held out. Parameters stay static per
training rollout regardless of the LSTM -- a hard constraint from this
project's finite-difference gradient-cost design, see
[notes/logs.md](notes/logs.md). Real result: train NSE `-0.05 -> +0.62`,
held-out NSE `+0.23 -> +0.58` over 25 epochs (PET via Hargreaves-Samani,
see below) -- held-out basins tracked training basins closely throughout,
no overfitting observed at this scale. See
[notes/logs.md](notes/logs.md) for the full data pipeline (basin
selection, PET derivation, training-window coverage verification, the
MLP-vs-LSTM+Hargreaves comparison) and these results in context.

Native-PyTorch HBV is **not** the current plan — it's the documented Aug
20 fallback only, see CLAUDE.md.

**Found and fixed one real, live upstream bug along the way:** a Fortran
implicit-`SAVE` state-leakage bug in SAC-SMA's `sac1.f90` (not gated behind
any disabled flag, unlike the Snow-17 issues below) — full writeup with an
empirical before/after proof in [notes/NOTES.md](notes/NOTES.md), fixed via
a disclosed, minimal, build-time-only patch (see
[patches/](patches/sac1_bypass_ratio_check_save_fix.patch)); the vendored
submodule itself is never modified.

## What this is

An LSTM-plus-MLP parameter network predicts parameters for both Snow-17
and SAC-SMA from CAMELS basin attributes and climatology. Snow-17 produces
RAIM (rain-plus-melt), the same coupling flux NOAA runs operationally into
SAC-SMA. Both are compiled Fortran, each wrapped as its own Tesseract
(`apply` + finite-difference `vector_jacobian_product`), composed so a
streamflow NSE loss backpropagates through both containers to the
parameter network. See CLAUDE.md for why two Tesseracts (not one merged
container, not a manufactured boundary) and for the finite-difference cost
analysis behind the gradient-coupling design.

**Methodological lineage, not code:** the LSTM-encoder-plus-physical-model
pattern follows the general differentiable-parameter-learning approach
described in Feng et al. (2022, WRR) and the broader MHPI dPL line of
work; architecture patterns were informed by studying
[NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
(Kratzert et al., BSD-3-Clause). MHPI's own `dPLHBVrelease` /
`generic_deltamodel` / `hydrodl2` are PSU Non-Commercial licensed and
incompatible with this project's Apache-2.0 submission (CLAUDE.md's
existing hard constraint) — their source was deliberately not read while
building this, cited as prior work only. No code from either source is
copied; this is an original implementation of a well-documented, published
modeling pattern.

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

**CAMELS data (multi-basin training):** not fetched by `make test` —
it's a separate, ~3.4GB one-time download, deliberately kept out of the
fast local test loop.

```bash
data/download_camels.sh                      # downloads attribute files + forcing/streamflow archive
.venv/bin/python data/select_basins.py       # -> data/camels/selected_basins.csv
.venv/bin/python data/build_attributes.py    # -> data/camels/basin_attributes.npz
.venv/bin/python data/build_pet.py           # -> data/camels/pet/{gauge_id}_pet.csv (Hargreaves)
.venv/bin/python data/build_climatology.py   # -> data/camels/basin_climatology.npz
.venv/bin/python src/train.py                # trains, prints per-epoch train/held-out NSE
```

`tests/test_train.py` skips gracefully (like the Docker container test
below) when this data isn't present.

**Docker note:** this development machine's account can't use Docker (not
in the `docker` group, no passwordless `sudo`, no `subuid`/`subgid`
entries for rootless Docker either — see notes/logs.md for the full
investigation). Day-to-day `apply()`/`vector_jacobian_product()`
development/testing runs through
`tesseract_core.Tesseract.from_tesseract_api()`, which imports
`tesseract_api.py` directly and needs no container — see
`tests/test_gradients.py`. Actual containerization (`tesseract build`,
needed for the real submission artifact) runs in CI instead —
GitHub-hosted runners have Docker preinstalled — see
[.github/workflows/ci.yml](.github/workflows/ci.yml) and
[tests/test_tesseract_build.py](tests/test_tesseract_build.py) (a
smoke test against the actual built images, skipped locally, run in CI
after each build).

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
src/coupling.py                   cross-model gradient orchestration (option 1.5, see CLAUDE.md)
src/pipeline.py                   wires the real Tesseracts into coupling.py, reused across basins
src/paramnet.py                   LSTM (12-mo climatology) + static attrs -> 27 bounded parameters
src/train.py                      multi-basin training loop + held-out evaluation
data/download_camels.sh           one-time ~3.4GB CAMELS download (not run by make test)
data/select_basins.py             picks snow-dominated basins by frac_snow, train/heldout split
data/build_attributes.py          builds the 39-feature normalized static attribute matrix
data/build_pet.py                 precomputes + caches Hargreaves PET per basin (data/camels/pet/)
data/build_climatology.py         builds the 12-month climatology sequence per basin (LSTM input)
data/camels_loader.py             per-basin forcing/streamflow loader + Hargreaves PET (cached)
tesseracts/snow17/                 Tesseract wrapper: apply() + finite-difference vector_jacobian_product()
tesseracts/sacsma/                 same, for SAC-SMA
tests/test_snow17_shim.py          determinism, state continuity, mass balance + reference cross-checks
tests/test_sacsma_shim.py          same, for the SAC-SMA shim, + a dedicated implicit-SAVE-patch proof
tests/test_gradients.py            VJP vs. manual perturbation + torch autograd integration, per Tesseract
tests/test_coupling_toy.py         validates the coupling mechanism against cheap stand-ins first
tests/test_pipeline_hhwm8.py       real HHWM8 chain: Snow17 -> SAC-SMA -> NSE -> backward, loss decreases
tests/test_paramnet.py             ParamNet output shapes/dtypes/bounds/gradient-flow
tests/test_train.py                multi-basin training loop regression check (skips w/o CAMELS data)
notes/NOTES.md                     upstream findings (TPREV, SCF, ADC, bypass_ratio_check, ...) -- writeup material
notes/logs.md                      rationale log for our own code/design decisions, kept live
```

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
