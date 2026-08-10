# dsnow17-hbv

Differentiable coupling of Snow-17 (legacy Fortran) and HBV (PyTorch) via
[Tesseract](https://github.com/pasteurlabs/tesseract), built for the
Pasteur Labs Tesseract Hackathon 2026 (Track 03: Hybrid ML + mechanistic
models). See [CLAUDE.md](CLAUDE.md) for the full design writeup and status
log.

**Status:** Fortran shim built and tested. Snow-17 Tesseract wrapper built
(`apply` + finite-difference `vector_jacobian_product`), gradient-checked
against manual perturbation, and verified through `tesseract-torch`'s
`apply_tesseract()` end to end (`.backward()` produces gradients matching
the VJP). HBV and the parameter-prediction LSTM are not yet implemented.
`tesseract build` (containerizing for submission) is blocked on this
development machine — see [Reproduce](#reproduce) and
[notes/logs.md](notes/logs.md).

## What this is

An LSTM predicts hydrological model parameters from basin attributes.
Those parameters drive Snow-17 (a compiled Fortran snow model, wrapped in
Tesseract so gradients can cross into it) and HBV (written from scratch in
PyTorch). Gradients flow from a streamflow loss all the way back through
both models to the LSTM. See [CLAUDE.md](CLAUDE.md) for why Tesseract is
load-bearing here, not decorative.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and `gfortran`.

```bash
git submodule update --init --recursive   # vendors NOAA-OWP/snow17, pinned commit
make test                                  # creates .venv via uv, builds libsnow17shim.so, runs pytest
```

`make test` builds the shim without `-fcheck=bounds` for speed. Use
`make build-checked` during development to catch out-of-bounds Fortran
array access (relevant to the `ADC(11)` vs `ADC(12)` issue noted in
[CLAUDE.md](CLAUDE.md) and [notes/NOTES.md](notes/NOTES.md)).

Tests extract `external/snow17/test_cases/ex1.tgz` on first run to get
reference forcing/parameters/output for validation — no manual step
needed.

**Docker note:** `tesseract build` (packaging the Tesseract as an OCI
container, needed for the final submission) requires Docker, which this
development machine's account doesn't have permission to use. All Tesseract
development/testing so far runs through
`tesseract_core.Tesseract.from_tesseract_api()`, which imports
`tesseract_api.py` directly and needs no container — see
`tests/test_tesseract_api.py`. Containerizing is unresolved; see
[notes/logs.md](notes/logs.md).

## Layout

```
external/snow17/                git submodule, pinned commit (Apache-2.0)
fortran/snow17_shim.f90         bind(C) loop around EXSNOW19, state threaded explicitly
src/snow17.py                    ctypes wrapper around the shim
tesseracts/snow17/               Tesseract wrapper: apply() + finite-difference vector_jacobian_product()
tests/test_shim.py               determinism, state continuity, mass balance + reference cross-checks
tests/test_tesseract_api.py      apply()/VJP correctness, VJP vs. manual perturbation, torch autograd integration
notes/NOTES.md                    upstream findings (TPREV, SCF, ADC) -- writeup material
notes/logs.md                     rationale log for our own code/design decisions, kept live
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) — this repo links
directly against unmodified Snow-17 source (NOAA-OWP, Apache-2.0). HBV is
implemented from scratch here, not imported from any non-commercially
licensed codebase (see [CLAUDE.md](CLAUDE.md) hard constraints).
