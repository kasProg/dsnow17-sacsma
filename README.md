# dsnow17-hbv

Differentiable coupling of Snow-17 (legacy Fortran) and HBV (PyTorch) via
[Tesseract](https://github.com/pasteurlabs/tesseract), built for the
Pasteur Labs Tesseract Hackathon 2026 (Track 03: Hybrid ML + mechanistic
models). See [CLAUDE.md](CLAUDE.md) for the full design writeup and status
log.

**Status:** Fortran shim built, three required tests passing (see below).
HBV, the Tesseract wrapper, and the parameter-prediction LSTM are not yet
implemented.

## What this is

An LSTM predicts hydrological model parameters from basin attributes.
Those parameters drive Snow-17 (a compiled Fortran snow model, wrapped in
Tesseract so gradients can cross into it) and HBV (written from scratch in
PyTorch). Gradients flow from a streamflow loss all the way back through
both models to the LSTM. See [CLAUDE.md](CLAUDE.md) for why Tesseract is
load-bearing here, not decorative.

## Reproduce

```bash
git submodule update --init --recursive   # vendors NOAA-OWP/snow17, pinned commit
pip install -r requirements.txt
make test                                  # builds fortran/libsnow17shim.so, runs pytest
```

`make test` builds the shim without `-fcheck=bounds` for speed. Use
`make build-checked` during development to catch out-of-bounds Fortran
array access (relevant to the `ADC(11)` vs `ADC(12)` issue noted in
[CLAUDE.md](CLAUDE.md) and [notes/NOTES.md](notes/NOTES.md)).

Tests extract `external/snow17/test_cases/ex1.tgz` on first run to get
reference forcing/parameters/output for validation — no manual step
needed.

## Layout

```
external/snow17/           git submodule, pinned commit (Apache-2.0)
fortran/snow17_shim.f90    bind(C) loop around EXSNOW19, state threaded explicitly
src/snow17.py               ctypes wrapper around the shim
tests/test_shim.py          determinism, state continuity, mass balance + reference cross-checks
notes/NOTES.md               upstream findings (TPREV, SCF, ADC) -- writeup material
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) — this repo links
directly against unmodified Snow-17 source (NOAA-OWP, Apache-2.0). HBV is
implemented from scratch here, not imported from any non-commercially
licensed codebase (see [CLAUDE.md](CLAUDE.md) hard constraints).
