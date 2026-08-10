# Project: Differentiable Snow17 + HBV via Tesseract

## What this is

A submission for the **Pasteur Labs Tesseract Hackathon 2026** (Aug 3–31, 2026;
submissions close Aug 31 11:59 PM AoE). Track 03: Hybrid ML + mechanistic models.

**Goal:** Train a neural network to predict hydrological model parameters from
basin attributes, where gradients must flow backwards through a legacy Fortran
snow model. Tesseract is the mechanism that lets PyTorch's autograd cross the
Fortran boundary.

```
basin attributes
      |
      v
   [ LSTM ]  predicts parameters for both models
      |
  +---+------------------+
  v                      v
[Snow17]  --rain+melt-> [HBV soil + routing] --> streamflow
 Fortran                 PyTorch                    |
 (Tesseract)                                        v
                                        NSE loss vs. CAMELS observed
                                                    |
   <------- gradients flow all the way back --------+
```

**Why Tesseract is load-bearing (this is the writeup's core argument):**
PyTorch's `.backward()` walks a recorded graph. Snow17 is a compiled Fortran
binary — nothing recorded, backprop stops, the LSTM never learns snow parameters.
Tesseract wraps Snow17 in a container exposing `apply` and
`vector_jacobian_product`, and `tesseract-torch`'s `apply_tesseract()` splices
it into the autograd graph as a normal differentiable layer.

Secondary framing worth using: **BMI (the NextGen interoperability standard)
lets Snow17 and dHBV exchange fluxes, but gradients cannot cross a BMI coupling.
Tesseract gives interoperability *and* differentiability.** This is a
differentiable BMI-style coupling.

---

## Hard constraints

### License — non-negotiable

- Submission must be **Apache-2.0**.
- Snow17 (NOAA-OWP) is Apache-2.0. Fine to depend on. Their README asks that
  adaptations credit the source repo — a NOTICE file is required.
- **Do NOT use MHPI's `generic_deltamodel` (δMG) or `hydrodl2`.** Both are under
  a PSU **Non-Commercial** license, incompatible with an Apache-2.0 submission
  and with the prize's research-collaboration component (Pasteur Labs is a
  company). HBV must be **written from scratch** in PyTorch (~200 lines;
  formulation per Feng et al. 2022, WRR). Cite δMG/dHBV as methodological
  lineage, do not import them.
- NOAA-OWP's CONTRIBUTING says all contributions to *their* repo are released to
  the public domain. **Never paste project code into a snow17 issue or PR.**

### Judging criteria (in their stated order)

1. Composition across a real boundary
2. Gradients doing real work (visible objective improving)
3. The why-Tesseract case — load-bearing, not a costume
4. A real application; undemoed domains score higher (**hydrology/climate is
   absent from their demo gallery — this is a structural advantage**)
5. Execution and technical depth
6. Reproducibility and communication

Explicit guidance from the organizers: *"We'd rather see a modest problem where
Tesseract is clearly load-bearing than an ambitious one where it's a costume."*

Deliverables: public GitHub repo, reproducible README, 2–4 page writeup naming
the track, optional ≤5 min demo video. Repo is currently **private** — flip to
public by **Aug 29**.

---

## Snow17: everything established so far

Upstream: `https://github.com/NOAA-OWP/snow17` (Apache-2.0, beta status).

### Use `EXSNOW19` directly — not the standalone driver, not BMI

The core physics is 13 plain FORTRAN-77 files in `src/snow19/`:

```
exsnow19.f  PACK19.f  SNOWPACK.f  SNEW.f     SNOWT.f   SNDEPTH.f  adjc19.f
aeco19.f    aesc19.f  melt19.f    rout19.f   updt19.f  zero19.f
```

`EXSNOW19` is the single-timestep, single-HRU entry point with everything passed
as explicit arguments. No derived types, no namelist, no file I/O, no
allocatables. **Build against these 13 files only** — do not link `src/share/`,
`src/bmi/`, or `src/driver/`.

### Signature

```fortran
SUBROUTINE EXSNOW19(IDTS,IDT,IDA,IMN,IYR,
     &    PCP,TMP,RAIM,SNEQV,SNOW,SNOWH,          ! forcing in / diagnostics out
     &    ALAT,SCF,MFMAX1,MFMIN1,UADJ1,SI,NMF1,TIPM1,
     &    MBASE,PXTEMP,PLWHC,DAYGM,ELEV1,PA,ADC,  ! parameters
     &    CS,TPREV)                               ! carryover state (in/out)
```

- `RAIM` = rain-plus-melt (mm per timestep). **This is the flux that feeds HBV.**
- `SNEQV` = SWE in metres (`TWE/1000.`) — diagnostic, useful for validation plots.
- `SNOWH` = snow depth in metres (`SNDPT/100.`) — diagnostic.

### Learnable parameters (13 scalars + 11-point curve)

`SCF, MFMAX1, MFMIN1, UADJ1, SI, NMF1, TIPM1, MBASE, PXTEMP, PLWHC, DAYGM`
plus `ADC(11)` (areal depletion curve). `ALAT`, `ELEV1`, `PA` are basin
attributes, not learnable.

### State contract — complete and explicit

```
CS(1..10)          = WE, NEGHS, LIQW, TINDEX, ACCMAX, SB, SBAESC, SBWS, STORGE, AEADJ
CS(11..10+NEXLAG)  = EXLAG(1..NEXLAG)
CS(11+NEXLAG)      = SNDPT
CS(12+NEXLAG)      = SNTMP
TPREV              = previous air temperature
```

`NEXLAG = 5/IDT + 2` (integer division). With daily forcing `IDT=24`, so
`NEXLAG = 2` and `CS` uses indices 1–14 of the 19 available.

`EXSNOW19` unpacks `CS` into COMMON block `/SNCO19/` at entry (lines ~89–106) and
repacks it at exit (lines ~168–184). **`CS(19)` + `TPREV` is the complete state.**
Thread them explicitly and the function is deterministic.

`/SNUP19/` correction factors (`MFC`, `UADJC`, `SFALLX`, `WINDC`) are set to 1.0
by `EXSNOW19` itself each call, along with `IUPWE=0`, `IUPSC=0`, `LMFV=0`,
`LAEC=0`. No external initialization needed. Cold start = pass `CS = 0.0`,
`TPREV = 0.0`.

### Units and calling convention

From `src/share/runSnow17.f90` lines ~158–170:

- `IDTS = dt` in **seconds** (86400 for daily)
- `IDT = dt/3600` in **hours** (24 for daily)
- `PCP` must be **depth per timestep in mm** (driver converts from mm/s by
  multiplying by `dt`)
- `RAIM` comes back as **depth in mm** (driver converts to mm/s by dividing)
- **Recommendation: work in mm/day throughout and skip both conversions.**
  Be explicit about units in the Tesseract schema.
- `ELEV1` in **metres** (`EXSNOW19` does `ELEV=ELEV1*0.01` internally)

**Melt factors are internally rescaled:** `MFMAX=(MFMAX1*IDT)/6.0`. With
`IDT=24` that's ×4. So learned `MFMAX1` is in **mm/°C/6hr**, not per day. Get
this right in the recovered-parameter figure or it's meaningless.

Surface pressure (from `constants.f90`, Anderson 2006), computed externally and
passed in as `PA`:

```fortran
pa = 33.86 * (29.9 - 0.335*(elev/100.0) + 0.00022*((elev/100.0)**2.4))
```

### KNOWN BUG — ADC dimension mismatch

- `exsnow19.f`: `REAL ADC(11)`
- `PACK19.f`: `DIMENSION ADC(12)`
- `runSnow17.f90` passes `parameters%adc(:,nh)` where `adc` is allocated `(11, n_hrus)`

`PACK19` reads one element past the end, every timestep, silently.

**Workaround:** pass a 12-element array from the shim, `adc12(1:11) = adc11`,
`adc12(12) = 1.0`. Document the choice. Not yet reported upstream — needs
confirmation on a second toolchain and a look at whether `PACK19` actually reads
element 12 before filing.

### Other upstream findings (macOS/arm64, gfortran 16.1.0 — may be compiler artifacts)

1. `test_cases/ex1` namelist supplies `start_datehr`/`end_datehr` unquoted, but
   they are declared `CHARACTER(len=10)` → "Missing quote while reading item 7".
   Fix: quote them.
2. After fixing (1), `read_snow17_parameters` fails with an invalid `adc` array
   descriptor despite `initParams` allocating it. **Unresolved.** Recheck on
   Linux/older gfortran.
3. `ioModule.f90` has commented-out transposed `adc` indexing — evidence of an
   incomplete refactor.
4. README documents build target `snow17_bmi`; actual target is `snow17bmi`.
   **Reported + PR filed.**
5. Open upstream PR #47 / issue #44 fix a *different* initialization bug class
   (implicit-SAVE from initialization-at-declaration) in `dateTimeUtilsModule.f90`
   and `ioModule.f90`. Not in our dependency path, but the bug class is a warning:
   watch for `DATA`-initialized variables in `snow19/` that are also assigned
   during execution.

**None of bugs 1–3 affect our build**, since we only link `src/snow19/`.

---

## Immediate task: the Fortran shim

Write `fortran/snow17_shim.f90`. Loop `EXSNOW19` over a time series with state
threaded explicitly, `bind(C)` for ctypes.

```fortran
subroutine snow17_run(n, idt, idts, iyr, imn, ida,             &
                      pcp, tmp,                                &
                      alat, elev, scf, mfmax, mfmin, uadj, si, &
                      nmf, tipm, mbase, pxtemp, plwhc, daygm,  &
                      adc11,                                   &
                      cs_io, tprev_io,                         &
                      raim, sneqv) bind(C, name="snow17_run")
```

Design notes:
- **Pass dates as arrays** (`iyr(n)`, `imn(n)`, `ida(n)`) built in pandas. Avoids
  writing Gregorian/leap-year arithmetic in Fortran.
- Compute `pa` inside the shim from `elev` using the formula above.
- Expand `adc11` to `adc12` with element 12 = 1.0 (see bug above).
- Check whether `PACK19` writes back to `TPREV` before deciding whether the shim
  should also set `tprev = tmp(t)` after each call — read `PACK19.f` for `TPREV`
  assignments.
- The F77 code uses default `REAL` (4-byte with gfortran). Do not pass
  `-fdefault-real-8`.

Build:

```bash
gfortran -shared -fPIC -O2 -o fortran/libsnow17shim.so \
  external/snow17/src/snow19/*.f fortran/snow17_shim.f90
```

Use `-fcheck=bounds` during development to surface the ADC issue; drop it for
the release build once the 12-element workaround is confirmed.

### THREE TESTS — write these before the Tesseract wrapper

Each catches a failure mode that produces *plausible but wrong* output:

1. **Determinism.** Call twice with identical inputs; assert bitwise-identical
   `raim`. Catches COMMON-block leakage between calls. Without this, a
   finite-difference VJP silently computes garbage — the perturbed run inherits
   state from the base run.
2. **State continuity.** One 100-day call must equal two chained 50-day calls
   (bitwise). Sharper version of (1); trust this one most.
3. **Mass balance.** Over a water year, `sum(raim) + delta_SWE ≈ sum(pcp)*scf`.
   Catches unit errors, which are the likeliest defect given the mm/s ↔ mm/step
   conventions we're deliberately skipping.

Validation target: `test_cases/ex1/output/snowout.orig.HHWM8.txt.dev` is a
reference output from the original NCAR version.

**Context on why these matter:** a prior debugging episode in this line of work
involved `CudnnLstmModel` in HydroDL silently returning zero gradients because
`torch._cudnn_rnn` bypasses autograd. It looked fine until gradients were checked
directly. Same failure shape here.

---

## Repo layout

```
dsnow17-hbv/
├── LICENSE                    # Apache-2.0
├── NOTICE                     # credit NOAA-OWP/snow17
├── README.md                  # one-command reproduction
├── external/snow17/           # git submodule, pinned commit
├── fortran/
│   └── snow17_shim.f90
├── tesseracts/snow17/
│   ├── tesseract_api.py
│   ├── tesseract_config.yaml
│   └── tesseract_requirements.txt
├── src/
│   ├── hbv.py                 # written from scratch, NOT from hydrodl2
│   ├── paramnet.py            # LSTM: attributes -> parameters
│   └── train.py
├── tests/
│   ├── test_shim.py           # the three tests above
│   └── test_gradients.py      # VJP vs. manual perturbation
└── notes/NOTES.md             # upstream bug log -> writeup motivation
```

Vendor snow17 as a **git submodule** pinned to a specific commit, so "which
version, with which bugs" is answerable.

---

## Tesseract specifics

- `apply`: forcings + parameters + initial state → `raim`, `sneqv`, final state.
- `vector_jacobian_product`: **finite differences** over the ~13 scalar
  parameters. The hackathon explicitly permits FD ("the solver may expose its
  Jacobian by autodiff or by finite differences; the composition with the
  inference engine is the contribution"). ~14 forward evaluations per VJP —
  affordable. **Do not attempt Enzyme/Flang** — it will consume the entire
  timeline. Mention it as future work.
- Verify the VJP against a manual perturbation before trusting it.
- Wrap at **full-rollout granularity**, never per-timestep. Tesseract's own docs
  say it targets kernels running at least several seconds.
- `abstract_eval` is cheap to implement and helps the JAX/Torch integration.

Rain/snow partitioning in Snow17 uses a hard temperature threshold (`PXTEMP`).
**Relax it with a sigmoid over ~1 °C** or `d(loss)/d(PXTEMP)` is exactly zero
almost everywhere and the LSTM will never learn it. Mention the relaxation
explicitly in the writeup — it reads as competence.

---

## Timeline and scope discipline

Remaining: ~22 days from Aug 9, realistically 8–10 working days.

| Days | Milestone |
|---|---|
| 1–2 | Build `snow19` + shim on HPC. Three tests pass. |
| 3–4 | Snow17 Tesseract with FD VJP. Gradient check vs. manual perturbation. |
| 5–6 | Minimal HBV in PyTorch. Chain: Tesseract → HBV → NSE → `.backward()`. Single basin, direct parameter optimization. **Loss goes down = submittable.** |
| 7–10 | LSTM parameter net, 30–50 snow-dominated CAMELS basins, held-out test. |
| 11–14 | Figures + 2–4 page writeup. |
| 15–20 | Buffer, submit, flip repo public (Aug 29), LinkedIn post tagging Pasteur Labs & ISI. |

**Hard stop: if end-to-end gradients are not flowing by Aug 20, drop the LSTM,
submit the single-basin parameter-optimization version, write it up honestly.**
A modest clean pipeline where Tesseract is obviously load-bearing scores better
than a broken ambitious one — that is literally the organizers' stated ranking.

Best-visual candidate figure: predicted melt factor (MFMAX1) vs. basin elevation
or mean winter temperature, showing the network recovered something physically
meaningful rather than fitting noise.

---

## How to work with me on this

- Be exacting and critical. Push back on bad ideas directly rather than
  softening them.
- Flag scope creep hard. The failure mode to watch for is reaching for a new
  interesting direction while the finishable thing sits unfinished.
- When something looks like it's working, ask what test would prove it *isn't*.
  Silent wrongness is the enemy here, not loud failure.