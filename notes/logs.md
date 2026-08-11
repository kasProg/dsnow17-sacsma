# Engineering log

Chronological record of what was built/changed and *why*, updated as work
happens (not written retroactively from git log). Distinct from
[NOTES.md](NOTES.md), which is upstream-bug findings specifically;
this is the "why did we make this choice" record for our own code.

---

## 2026-08-10 — Repo skeleton + snow17 submodule

**What:** `git submodule add` for `https://github.com/NOAA-OWP/snow17`,
pinned at `ea142f94a4788ddf0be7b4d7da59c1f9f552df45`. Created
`fortran/`, `tesseracts/snow17/`, `src/`, `tests/`, `notes/` per the
repo layout CLAUDE.md specifies.

**Why a submodule instead of copying files in:** CLAUDE.md's own
reasoning — "which version, with which bugs" needs to be an answerable
question, both for reproducibility and because we found (and will keep
finding) upstream quirks that are commit-specific. Vendoring by copy would
lose that traceability and make "did we modify this" ambiguous, which
matters for the Apache-2.0 NOTICE obligation.

**Why pin to a specific commit rather than track a branch:** the whole
point of a submodule here is determinism — if upstream fixes the ADC bug
or the TPREV wiring after we've built around their current behavior, we
want that to be a deliberate `git submodule update`, not something that
silently changes our test baseline.

---

## 2026-08-10 — `fortran/snow17_shim.f90`

**What:** new Fortran source, ~110 lines. Loops the vendored, *unmodified*
`EXSNOW19` (from `external/snow17/src/snow19/exsnow19.f`) over a time
series for one HRU, exposed as a single `bind(C)` entry point
(`snow17_run`).

**Why write a shim instead of calling `EXSNOW19` directly from Python/C:**
`EXSNOW19` is single-timestep, single-HRU, with ~30 positional arguments
and an explicit state array the caller must thread by hand — there's no
way to call it usefully without *some* looping wrapper. The upstream
driver (`runSnow17.f90`) already is that wrapper, but it's built around
namelists, multi-HRU config, and file I/O we don't want; reusing it would
mean linking `src/share`, which CLAUDE.md's build-scope decision
explicitly excludes (keeps the dependency footprint to exactly the 13
physics files we can audit and attribute cleanly).

**Why `bind(C)` rather than an `iso_c_binding`-free F90 interface:** the
Tesseract/PyTorch side of this project is Python. `bind(C)` gives a flat C
ABI (`ctypes.CDLL` can load it directly) instead of requiring a
Fortran-aware FFI. This is also what makes it possible to build with plain
`gfortran -shared -fPIC`, no Fortran-specific packaging.

**Why the state (`CS(19)` + `TPREV`) is threaded explicitly by the
caller, not hidden inside the shim:** two reasons. (1) `EXSNOW19` itself
already requires this — it unpacks `CS` into a COMMON block at entry and
repacks at exit, so state has to come from *somewhere* outside the
function on every call; hiding it inside the shim would just relocate the
same problem one layer down and make chained/resumable runs impossible.
(2) It's what the [determinism and state-continuity tests](test_snow17_shim.py)
actually verify — if state were implicit (e.g. a module-level variable
retained between calls), those tests would be checking the shim's
plumbing instead of the physics, and would stop being a meaningful guard
against COMMON-block leakage.

**Why `tprev_io = tmp(t)` happens *after* the call, not before:** confirmed
by reading `PACK19.f` (`DTA = TA - TPREV` is the only read site) that
`TPREV` means "the previous timestep's air temperature," and neither
`PACK19.f` nor `exsnow19.f` ever assigns it — see the state-contract note
in CLAUDE.md that flagged this as an open question. The call at step `t`
needs `TPREV = T(t-1)`, so the update to `T(t)` has to happen after, not
before, or the model would see the *current* step's temperature as its own
"previous" value. Verified against real data later — see the 2026-08-10
"reference cross-check" entry below for the divergence this choice causes
relative to upstream's actual (buggier) behavior, and why the choice was
kept anyway.

**Why `PA` is computed inside the shim from `elev`, not passed in as a
forcing:** traced every use of the internal `ELEV` variable in
`exsnow19.f` and found the line that would compute `PA` from it is
commented out — `PA` has to come from outside, exactly as CLAUDE.md's
design note said, and this was confirmed against
`external/snow17/src/share/ioModule.f90`'s `sfc_pressure` subroutine
(same formula, same constants, down to the `Anderson 2006` sourcing) so
the shim's version isn't a guess, it's a re-implementation of the same
five-constant formula the real driver uses.

**Why `ADC` is padded from 11 to 12 elements before the call:**
`exsnow19.f` declares `ADC(11)`, `PACK19.f` declares the same dummy
argument as `ADC(12)` — a real declaration-size mismatch. Traced every
site in `snow19/` that indexes `ADC` directly (`adjc19.f`, `aesc19.f`,
`aeco19.f`, `updt19.f`) and found the only path that would touch index 12
(`UPDT19`, reached only when `IUPWE`/`IUPSC` data-assimilation flags are
set) is unreachable here — `EXSNOW19` hardcodes both flags to `0` on every
call. So this padding fixes a latent mismatch that isn't live corruption
in our usage, but it costs nothing and removes the mismatch outright
rather than relying on "we checked, it's fine" holding forever as the
model is used in new ways later (e.g. if data assimilation is ever turned
on). Full trace in [NOTES.md](NOTES.md).

**Why `-fdefault-real-8` must never be added to the build:** `EXSNOW19`
and everything it calls use gfortran's default `REAL` (4 bytes). Our
`bind(C)` dummy arguments use `c_float` (4 bytes) to match. Compiling with
`-fdefault-real-8` would silently reinterpret every `REAL` in the F77
files as 8 bytes, breaking argument association with our `c_float` dummy
args without any compile error — the exact "plausible but wrong" failure
mode this whole project is trying to avoid at the state/units layer.
Called out explicitly in a source comment for this reason, not just in
CLAUDE.md.

---

## 2026-08-10 — `fortran/build.sh`

**What:** builds `libsnow17shim.so` from the 13 vendored `snow19/*.f`
files plus the shim, in one `gfortran -shared -fPIC` command.

**Why a script instead of inlining the command in the README/Makefile:**
wanted the bounds-check toggle (`SNOW17_BOUNDS_CHECK=1`) available without
having two near-duplicate build commands to keep in sync (one for
day-to-day fast builds, one with `-fcheck=bounds` for catching Fortran
array issues during development, per CLAUDE.md's suggested workflow). A
single script with an env-var switch keeps that DRY, and the Makefile
(`build` vs `build-checked`) just sets the flag rather than duplicating
compiler invocations.

---

## 2026-08-10 — `src/snow17.py`

**What:** `ctypes` wrapper (`run_snow17`) around `snow17_run`, plus
`Snow17Params`/`Snow17Output` dataclasses.

**Why dataclasses instead of passing 13 positional scalars around:** the
Fortran side has an inherent flat-scalar signature (unavoidable — that's
`EXSNOW19`'s own argument list, and the shim mirrors it 1:1 so the
Fortran-to-Python argument mapping stays easy to audit by eye). On the
Python side, callers will be constructing parameter sets from an LSTM's
output vector and from a CSV-loaded basin-attributes table, both of which
are natural to bind to named fields — the dataclass exists to stop that
mapping from being 13 unlabeled positional floats at every call site.

**Why `np.ctypeslib.ndpointer` for the arrays instead of raw
`ctypes.POINTER(c_float)`:** `ndpointer` enforces `dtype`,
contiguity, and dimensionality *before* the call, so a caller who passes a
non-contiguous slice or the wrong dtype gets a clear Python `TypeError`
immediately, not silent memory corruption or a segfault three function
calls later inside Fortran. Given the project's stated failure mode to
guard against (silent wrongness, per the `CudnnLstmModel` cautionary
note in CLAUDE.md), the cost of the extra strictness is worth it.

**Why `cs0` defaults to `None` -> zeros rather than requiring the caller
to pass it:** matches `EXSNOW19`'s own documented cold-start convention
(`CS = 0.0`, `TPREV = 0.0`, no external initialization needed — see
CLAUDE.md's state-contract section). Keeping that as the Python default
means a caller doing a fresh single-basin run doesn't have to know the
internal state layout at all; only the multi-segment / warm-start /
Tesseract-VJP-perturbation callers need to pass `cs0` explicitly.

---

## 2026-08-10 — `tests/test_snow17_shim.py`

**What:** 7 tests — determinism, state continuity, mass balance (the
three CLAUDE.md calls out as required before the Tesseract wrapper), a
reference cross-check against `test_cases/ex1`'s per-HRU output, and an
explicit "prove the snowh divergence" test.

**Why synthetic data for determinism/continuity but real ex1 data for
mass-balance/reference checks:** determinism and state-continuity are
checks on the *shim's own plumbing* (does calling the function twice give
the same answer; does splitting one call into two give the same answer) —
they don't need physically realistic forcing, just forcing that exercises
both accumulation and melt branches so state actually moves through its
different code paths. Mass balance and the reference cross-check are
checks on *physical correctness*, which synthetic data can't validate —
only real forcing with a known-good reference output can.

**Why the reference-cross-check test doesn't just loosen tolerances until
`snowh` passes:** loosening a tolerance until a test passes without
knowing why the mismatch exists is exactly the "silent wrongness" failure
mode this project is trying to avoid — a green test that's green because
the assertion doesn't actually constrain anything isn't a test, just
decoration. Instead, `test_snowh_divergence_is_explained_by_upstream_tprev_bug`
replays the exact same run with `TPREV` pinned at `0.0` (matching what
tracing the upstream driver showed it actually does for a cold start) and
asserts *that* matches the reference — which turns the mismatch from "our
shim is being sloppy" into "we understand precisely why upstream's number
differs from the physically correct one, we've proven it, and we're
deliberately not replicating their gap." See NOTES.md for the full trace.

**Why the mass-balance test computes `corrected_pcp` with a per-timestep
rain/snow split instead of a blanket `pcp * scf`:** the first version of
this test used `sum(pcp) * scf` as expected input and was off by ~25%.
Rather than loosen the tolerance, traced `PACK19.f` and found `SCF` is
only applied to the snowfall fraction (`SFALL = PXI*FRACS*SCF`,
`PACK19.f:134`), not rain — matching the real-world motivation for the
correction (gauges under-catch wind-blown snow, not rain). The test now
encodes that split explicitly rather than treating `scf` as a blanket
precip multiplier, and closes to float32 precision once that's fixed.

---

## 2026-08-10 — `NOTICE`, `README.md`, `Makefile`, `.gitignore`,
`requirements.txt`

**What:** repo scaffolding for a one-command reproduction path.

**Why `.gitignore` excludes `external/snow17/test_cases/ex1/`:** the
extracted directory is a side effect of `tar xzf`-ing a file that already
lives (compressed) inside the pinned submodule — committing the extracted
copy to *our* repo would duplicate data that's already version-controlled
upstream (in the submodule) and would go stale if the pinned commit ever
changes. `tests/test_snow17_shim.py::_ensure_ex1_extracted` extracts it on first
test run instead, so a fresh clone + `git submodule update --init` still
reproduces without a manual step — which is the actual requirement
(`README.md` promises one-command reproduction, and CLAUDE.md's
deliverables list requires it), not "commit everything so nothing can
fail."

**Why `NOTICE` explicitly states no `snow19/` source files were
modified:** required by NOAA-OWP's own README (adaptations must credit
the source repo) and is the operative Apache-2.0 obligation for a
derivative work; also directly checkable — anyone can diff
`external/snow17` against upstream at the pinned commit and confirm the
claim rather than take it on faith.

---

## 2026-08-10 — Project environment: `uv`, and the Docker access gap

**What:** `.venv/` created via `uv venv --python 3.11`, dependencies
installed via `uv pip install -r requirements.txt`. `Makefile`'s `env`
target wraps this; `test` depends on it.

**Why `uv` instead of the base conda env or a plain `venv`:** the base
Python (`/usr/local/anaconda3/bin/python3`) is a shared system
environment on this machine — installing `tesseract-core`/`tesseract-torch`
into it would affect anything else using that interpreter and makes "what
exactly does this repo need" unanswerable from the repo alone. A
project-local `.venv` fixes that; `uv` over stdlib `venv`+`pip` because
it's already available on this machine and meaningfully faster for
repeated installs during iteration (relevant since `make test` re-runs
`uv pip install` every time to stay a true one-command reproduction path).

**What we found: this account can't use Docker.** `docker ps` ->
permission denied; `getent group docker` confirms the account isn't a
member. Relevant because Tesseract's standard workflow (`tesseract
build`) packages a Tesseract as an OCI container, and that's what the
hackathon's "package it as a real Tesseract" expectation presumably means
for the final submission.

**Why this didn't block Day 3-4 work:** `tesseract_core.Tesseract` has a
documented `from_tesseract_api()` constructor -- "This does not use a
containerized Tesseract, but rather imports the Tesseract API directly.
This is useful for debugging, but requires a matching runtime environment
+ all dependencies to be installed locally." That's exactly the
constraint our own `.venv` already satisfies. All of
`tests/test_gradients.py` runs through this path -- `apply()`,
`vector_jacobian_product()`, `abstract_eval()`, and the
`tesseract_torch.apply_tesseract()` autograd integration are all verified
working without touching Docker.

**What's still unresolved:** `tesseract build` itself, needed to
containerize for actual submission. Options going forward: request
`docker` group membership from HPC admin, use a different machine with
Docker access for that one step, or check whether rootless
Docker/Podman is viable here. Flagged in `tesseract_config.yaml`'s
`build_config` comments and in `README.md` rather than silently deferred
-- the risk is discovering this gap at submission time instead of now.

---

## 2026-08-10 — `tesseracts/snow17/tesseract_api.py`

**What:** the Snow-17 Tesseract wrapper -- `InputSchema`/`OutputSchema`,
`apply()`, `vector_jacobian_product()` (forward-difference FD), and
`abstract_eval()`.

**Why scaffolded via `tesseract init --recipe base` rather than written
from scratch:** wanted the actual expected shapes of `InputSchema`,
`Differentiable[...]` annotations, and the `vjp_inputs`/`vjp_outputs`/
`cotangent_vector` signature confirmed from the installed `tesseract-core`
version rather than guessed from CLAUDE.md's prose description -- prose
can drift from what a specific pinned package version actually expects
across a piece of software still stating a beta-ish version number.
Generated both `base` and `pytorch` recipe templates to compare: `base`
uses a plain user-supplied `vector_jacobian_product`, `pytorch` wraps
`torch.func.vjp` around a user-supplied `evaluate()` and expects the
whole computation to be torch-native/traceable. Ours isn't -- gradients
come from finite-differencing calls into compiled Fortran through
`ctypes`, not from anything `torch.func` can trace -- so `base` is the
correct recipe, not `pytorch`, despite this project's autograd endpoint
being PyTorch.

**Why `ADC` (11-point areal depletion curve) is NOT in
`DIFFERENTIABLE_PARAMS`:** asked the user directly rather than resolve a
real inconsistency in CLAUDE.md silently -- the "Learnable parameters"
section lists 11 named scalars *plus* `ADC(11)` and calls it "13 scalars",
while the "Tesseract specifics" section scopes the FD VJP to "~13 scalar
parameters... ~14 forward evaluations," which is only consistent with
forward-difference over roughly 13 differentiable values, not 11+11=22.
User chose: ADC fixed for v1. `vector_jacobian_product` raises `ValueError`
if `adc` is requested as a `vjp_input`, rather than silently returning
`0.0` -- an explicit error is far cheaper to debug than a gradient that's
quietly always zero and looks like a converged/uninformative parameter
during training.

**Why the FD step is a relative step with a floor
(`max(abs(value)*1e-3, 1e-4)`), not a fixed epsilon:** the 11 differentiable
parameters span very different scales (e.g. `SI` ~500, `MBASE` often
exactly `0.0`, `PXTEMP` ~O(1)). A single fixed epsilon would be
simultaneously too large relative to small parameters (poor local-slope
estimate) and swamped by float32 rounding for near-zero ones without the
floor (`MBASE=0.0` would get `eps=0`, a zero step, dividing by which is
undefined). Values are not yet tuned against a real training loop --
flagged as worth revisiting once loss curves are in hand and it's
possible to tell whether FD noise is limiting learning, per CLAUDE.md's
"verify the VJP against a manual perturbation before trusting it," which
this file's tests do, but "trusted for correctness" and "tuned for
optimization" are different bars.

**Why `vector_jacobian_product` explicitly casts to `float64` before
subtracting `perturbed - base`, rather than working in the native
`float32` of the rollout outputs:** found empirically, not designed in
up front -- see the `tests/test_gradients.py` entry below. `perturbed`
and `base` are two nearly-identical float32 arrays (they differ only by
one small parameter perturbation), so summing each in float32 and then
subtracting the two sums hits catastrophic cancellation. Subtracting
element-wise first (while both operands are still close to each other,
so the subtraction itself is well-conditioned) and only then summing --
in float64, so the *accumulation* of many small differences doesn't
itself lose precision -- avoids it. This is generic numerical-methods
practice for finite differences of near-identical values, not something
specific to this codebase, but it wasn't in the first version of this
function and its absence was the actual root cause of an apparent test
failure (see below), not a bug in the test.

---

## 2026-08-10 — `tests/test_gradients.py`

**What:** `apply()`-matches-shim-directly, `abstract_eval()`-matches-`apply()`
shapes, `vector_jacobian_product()`-vs-manual-perturbation (parametrized
over all 11 differentiable params), linearity-in-cotangent, ADC-rejection,
and a documented near-zero-gradient check for `PXTEMP`.

**Why the synthetic fixture spans 200 days from day-of-year 0, not a
shorter window:** the first version used `n=60`. Checked (didn't assume)
whether that window actually exercises melt: `(tmp > 0).sum()` over that
window was `0/60` -- deep winter only, by construction of a `sin`-based
seasonal temperature curve starting at day 0. That makes any melt-only
parameter's (e.g. `MFMAX`) true gradient exactly zero in that window --
correctly computed as zero by both the VJP and by manual perturbation
during ad hoc verification, but a real `0 == 0` isn't a meaningful test of
whether the VJP's *wiring* (right parameter perturbed, right output read,
right cotangent applied) is correct, since a wiring bug could also
produce zero by accident. Extended to `n=200`, matching the window
`tests/test_snow17_shim.py`'s own synthetic series already uses and had already
confirmed reaches positive temperatures.

**Why `test_vjp_matches_manual_perturbation` reads `base_value` from a
validated `InputSchema` instance rather than the raw input dict:** first
attempt used the raw Python float (e.g. `1.05`) to compute the FD step,
while `vector_jacobian_product` internally reads the *pydantic-coerced*
`float32` value (`np.float32(1.05) == 1.0499999523...`) -- a different
starting point for the same nominal parameter. Investigated in case this
alone explained an observed mismatch; it turned out not to be the
dominant effect (the real cause was the cancellation issue below), but
using the coerced value is still the correct apples-to-apples comparison
and was kept.

**Why this test almost shipped with a wrong "manual perturbation" formula,
and how it was caught:** the first version computed
`(perturbed["raim"].sum() - base["raim"].sum()) / eps` using the raw
float32 rollout outputs. For 7 of 11 parameters (specifically the
melt/lag-related ones that actually got exercised once the fixture
covered real melt) this disagreed with `vector_jacobian_product`'s answer
by up to ~0.5% -- above a `rtol=1e-4` tolerance that was deliberately
tight because both sides were supposed to be doing the identical
computation. Rather than loosen the tolerance (which would have quietly
hidden either a real bug or a real numerical issue, indistinguishably),
reproduced `vector_jacobian_product`'s internal arithmetic line by line
in a scratch script and confirmed it matched the wrapper's own output
bit-for-bit -- meaning the wrapper was correct and the test's independent
check was not, specifically because it summed float32 arrays before
subtracting (catastrophic cancellation) rather than subtracting
element-wise in float64 first. Fixed the test to match the numerically
correct approach (see the `tesseract_api.py` entry above) rather than
either loosening the tolerance or leaving a subtly-wrong "independent"
check in place -- a manual check that's itself numerically unsound isn't
independent verification, it just moves the bug to a different file.

**Why `test_pxtemp_gradient_is_structurally_near_zero` exists as a
passing test rather than being left undocumented:** CLAUDE.md already
identifies that `PXTEMP`'s hard threshold means its gradient is ~zero
almost everywhere until the sigmoid relaxation lands. Rather than leave
that as a fact someone has to remember from a planning doc, encoded it as
a test that will fail loudly (forcing an update, not a silent pass) once
the relaxation actually changes this behavior -- the assertion compares
`PXTEMP`'s gradient magnitude against `MFMAX`'s (a parameter confirmed to
respond continuously in the same window) rather than against a fixed
threshold, so it stays meaningful regardless of the absolute scale of
gradients produced by future changes elsewhere in the pipeline.

**Why `test_backward_through_apply_tesseract` exists as its own test,
separate from the `vector_jacobian_product`-level ones:** first checked
this ad hoc (not committed) and it worked -- `d(loss)/d(scf)` via
`tesseract_torch.apply_tesseract()` + `.backward()` matched the manual-FD
value to ~0.02%. But CLAUDE.md's whole argument is that Tesseract is
*load-bearing*: gradients have to actually cross the Fortran boundary
through PyTorch's autograd graph, not just be computable by calling
`vector_jacobian_product` directly (which never proves the graph-splicing
machinery itself works -- a broken `apply_tesseract()` integration could
coexist with a perfectly correct `vector_jacobian_product`). Left as
scratch verification, that distinction isn't captured anywhere durable.
Promoted into a committed test that builds a real `torch` graph, calls
`.backward()`, and asserts both that the resulting gradient matches
`vector_jacobian_product` *and* that it's nonzero -- the second assertion
specifically to catch the `CudnnLstmModel`-style failure mode CLAUDE.md
names as the cautionary example (an integration that runs without error
but silently returns zero gradients, which "gradients match" alone
wouldn't catch if both sides happened to independently return zero).

---

## 2026-08-11 — CLAUDE.md rewrite: SAC-SMA replaces HBV, two Tesseracts required

**What:** user rewrote CLAUDE.md — HBV is now the Aug-20 fallback only;
SAC-SMA is the primary downstream model, chained Snow17 -> SAC-SMA via
RAIM, each its own Tesseract (judging criterion 1 requires composing two
or more). Renamed `tests/test_shim.py` -> `tests/test_snow17_shim.py` and
`tests/test_tesseract_api.py` -> `tests/test_gradients.py` to match the
new repo layout CLAUDE.md specifies, fixed all internal references
(README, docstrings, other notes files) accordingly, and updated
README/HBV mentions to reflect SAC-SMA as the primary plan rather than
rewriting history in already-dated log entries above this one.

**Why rename now instead of leaving it for later:** about to add the
SAC-SMA equivalents (`test_sacsma_shim.py`, and eventually a
`tesseracts/sacsma/` alongside a differently-scoped `test_gradients.py`).
Adding new files in the new naming convention while the Snow17 ones
still used the old names would mean drifting further from CLAUDE.md's
layout with every commit instead of converging on it.

---

## 2026-08-11 — `external/sac-sma` submodule + archaeology

**What:** vendored `https://github.com/NOAA-OWP/sac-sma`, pinned at
`975902e3d44785f3b3503f29adfb5755120f5bf`. Read `src/sac/ex_sac1.f90`
(the `EXSAC` entry point, single-timestep/single-HRU, mirrors
`EXSNOW19`'s role exactly), `src/sac/sac1.f90` (the physics), and
`src/sac/sac_data_mod.f90` (the only module either of those `USE`s).
Confirmed via sac-sma's own `CMakeLists.txt` (`MODEL_SOURCES`) that the
physics-core build needs exactly these 3 files — smaller and simpler
than Snow17's 13-file F77 core.

**Why DOUBLE PRECISION, not the default REAL snow17 uses:** read the
actual type declarations rather than assuming symmetry with the Snow17
shim — `ex_sac1.f90`/`sac1.f90` declare every real value
`DOUBLE PRECISION` explicitly (`sac_data_mod.f90`'s `dp =
SELECTED_REAL_KIND(15, 307)` confirms 8-byte). `fortran/sacsma_shim.f90`
and `src/sacsma.py` use `c_double`/`float64` throughout — mixing this up
with Snow17's `c_float`/`float32` convention would be exactly the kind
of silent argument-association corruption the "never add
-fdefault-real-8" note on the snow17 shim exists to prevent, just from
the opposite direction.

**Why state is a flat 6-element array (`UZTWC, UZFWC, LZTWC, LZFSC,
LZFPC, ADIMC`), not a packed/indexed structure like Snow17's `CS(19)`:**
`EXSAC` itself passes these as 6 individually-named `INTENT(INOUT)`
arguments — there's no packed array to unpack/repack, no `NEXLAG`-style
timestep-dependent indexing. `state_io(6)` in the shim is purely a
calling-convention choice (uniform with `cs_io` on the Python side, one
array in/out instead of 6 loose scalars) — the physics itself has no
`CS`-equivalent bookkeeping to get wrong.

**Why `TMP` is still threaded through the shim/wrapper despite having
zero effect on output:** confirmed by reading `SAC1`'s body — `TA` (its
internal name for `TMP`) is only read inside `FROST1`, called only `IF
(IFRZE .GT. 0)`, and `EXSAC` hardcodes `IFRZE = 0` on every call. Same
dead-parameter shape as Snow17's `ELEV1`. Kept in the interface anyway
(rather than dropped) so the signature doesn't need to change if frozen-
ground support is ever turned on later — cheap to keep, and mirrors
`EXSAC`'s own argument list exactly, which keeps the Fortran-to-Python
mapping easy to audit by eye (same reasoning as the snow17 shim's
1:1-with-`EXSNOW19` design).

**Why `DUAMEL` (unit hydrograph routing) is NOT part of the shim:**
checked whether the reference driver (`runSac.f90`'s `solve_sac`) calls
it before assuming routing is needed to get from `EXSAC`'s `TCI` output
to something loss-comparable — it doesn't. `solve_sac` calls `exsac(...)`
directly and does its own mass-balance bookkeeping straight off `TCI`.
`DUAMEL` also isn't in `CMakeLists.txt`'s `MODEL_SOURCES` at all. So
`TCI` (exposed as `q` in `sacsma_shim.f90`/`src/sacsma.py`) is "runoff"
for the NSE loss, matching CLAUDE.md's diagram, with no separate routing
Tesseract needed for v1. Worth a caveat in the eventual writeup: `TCI` is
unrouted total channel inflow, not timing-lagged to match a USGS gauge
hydrograph the way `DUAMEL` output would be — a real limitation, not
hidden, just out of scope for now (CLAUDE.md's own 16-parameter SAC-SMA
list doesn't include `DUAMEL`'s unit-hydrograph shape parameters either,
so this reading is consistent with the plan as written, not a unilateral
scope cut).

---

## 2026-08-11 — `bypass_ratio_check` implicit-SAVE bug: found, proven, patched

**What:** found a live (not dead-code-gated) state-leakage bug in
`sac1.f90` — full writeup in [NOTES.md](NOTES.md). Fixed via
`patches/sac1_bypass_ratio_check_save_fix.patch`, applied to a
build-time-only copy of the file by `fortran/sacsma_build.sh`.

**Why patch a vendored dependency at all, given CLAUDE.md's Apache-2.0
framing leans on "Snow17 and SAC-SMA are unmodified upstream
dependencies":** there was no alternative that didn't sacrifice either
correctness or the whole point of having a fast FD-based Tesseract.
`bypass_ratio_check` has no argument, `COMMON` block, or module interface
exposing it — nothing our shim could poke from outside to reset it, the
way `TPREV` or `CS` could be threaded correctly for Snow17. The only
other options were: (a) don't fix it and hope basin data / parameter
search never triggers the precondition -- rejected outright, "hope it
doesn't trigger" is precisely the silent-wrongness failure mode this
whole project's testing philosophy exists to rule out, and gradient-based
parameter search will explore extreme parameter values that a calibrated
parameter set never would; (b) reload the shared library (or spawn a
fresh process) before every single `EXSAC` call to force a true reset of
all static state -- technically addresses it but destroys the performance
FD-based VJPs depend on (already budgeted at up to ~45 forward runs per
gradient per CLAUDE.md's cost analysis; per-call process spawns would
dominate that cost by orders of magnitude). A one-line source patch,
fully disclosed, applied only to a build-time copy, was the only option
that fixes the actual defect without either of those costs.

**Why apply the patch to a staged copy (`fortran/_sacsma_patched/`,
gitignored) instead of committing an edited file into `external/sac-sma`
directly:** the submodule pin is supposed to answer "which version, with
which bugs" unambiguously (same reasoning as the original decision to
vendor via submodule at all, logged above under "Repo skeleton + snow17
submodule") — editing the submodule's working tree directly would make
`git submodule status` show a dirty/detached state that doesn't match
any real upstream commit, and `git submodule update` would silently wipe
the edit on the next sync. A patch file applied at build time keeps
`external/sac-sma` byte-identical to the pinned commit *and* keeps the
fix auditable as its own diffable, reviewable artifact — closer to how a
Linux distro package patches upstream source than to silently forking it.

**Why the "unmodified dependency" README/NOTICE language needs updating,
and what it should say instead:** it's simply no longer accurate for
SAC-SMA. Apache-2.0 explicitly permits modification given attribution
(that's the license working as intended, not a workaround) — the honest
framing is "SAC-SMA is vendored unmodified as a pinned submodule; one
disclosed, minimal, build-time-only patch is applied to fix a confirmed
upstream defect, documented in `notes/NOTES.md` and
`patches/sac1_bypass_ratio_check_save_fix.patch`." Updated NOTICE and
README to say exactly that rather than leave the stronger "unmodified"
claim standing.

**How the patch itself was generated, and why not hand-written directly
as a unified diff:** wrote the target fix as a small Python
find-and-replace against a copy of the real file, then generated the
patch via `diff -u` against the original -- guarantees byte-exact
context lines (a hand-written patch's context has to match whitespace
and line content exactly or `patch` rejects it; got this wrong on the
first hand-written attempt -- `patch --dry-run` failed on a malformed
hunk before the second, diff-generated version applied cleanly).

---

## 2026-08-11 — `tests/test_sacsma_shim.py`

**What:** the same three required tests as `test_snow17_shim.py`
(determinism, state continuity, mass balance -- using `runSac.f90`'s own
mass-balance formula, read directly from its `derived%mass_balance`
computation rather than re-derived from scratch), plus a dedicated test
proving the `bypass_ratio_check` patch fixes the exact scenario in
NOTES.md through the real compiled shim (not just the standalone repro
program used to first confirm the bug), plus a reference cross-check
against ex1's `output.sacbmi.HHWM8I{L,U}.txt`.

**Why the mass-balance test's tolerance is `1e-2 * total_precip` rather
than something tighter:** checked the reference's own `mass_bal.csv`
before picking a number rather than guessing -- it shows non-zero
residuals itself (up to a few tenths of a mm on some days, computed by
upstream's own driver), so demanding a tighter closure from our shim
than upstream's own reference achieves would be testing against a
standard the reference doesn't meet either.

**Why `test_matches_upstream_reference` passing tightly isn't treated as
"the patch doesn't matter" evidence:** checked directly (see NOTES.md) --
`UZTWC` never hits exactly `0.0` anywhere in the 46-year HHWM8 state
trace, so the branch that sets `bypass_ratio_check` never fires for this
basin at all. A passing cross-check here demonstrates the shim's
plumbing is correct, not that the bug was inconsequential -- those are
different claims, and NOTES.md states which one is actually supported by
the evidence.

---

## 2026-08-11 — `src/coupling.py`: cross-container gradient coupling, prototyped first

**What:** `CoupledTwoStageFunction`, a single `torch.autograd.Function`
implementing CLAUDE.md's "option 1.5" gradient-coupling design, plus
`FDConfig` (per-parameter relative-step, floored, central-difference
config). Validated against cheap toy stand-ins
(`tests/test_coupling_toy.py`) before ever touching the real Tesseracts,
per CLAUDE.md's explicit "prototype on day 3-4 before committing."

**Why one `Function` instead of the split I originally proposed (custom
autograd for theta_A, normal `apply_tesseract()` composition for
theta_B):** user correction, and right — that split leaves a real open
question about how two structurally different gradient computations
(one from a hand-rolled `backward()`, one from Tesseract's own VJP
machinery composed automatically) correctly merge into the same upstream
network's `.backward()` without double-counting or sign errors. Folding
both parameter blocks into one `Function`'s `backward()` makes that
question not exist: both blocks reach the shared LSTM/MLP through the
exact same, single autograd node, and both are computed with the same
manual FD-sweep code path (`_fd_vjp_block`) -- theta_B's block is just a
shorter version of theta_A's (no `stage_a` re-run, RAIM held at a cached
base), not a different mechanism entirely. "Symmetry over hybrid," per
the correction.

**Why the `Function` returns `runoff` (via a real VJP: perturb, rerun,
dot the resulting *output* series against the incoming cotangent) rather
than differencing the loss directly:** second correction, also right.
Differencing the loss directly (`(loss(theta+eps) - loss(theta))/eps`)
would have baked a specific loss function into the coupling layer --
swapping NSE for KGE, adding a regularizer, or differentiating something
downstream of runoff (a routing step, a multi-basin aggregation) would
all require editing `coupling.py`. Returning `runoff` as a normal
`torch.Tensor` node and computing `d(runoff)/d(theta)` (dotted with
whatever cotangent arrives from upstream, which is exactly what a VJP
is) keeps the loss entirely in ordinary autograd-land, outside this
file, and is the honest mathematical object rather than a hidden
shortcut -- worth stating plainly in the writeup, per the user's note.
Cost is identical either way: each perturbed run already produces the
full `runoff` series, so keeping it (instead of collapsing straight to
a scalar loss inside the sweep) costs nothing extra, just a dot product.

**Why `FDConfig` supports per-parameter `rel`/`floor` arrays instead of
one scalar step size:** real Snow17/SAC-SMA parameters span very
different scales in the same gradient computation (`UZTWM` ~O(100) mm,
`UZK` ~O(0.1)) -- a single fixed `eps` is simultaneously too coarse
for the small-scale parameters (poor local-slope estimate) and
needlessly fine for the large-scale ones. This mirrors why
`tesseracts/snow17/tesseract_api.py`'s own `_fd_step` already does
relative-step-with-floor for its 11 parameters; `FDConfig` generalizes
that so `coupling.py` doesn't reinvent it differently when the real
wrappers get plugged in.

**Why central differences, not forward, for both blocks:** user's
correction -- Snow17 is float32, and forward differences carry a
leading O(eps) truncation error that's especially noisy at float32
precision; central differences (`(f(x+eps)-f(x-eps))/2eps`) kill that
leading term at the cost of one extra evaluation per parameter (26->52
runs for the real 13-parameter theta_A block -- still cheap). Applied
central to the theta_B block too, for the "symmetry over hybrid"
reasoning above, even though SAC-SMA's float64 precision would tolerate
forward differences better on its own -- consistency of method across
both blocks was judged more valuable than the small extra saving,
and `FDConfig(central=False)` remains available (with the cached base
output reused, not recomputed) if that tradeoff needs revisiting once
real timing numbers exist.

**How the toy prototype validates it, and why three independent
checks:** `tests/test_coupling_toy.py` builds cheap numpy stand-ins for
both stages (a sigmoid-relaxed rain/snow threshold for "Snow17," a
leaky-reservoir recursion with real memory for "SAC-SMA," matching the
real problem's shape on purpose -- nonlinear, stateful, heterogeneous
parameter scales including one deliberately near zero to exercise the
FD floor) plus torch-native equivalents of the identical math, so
autograd gives an exact ground-truth gradient. `CoupledTwoStageFunction`'s
FD-based gradient is checked against that autograd ground truth *and*
against an independently hand-rolled brute-force central difference of
the whole pipeline's loss, computed at a different eps than what
`FDConfig` itself uses in production for that run. The "different eps"
part is the point, per the user's correction: comparing an FD
computation to itself at the same step size is tautological (a wiring
bug -- wrong sign, wrong parameter, wrong output slice -- doesn't
necessarily shrink or grow with eps, so self-consistency at one eps
doesn't rule it out); two independently-implemented computations at two
different step sizes converging to the same answer is real evidence.
Measured the actual achieved precision before picking a test tolerance
rather than guessing one (relative errors were 1e-14 to 6e-5 across both
eps values tested) and set `rtol=1e-3` -- tight enough to actually catch
a moderate-sized regression, not just loose headroom that would pass
almost anything.

**Why there's also a call-counting structural test
(`test_raim_never_requires_grad_on_theta_b_path`), separate from the
numerical gradient checks:** a numerical match alone wouldn't catch a
version of this that "accidentally" got the right answer while still
being expensive -- e.g., if `stage_a` (Snow17) were silently re-invoked
during the theta_B sweep for some unrelated reason, gradients could
still come out numerically correct while defeating the entire point of
option 1.5 (avoiding N re-runs of the expensive stage on the cheap
parameter block). Wrapping both stages in a call-counting shim and
asserting exact expected call counts (`1 + 2*n_a` for stage_a, matching
forward + the central-diff theta_A sweep, never touching stage_a again
during the theta_B block) makes the cost claim itself a tested
invariant, not just an assumption from reading the code.

**What's still open going into the real wrapper integration:** the toy
uses float64 throughout for numerical cleanliness in the ground-truth
comparison; the real integration has Snow17 predicting/consuming
float32 and SAC-SMA float64 (see the snow17.py vs sacsma.py dtype notes
above), so `coupling.py`'s dtype handling at that boundary needs
explicit attention once `tesseracts/sacsma/` exists and both wrappers
get plugged in as the real `stage_a`/`stage_b`.
