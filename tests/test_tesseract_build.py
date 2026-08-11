"""Smoke test for the actual containerized Tesseract images -- built via
`tesseract build tesseracts/snow17` / `tesseract build tesseracts/sacsma`
in CI (.github/workflows/ci.yml), since this project's local dev machine
can't use Docker (see notes/logs.md for the full investigation).

Deliberately skips rather than fails when Docker or the built images
aren't available -- this is NOT part of `make test`'s normal local run
(every other test file runs fine without Docker; this one exists
specifically to verify the containerized artifact once it can be built).

Runs real apply() calls against the built images, not just confirming
`docker build` succeeded -- a build can succeed while the runtime is
still broken (wrong path inside the container despite building, a
missing runtime dependency, etc). This is the check that actually
verifies tesseract_config.yaml's package_data/custom_build_steps/env
wiring (see notes/logs.md for the container path-resolution fix that
wiring depends on) produces a working container, not just one that
builds.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tesseract_core import Tesseract  # noqa: E402


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=True)
        return True
    except Exception:
        return False


def _image_exists(tag: str) -> bool:
    if not _docker_available():
        return False
    result = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    return result.returncode == 0


@pytest.mark.skipif(
    not _image_exists("snow17:latest"),
    reason="snow17:latest not built -- run `tesseract build tesseracts/snow17` (needs Docker)",
)
def test_snow17_container_apply():
    n = 60
    rng = np.random.default_rng(0)
    inputs = {
        "idt": 24, "idts": 86400,
        "iyr": np.full(n, 2001, dtype=np.int32),
        "imn": np.full(n, 1, dtype=np.int32),
        "ida": (np.arange(n, dtype=np.int32) % 28) + 1,
        "pcp": np.abs(rng.normal(3, 2, n)).astype(np.float32),
        "tmp": np.linspace(-10, 5, n).astype(np.float32),
        "alat": 47.78, "elev": 1612.5,
        "scf": 1.15, "mfmax": 1.05, "mfmin": 0.15, "uadj": 0.04, "si": 500.0,
        "nmf": 0.15, "tipm": 0.2, "mbase": 0.0, "pxtemp": 1.0, "plwhc": 0.03, "daygm": 0.0,
        "adc": np.array(
            [0.05, 0.09, 0.16, 0.31, 0.54, 0.74, 0.84, 0.89, 0.93, 0.97, 1.0], dtype=np.float32
        ),
        "cs0": np.zeros(19, dtype=np.float32),
        "tprev0": 0.0,
    }
    with Tesseract.from_image("snow17:latest") as tess:
        out = tess.apply(inputs)

    raim = np.asarray(out["raim"])
    assert raim.shape == (n,)
    assert np.all(np.isfinite(raim))


@pytest.mark.skipif(
    not _image_exists("sacsma:latest"),
    reason="sacsma:latest not built -- run `tesseract build tesseracts/sacsma` (needs Docker)",
)
def test_sacsma_container_apply():
    n = 60
    rng = np.random.default_rng(0)
    inputs = {
        "dtm": 86400.0,
        "pcp": rng.gamma(1.0, 3.0, n),
        "tmp": rng.normal(10, 8, n),
        "etp": np.abs(rng.gamma(1.0, 2.0, n)),
        "uztwm": 30.0, "uzfwm": 25.0, "uzk": 0.3, "pctim": 0.01, "adimp": 0.05,
        "riva": 0.01, "zperc": 100.0, "rexp": 3.0, "lztwm": 130.0, "lzfsm": 25.0,
        "lzfpm": 60.0, "lzsk": 0.05, "lzpk": 0.01, "pfree": 0.15, "side": 0.0, "rserv": 0.3,
        "state0": np.zeros(6),
    }
    with Tesseract.from_image("sacsma:latest") as tess:
        out = tess.apply(inputs)

    q = np.asarray(out["q"])
    assert q.shape == (n,)
    assert np.all(np.isfinite(q))
