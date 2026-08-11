.PHONY: env build build-checked test clean

# Project-local venv, managed by uv. Created once; `make test` depends on
# it so a fresh clone just needs `make test` after the submodule is
# checked out. `uv venv` errors if .venv already exists (no implicit
# reuse), so guard creation -- otherwise every `make test` after the
# first fails outright instead of just reinstalling requirements.
env:
	[ -d .venv ] || uv venv .venv --python 3.11
	uv pip install --python .venv/bin/python -r requirements.txt

build:
	./fortran/build.sh
	./fortran/sacsma_build.sh

build-checked:
	BOUNDS_CHECK=1 ./fortran/build.sh
	BOUNDS_CHECK=1 ./fortran/sacsma_build.sh

test: env build
	.venv/bin/python -m pytest tests/ -v

clean:
	rm -f fortran/*.so fortran/*.o *.mod
	rm -rf fortran/_sacsma_patched
