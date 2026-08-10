.PHONY: env build build-checked test clean

# Project-local venv, managed by uv. Created once; `make test` depends on
# it so a fresh clone just needs `make test` after the submodule is
# checked out.
env:
	uv venv .venv --python 3.11
	uv pip install --python .venv/bin/python -r requirements.txt

build:
	./fortran/build.sh

build-checked:
	SNOW17_BOUNDS_CHECK=1 ./fortran/build.sh

test: env build
	.venv/bin/python -m pytest tests/ -v

clean:
	rm -f fortran/*.so fortran/*.o *.mod
