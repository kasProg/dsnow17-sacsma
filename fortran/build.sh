#!/usr/bin/env bash
# Build fortran/libsnow17shim.so against the vendored snow19 sources.
#
# Deliberately links only external/snow17/src/snow19/*.f (the 13 plain
# FORTRAN-77 physics files) plus our shim -- not src/share, src/bmi, or
# src/driver.
#
# Do NOT add -fdefault-real-8: EXSNOW19 uses gfortran's default REAL
# (4 bytes) throughout.
set -euo pipefail
cd "$(dirname "$0")/.."

FLAGS="-shared -fPIC -O2"
if [[ "${BOUNDS_CHECK:-0} " == "1 " ]]; then
  FLAGS="$FLAGS -fcheck=bounds"
fi

gfortran $FLAGS -o fortran/libsnow17shim.so \
  external/snow17/src/snow19/*.f fortran/snow17_shim.f90
echo "Built fortran/libsnow17shim.so"
