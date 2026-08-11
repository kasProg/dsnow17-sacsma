#!/usr/bin/env bash
# Build fortran/libsacsmashim.so against the vendored sac-sma sources.
#
# Links only external/sac-sma/src/sac/{sac_data_mod,sac1,ex_sac1}.f90 --
# confirmed via sac-sma's own CMakeLists.txt MODEL_SOURCES -- plus our
# shim. Never links src/share, src/bmi, src/driver, or duamel.f (unit
# hydrograph routing; confirmed NOT part of the reference driver's
# per-timestep call chain -- see notes/logs.md).
#
# Applies patches/sac1_bypass_ratio_check_save_fix.patch to a BUILD-TIME
# COPY of sac1.f90, staged under fortran/_sacsma_patched/ (gitignored).
# external/sac-sma itself is never modified -- `git submodule status`
# stays clean and byte-identical to the pinned upstream commit. See
# notes/NOTES.md for why this patch exists (a real, live implicit-SAVE
# bug, empirically confirmed, not a defensive/precautionary patch).
#
# Do NOT add -fdefault-real-8 or similar: EXSAC/SAC1 use explicit
# DOUBLE PRECISION already: our shim matches with c_double. (Also
# irrelevant here since sac-sma's kind is explicit, not default REAL --
# unlike snow17, where that flag would be actively dangerous.)
set -euo pipefail
cd "$(dirname "$0")/.."

SAC_SRC=external/sac-sma/src/sac
STAGE=fortran/_sacsma_patched

rm -rf "$STAGE"
mkdir -p "$STAGE"
cp "$SAC_SRC/sac_data_mod.f90" "$SAC_SRC/sac1.f90" "$SAC_SRC/ex_sac1.f90" "$STAGE/"
patch -p2 -d "$STAGE" < patches/sac1_bypass_ratio_check_save_fix.patch

FLAGS="-shared -fPIC -O2"
if [[ "${BOUNDS_CHECK:-0} " == "1 " ]]; then
  FLAGS="$FLAGS -fcheck=bounds"
fi

# Order matters: sac_data_mod.f90 (the module) must compile before the
# files that USE it, in this single gfortran invocation.
gfortran $FLAGS -o fortran/libsacsmashim.so \
  "$STAGE/sac_data_mod.f90" "$STAGE/sac1.f90" "$STAGE/ex_sac1.f90" \
  fortran/sacsma_shim.f90
echo "Built fortran/libsacsmashim.so (sac1.f90 patched at build time; see patches/)"
