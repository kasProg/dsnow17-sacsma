#!/usr/bin/env bash
# Downloads the CAMELS data this project needs: small basin-attribute
# text files (climate indices, topography, soil, vegetation, geology,
# names) plus the ~3.4GB forcing+observed-streamflow archive.
#
# Source: https://dx.doi.org/10.5065/D6MW2F4D (CAMELS v2.0, UCAR),
# mirrored on Zenodo at https://zenodo.org/records/15529996 -- there is
# no per-basin download; the forcing/flow archive is one bulk file
# covering all 671 basins even though data/select_basins.py only uses a
# ~45-basin subset (see that script for the snow-dominated selection).
#
# Not run automatically by `make test` -- this is a one-time, ~3.4GB
# download, deliberately separate from the fast local test loop.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p camels
cd camels

BASE_URL="https://zenodo.org/records/15529996/files"

for f in camels_clim.txt camels_topo.txt camels_soil.txt camels_vege.txt camels_geol.txt camels_hydro.txt camels_name.txt; do
  if [[ ! -f "$f" ]]; then
    echo "Downloading $f..."
    curl -sL "${BASE_URL}/${f}?download=1" -o "$f"
  fi
done

ZIP=basin_timeseries_v1p2_metForcing_obsFlow.zip
if [[ ! -f "$ZIP" ]]; then
  echo "Downloading $ZIP (~3.4GB, this takes a while)..."
  curl -L "${BASE_URL}/${ZIP}?download=1" -o "$ZIP"
fi

if [[ ! -d basin_dataset_public_v1p2 ]]; then
  echo "Extracting $ZIP..."
  unzip -q "$ZIP"
fi

echo "Done. Run data/select_basins.py to (re)generate data/camels/selected_basins.csv."
