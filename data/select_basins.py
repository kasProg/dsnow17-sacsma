"""Select snow-dominated CAMELS basins for training/held-out evaluation.

Target: 30-50 snow-dominated CAMELS basins, with some held out.
"Snow-dominated" is operationalized here as high frac_snow (fraction of
precipitation falling as snow) from CAMELS' own climate-indices
attribute file -- the CAMELS authors' own climatology summary, not
something we're deriving ourselves.

Run after data/camels/camels_clim.txt, camels_topo.txt, camels_name.txt
have been downloaded (see notes/logs.md for the download source and
why the full ~3.4GB metForcing_obsFlow archive is needed even though we
only use a subset of basins -- CAMELS doesn't offer a per-basin
download, only the bulk archive).

Writes data/camels/selected_basins.csv: gauge_id, gauge_name, frac_snow,
elev_mean, gauge_lat, gauge_lon, area_gages2, split (train/heldout).
"""

from pathlib import Path

import numpy as np
import pandas as pd

CAMELS_DIR = Path(__file__).resolve().parent / "camels"

# Top N by frac_snow, split ~80/20 train/held-out. N=45 sits in the
# middle of the 30-50 target range. A minimum area filter avoids
# tiny/flashy headwater catchments (a handful of CAMELS' highest-frac_snow
# basins are <5 sq mi) that are more likely to be numerically finicky for
# reasons unrelated to what we're actually testing (gradient flow through
# two Fortran models). Fixed seed for a reproducible split.
N_BASINS = 45
N_HELDOUT = 10
MIN_AREA_SQKM = 20.0
SEED = 0


def select_basins() -> pd.DataFrame:
    clim = pd.read_csv(CAMELS_DIR / "camels_clim.txt", sep=";", dtype={"gauge_id": str})
    topo = pd.read_csv(CAMELS_DIR / "camels_topo.txt", sep=";", dtype={"gauge_id": str})
    name = pd.read_csv(CAMELS_DIR / "camels_name.txt", sep=";", dtype={"gauge_id": str})

    df = clim.merge(topo, on="gauge_id").merge(name, on="gauge_id")
    df["gauge_id"] = df["gauge_id"].str.zfill(8)  # CAMELS gauge IDs are 8-digit, zero-padded

    df = df[df["area_gages2"] >= MIN_AREA_SQKM]
    top = df.sort_values("frac_snow", ascending=False).head(N_BASINS).reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    heldout_idx = rng.choice(len(top), size=N_HELDOUT, replace=False)
    top["split"] = "train"
    top.loc[heldout_idx, "split"] = "heldout"

    cols = [
        "gauge_id", "gauge_name", "huc_02", "frac_snow", "pet_mean", "aridity",
        "elev_mean", "gauge_lat", "gauge_lon", "area_gages2", "split",
    ]
    return top[cols].sort_values("frac_snow", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    result = select_basins()
    out_path = CAMELS_DIR / "selected_basins.csv"
    result.to_csv(out_path, index=False)
    print(f"Selected {len(result)} basins ({(result.split == 'train').sum()} train, "
          f"{(result.split == 'heldout').sum()} heldout) -> {out_path}")
    print(result[["gauge_id", "gauge_name", "frac_snow", "split"]].to_string(index=False))
