"""Loads per-basin CAMELS forcing + observed streamflow, and derives
daily PET (not provided directly by CAMELS' Daymet forcing -- see
notes/logs.md).

File formats confirmed by reading actual extracted files directly, not
assumed from documentation:

Forcing (basin_mean_forcing/daymet/{HUC}/{gauge_id}_lump_cida_forcing_leap.txt):
    line 1: basin-mean latitude (deg)
    line 2: basin-mean elevation (m)
    line 3: basin area (m^2)
    line 4: column header
    data:   Year Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)
    whitespace-separated, one row per day, 1980-01-01 through 2014-12-31.

Streamflow (usgs_streamflow/{HUC}/{gauge_id}_streamflow_qc.txt):
    no header. columns: gauge_id year month day discharge_cfs qc_flag
    whitespace-separated. Missing days: discharge = -999.00, flag = "M"
    (confirmed by grep across the archive, e.g. gauge 01013500 2014-10).
    Observed flags in this archive: "A" (approved), "A:e" (approved,
    estimated), "M" (missing).

HUC subfolder does NOT reliably match the gauge_id's own leading digits
(e.g. gauge 06614800's forcing file lives under .../daymet/10/, not
.../daymet/06/) -- located by glob, not by computing the HUC from the ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CAMELS_DIR = Path(__file__).resolve().parent / "camels"
DATASET_DIR = CAMELS_DIR / "basin_dataset_public_v1p2"

CFS_TO_MM_PER_DAY_PER_KM2 = 2.446575  # see module docstring derivation in notes/logs.md

# Hamon (1963) PET, as given by USACE HEC-HMS's technical reference
# (authoritative, unambiguous source for the exact constants -- several
# variants with different-looking constants circulate in the literature,
# all algebraically equivalent once you track units through consistently).
_HAMON_C = 0.1651  # mm/day per g/m^3, at N=12hr daylight


def _find_file(subdir: Path, gauge_id: str, suffix_glob: str) -> Path:
    matches = list(subdir.glob(f"*/{gauge_id}{suffix_glob}"))
    if not matches:
        raise FileNotFoundError(f"No file matching {gauge_id}{suffix_glob} under {subdir}")
    assert len(matches) == 1, f"multiple matches for {gauge_id}: {matches}"
    return matches[0]


def hamon_pet(tmax_c: np.ndarray, tmin_c: np.ndarray, dayl_s: np.ndarray) -> np.ndarray:
    """Daily PET (mm/day) via Hamon (1963). tmax_c/tmin_c: deg C.
    dayl_s: day length, seconds (Daymet's own `dayl(s)` column -- already
    accounts for the basin's latitude via Daymet's own solar geometry
    model, so no separate latitude term is needed here)."""
    t_mean = (tmax_c + tmin_c) / 2.0
    n_hours = dayl_s / 3600.0
    e_s_mb = 6.108 * np.exp(17.27 * t_mean / (t_mean + 237.3))  # saturation vapor pressure, mb
    p_t = 216.7 * e_s_mb / (t_mean + 273.3)  # saturation vapor density, g/m^3
    pet = _HAMON_C * (n_hours / 12.0) * p_t
    return np.maximum(pet, 0.0)  # guard against a pathological negative from extreme T


@dataclass
class BasinTimeseries:
    dates: pd.DatetimeIndex
    prcp: np.ndarray       # mm/day
    tmax: np.ndarray       # deg C
    tmin: np.ndarray       # deg C
    tmean: np.ndarray      # deg C, (tmax+tmin)/2 -- what we feed as Snow17's/SAC-SMA's TMP
    pet: np.ndarray        # mm/day, Hamon
    q_obs: np.ndarray      # mm/day, NaN where missing/QC-flagged "M"
    lat: float
    elev: float
    area_km2: float


def load_basin_timeseries(gauge_id: str) -> BasinTimeseries:
    forcing_path = _find_file(
        DATASET_DIR / "basin_mean_forcing" / "daymet", gauge_id, "_lump_cida_forcing_leap.txt"
    )
    with open(forcing_path) as f:
        lat = float(f.readline().strip())
        elev = float(f.readline().strip())
        area_m2 = float(f.readline().strip())
    forcing = pd.read_csv(forcing_path, sep=r"\s+", skiprows=3)
    dates = pd.to_datetime(
        forcing[["Year", "Mnth", "Day"]].rename(columns={"Mnth": "month", "Day": "day", "Year": "year"})
    )
    prcp = forcing["prcp(mm/day)"].to_numpy(dtype=np.float64)
    tmax = forcing["tmax(C)"].to_numpy(dtype=np.float64)
    tmin = forcing["tmin(C)"].to_numpy(dtype=np.float64)
    dayl = forcing["dayl(s)"].to_numpy(dtype=np.float64)
    tmean = (tmax + tmin) / 2.0
    pet = hamon_pet(tmax, tmin, dayl)

    area_km2 = area_m2 / 1e6

    flow_path = _find_file(DATASET_DIR / "usgs_streamflow", gauge_id, "_streamflow_qc.txt")
    flow = pd.read_csv(
        flow_path, sep=r"\s+", header=None,
        names=["gauge_id", "year", "month", "day", "q_cfs", "flag"],
        dtype={"gauge_id": str},
    )
    flow_dates = pd.to_datetime(flow[["year", "month", "day"]])
    q_mm_day = flow["q_cfs"].to_numpy(dtype=np.float64) * CFS_TO_MM_PER_DAY_PER_KM2 / area_km2
    q_mm_day[(flow["flag"] == "M").to_numpy() | (flow["q_cfs"].to_numpy() < 0)] = np.nan

    # Align forcing and streamflow on date (both nominally span 1980-01-01
    # to 2014-12-31 daily, but don't assume -- merge explicitly).
    forcing_df = pd.DataFrame({
        "date": dates, "prcp": prcp, "tmax": tmax, "tmin": tmin,
        "tmean": tmean, "pet": pet,
    })
    flow_df = pd.DataFrame({"date": flow_dates, "q_obs": q_mm_day})
    merged = forcing_df.merge(flow_df, on="date", how="inner").sort_values("date")

    return BasinTimeseries(
        dates=pd.DatetimeIndex(merged["date"]),
        prcp=merged["prcp"].to_numpy(),
        tmax=merged["tmax"].to_numpy(),
        tmin=merged["tmin"].to_numpy(),
        tmean=merged["tmean"].to_numpy(),
        pet=merged["pet"].to_numpy(),
        q_obs=merged["q_obs"].to_numpy(),
        lat=lat, elev=elev, area_km2=area_km2,
    )
