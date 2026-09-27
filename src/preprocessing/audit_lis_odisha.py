from pathlib import Path
import sys
import glob
import os

import numpy as np
import xarray as xr


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIS_DIR = (
    PROJECT_ROOT
    / "restdataset"
    / "DATA,RAW,LIGHTNING NASA_ISS_LIS 2020-05-01_03"
)


# ============================================================
# ODISHА PROTOTYPE DOMAIN
# ============================================================

LAT_MIN = 20.0
LAT_MAX = 21.0
LON_MIN = 86.0
LON_MAX = 88.0


# ============================================================
# FIND LIS FILES
# ============================================================

files = sorted(LIS_DIR.glob("*.nc"))

print("=" * 120)
print("LIS → ODISHA DOMAIN AUDIT")
print("=" * 120)

print(f"LIS directory : {LIS_DIR}")
print(f"Total files   : {len(files)}")
print(
    f"Domain        : {LAT_MIN}–{LAT_MAX}°N, "
    f"{LON_MIN}–{LON_MAX}°E"
)

print()


# ============================================================
# HEADER
# ============================================================

print(
    f"{'FILE':52} "
    f"{'TOTAL':>8} "
    f"{'DOMAIN':>9} "
    f"{'LAT RANGE':>20} "
    f"{'LON RANGE':>20}"
)

print("-" * 120)


# ============================================================
# AUDIT EACH FILE
# ============================================================

domain_files = []

for path in files:

    try:
        ds = xr.open_dataset(path)

        lat = ds["lightning_flash_lat"].values
        lon = ds["lightning_flash_lon"].values

        # Remove invalid values
        valid = np.isfinite(lat) & np.isfinite(lon)

        lat_valid = lat[valid]
        lon_valid = lon[valid]

        if len(lat_valid) == 0:
            print(
                f"{path.name:52} "
                f"{0:8} "
                f"{0:9} "
                f"{'N/A':>20} "
                f"{'N/A':>20}"
            )
            ds.close()
            continue

        # Domain mask
        inside = (
            (lat_valid >= LAT_MIN)
            & (lat_valid <= LAT_MAX)
            & (lon_valid >= LON_MIN)
            & (lon_valid <= LON_MAX)
        )

        total_flashes = len(lat_valid)
        domain_flashes = int(inside.sum())

        lat_range = (
            f"{lat_valid.min():.2f}–{lat_valid.max():.2f}"
        )

        lon_range = (
            f"{lon_valid.min():.2f}–{lon_valid.max():.2f}"
        )

        print(
            f"{path.name:52} "
            f"{total_flashes:8} "
            f"{domain_flashes:9} "
            f"{lat_range:>20} "
            f"{lon_range:>20}"
        )

        if domain_flashes > 0:
            domain_files.append(
                (path.name, domain_flashes)
            )

        ds.close()

    except Exception as exc:

        print(
            f"{path.name:52} ERROR: {exc}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 120)
print("SUMMARY")
print("=" * 120)

print(
    f"Files containing at least one Odisha-domain flash: "
    f"{len(domain_files)}"
)

if domain_files:

    print()
    print("Files with Odisha-domain lightning:")

    total_domain_flashes = 0

    for name, count in domain_files:
        print(f"  {name} → {count} flashes")
        total_domain_flashes += count

    print()
    print(
        f"Total flashes inside domain across these files: "
        f"{total_domain_flashes}"
    )

else:

    print()
    print("No LIS flashes were found inside the Odisha domain.")


print()
print("=" * 120)
print("AUDIT COMPLETE")
print("=" * 120)