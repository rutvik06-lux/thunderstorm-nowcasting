from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIS_DIR = (
    PROJECT_ROOT
    / "restdataset"
    / "DATA,RAW,LIGHTNING NASA_ISS_LIS 2020-05-01_03"
)


# ============================================================
# ODISHA PROTOTYPE DOMAIN
# ============================================================

LAT_MIN = 20.0
LAT_MAX = 21.0
LON_MIN = 86.0
LON_MAX = 88.0


# ============================================================
# FIND FILES
# ============================================================

files = sorted(LIS_DIR.glob("*.nc"))


print("=" * 120)
print("LIS → ODISHA OBSERVATION COVERAGE AUDIT")
print("=" * 120)

print(f"Files : {len(files)}")
print(
    f"Domain: {LAT_MIN}–{LAT_MAX}°N, "
    f"{LON_MIN}–{LON_MAX}°E"
)

print()


# ============================================================
# HEADER
# ============================================================

print(
    f"{'FILE':52} "
    f"{'VIEWTIME':>10} "
    f"{'IN DOMAIN':>10} "
    f"{'MAX OBS(s)':>12} "
    f"{'START':>22} "
    f"{'END':>22}"
)

print("-" * 135)


coverage_files = []


# ============================================================
# AUDIT
# ============================================================

for path in files:

    try:

        ds = xr.open_dataset(path)

        lat = ds["viewtime_lat"].values
        lon = ds["viewtime_lon"].values

        start = ds["viewtime_TAI93_start"].values
        end = ds["viewtime_TAI93_end"].values

        effective_obs = ds["viewtime_effective_obs"].values

        valid = (
            np.isfinite(lat)
            & np.isfinite(lon)
        )

        inside = (
            valid
            & (lat >= LAT_MIN)
            & (lat <= LAT_MAX)
            & (lon >= LON_MIN)
            & (lon <= LON_MAX)
        )

        count_total = int(valid.sum())
        count_domain = int(inside.sum())

        if count_domain > 0:

            obs = effective_obs[inside]

            finite_obs = obs[np.isfinite(obs)]

            if len(finite_obs) > 0:
                max_obs = float(finite_obs.max())
            else:
                max_obs = 0.0

            start_domain = start[inside]
            end_domain = end[inside]

            start_text = str(start_domain.min())[:19]
            end_text = str(end_domain.max())[:19]

            coverage_files.append(
                (
                    path.name,
                    count_domain,
                    max_obs,
                    start_text,
                    end_text,
                )
            )

        else:

            max_obs = 0.0
            start_text = "N/A"
            end_text = "N/A"

        print(
            f"{path.name:52} "
            f"{count_total:10} "
            f"{count_domain:10} "
            f"{max_obs:12.2f} "
            f"{start_text:>22} "
            f"{end_text:>22}"
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
    f"LIS files observing the Odisha domain: "
    f"{len(coverage_files)}"
)

if coverage_files:

    print()
    print("Observation passes:")

    total_records = 0

    for (
        name,
        count,
        max_obs,
        start,
        end,
    ) in coverage_files:

        total_records += count

        print()
        print(f"  {name}")
        print(f"    Viewtime records : {count}")
        print(f"    Max effective obs: {max_obs:.2f} seconds")
        print(f"    Start             : {start}")
        print(f"    End               : {end}")

    print()
    print(
        f"Total Odisha viewtime records: "
        f"{total_records}"
    )

else:

    print()
    print("No LIS observation coverage found.")


print()
print("=" * 120)
print("COVERAGE AUDIT COMPLETE")
print("=" * 120)