from pathlib import Path
import re
import xarray as xr
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

IMERG_DIR = ROOT / "data" / "processed" / "imerg"
LIS_ROOT = ROOT / "restdataset"

OUT = ROOT / "outputs" / "thunderstorm_event_candidates.csv"

LAT_MIN, LAT_MAX = 20.0, 21.0
LON_MIN, LON_MAX = 86.0, 88.0

IMERG_THRESHOLD = 5.0


def extract_date(name):
    match = re.search(r"(20\d{6})", name)

    if match:
        return pd.to_datetime(
            match.group(1),
            format="%Y%m%d"
        ).date()

    return None


def find_lis_files():
    files = []

    if not LIS_ROOT.exists():
        return files

    for path in LIS_ROOT.rglob("*.nc"):
        if "lis" in path.name.lower():
            files.append(path)

    return sorted(files)


def find_var(ds, names):
    for name in names:
        if name in ds.variables:
            return name

    return None


print("=" * 70)
print("THUNDERSTORM EVENT DISCOVERY")
print("=" * 70)

# =========================================================
# 1. DISCOVER AVAILABLE DATES FROM FILENAMES
# =========================================================

print("\n[1/3] Discovering available IMERG dates...")

imerg_files = sorted(
    IMERG_DIR.rglob("*.nc")
)

imerg_dates = {}

for path in imerg_files:

    date = extract_date(path.name)

    if date is not None:

        if date not in imerg_dates:
            imerg_dates[date] = []

        imerg_dates[date].append(path)


print(
    f"IMERG NetCDF files found: "
    f"{len(imerg_files)}"
)

for date in sorted(imerg_dates):

    print(
        f"  {date} | "
        f"{len(imerg_dates[date])} files"
    )


# =========================================================
# 2. LIS DATE INVENTORY
# =========================================================

print("\n[2/3] Discovering available LIS dates...")

lis_files = find_lis_files()

lis_dates = {}

for path in lis_files:

    date = extract_date(path.name)

    if date is not None:

        if date not in lis_dates:
            lis_dates[date] = []

        lis_dates[date].append(path)


print(
    f"LIS NetCDF files found: "
    f"{len(lis_files)}"
)

for date in sorted(lis_dates):

    print(
        f"  {date} | "
        f"{len(lis_dates[date])} files"
    )


# =========================================================
# 3. ONLY INSPECT LIS FOR DATES WITH IMERG
# =========================================================

print("\n[3/3] Checking LIS lightning activity...")

common_dates = sorted(
    set(imerg_dates.keys())
    & set(lis_dates.keys())
)

print(
    f"\nDates with both IMERG + LIS: "
    f"{len(common_dates)}"
)

lis_counts = {}

for date in common_dates:

    total = 0

    for path in lis_dates[date]:

        try:

            ds = xr.open_dataset(
                path,
                decode_times=False
            )

            lat_name = find_var(
                ds,
                [
                    "flash_lat",
                    "flash_latitude",
                    "latitude",
                    "lat"
                ]
            )

            lon_name = find_var(
                ds,
                [
                    "flash_lon",
                    "flash_longitude",
                    "longitude",
                    "lon"
                ]
            )

            if lat_name is None or lon_name is None:
                ds.close()
                continue

            lat = np.asarray(
                ds[lat_name].values
            ).reshape(-1)

            lon = np.asarray(
                ds[lon_name].values
            ).reshape(-1)

            valid = (
                np.isfinite(lat)
                & np.isfinite(lon)
            )

            domain = (
                valid
                & (lat >= LAT_MIN)
                & (lat <= LAT_MAX)
                & (lon >= LON_MIN)
                & (lon <= LON_MAX)
            )

            total += int(
                np.sum(domain)
            )

            ds.close()

        except Exception:
            continue

    lis_counts[date] = total

    print(
        f"  {date} | "
        f"LIS domain flashes="
        f"{total}"
    )


# =========================================================
# 4. BUILD SIMPLE CANDIDATE TABLE
# =========================================================

rows = []

for date in sorted(
    set(imerg_dates.keys())
    | set(lis_dates.keys())
):

    has_imerg = date in imerg_dates
    has_lis = date in lis_dates

    lis_flashes = lis_counts.get(
        date,
        0
    )

    # We deliberately do NOT calculate a complicated
    # precipitation score here.
    #
    # Candidate = date has both datasets AND
    # LIS observed lightning.
    #
    # IMERG is retained as supporting data.

    candidate = (
        has_imerg
        and has_lis
        and lis_flashes > 0
    )

    rows.append({
        "date": date,
        "imerg_available": has_imerg,
        "lis_available": has_lis,
        "lis_domain_flashes": lis_flashes,
        "candidate": candidate
    })


df = pd.DataFrame(rows)


# =========================================================
# 5. RESULTS
# =========================================================

print("\n" + "=" * 70)
print("EVENT DISCOVERY RESULTS")
print("=" * 70)

if df.empty:

    print(
        "No dates were discovered."
    )

else:

    print(
        "\nDATE         IMERG    LIS     "
        "DOMAIN FLASHES    CANDIDATE"
    )

    print("-" * 70)

    for _, row in df.iterrows():

        print(
            f"{row['date']}   "
            f"{'YES' if row['imerg_available'] else 'NO ':>5}   "
            f"{'YES' if row['lis_available'] else 'NO ':>5}   "
            f"{int(row['lis_domain_flashes']):>14}   "
            f"{str(row['candidate']):>9}"
        )


# =========================================================
# 6. CANDIDATES
# =========================================================

candidates = df[
    df["candidate"] == True
].copy()


print("\n" + "=" * 70)
print("CANDIDATE EVENTS")
print("=" * 70)

if candidates.empty:

    print(
        "No additional candidate events "
        "were found."
    )

else:

    for _, row in candidates.iterrows():

        print(
            f"{row['date']} | "
            f"LIS flashes="
            f"{int(row['lis_domain_flashes'])}"
        )


# =========================================================
# 7. SAVE
# =========================================================

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUT,
    index=False
)

print(
    f"\nSaved inventory/candidate table:"
    f"\n{OUT}"
)

print("\nNEXT STEP:")

if candidates.empty:

    print(
        "No new event is currently available "
        "from the existing IMERG + LIS data."
    )

else:

    print(
        "Acquire INSAT + ERA5 for the "
        "candidate dates."
    )

print("=" * 70)
