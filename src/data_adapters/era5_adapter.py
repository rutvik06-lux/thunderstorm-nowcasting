from pathlib import Path

import numpy as np
import xarray as xr


ERA5_DIR = Path(r"restdataset\DATA,RAW,ERA5 2020-05-01_03")
INSTANT_FILE = ERA5_DIR / "data_stream-oper_stepType-instant.nc"

OUTPUT_DIR = Path(r"data\processed\era5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "era5_odisha_27x27_20200501_20200503.nc"


# ============================================================
# COMMON ODisha PROTOTYPE GRID
# ============================================================

LAT_MIN = 20.0
LAT_MAX = 21.0

LON_MIN = 86.0
LON_MAX = 88.0

GRID_SIZE = 27


# 27 latitude points from north to south
TARGET_LAT = np.linspace(
    LAT_MAX,
    LAT_MIN,
    GRID_SIZE,
    dtype=np.float32,
)

# 27 longitude points from west to east
TARGET_LON = np.linspace(
    LON_MIN,
    LON_MAX,
    GRID_SIZE,
    dtype=np.float32,
)


TARGET_LAT_2D, TARGET_LON_2D = np.meshgrid(
    TARGET_LAT,
    TARGET_LON,
    indexing="ij",
)


VARIABLES = [
    "t2m",
    "d2m",
    "u10",
    "v10",
    "msl",
    "sp",
    "tcc",
    "cape",
]


# ============================================================
# LOAD ERA5
# ============================================================

def load_era5():

    if not INSTANT_FILE.exists():
        raise FileNotFoundError(INSTANT_FILE)

    return xr.open_dataset(INSTANT_FILE)


# ============================================================
# REGRID ERA5 TO COMMON ODisha GRID
# ============================================================

def regrid_era5_to_common_grid(ds):

    lat_points = xr.DataArray(
        TARGET_LAT,
        dims="y",
    )

    lon_points = xr.DataArray(
        TARGET_LON,
        dims="x",
    )

    output = {}

    for variable in VARIABLES:

        print(f"Regridding {variable}...")

        values = ds[variable].interp(
            latitude=lat_points,
            longitude=lon_points,
            method="linear",
        )

        output[variable] = values.values.astype(
            np.float32
        )

    return output


# ============================================================
# BUILD OUTPUT DATASET
# ============================================================

def build_dataset(ds, data):

    dataset = xr.Dataset(
        {
            variable: (
                ("time", "y", "x"),
                values,
            )
            for variable, values in data.items()
        },

        coords={
            "time": ds.valid_time.values,

            "latitude": (
                ("y", "x"),
                TARGET_LAT_2D,
            ),

            "longitude": (
                ("y", "x"),
                TARGET_LON_2D,
            ),
        },

        attrs={
            "source": "ERA5",
            "source_dataset": "ECMWF ERA5",

            "target_grid": (
                "Common 27x27 Odisha prototype grid"
            ),

            "spatial_interpolation": "linear",

            "temporal_resolution": "hourly",

            "region": (
                "Odisha prototype domain "
                "(20-21N, 86-88E)"
            ),
        },
    )

    return dataset


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ERA5 -> COMMON ODisha GRID -> SAVED DATASET")
    print("=" * 60)

    print("\nTarget grid:")

    print(f"Latitude : {LAT_MIN} -> {LAT_MAX}")
    print(f"Longitude: {LON_MIN} -> {LON_MAX}")
    print(f"Grid     : {GRID_SIZE} x {GRID_SIZE}")

    print("\nLoading ERA5...")

    ds = load_era5()

    print("\nRegridding ERA5...")

    data = regrid_era5_to_common_grid(ds)

    output_ds = build_dataset(
        ds,
        data,
    )

    print("\nSaving:")
    print(OUTPUT_FILE)

    output_ds.to_netcdf(
        OUTPUT_FILE,
        engine="netcdf4",
    )

    print("\nSaved successfully.")

    print("\nDataset:")
    print(output_ds)

    print("\nOutput size:")

    print(
        f"{OUTPUT_FILE.stat().st_size / (1024 ** 2):.2f} MB"
    )

    ds.close()
    output_ds.close()


if __name__ == "__main__":
    main()