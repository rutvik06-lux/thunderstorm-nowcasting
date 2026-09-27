from pathlib import Path

import h5py
import numpy as np
import xarray as xr

# ============================================================

# PROJECT PATHS

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
PROJECT_ROOT
/ "restdataset"
/ "DATA,RAW,GPM_IMERG 2020-05-01_03"
)

OUTPUT_DIR = (
PROJECT_ROOT
/ "data"
/ "processed"
/ "imerg"
)

OUTPUT_FILE = (
OUTPUT_DIR
/ "imerg_odisha_27x27_20200501_20200503.nc"
)

# ============================================================

# COMMON ODisha PROTOTYPE GRID

# ============================================================

#

# This MUST match the geographic domain used by:

#

# INSAT

# LIS

# ERA5

#

# All modalities will eventually use these same coordinates.

# ============================================================

LAT_MIN = 20.0
LAT_MAX = 21.0

LON_MIN = 86.0
LON_MAX = 88.0

TARGET_SIZE = 27

# ============================================================

# IMERG TIME REFERENCE

# ============================================================

IMERG_EPOCH = np.datetime64("1980-01-06T00:00:00")

# ============================================================

# BUILD COMMON GEOGRAPHIC GRID

# ============================================================


def build_target_grid():
    """Build the common Odisha geographic grid."""
    target_lat = np.linspace(
        LAT_MIN,
        LAT_MAX,
        TARGET_SIZE,
        dtype=np.float32,
    )

    target_lon = np.linspace(
        LON_MIN,
        LON_MAX,
        TARGET_SIZE,
        dtype=np.float32,
    )

    return target_lat, target_lon


# ============================================================

# INTERPOLATION

# ============================================================


def interpolate_to_target(
    data,
    source_lat,
    source_lon,
    target_lat,
    target_lon,
):
    """
    Bilinear interpolation from native IMERG grid
    to the common 27x27 geographic grid.

    Input:
        data       -> [lat, lon]
        source_lat -> 1D
        source_lon -> 1D

    Output:
        [target_lat, target_lon]
    """

    source_lat = np.asarray(source_lat, dtype=np.float64)
    source_lon = np.asarray(source_lon, dtype=np.float64)
    data = np.asarray(data, dtype=np.float32)

    # --------------------------------------------------------
    # Make sure latitude/longitude are increasing.
    # --------------------------------------------------------

    if source_lat[0] > source_lat[-1]:
        source_lat = source_lat[::-1]
        data = data[::-1, :]

    if source_lon[0] > source_lon[-1]:
        source_lon = source_lon[::-1]
        data = data[:, ::-1]

    # --------------------------------------------------------
    # Longitude interpolation
    # --------------------------------------------------------

    temp = np.full(
        (
            len(source_lat),
            len(target_lon),
        ),
        np.nan,
        dtype=np.float32,
    )

    for i in range(len(source_lat)):
        row = data[i, :]
        valid = np.isfinite(row)

        if np.count_nonzero(valid) < 2:
            continue

        temp[i, :] = np.interp(
            target_lon,
            source_lon[valid],
            row[valid],
            left=np.nan,
            right=np.nan,
        )

    # --------------------------------------------------------
    # Latitude interpolation
    # --------------------------------------------------------

    output = np.full(
        (
            len(target_lat),
            len(target_lon),
        ),
        np.nan,
        dtype=np.float32,
    )

    for j in range(len(target_lon)):
        column = temp[:, j]
        valid = np.isfinite(column)

        if np.count_nonzero(valid) < 2:
            continue

        output[:, j] = np.interp(
            target_lat,
            source_lat[valid],
            column[valid],
            left=np.nan,
            right=np.nan,
        )

    return output


# ============================================================

# READ ONE IMERG FILE

# ============================================================


def read_imerg_file(
    path,
    target_lat,
    target_lon,
):
    """Read one IMERG HDF5 file and regrid it to the target grid."""
    with h5py.File(path, "r") as f:

        # ----------------------------------------------------
        # Native coordinates
        # ----------------------------------------------------

        lat = np.asarray(f["Grid/lat"][:], dtype=np.float32)
        lon = np.asarray(f["Grid/lon"][:], dtype=np.float32)

        # ----------------------------------------------------
        # Native precipitation
        #
        # IMERG sample:
        #
        # [time, lon, lat]
        #
        # One time slice per HDF5 file.
        # ----------------------------------------------------

        precipitation = np.asarray(f["Grid/precipitation"][0], dtype=np.float32)

        # Convert:
        #
        # [lon, lat]
        #
        # ->
        #
        # [lat, lon]
        #
        precipitation = precipitation.T

        # ----------------------------------------------------
        # Invalid / fill values
        # ----------------------------------------------------

        precipitation[precipitation <= -9990] = np.nan
        precipitation[precipitation < 0] = np.nan

        # ----------------------------------------------------
        # Select Odisha domain
        # ----------------------------------------------------

        lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
        lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)

        source_lat = lat[lat_mask]
        source_lon = lon[lon_mask]

        regional = precipitation[np.ix_(lat_mask, lon_mask)]

        if regional.size == 0:
            raise RuntimeError(
                "No IMERG pixels found inside "
                f"Odisha domain: {path.name}"
            )

        if len(source_lat) < 2 or len(source_lon) < 2:
            raise RuntimeError(
                "Not enough native IMERG pixels "
                f"for interpolation: {path.name}"
            )

        # ----------------------------------------------------
        # Interpolate onto common geographic grid
        # ----------------------------------------------------

        output = interpolate_to_target(
            regional,
            source_lat,
            source_lon,
            target_lat,
            target_lon,
        )

        # ----------------------------------------------------
        # Availability mask
        # ----------------------------------------------------

        availability = np.isfinite(output).astype(np.uint8)

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        time_values = np.asarray(f["Grid/time"][:])

        if time_values.size == 0:
            raise RuntimeError(f"No time value found: {path.name}")

        time_seconds = int(time_values.reshape(-1)[0])
        timestamp = IMERG_EPOCH + np.timedelta64(time_seconds, "s")

        return timestamp, output.astype(np.float32), availability


# ============================================================

# MAIN

# ============================================================


def main():
    print("=" * 70)
    print("IMERG -> ODISHA 27x27 ADAPTER")
    print("=" * 70)

    # --------------------------------------------------------
    # Check raw directory
    # --------------------------------------------------------

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"\nIMERG directory not found:\n{RAW_DIR}")

    files = sorted(RAW_DIR.glob("*.HDF5"))

    if not files:
        raise FileNotFoundError(f"\nNo IMERG HDF5 files found:\n{RAW_DIR}")

    print(f"\n[INFO] Raw directory:")
    print(f"       {RAW_DIR}")
    print(f"[INFO] IMERG files found: {len(files)}")

    # --------------------------------------------------------
    # Common geographic grid
    # --------------------------------------------------------

    target_lat, target_lon = build_target_grid()

    print("\n[INFO] Common geographic grid:")
    print(f"       Size      : {TARGET_SIZE} x {TARGET_SIZE}")
    print(f"       Latitude  : {target_lat[0]:.6f} -> {target_lat[-1]:.6f}")
    print(f"       Longitude : {target_lon[0]:.6f} -> {target_lon[-1]:.6f}")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    timestamps = []
    rainfall_frames = []
    availability_frames = []

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    for index, path in enumerate(files, start=1):
        print(f"\n[{index:03d}/{len(files):03d}] {path.name}")

        try:
            timestamp, rainfall, availability = read_imerg_file(
                path,
                target_lat,
                target_lon,
            )

            timestamps.append(timestamp)
            rainfall_frames.append(rainfall)
            availability_frames.append(availability)

            valid = rainfall[np.isfinite(rainfall)]

            print(f"       time={timestamp}")

            if valid.size > 0:
                print(f"       max={float(valid.max()):.3f} mm/hr")
                print(f"       mean={float(valid.mean()):.3f} mm/hr")
                print(f"       valid={int(availability.sum())}/{availability.size}")
            else:
                print("       WARNING: no valid rainfall pixels")

        except Exception as exc:
            print(f"       ERROR: {exc}")

    # --------------------------------------------------------
    # Check successful processing
    # --------------------------------------------------------

    if not rainfall_frames:
        raise RuntimeError("\nNo IMERG files were successfully processed.")

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    rainfall_array = np.stack(rainfall_frames, axis=0)
    availability_array = np.stack(availability_frames, axis=0)
    timestamps = np.asarray(timestamps, dtype="datetime64[s]")

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    rainfall_array = rainfall_array[order]
    availability_array = availability_array[order]

    # --------------------------------------------------------
    # Remove duplicate timestamps
    #
    # This prevents accidental duplicate source files from
    # creating repeated temporal frames.
    # --------------------------------------------------------

    unique_times, unique_indices = np.unique(timestamps, return_index=True)

    if len(unique_times) != len(timestamps):
        print("\n[INFO] Duplicate timestamps detected.")
        print(f"       Original frames: {len(timestamps)}")
        print(f"       Unique frames  : {len(unique_times)}")

        timestamps = unique_times
        rainfall_array = rainfall_array[unique_indices]
        availability_array = availability_array[unique_indices]

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Delete old output if present
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    # --------------------------------------------------------
    # Create xarray dataset
    # --------------------------------------------------------

    dataset = xr.Dataset(
        data_vars={
            "precipitation": (("time", "y", "x"), rainfall_array),
            "precipitation_available": (("time", "y", "x"), availability_array),
        },
        coords={
            "time": timestamps,
            "latitude": ("y", target_lat),
            "longitude": ("x", target_lon),
        },
        attrs={
            "description": (
                "NASA GPM IMERG precipitation "
                "regridded onto the common Odisha "
                "27x27 prototype grid."
            ),
            "source": "NASA GPM IMERG V07B",
            "native_resolution": "0.1 degree",
            "target_grid": "27x27",
            "domain": "20-21N, 86-88E",
            "time_reference": "UTC",
            "rainfall_units": "mm/hr",
            "spatial_interpolation": "bilinear",
            "common_grid_note": (
                "Latitude and longitude coordinates "
                "are shared with the Odisha prototype "
                "domain used by INSAT and LIS."
            ),
        },
    )

    # --------------------------------------------------------
    # Variable metadata
    # --------------------------------------------------------

    dataset["precipitation"].attrs.update(
        {
            "long_name": "IMERG precipitation rate",
            "units": "mm/hr",
            "missing_value": "NaN",
        }
    )

    dataset["precipitation_available"].attrs.update(
        {
            "long_name": "IMERG precipitation availability mask",
            "description": "1 = valid interpolated IMERG observation, 0 = unavailable.",
        }
    )

    dataset["latitude"].attrs.update(
        {
            "standard_name": "latitude",
            "units": "degrees_north",
        }
    )

    dataset["longitude"].attrs.update(
        {
            "standard_name": "longitude",
            "units": "degrees_east",
        }
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    dataset.to_netcdf(OUTPUT_FILE, engine="netcdf4")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IMERG PROCESSING COMPLETE")
    print("=" * 70)

    print(f"\nOutput:")
    print(f"  {OUTPUT_FILE}")

    print("\nDimensions:")
    print(f"  time={dataset.sizes['time']}")
    print(f"  y={dataset.sizes['y']}")
    print(f"  x={dataset.sizes['x']}")

    print("\nGeographic domain:")
    print(f"  latitude : {target_lat[0]:.6f} -> {target_lat[-1]:.6f}")
    print(f"  longitude: {target_lon[0]:.6f} -> {target_lon[-1]:.6f}")

    print("\nTime range:")
    print(f"  {timestamps[0]} -> {timestamps[-1]}")

    print("\nArray shape:")
    print(f"  precipitation: {rainfall_array.shape}")
    print(f"  availability : {availability_array.shape}")

    valid_values = rainfall_array[np.isfinite(rainfall_array)]

    if valid_values.size > 0:
        print("\nRainfall statistics:")
        print(f"  overall max : {float(valid_values.max()):.3f} mm/hr")
        print(f"  overall mean: {float(valid_values.mean()):.3f} mm/hr")
        print(f"  valid values: {valid_values.size}")

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()
