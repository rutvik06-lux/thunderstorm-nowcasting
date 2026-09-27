
"""
SIH26072YELLOW
==============================================================

REAL MULTIMODAL THUNDERSTORM / LIGHTNING NOWCASTING DATASET

Current prototype event
-----------------------
Region      : Odisha
Latitude    : 20N - 21N
Longitude   : 86E - 88E
Grid        : 27 x 27
Cadence     : 30 minutes
Date        : 01 May 2020
Time        : 04:00 - 06:30 UTC

Modalities
----------
1. INSAT-3D
2. GPM IMERG
3. ERA5
4. NASA ISS LIS

Important
---------
The source files are already spatially processed to the common
27 x 27 model grid.

Therefore this builder:

    DOES NOT spatially interpolate IMERG/ERA5/LIS again.

It only:

    1. normalizes coordinate orientation
    2. selects the common time period
    3. interpolates ERA5 in TIME from hourly to 30-minute cadence
    4. merges all modalities
    5. validates the final 18-channel model input

No synthetic atmospheric or lightning observations are generated.
"""


from pathlib import Path
import traceback

import numpy as np
import xarray as xr


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INSAT_FILE = (
    PROCESSED_DIR
    / "insat"
    / "insat_odisha_27x27_20200501.nc"
)

IMERG_FILE = (
    PROCESSED_DIR
    / "imerg"
    / "imerg_odisha_27x27_20200501_20200503.nc"
)

ERA5_FILE = (
    PROCESSED_DIR
    / "era5"
    / "era5_odisha_27x27_20200501_20200503.nc"
)

LIS_FILE = (
    PROCESSED_DIR
    / "lis"
    / "lis_odisha_27x27_20200501.nc"
)

OUTPUT_DIR = (
    PROCESSED_DIR
    / "multimodal"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "odisha_multimodal_20200501.nc"
)


# ======================================================================
# COMMON MODEL GRID
# ======================================================================

TARGET_NY = 27
TARGET_NX = 27

# Canonical model orientation:
#
# latitude:
#     21.0 -> 20.0
#
# longitude:
#     86.0 -> 88.0

TARGET_LAT = np.linspace(
    21.0,
    20.0,
    TARGET_NY,
    dtype=np.float32,
)

TARGET_LON = np.linspace(
    86.0,
    88.0,
    TARGET_NX,
    dtype=np.float32,
)


# ======================================================================
# COMMON TIME AXIS
# ======================================================================

TARGET_TIMES = np.arange(
    np.datetime64("2020-05-01T04:00:00"),
    np.datetime64("2020-05-01T07:00:00"),
    np.timedelta64(30, "m"),
).astype("datetime64[ns]")


# ======================================================================
# EXPECTED 18 MODEL CHANNELS
# ======================================================================

EXPECTED_CHANNELS = [

    # INSAT
    "insat_tir1",
    "insat_tir2",
    "insat_wv",
    "insat_vis",
    "insat_availability",

    # IMERG
    "imerg_precipitation",
    "imerg_availability",

    # ERA5
    "era5_u10",
    "era5_v10",
    "era5_d2m",
    "era5_t2m",
    "era5_msl",
    "era5_sp",
    "era5_tcc",
    "era5_cape",
    "era5_availability",

    # LIS
    "lis_lightning_density",
    "lis_lightning_availability",
]


# ======================================================================
# GENERAL HELPERS
# ======================================================================

def section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check_file(path, name):

    if not path.exists():

        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )

    print(
        f"[OK] {name}: {path}"
    )


def normalize_time(ds, name):

    if "time" not in ds.coords:

        raise ValueError(
            f"{name}: time coordinate is missing."
        )

    ds = ds.assign_coords(
        time=ds["time"].values.astype(
            "datetime64[ns]"
        )
    )

    return ds


def print_grid(name, ds):

    print()
    print(f"{name}:")
    print(
        f"  dimensions = {dict(ds.sizes)}"
    )

    if "latitude" in ds.coords:

        lat = np.asarray(
            ds["latitude"].values
        )

        print(
            f"  latitude shape = {lat.shape}"
        )

        print(
            f"  latitude range = "
            f"{float(np.nanmin(lat)):.6f} -> "
            f"{float(np.nanmax(lat)):.6f}"
        )

    if "longitude" in ds.coords:

        lon = np.asarray(
            ds["longitude"].values
        )

        print(
            f"  longitude shape = {lon.shape}"
        )

        print(
            f"  longitude range = "
            f"{float(np.nanmin(lon)):.6f} -> "
            f"{float(np.nanmax(lon)):.6f}"
        )

    if "time" in ds.coords:

        if ds.sizes.get("time", 0) > 0:

            print(
                f"  time = "
                f"{ds.time.values[0]} -> "
                f"{ds.time.values[-1]}"
            )


# ======================================================================
# FIND GEOGRAPHIC COORDINATES
# ======================================================================

def get_lat_lon(ds, name):

    """
    Return latitude and longitude coordinate arrays.

    Supports:

        latitude(y)
        longitude(x)

    and:

        latitude(y,x)
        longitude(y,x)

    ERA5 commonly uses the second form.
    """

    if "latitude" not in ds.coords:

        raise ValueError(
            f"{name}: latitude coordinate missing."
        )

    if "longitude" not in ds.coords:

        raise ValueError(
            f"{name}: longitude coordinate missing."
        )

    lat = np.asarray(
        ds["latitude"].values,
        dtype=np.float64,
    )

    lon = np.asarray(
        ds["longitude"].values,
        dtype=np.float64,
    )

    return lat, lon


# ======================================================================
# NORMALIZE 1-D GEOGRAPHIC GRID
# ======================================================================

def normalize_1d_grid(
    ds,
    name,
    lat,
    lon,
):

    """
    Normalize a dataset where:

        latitude = latitude(y)
        longitude = longitude(x)

    Handles reversed latitude or longitude.
    """

    if lat.ndim != 1:

        raise ValueError(
            f"{name}: expected 1-D latitude."
        )

    if lon.ndim != 1:

        raise ValueError(
            f"{name}: expected 1-D longitude."
        )

    if len(lat) != TARGET_NY:

        raise ValueError(
            f"{name}: expected 27 latitude points, "
            f"got {len(lat)}"
        )

    if len(lon) != TARGET_NX:

        raise ValueError(
            f"{name}: expected 27 longitude points, "
            f"got {len(lon)}"
        )

    # --------------------------------------------------------------
    # Latitude
    # --------------------------------------------------------------

    if np.allclose(
        lat,
        TARGET_LAT,
        atol=1e-5,
    ):

        print(
            f"{name}: latitude = north -> south."
        )

    elif np.allclose(
        lat[::-1],
        TARGET_LAT,
        atol=1e-5,
    ):

        print(
            f"{name}: latitude = south -> north."
        )

        print(
            f"{name}: reversing y dimension."
        )

        ds = ds.isel(
            y=slice(None, None, -1)
        )

    else:

        raise ValueError(
            f"{name}: latitude values do not match "
            f"the expected 20-21N model grid.\n\n"
            f"Expected:\n{TARGET_LAT}\n\n"
            f"Got:\n{lat}"
        )

    # --------------------------------------------------------------
    # Longitude
    # --------------------------------------------------------------

    if np.allclose(
        lon,
        TARGET_LON,
        atol=1e-5,
    ):

        print(
            f"{name}: longitude = west -> east."
        )

    elif np.allclose(
        lon[::-1],
        TARGET_LON,
        atol=1e-5,
    ):

        print(
            f"{name}: longitude = east -> west."
        )

        print(
            f"{name}: reversing x dimension."
        )

        ds = ds.isel(
            x=slice(None, None, -1)
        )

    else:

        raise ValueError(
            f"{name}: longitude values do not match "
            f"the expected 86-88E model grid.\n\n"
            f"Expected:\n{TARGET_LON}\n\n"
            f"Got:\n{lon}"
        )

    # --------------------------------------------------------------
    # Assign clean model coordinates
    # --------------------------------------------------------------

    ds = ds.assign_coords(
        y=np.arange(TARGET_NY),
        x=np.arange(TARGET_NX),
    )

    ds = ds.assign_coords(
        latitude=(
            "y",
            TARGET_LAT,
        ),
        longitude=(
            "x",
            TARGET_LON,
        ),
    )

    return ds


# ======================================================================
# NORMALIZE 2-D GEOGRAPHIC GRID
# ======================================================================

def normalize_2d_grid(
    ds,
    name,
    lat,
    lon,
):

    """
    Normalize a dataset where latitude/longitude are 2-D:

        latitude(y,x)
        longitude(y,x)

    This is the important case for ERA5.

    We verify that the 2-D geographic coordinates represent the
    expected 27x27 grid.

    We then determine whether y/x need reversing.

    No spatial interpolation is performed.
    """

    if lat.ndim != 2:

        raise ValueError(
            f"{name}: expected 2-D latitude."
        )

    if lon.ndim != 2:

        raise ValueError(
            f"{name}: expected 2-D longitude."
        )

    expected_lat_2d = np.repeat(
        TARGET_LAT[:, None],
        TARGET_NX,
        axis=1,
    )

    expected_lon_2d = np.repeat(
        TARGET_LON[None, :],
        TARGET_NY,
        axis=0,
    )

    # --------------------------------------------------------------
    # Already canonical
    # --------------------------------------------------------------

    if (
        np.allclose(
            lat,
            expected_lat_2d,
            atol=1e-5,
        )
        and
        np.allclose(
            lon,
            expected_lon_2d,
            atol=1e-5,
        )
    ):

        print(
            f"{name}: 2-D geographic grid already "
            f"matches canonical orientation."
        )

        return ds

    # --------------------------------------------------------------
    # Try latitude reversed
    # --------------------------------------------------------------

    lat_y_reversed = expected_lat_2d[::-1, :]

    if (
        np.allclose(
            lat,
            lat_y_reversed,
            atol=1e-5,
        )
        and
        np.allclose(
            lon,
            expected_lon_2d,
            atol=1e-5,
        )
    ):

        print(
            f"{name}: 2-D latitude is south -> north."
        )

        print(
            f"{name}: reversing y dimension."
        )

        ds = ds.isel(
            y=slice(None, None, -1)
        )

        return ds

    # --------------------------------------------------------------
    # Try longitude reversed
    # --------------------------------------------------------------

    lon_x_reversed = expected_lon_2d[:, ::-1]

    if (
        np.allclose(
            lat,
            expected_lat_2d,
            atol=1e-5,
        )
        and
        np.allclose(
            lon,
            lon_x_reversed,
            atol=1e-5,
        )
    ):

        print(
            f"{name}: 2-D longitude is east -> west."
        )

        print(
            f"{name}: reversing x dimension."
        )

        ds = ds.isel(
            x=slice(None, None, -1)
        )

        return ds

    # --------------------------------------------------------------
    # Try both reversed
    # --------------------------------------------------------------

    if (
        np.allclose(
            lat,
            lat_y_reversed,
            atol=1e-5,
        )
        and
        np.allclose(
            lon,
            lon_x_reversed,
            atol=1e-5,
        )
    ):

        print(
            f"{name}: both latitude and longitude "
            f"orientations are reversed."
        )

        print(
            f"{name}: reversing y and x dimensions."
        )

        ds = ds.isel(
            y=slice(None, None, -1),
            x=slice(None, None, -1),
        )

        return ds

    # --------------------------------------------------------------
    # If we reach here, the geographic grid really differs.
    # --------------------------------------------------------------

    raise ValueError(
        f"{name}: 2-D geographic grid mismatch.\n\n"
        f"Latitude shape: {lat.shape}\n"
        f"Longitude shape: {lon.shape}\n\n"
        f"Latitude range: "
        f"{np.nanmin(lat):.6f} -> "
        f"{np.nanmax(lat):.6f}\n\n"
        f"Longitude range: "
        f"{np.nanmin(lon):.6f} -> "
        f"{np.nanmax(lon):.6f}"
    )


# ======================================================================
# UNIVERSAL GRID NORMALIZATION
# ======================================================================

def normalize_geographic_grid(
    ds,
    name,
):

    """
    Universal geographic-grid normalization.

    Supports both:

        1-D latitude/longitude

    and:

        2-D latitude/longitude

    The final output always uses:

        dimensions:
            y = 27
            x = 27

        coordinates:
            latitude(y)
            longitude(x)

    No spatial interpolation.
    """

    if "y" not in ds.dims:

        raise ValueError(
            f"{name}: y dimension missing.\n"
            f"Dimensions: {dict(ds.sizes)}"
        )

    if "x" not in ds.dims:

        raise ValueError(
            f"{name}: x dimension missing.\n"
            f"Dimensions: {dict(ds.sizes)}"
        )

    if ds.sizes["y"] != TARGET_NY:

        raise ValueError(
            f"{name}: expected y=27, "
            f"got {ds.sizes['y']}"
        )

    if ds.sizes["x"] != TARGET_NX:

        raise ValueError(
            f"{name}: expected x=27, "
            f"got {ds.sizes['x']}"
        )

    lat, lon = get_lat_lon(
        ds,
        name,
    )

    print(
        f"{name}: latitude dimensions = "
        f"{lat.shape}"
    )

    print(
        f"{name}: longitude dimensions = "
        f"{lon.shape}"
    )

    # --------------------------------------------------------------
    # 1-D geographic coordinates
    # --------------------------------------------------------------

    if lat.ndim == 1 and lon.ndim == 1:

        ds = normalize_1d_grid(
            ds,
            name,
            lat,
            lon,
        )

    # --------------------------------------------------------------
    # 2-D geographic coordinates
    # --------------------------------------------------------------

    elif lat.ndim == 2 and lon.ndim == 2:

        ds = normalize_2d_grid(
            ds,
            name,
            lat,
            lon,
        )

        # ----------------------------------------------------------
        # After spatial orientation is corrected, replace the
        # 2-D source coordinates with the canonical 1-D model
        # coordinates.
        # ----------------------------------------------------------

        ds = ds.assign_coords(
            y=np.arange(TARGET_NY),
            x=np.arange(TARGET_NX),
        )

        ds = ds.assign_coords(
            latitude=(
                "y",
                TARGET_LAT,
            ),
            longitude=(
                "x",
                TARGET_LON,
            ),
        )

    else:

        raise ValueError(
            f"{name}: unsupported latitude/longitude "
            f"dimensions.\n"
            f"latitude shape = {lat.shape}\n"
            f"longitude shape = {lon.shape}"
        )

    # --------------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------------

    final_lat = np.asarray(
        ds.latitude.values
    )

    final_lon = np.asarray(
        ds.longitude.values
    )

    if not np.allclose(
        final_lat,
        TARGET_LAT,
        atol=1e-5,
    ):

        raise ValueError(
            f"{name}: final latitude normalization failed."
        )

    if not np.allclose(
        final_lon,
        TARGET_LON,
        atol=1e-5,
    ):

        raise ValueError(
            f"{name}: final longitude normalization failed."
        )

    print(
        f"{name}: [PASS] canonical 27x27 grid."
    )

    return ds


# ======================================================================
# TIME NORMALIZATION
# ======================================================================

def select_target_times(
    ds,
    name,
):

    ds = normalize_time(
        ds,
        name,
    )

    available = ds.time.values

    missing = [
        t
        for t in TARGET_TIMES
        if t not in available
    ]

    if missing:

        raise ValueError(
            f"{name}: required prototype timestamps "
            f"are missing:\n{missing}"
        )

    return ds.sel(
        time=TARGET_TIMES
    )


# ======================================================================
# INSAT
# ======================================================================

def load_insat():

    section(
        "1. INSAT-3D"
    )

    check_file(
        INSAT_FILE,
        "INSAT",
    )

    ds = xr.open_dataset(
        INSAT_FILE
    )

    ds = normalize_time(
        ds,
        "INSAT",
    )

    print_grid(
        "INSAT source",
        ds,
    )

    required = [
        "IMG_TIR1",
        "IMG_TIR2",
        "IMG_WV",
        "IMG_VIS",
    ]

    missing = [
        v
        for v in required
        if v not in ds.data_vars
    ]

    if missing:

        ds.close()

        raise ValueError(
            f"INSAT missing variables: {missing}"
        )

    ds = ds[
        required
    ]

    ds = normalize_geographic_grid(
        ds,
        "INSAT",
    )

    ds = select_target_times(
        ds,
        "INSAT",
    )

    ds = ds.rename(
        {
            "IMG_TIR1":
                "insat_tir1",

            "IMG_TIR2":
                "insat_tir2",

            "IMG_WV":
                "insat_wv",

            "IMG_VIS":
                "insat_vis",
        }
    )

    ds["insat_availability"] = (
        (
            "time",
            "y",
            "x",
        ),
        np.ones(
            (
                len(TARGET_TIMES),
                TARGET_NY,
                TARGET_NX,
            ),
            dtype=np.uint8,
        ),
    )

    print(
        f"INSAT final: {dict(ds.sizes)}"
    )

    return ds


# ======================================================================
# IMERG
# ======================================================================

def load_imerg():

    section(
        "2. GPM IMERG"
    )

    check_file(
        IMERG_FILE,
        "IMERG",
    )

    ds = xr.open_dataset(
        IMERG_FILE
    )

    ds = normalize_time(
        ds,
        "IMERG",
    )

    print_grid(
        "IMERG source",
        ds,
    )

    if "precipitation" not in ds.data_vars:

        ds.close()

        raise ValueError(
            "IMERG precipitation variable missing."
        )

    availability_name = None

    for candidate in [
        "precipitation_available",
        "imerg_availability",
        "availability",
    ]:

        if candidate in ds.data_vars:

            availability_name = candidate
            break

    if availability_name is None:

        ds.close()

        raise ValueError(
            "IMERG availability variable missing."
        )

    ds = ds[
        [
            "precipitation",
            availability_name,
        ]
    ]

    if availability_name != "imerg_availability":

        ds = ds.rename(
            {
                availability_name:
                    "imerg_availability"
            }
        )

    # --------------------------------------------------------------
    # IMERG is already spatially processed.
    # This also handles its south -> north latitude orientation.
    # --------------------------------------------------------------

    ds = normalize_geographic_grid(
        ds,
        "IMERG",
    )

    ds = select_target_times(
        ds,
        "IMERG",
    )

    ds = ds.rename(
        {
            "precipitation":
                "imerg_precipitation",
        }
    )

    print(
        f"IMERG final: {dict(ds.sizes)}"
    )

    return ds


# ======================================================================
# ERA5
# ======================================================================

def load_era5():

    section(
        "3. ERA5"
    )

    check_file(
        ERA5_FILE,
        "ERA5",
    )

    ds = xr.open_dataset(
        ERA5_FILE
    )

    ds = normalize_time(
        ds,
        "ERA5",
    )

    print_grid(
        "ERA5 source",
        ds,
    )

    required = [
        "u10",
        "v10",
        "d2m",
        "t2m",
        "msl",
        "sp",
        "tcc",
        "cape",
    ]

    missing = [
        v
        for v in required
        if v not in ds.data_vars
    ]

    if missing:

        ds.close()

        raise ValueError(
            f"ERA5 missing variables: {missing}"
        )

    ds = ds[
        required
    ]

    # --------------------------------------------------------------
    # ERA5 may have 2-D lat/lon.
    #
    # normalize_geographic_grid() handles:
    #
    #     latitude(y,x)
    #     longitude(y,x)
    #
    # and converts them to:
    #
    #     latitude(y)
    #     longitude(x)
    # --------------------------------------------------------------

    ds = normalize_geographic_grid(
        ds,
        "ERA5",
    )

    # --------------------------------------------------------------
    # ERA5 source is hourly.
    #
    # We interpolate ONLY in time.
    # --------------------------------------------------------------

    print()

    print(
        "ERA5: converting hourly data "
        "to 30-minute cadence..."
    )

    ds = ds.interp(
        time=TARGET_TIMES,
        method="linear",
    )

    ds = ds.rename(
        {
            "u10":
                "era5_u10",

            "v10":
                "era5_v10",

            "d2m":
                "era5_d2m",

            "t2m":
                "era5_t2m",

            "msl":
                "era5_msl",

            "sp":
                "era5_sp",

            "tcc":
                "era5_tcc",

            "cape":
                "era5_cape",
        }
    )

    ds["era5_availability"] = (
        (
            "time",
            "y",
            "x",
        ),
        np.ones(
            (
                len(TARGET_TIMES),
                TARGET_NY,
                TARGET_NX,
            ),
            dtype=np.uint8,
        ),
    )

    print(
        f"ERA5 final: {dict(ds.sizes)}"
    )

    return ds


# ======================================================================
# LIS
# ======================================================================

def load_lis():

    section(
        "4. NASA ISS LIS"
    )

    check_file(
        LIS_FILE,
        "LIS",
    )

    ds = xr.open_dataset(
        LIS_FILE
    )

    ds = normalize_time(
        ds,
        "LIS",
    )

    print_grid(
        "LIS source",
        ds,
    )

    required = [
        "lightning_density",
        "lightning_availability",
    ]

    missing = [
        v
        for v in required
        if v not in ds.data_vars
    ]

    if missing:

        ds.close()

        raise ValueError(
            f"LIS missing variables: {missing}"
        )

    variables = [
        "lightning_density",
        "lightning_availability",
    ]

    if "lightning_observation_seconds" in ds.data_vars:

        variables.append(
            "lightning_observation_seconds"
        )

    ds = ds[
        variables
    ]

    ds = normalize_geographic_grid(
        ds,
        "LIS",
    )

    ds = select_target_times(
        ds,
        "LIS",
    )

    rename_map = {

        "lightning_density":
            "lis_lightning_density",

        "lightning_availability":
            "lis_lightning_availability",
    }

    if "lightning_observation_seconds" in ds.data_vars:

        rename_map[
            "lightning_observation_seconds"
        ] = "lis_observation_seconds"

    ds = ds.rename(
        rename_map
    )

    print(
        f"LIS final: {dict(ds.sizes)}"
    )

    return ds


# ======================================================================
# MODALITY VALIDATION
# ======================================================================

def validate_modality(
    ds,
    name,
):

    if ds.sizes.get("time") != len(
        TARGET_TIMES
    ):

        raise ValueError(
            f"{name}: expected "
            f"{len(TARGET_TIMES)} time frames, "
            f"got {ds.sizes.get('time')}"
        )

    if ds.sizes.get("y") != TARGET_NY:

        raise ValueError(
            f"{name}: expected y=27"
        )

    if ds.sizes.get("x") != TARGET_NX:

        raise ValueError(
            f"{name}: expected x=27"
        )

    if not np.array_equal(
        ds.time.values,
        TARGET_TIMES,
    ):

        raise ValueError(
            f"{name}: time grid mismatch."
        )

    if not np.allclose(
        ds.latitude.values,
        TARGET_LAT,
        atol=1e-5,
    ):

        raise ValueError(
            f"{name}: latitude grid mismatch."
        )

    if not np.allclose(
        ds.longitude.values,
        TARGET_LON,
        atol=1e-5,
    ):

        raise ValueError(
            f"{name}: longitude grid mismatch."
        )

    print(
        f"[PASS] {name}"
    )


# ======================================================================
# MERGE
# ======================================================================

def merge_modalities(
    insat,
    imerg,
    era5,
    lis,
):

    section(
        "5. MULTIMODAL FUSION"
    )

    validate_modality(
        insat,
        "INSAT",
    )

    validate_modality(
        imerg,
        "IMERG",
    )

    validate_modality(
        era5,
        "ERA5",
    )

    validate_modality(
        lis,
        "LIS",
    )

    print()

    print(
        "All four modalities have:"
    )

    print(
        "  6 time frames"
    )

    print(
        "  27 x 27 spatial grid"
    )

    print(
        "  30-minute cadence"
    )

    print()

    print(
        "Merging..."
    )

    merged = xr.merge(
        [
            insat,
            imerg,
            era5,
            lis,
        ],
        join="exact",
        compat="override",
    )

    print(
        "[PASS] Modalities merged."
    )

    return merged


# ======================================================================
# CHANNEL VALIDATION
# ======================================================================

def validate_model_channels(
    ds,
):

    section(
        "MODEL CHANNEL VALIDATION"
    )

    missing = [
        channel
        for channel in EXPECTED_CHANNELS
        if channel not in ds.data_vars
    ]

    if missing:

        raise ValueError(
            "Missing expected model channels:\n"
            + "\n".join(missing)
        )

    print(
        f"Expected channels: "
        f"{len(EXPECTED_CHANNELS)}"
    )

    for index, channel in enumerate(
        EXPECTED_CHANNELS
    ):

        print(
            f"  {index:02d}  {channel}"
        )

    print()

    print(
        "[PASS] 18-channel multimodal input."
    )


# ======================================================================
# LIS DIAGNOSTICS
# ======================================================================

def lis_statistics(
    ds,
):

    section(
        "LIS COVERAGE DIAGNOSTICS"
    )

    lightning = ds[
        "lis_lightning_density"
    ].values

    availability = ds[
        "lis_lightning_availability"
    ].values

    for i, timestamp in enumerate(
        ds.time.values
    ):

        lightning_frame = lightning[i]

        availability_frame = availability[i]

        total_lightning = float(
            np.nansum(
                lightning_frame
            )
        )

        active_cells = int(
            np.count_nonzero(
                lightning_frame > 0
            )
        )

        available_cells = int(
            np.count_nonzero(
                availability_frame > 0
            )
        )

        print(
            f"{timestamp} | "
            f"lightning={total_lightning:.1f} | "
            f"active_cells={active_cells} | "
            f"available_cells={available_cells}"
        )

    print()

    print(
        "LIS meaning:"
    )

    print(
        "  availability=1 + lightning>0 "
        "=> observed lightning"
    )

    print(
        "  availability=1 + lightning=0 "
        "=> observed no lightning"
    )

    print(
        "  availability=0 "
        "=> observation unavailable"
    )

    print()

    print(
        "Native LIS spatial resolution remains "
        "approximately 0.5 degree."
    )

    print(
        "27x27 is the common MODEL grid."
    )


# ======================================================================
# DATA QUALITY
# ======================================================================

def print_statistics(
    ds,
):

    section(
        "6. DATA QUALITY"
    )

    for variable in ds.data_vars:

        values = ds[
            variable
        ].values

        finite = np.isfinite(
            values
        )

        valid = int(
            finite.sum()
        )

        total = int(
            values.size
        )

        if valid == 0:

            print(
                f"{variable:35s} "
                f"NO FINITE VALUES"
            )

            continue

        print(
            f"{variable:35s} "
            f"valid={valid}/{total} "
            f"min={np.nanmin(values):.4f} "
            f"max={np.nanmax(values):.4f} "
            f"mean={np.nanmean(values):.4f}"
        )


# ======================================================================
# METADATA
# ======================================================================

def add_metadata(
    ds,
):

    ds.attrs.update(
        {

            "project":
                "SIH26072YELLOW",

            "problem":
                "AIML based Nowcasting of Thunderstorm "
                "and Lightning using Atmospheric Observation",

            "prototype_region":
                "Odisha",

            "domain":
                "20N-21N, 86E-88E",

            "grid":
                "27x27",

            "cadence":
                "30 minutes",

            "period":
                "2020-05-01 04:00-06:30 UTC",

            "modalities":
                "INSAT-3D, GPM IMERG, ERA5, NASA ISS LIS",

            "channel_count":
                "18",

            "data_type":
                "Real observations",

            "synthetic_data":
                "None",

            "spatial_processing":
                "Input modality files were already processed "
                "to the common 27x27 model grid. "
                "The multimodal builder only normalizes "
                "coordinate orientation.",

            "temporal_processing":
                "ERA5 hourly data is linearly interpolated "
                "to the common 30-minute time axis.",

            "lis_native_resolution":
                "approximately 0.5 degree",

            "lis_availability":
                "Observation coverage. "
                "Unavailable LIS coverage is not interpreted "
                "as zero lightning.",

            "target_semantics":
                "Lightning targets represent lightning "
                "observed within the corresponding "
                "30-minute interval.",

            "prototype_status":
                "Six-frame real-data multimodal prototype. "
                "Not yet a statistically sufficient training corpus.",
        }
    )

    return ds


# ======================================================================
# SAVE
# ======================================================================

def save_dataset(
    ds,
):

    section(
        "7. SAVING MULTIMODAL DATASET"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoding = {}

    for variable in ds.data_vars:

        dtype = ds[
            variable
        ].dtype

        if np.issubdtype(
            dtype,
            np.floating,
        ):

            encoding[
                variable
            ] = {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            }

        elif np.issubdtype(
            dtype,
            np.integer,
        ):

            encoding[
                variable
            ] = {
                "zlib": True,
                "complevel": 4,
            }

    ds.to_netcdf(
        OUTPUT_FILE,
        engine="netcdf4",
        encoding=encoding,
    )

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    return OUTPUT_FILE


# ======================================================================
# FINAL SAVED FILE CHECK
# ======================================================================

def validate_saved_file(
    path,
):

    section(
        "8. FINAL FILE VALIDATION"
    )

    ds = xr.open_dataset(
        path
    )

    try:

        print(
            ds
        )

        validate_modality(
            ds,
            "SAVED DATASET",
        )

        validate_model_channels(
            ds
        )

        print()

        print(
            "Final time:"
        )

        print(
            f"  {ds.time.values[0]}"
        )

        print(
            f"  {ds.time.values[-1]}"
        )

        print()

        print(
            "Final grid:"
        )

        print(
            f"  latitude  = "
            f"{ds.latitude.values[0]:.6f} -> "
            f"{ds.latitude.values[-1]:.6f}"
        )

        print(
            f"  longitude = "
            f"{ds.longitude.values[0]:.6f} -> "
            f"{ds.longitude.values[-1]:.6f}"
        )

        print()

        print(
            "[PASS] Saved dataset is valid."
        )

    finally:

        ds.close()


# ======================================================================
# MAIN BUILD
# ======================================================================

def build_dataset():

    section(
        "SIH26072YELLOW"
    )

    print(
        "REAL MULTIMODAL THUNDERSTORM NOWCASTING "
        "PROTOTYPE"
    )

    print()

    print(
        "Region     : Odisha"
    )

    print(
        "Latitude   : 21N -> 20N"
    )

    print(
        "Longitude  : 86E -> 88E"
    )

    print(
        "Grid       : 27 x 27"
    )

    print(
        "Cadence    : 30 minutes"
    )

    print(
        f"Frames     : {len(TARGET_TIMES)}"
    )

    print(
        "Period     : 2020-05-01 04:00-06:30 UTC"
    )

    print()

    print(
        "Modalities:"
    )

    print(
        "  INSAT-3D"
    )

    print(
        "  GPM IMERG"
    )

    print(
        "  ERA5"
    )

    print(
        "  NASA ISS LIS"
    )

    print()

    print(
        "Spatial interpolation: NONE"
    )

    print(
        "ERA5 temporal interpolation: YES"
    )

    print(
        "Synthetic observations: NONE"
    )

    # ==============================================================
    # LOAD
    # ==============================================================

    insat = load_insat()

    imerg = load_imerg()

    era5 = load_era5()

    lis = load_lis()

    # ==============================================================
    # MERGE
    # ==============================================================

    merged = merge_modalities(
        insat,
        imerg,
        era5,
        lis,
    )

    # ==============================================================
    # CHANNELS
    # ==============================================================

    validate_model_channels(
        merged
    )

    # ==============================================================
    # LIS
    # ==============================================================

    lis_statistics(
        merged
    )

    # ==============================================================
    # METADATA
    # ==============================================================

    merged = add_metadata(
        merged
    )

    # ==============================================================
    # STATISTICS
    # ==============================================================

    print_statistics(
        merged
    )

    # ==============================================================
    # SHOW FINAL DATASET
    # ==============================================================

    section(
        "FINAL MULTIMODAL DATASET"
    )

    print(
        merged
    )

    # ==============================================================
    # SAVE
    # ==============================================================

    output = save_dataset(
        merged
    )

    # ==============================================================
    # VALIDATE SAVED FILE
    # ==============================================================

    validate_saved_file(
        output
    )

    # ==============================================================
    # SUCCESS
    # ==============================================================

    section(
        "SUCCESS"
    )

    print(
        "REAL MULTIMODAL DATASET CREATED"
    )

    print()

    print(
        f"Output:\n{output}"
    )

    print()

    print(
        "18-channel model input is ready."
    )

    print()

    print(
        "Pipeline:"
    )

    print(
        "INSAT + IMERG + ERA5 + LIS"
    )

    print(
        "        ↓"
    )

    print(
        "common 27x27 grid"
    )

    print(
        "        ↓"
    )

    print(
        "30-minute cadence"
    )

    print(
        "        ↓"
    )

    print(
        "18-channel multimodal tensor"
    )

    print(
        "        ↓"
    )

    print(
        "temporal window"
    )

    print(
        "        ↓"
    )

    print(
        "ConvLSTM / multimodal nowcaster"
    )

    return output


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    try:

        build_dataset()

    except Exception as exc:

        print()

        print(
            "=" * 78
        )

        print(
            "BUILD FAILED"
        )

        print(
            "=" * 78
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        traceback.print_exc()

        raise   