from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path(
    "restdataset/DATA,RAW,LIGHTNING NASA_ISS_LIS 2020-05-01_03"
)

OUTPUT_DIR = Path(
    "data/processed/lis"
)

OUTPUT_FILE = OUTPUT_DIR / (
    "lis_odisha_27x27_20200501.nc"
)


# ============================================================
# REGION
# ============================================================

LAT_MIN = 20.0
LAT_MAX = 21.0

LON_MIN = 86.0
LON_MAX = 88.0

TARGET_SIZE = 27


# ============================================================
# TIME WINDOW
# ============================================================

START_TIME = np.datetime64(
    "2020-05-01T04:00:00"
)

END_TIME = np.datetime64(
    "2020-05-01T06:30:00"
)

CADENCE = np.timedelta64(
    30,
    "m"
)


# ============================================================
# TARGET GRID
# ============================================================

TARGET_LATS = np.linspace(
    LAT_MIN,
    LAT_MAX,
    TARGET_SIZE
)

TARGET_LONS = np.linspace(
    LON_MIN,
    LON_MAX,
    TARGET_SIZE
)


# ============================================================
# LIS NATIVE VIEWTIME GRID
# ============================================================

# NASA ISS-LIS viewtime records describe approximately
# 0.5° × 0.5° native observation cells.

LIS_CELL_SIZE = 0.5
LIS_HALF_CELL = LIS_CELL_SIZE / 2.0


# ============================================================
# HELPERS
# ============================================================

def find_variable(ds, candidates):
    """
    Return the first variable that exists in the dataset.
    """

    for name in candidates:

        if name in ds.variables:
            return name

    return None


def to_datetime64(values):
    """
    Convert LIS time values to numpy datetime64[ns].

    LIS TAI93 values are interpreted as seconds since
    1993-01-01 00:00:00 UTC when they are numeric.
    """

    values = np.asarray(values).reshape(-1)

    if np.issubdtype(
        values.dtype,
        np.datetime64
    ):

        return values.astype(
            "datetime64[ns]"
        )

    values = values.astype(
        np.float64
    )

    epoch = np.datetime64(
        "1993-01-01T00:00:00",
        "ns"
    )

    seconds = np.rint(
        values
    ).astype(
        np.int64
    )

    return (
        epoch
        + seconds.astype(
            "timedelta64[s]"
        )
    ).astype(
        "datetime64[ns]"
    )


def get_viewtime_arrays(ds):
    """
    Extract NASA ISS-LIS viewtime information.

    Returns:

        view_lat
        view_lon
        view_start
        view_end
        view_effective_obs

    or None when the required viewtime variables are absent.
    """

    required = [
        "viewtime_lat",
        "viewtime_lon",
        "viewtime_TAI93_start",
        "viewtime_TAI93_end",
        "viewtime_effective_obs",
    ]

    if not all(
        name in ds.variables
        for name in required
    ):

        return None

    lat = np.asarray(
        ds["viewtime_lat"].values
    ).reshape(-1)

    lon = np.asarray(
        ds["viewtime_lon"].values
    ).reshape(-1)

    start = to_datetime64(
        ds["viewtime_TAI93_start"].values
    )

    end = to_datetime64(
        ds["viewtime_TAI93_end"].values
    )

    effective_obs = np.asarray(
        ds["viewtime_effective_obs"].values,
        dtype=np.float64
    ).reshape(-1)

    n = min(
        len(lat),
        len(lon),
        len(start),
        len(end),
        len(effective_obs)
    )

    lat = lat[:n]
    lon = lon[:n]
    start = start[:n]
    end = end[:n]
    effective_obs = effective_obs[:n]

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(effective_obs)
        & (effective_obs > 0)
        & (end >= start)
    )

    return (
        lat[valid],
        lon[valid],
        start[valid],
        end[valid],
        effective_obs[valid],
    )


def get_lightning_arrays(ds):
    """
    Extract LIS lightning observations.

    The adapter supports the common NASA ISS-LIS variable names
    used in the downloaded prototype files.
    """

    lat_name = find_variable(
        ds,
        [
            "lightning_flash_lat",
            "flash_lat",
            "lightning_lat",
            "event_lat",
        ]
    )

    lon_name = find_variable(
        ds,
        [
            "lightning_flash_lon",
            "flash_lon",
            "lightning_lon",
            "event_lon",
        ]
    )

    time_name = find_variable(
        ds,
        [
            "lightning_flash_TAI93_time",
            "flash_TAI93_time",
            "lightning_TAI93_time",
            "event_TAI93_time",
            "flash_time",
        ]
    )

    if (
        lat_name is None
        or lon_name is None
        or time_name is None
    ):

        return None

    lat = np.asarray(
        ds[lat_name].values
    ).reshape(-1)

    lon = np.asarray(
        ds[lon_name].values
    ).reshape(-1)

    times = to_datetime64(
        ds[time_name].values
    )

    n = min(
        len(lat),
        len(lon),
        len(times)
    )

    lat = lat[:n]
    lon = lon[:n]
    times = times[:n]

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= LAT_MIN)
        & (lat <= LAT_MAX)
        & (lon >= LON_MIN)
        & (lon <= LON_MAX)
        & (times >= START_TIME)
        & (times <= END_TIME + CADENCE)
    )

    return (
        lat[valid],
        lon[valid],
        times[valid],
        lat_name,
    )


def read_lis_file(path):
    """
    Read one NASA ISS-LIS orbit file.

    Returns a dictionary containing lightning observations and
    native LIS viewtime coverage.
    """

    try:

        ds = xr.open_dataset(
            path,
            decode_times=True
        )

    except Exception as exc:

        print(
            f"      [ERROR] Could not open file: {exc}"
        )

        return None

    try:

        lightning = get_lightning_arrays(
            ds
        )

        if lightning is None:

            ds.close()

            return None

        (
            lightning_lat,
            lightning_lon,
            lightning_time,
            lightning_variable,
        ) = lightning

        viewtime = get_viewtime_arrays(
            ds
        )

    finally:

        ds.close()

    # --------------------------------------------------------
    # No viewtime data
    # --------------------------------------------------------

    if viewtime is None:

        view_lat = np.empty(
            0,
            dtype=np.float64
        )

        view_lon = np.empty(
            0,
            dtype=np.float64
        )

        view_start = np.empty(
            0,
            dtype="datetime64[ns]"
        )

        view_end = np.empty(
            0,
            dtype="datetime64[ns]"
        )

        view_effective_obs = np.empty(
            0,
            dtype=np.float64
        )

    else:

        (
            view_lat,
            view_lon,
            view_start,
            view_end,
            view_effective_obs,
        ) = viewtime

        # ----------------------------------------------------
        # Keep only native LIS cells that can intersect our
        # region and requested time range.
        # ----------------------------------------------------

        valid = (
            (view_lat >= LAT_MIN - LIS_HALF_CELL)
            & (view_lat <= LAT_MAX + LIS_HALF_CELL)
            & (view_lon >= LON_MIN - LIS_HALF_CELL)
            & (view_lon <= LON_MAX + LIS_HALF_CELL)
            & (view_end >= START_TIME)
            & (view_start <= END_TIME + CADENCE)
        )

        view_lat = view_lat[valid]
        view_lon = view_lon[valid]
        view_start = view_start[valid]
        view_end = view_end[valid]
        view_effective_obs = view_effective_obs[valid]

    return {
        "lightning_lat": lightning_lat,
        "lightning_lon": lightning_lon,
        "lightning_time": lightning_time,
        "lightning_variable": lightning_variable,

        "view_lat": view_lat,
        "view_lon": view_lon,
        "view_start": view_start,
        "view_end": view_end,
        "view_effective_obs": view_effective_obs,
    }


# ============================================================
# NATIVE LIS COVERAGE
# ============================================================

def build_native_viewtime_coverage(
    view_lat,
    view_lon,
    view_start,
    view_end,
    view_effective_obs,
    timeline,
):
    """
    Build LIS coverage at the native 0.5° observation-cell level.

    Important:

    NASA LIS viewtime cells are the actual observation footprint.
    Multiple viewtime records can refer to the same native cell.

    Therefore we first aggregate records by:

        time interval + native LIS cell center

    rather than independently accumulating every record.

    Returns:

        native_available
        native_observation_seconds

    Shapes:

        [time, native_lat, native_lon]

    The native grid is determined from the unique viewtime cell
    centers present in the selected region.
    """

    if len(view_lat) == 0:

        return (
            np.empty(
                (
                    len(timeline),
                    0,
                    0
                ),
                dtype=np.float32
            ),
            np.empty(
                (
                    len(timeline),
                    0,
                    0
                ),
                dtype=np.float32
            ),
            np.empty(
                0,
                dtype=np.float64
            ),
            np.empty(
                0,
                dtype=np.float64
            ),
        )

    # --------------------------------------------------------
    # Native cell centers
    # --------------------------------------------------------

    native_lats = np.unique(
        np.round(
            view_lat,
            decimals=6
        )
    )

    native_lons = np.unique(
        np.round(
            view_lon,
            decimals=6
        )
    )

    native_lats.sort()
    native_lons.sort()

    lat_index = {
        value: i
        for i, value in enumerate(native_lats)
    }

    lon_index = {
        value: i
        for i, value in enumerate(native_lons)
    }

    native_available = np.zeros(
        (
            len(timeline),
            len(native_lats),
            len(native_lons)
        ),
        dtype=np.float32
    )

    native_observation_seconds = np.zeros(
        (
            len(timeline),
            len(native_lats),
            len(native_lons)
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Process each native viewtime record
    # --------------------------------------------------------

    for (
        vlat,
        vlon,
        vstart,
        vend,
        veffective,
    ) in zip(
        view_lat,
        view_lon,
        view_start,
        view_end,
        view_effective_obs,
    ):

        vlat = float(
            np.round(
                vlat,
                6
            )
        )

        vlon = float(
            np.round(
                vlon,
                6
            )
        )

        if (
            vlat not in lat_index
            or vlon not in lon_index
        ):

            continue

        yi = lat_index[vlat]
        xi = lon_index[vlon]

        for ti, t0 in enumerate(timeline):

            t1 = (
                t0
                + CADENCE
            )

            overlap_start = max(
                vstart,
                t0
            )

            overlap_end = min(
                vend,
                t1
            )

            if overlap_end <= overlap_start:

                continue

            overlap_seconds = float(
                (
                    overlap_end
                    - overlap_start
                )
                / np.timedelta64(
                    1,
                    "s"
                )
            )

            if overlap_seconds <= 0:

                continue

            # effective_obs is the effective observation time
            # associated with the native LIS cell.
            #
            # We DO NOT sum this repeatedly across model pixels.
            # At native-cell level, retain the largest valid
            # effective observation value for the interval.

            effective_seconds = min(
                overlap_seconds,
                float(veffective)
            )

            native_available[
                ti,
                yi,
                xi
            ] = 1.0

            native_observation_seconds[
                ti,
                yi,
                xi
            ] = max(
                native_observation_seconds[
                    ti,
                    yi,
                    xi
                ],
                np.float32(
                    effective_seconds
                )
            )

    return (
        native_available,
        native_observation_seconds,
        native_lats,
        native_lons,
    )


# ============================================================
# UPSAMPLE NATIVE LIS COVERAGE TO MODEL GRID
# ============================================================

def map_native_coverage_to_target_grid(
    native_available,
    native_observation_seconds,
    native_lats,
    native_lons,
):
    """
    Project native 0.5° LIS coverage onto the 27×27 model grid.

    IMPORTANT:

    This does NOT create new LIS observations.

    It creates a model-grid coverage field indicating whether
    the center of a 27×27 model cell lies inside an observed
    native LIS footprint.

    Observation seconds are copied from the corresponding
    native LIS cell. They are NOT summed across fine pixels.
    """

    availability = np.zeros(
        (
            native_available.shape[0],
            TARGET_SIZE,
            TARGET_SIZE
        ),
        dtype=np.float32
    )

    observation_seconds = np.zeros(
        (
            native_available.shape[0],
            TARGET_SIZE,
            TARGET_SIZE
        ),
        dtype=np.float32
    )

    if (
        len(native_lats) == 0
        or len(native_lons) == 0
    ):

        return (
            availability,
            observation_seconds
        )

    # --------------------------------------------------------
    # Process every native LIS cell
    # --------------------------------------------------------

    for yi_native, vlat in enumerate(
        native_lats
    ):

        for xi_native, vlon in enumerate(
            native_lons
        ):

            lat_low = (
                vlat
                - LIS_HALF_CELL
            )

            lat_high = (
                vlat
                + LIS_HALF_CELL
            )

            lon_low = (
                vlon
                - LIS_HALF_CELL
            )

            lon_high = (
                vlon
                + LIS_HALF_CELL
            )

            # ------------------------------------------------
            # Model-grid centers inside native LIS footprint
            # ------------------------------------------------

            y_indices = np.where(
                (TARGET_LATS >= lat_low)
                & (TARGET_LATS <= lat_high)
                & (TARGET_LATS >= LAT_MIN)
                & (TARGET_LATS <= LAT_MAX)
            )[0]

            x_indices = np.where(
                (TARGET_LONS >= lon_low)
                & (TARGET_LONS <= lon_high)
                & (TARGET_LONS >= LON_MIN)
                & (TARGET_LONS <= LON_MAX)
            )[0]

            if (
                len(y_indices) == 0
                or len(x_indices) == 0
            ):

                continue

            for y in y_indices:

                for x in x_indices:

                    # ----------------------------------------
                    # Copy native availability.
                    # ----------------------------------------

                    availability[
                        :,
                        y,
                        x
                    ] = np.maximum(
                        availability[
                            :,
                            y,
                            x
                        ],
                        native_available[
                            :,
                            yi_native,
                            xi_native
                        ]
                    )

                    # ----------------------------------------
                    # DO NOT SUM observation time across fine
                    # pixels. Retain the corresponding native
                    # cell coverage time.
                    # ----------------------------------------

                    observation_seconds[
                        :,
                        y,
                        x
                    ] = np.maximum(
                        observation_seconds[
                            :,
                            y,
                            x
                        ],
                        native_observation_seconds[
                            :,
                            yi_native,
                            xi_native
                        ]
                    )

    return (
        availability,
        observation_seconds
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ISS-LIS LIGHTNING ADAPTER")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"\nLIS directory not found:\n{INPUT_DIR}"
        )

    files = sorted(
        INPUT_DIR.glob("*.nc")
    )

    print(
        f"\nFound {len(files)} LIS files."
    )

    if not files:

        raise FileNotFoundError(
            "\nNo LIS NetCDF files found."
        )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_lat = []
    all_lon = []
    all_times = []

    all_view_lat = []
    all_view_lon = []
    all_view_start = []
    all_view_end = []
    all_view_effective_obs = []

    variable_used = None

    # --------------------------------------------------------
    # Read every orbit
    # --------------------------------------------------------

    for index, path in enumerate(
        files,
        start=1
    ):

        print(
            f"[{index:02d}/{len(files):02d}] "
            f"{path.name}"
        )

        result = read_lis_file(
            path
        )

        if result is None:

            print(
                "      [SKIP] Required lightning variables unavailable."
            )

            continue

        lightning_lat = result[
            "lightning_lat"
        ]

        lightning_lon = result[
            "lightning_lon"
        ]

        lightning_time = result[
            "lightning_time"
        ]

        view_lat = result[
            "view_lat"
        ]

        view_lon = result[
            "view_lon"
        ]

        view_start = result[
            "view_start"
        ]

        view_end = result[
            "view_end"
        ]

        view_effective_obs = result[
            "view_effective_obs"
        ]

        variable_used = result[
            "lightning_variable"
        ]

        if len(lightning_lat) > 0:

            all_lat.append(
                lightning_lat
            )

            all_lon.append(
                lightning_lon
            )

            all_times.append(
                lightning_time
            )

        if len(view_lat) > 0:

            all_view_lat.append(
                view_lat
            )

            all_view_lon.append(
                view_lon
            )

            all_view_start.append(
                view_start
            )

            all_view_end.append(
                view_end
            )

            all_view_effective_obs.append(
                view_effective_obs
            )

        print(
            f"      lightning in region/time: "
            f"{len(lightning_lat)}"
        )

        print(
            f"      viewtime cells intersecting domain: "
            f"{len(view_lat)}"
        )

    # --------------------------------------------------------
    # Lightning must exist
    # --------------------------------------------------------

    if len(all_lat) == 0:

        raise RuntimeError(
            "\nNo LIS lightning observations found "
            "in selected region/time."
        )

    # --------------------------------------------------------
    # Combine lightning
    # --------------------------------------------------------

    lat = np.concatenate(
        all_lat
    )

    lon = np.concatenate(
        all_lon
    )

    times = np.concatenate(
        all_times
    )

    # --------------------------------------------------------
    # Combine viewtime
    # --------------------------------------------------------

    if len(all_view_lat) > 0:

        view_lat = np.concatenate(
            all_view_lat
        )

        view_lon = np.concatenate(
            all_view_lon
        )

        view_start = np.concatenate(
            all_view_start
        )

        view_end = np.concatenate(
            all_view_end
        )

        view_effective_obs = np.concatenate(
            all_view_effective_obs
        )

    else:

        view_lat = np.empty(
            0,
            dtype=np.float64
        )

        view_lon = np.empty(
            0,
            dtype=np.float64
        )

        view_start = np.empty(
            0,
            dtype="datetime64[ns]"
        )

        view_end = np.empty(
            0,
            dtype="datetime64[ns]"
        )

        view_effective_obs = np.empty(
            0,
            dtype=np.float64
        )

    # --------------------------------------------------------
    # Remove exact duplicate lightning observations
    # --------------------------------------------------------

    print(
        "\nRemoving exact duplicate lightning observations..."
    )

    before = len(lat)

    records = np.rec.fromarrays(
        [
            lat,
            lon,
            times
        ],
        names=[
            "lat",
            "lon",
            "time"
        ]
    )

    _, unique_indices = np.unique(
        records,
        return_index=True
    )

    unique_indices = np.sort(
        unique_indices
    )

    lat = lat[
        unique_indices
    ]

    lon = lon[
        unique_indices
    ]

    times = times[
        unique_indices
    ]

    after = len(lat)

    print(
        f"      before: {before}"
    )

    print(
        f"      after : {after}"
    )

    print(
        f"      removed: {before - after}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LIGHTNING SUMMARY")
    print("=" * 70)

    print(
        f"\nLightning variable: "
        f"{variable_used}"
    )

    print(
        f"\nTotal lightning observations: "
        f"{len(lat)}"
    )

    print(
        f"Latitude: "
        f"{lat.min():.4f} -> {lat.max():.4f}"
    )

    print(
        f"Longitude: "
        f"{lon.min():.4f} -> {lon.max():.4f}"
    )

    print(
        f"Time: "
        f"{times.min()} -> {times.max()}"
    )

    print(
        f"\nTotal LIS viewtime records intersecting domain: "
        f"{len(view_lat)}"
    )

    # --------------------------------------------------------
    # Timeline
    # --------------------------------------------------------

    timeline = np.arange(
        START_TIME,
        END_TIME + CADENCE,
        CADENCE
    )

    print(
        "\n30-minute timestamps:"
    )

    for t in timeline:

        print(
            f"  {t}"
        )

    # --------------------------------------------------------
    # Output arrays
    # --------------------------------------------------------

    lightning_density = np.zeros(
        (
            len(timeline),
            TARGET_SIZE,
            TARGET_SIZE
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Map lightning flashes
    # --------------------------------------------------------

    lat_step = (
        LAT_MAX - LAT_MIN
    ) / (
        TARGET_SIZE - 1
    )

    lon_step = (
        LON_MAX - LON_MIN
    ) / (
        TARGET_SIZE - 1
    )

    for la, lo, timestamp in zip(
        lat,
        lon,
        times
    ):

        elapsed_minutes = (
            timestamp - START_TIME
        ) / np.timedelta64(
            1,
            "m"
        )

        bin_index = int(
            np.floor(
                elapsed_minutes / 30
            )
        )

        if timestamp == END_TIME:

            bin_index = (
                len(timeline) - 1
            )

        if (
            bin_index < 0
            or bin_index >= len(timeline)
        ):

            continue

        y = int(
            round(
                (la - LAT_MIN)
                / lat_step
            )
        )

        x = int(
            round(
                (lo - LON_MIN)
                / lon_step
            )
        )

        y = int(
            np.clip(
                y,
                0,
                TARGET_SIZE - 1
            )
        )

        x = int(
            np.clip(
                x,
                0,
                TARGET_SIZE - 1
            )
        )

        lightning_density[
            bin_index,
            y,
            x
        ] += 1.0

    # ========================================================
    # BUILD NATIVE LIS COVERAGE
    # ========================================================

    print(
        "\nBuilding native LIS viewtime coverage..."
    )

    (
        native_available,
        native_observation_seconds,
        native_lats,
        native_lons,
    ) = build_native_viewtime_coverage(
        view_lat=view_lat,
        view_lon=view_lon,
        view_start=view_start,
        view_end=view_end,
        view_effective_obs=view_effective_obs,
        timeline=timeline,
    )

    print(
        f"\nNative LIS latitude cells: "
        f"{len(native_lats)}"
    )

    print(
        f"Native LIS longitude cells: "
        f"{len(native_lons)}"
    )

    print(
        f"Native LIS cells: "
        f"{len(native_lats) * len(native_lons)}"
    )

    # ========================================================
    # PROJECT NATIVE COVERAGE TO MODEL GRID
    # ========================================================

    print(
        "\nProjecting native LIS coverage onto 27x27 model grid..."
    )

    (
        lightning_availability,
        observation_seconds,
    ) = map_native_coverage_to_target_grid(
        native_available=native_available,
        native_observation_seconds=native_observation_seconds,
        native_lats=native_lats,
        native_lons=native_lons,
    )

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    ds = xr.Dataset(

        data_vars={

            "lightning_density": (
                ("time", "y", "x"),
                lightning_density
            ),

            "lightning_availability": (
                ("time", "y", "x"),
                lightning_availability
            ),

            "lightning_observation_seconds": (
                ("time", "y", "x"),
                observation_seconds
            ),
        },

        coords={

            "time": timeline,

            "y": np.arange(
                TARGET_SIZE
            ),

            "x": np.arange(
                TARGET_SIZE
            ),

            "latitude": (
                "y",
                TARGET_LATS
            ),

            "longitude": (
                "x",
                TARGET_LONS
            ),
        },

        attrs={

            "source":
                "NASA ISS-LIS",

            "product":
                "ISS Lightning Imaging Sensor",

            "observation_variable":
                variable_used,

            "region":
                "20-21N, 86-88E",

            "grid":
                "27x27",

            "cadence":
                "30 minutes",

            "description":
                "NASA ISS-LIS lightning observations "
                "mapped to a 27x27 geographic prototype grid.",

            "density_definition":
                "Count of LIS lightning flash observations "
                "assigned to each model grid cell and "
                "30-minute interval.",

            "availability_definition":
                "Model-grid representation of native NASA "
                "ISS-LIS viewtime coverage. A model cell is "
                "marked available when its center falls within "
                "an observed native approximately 0.5-degree "
                "LIS viewtime footprint during the interval. "
                "This is a coverage mask, not independent "
                "27x27 LIS measurement resolution.",

            "observation_seconds_definition":
                "Effective LIS observation time inherited "
                "from the corresponding native LIS viewtime "
                "cell. Values are not spatially summed across "
                "the upsampled model grid.",

            "native_lis_resolution":
                "Approximately 0.5 degree viewtime cells",

            "scientific_note":
                "ISS-LIS is an orbital lightning sensor. "
                "LIS coverage is intermittent and restricted "
                "to satellite overpasses. A zero lightning "
                "count outside observed viewtime must not be "
                "interpreted as no lightning.",

            "target_semantics":
                "Lightning density represents observed LIS "
                "lightning flashes occurring within each "
                "30-minute interval.",

            "time_semantics":
                "Each timestamp represents the beginning of "
                "the corresponding 30-minute observation "
                "interval. For example, 05:00 represents "
                "05:00-05:30 UTC.",

            "viewtime_source":
                "NASA ISS-LIS viewtime_lat, viewtime_lon, "
                "viewtime_TAI93_start, viewtime_TAI93_end "
                "and viewtime_effective_obs.",
        }
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_FILE.exists():

        OUTPUT_FILE.unlink()

    ds.to_netcdf(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nTotal lightning observations:"
    )

    print(
        int(
            np.sum(
                lightning_density
            )
        )
    )

    print(
        "\nMaximum lightning count "
        "in one grid cell/time bin:"
    )

    print(
        int(
            np.max(
                lightning_density
            )
        )
    )

    print(
        "\nNative LIS cells observed by timestamp:"
    )

    for i, t in enumerate(
        timeline
    ):

        native_cells = int(
            np.count_nonzero(
                native_available[i]
            )
        )

        available_cells = int(
            np.count_nonzero(
                lightning_availability[i]
            )
        )

        max_obs_seconds = float(
            np.max(
                observation_seconds[i]
            )
        )

        native_total_obs_seconds = float(
            np.sum(
                native_observation_seconds[i]
            )
        )

        lightning_count = int(
            np.sum(
                lightning_density[i]
            )
        )

        active_cells = int(
            np.count_nonzero(
                lightning_density[i]
            )
        )

        print(
            f"  {t} | "
            f"lightning={lightning_count} | "
            f"active_cells={active_cells} | "
            f"native_LIS_cells={native_cells} | "
            f"model_available_cells={available_cells} | "
            f"max_obs_sec={max_obs_seconds:.2f} | "
            f"native_total_obs_sec={native_total_obs_seconds:.2f}"
        )

    print(
        "\nGrid:"
    )

    print(
        f"  latitude : {LAT_MIN} -> {LAT_MAX}"
    )

    print(
        f"  longitude: {LON_MIN} -> {LON_MAX}"
    )

    print(
        f"  size     : {TARGET_SIZE} x {TARGET_SIZE}"
    )

    print(
        "\nLIS availability meaning:"
    )

    print(
        "  1 = the model-cell center falls inside a "
        "native LIS footprint observed during the interval"
    )

    print(
        "  0 = LIS coverage unavailable for that model cell"
    )

    print(
        "\nLightning meaning:"
    )

    print(
        "  lightning_density > 0 = observed LIS lightning"
    )

    print(
        "  lightning_density == 0 with availability=1 "
        "= observed no lightning in that covered interval"
    )

    print(
        "  lightning_density == 0 with availability=0 "
        "= unknown because LIS was not observing there"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  lightning_availability is an upsampled native "
        "LIS coverage field, NOT 27x27 native LIS resolution."
    )

    print(
        "  Native LIS resolution remains approximately "
        "0.5 degree."
    )


if __name__ == "__main__":
    main()