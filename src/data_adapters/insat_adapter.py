from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# PATHS
# ============================================================

INSAT_DIR = Path(
    r"data\raw\insat\2020\0501"
)

OUTPUT_DIR = Path(
    r"data\processed\insat"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "insat_odisha_27x27_20200501.nc"
)


# ============================================================
# COMMON GRID
# Same physical grid used by IMERG / ERA5 / LIS
# ============================================================

LAT_MIN = 20.0
LAT_MAX = 21.0

LON_MIN = 86.0
LON_MAX = 88.0

GRID_SIZE = 27


TARGET_LAT = np.linspace(
    LAT_MAX,
    LAT_MIN,
    GRID_SIZE,
    dtype=np.float32,
)

TARGET_LON = np.linspace(
    LON_MIN,
    LON_MAX,
    GRID_SIZE,
    dtype=np.float32,
)


# ============================================================
# INSAT-3D MERCATOR PROJECTION PARAMETERS
# ============================================================

EARTH_RADIUS = 6378137.0

LON_ORIGIN = 75.0

STANDARD_PARALLEL = 17.75

X_MIN = -3234623.003937
X_MAX = 3234623.003937

Y_MIN = -1058893.687497
Y_MAX = 5401854.420193


# ============================================================
# CONVERT INSAT MERCATOR X/Y -> LAT/LON
# ============================================================

def mercator_xy_to_latlon(x, y):

    # Longitude
    lon = (
        LON_ORIGIN
        + np.degrees(
            x / EARTH_RADIUS
        )
    )

    # Mercator latitude
    lat = np.degrees(
        2.0
        * np.arctan(
            np.exp(
                y / EARTH_RADIUS
            )
        )
        - np.pi / 2.0
    )

    return lat, lon


# ============================================================
# BUILD NATIVE INSAT GEOGRAPHIC GRID
# ============================================================

def get_insat_latlon(ds):

    x = ds["X"].values
    y = ds["Y"].values

    lon_1d = (
        LON_ORIGIN
        + np.degrees(
            x / EARTH_RADIUS
        )
    )

    lat_1d = np.degrees(
        2.0
        * np.arctan(
            np.exp(
                y / EARTH_RADIUS
            )
        )
        - np.pi / 2.0
    )

    lon_2d, lat_2d = np.meshgrid(
        lon_1d,
        lat_1d
    )

    return (
        lat_2d.astype(np.float32),
        lon_2d.astype(np.float32),
    )


# ============================================================
# FIND INSAT FILES
# ============================================================

def get_files():

    files = sorted(
        INSAT_DIR.glob(
            "3DIMG_*_L1C_ASIA_MER_V01R00.h5"
        )
    )

    # Only event-period files:
    #
    # 04:00
    # 04:30
    # 05:00
    # 05:30
    # 06:00
    # 06:30
    #
    selected = []

    for file in files:

        name = file.name

        if any(
            f"_{hour:02d}{minute:02d}_"
            in name
            for hour in range(4, 7)
            for minute in (0, 30)
        ):
            selected.append(file)

    return selected


# ============================================================
# CALIBRATION
# ============================================================

def apply_lut(
    counts,
    lut,
    fill_value=None
):

    counts = counts.astype(
        np.int64
    )

    invalid = (
        counts < 0
    )

    if fill_value is not None:

        invalid |= (
            counts == fill_value
        )

    invalid |= (
        counts >= len(lut)
    )

    safe = counts.copy()

    safe[invalid] = 0

    physical = lut[
        safe
    ].astype(
        np.float32
    )

    physical[invalid] = np.nan

    return physical


# ============================================================
# READ ONE INSAT FILE
# ============================================================

def read_file(file):

    ds = xr.open_dataset(
        file,
        engine="h5netcdf"
    )

    lat_native, lon_native = (
        get_insat_latlon(ds)
    )

    variables = {}

    # --------------------------------------------------------
    # TIR1
    # --------------------------------------------------------

    tir1_counts = (
        ds["IMG_TIR1"]
        .values[0]
    )

    tir1_lut = (
        ds["IMG_TIR1_TEMP"]
        .values
    )

    variables["IMG_TIR1"] = apply_lut(
        tir1_counts,
        tir1_lut,
        fill_value=1023
    )

    # --------------------------------------------------------
    # TIR2
    # --------------------------------------------------------

    tir2_counts = (
        ds["IMG_TIR2"]
        .values[0]
    )

    tir2_lut = (
        ds["IMG_TIR2_TEMP"]
        .values
    )

    variables["IMG_TIR2"] = apply_lut(
        tir2_counts,
        tir2_lut,
        fill_value=1023
    )

    # --------------------------------------------------------
    # WV
    # --------------------------------------------------------

    wv_counts = (
        ds["IMG_WV"]
        .values[0]
    )

    wv_lut = (
        ds["IMG_WV_TEMP"]
        .values
    )

    variables["IMG_WV"] = apply_lut(
        wv_counts,
        wv_lut,
        fill_value=1023
    )

    # --------------------------------------------------------
    # VIS
    # --------------------------------------------------------

    vis_counts = (
        ds["IMG_VIS"]
        .values[0]
    )

    vis_lut = (
        ds["IMG_VIS_ALBEDO"]
        .values
    )

    variables["IMG_VIS"] = apply_lut(
        vis_counts,
        vis_lut,
        fill_value=0
    )

    timestamp = (
        ds["time"]
        .values[0]
    )

    ds.close()

    return (
        timestamp,
        lat_native,
        lon_native,
        variables
    )


# ============================================================
# MAP NATIVE INSAT PIXELS TO COMMON 27x27 GRID
# ============================================================

def regrid_to_common_grid(
    values,
    lat_native,
    lon_native
):

    output = np.full(
        (
            GRID_SIZE,
            GRID_SIZE
        ),
        np.nan,
        dtype=np.float32
    )

    for iy, lat in enumerate(
        TARGET_LAT
    ):

        for ix, lon in enumerate(
            TARGET_LON
        ):

            distance = (
                (lat_native - lat) ** 2
                +
                (lon_native - lon) ** 2
            )

            index = np.nanargmin(
                distance
            )

            native_y, native_x = (
                np.unravel_index(
                    index,
                    lat_native.shape
                )
            )

            output[iy, ix] = (
                values[
                    native_y,
                    native_x
                ]
            )

    return output


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset():

    files = get_files()

    print(
        f"\nINSAT files found: {len(files)}"
    )

    if not files:

        raise FileNotFoundError(
            "No INSAT event-period files found."
        )

    timestamps = []

    data = {
        "IMG_TIR1": [],
        "IMG_TIR2": [],
        "IMG_WV": [],
        "IMG_VIS": [],
    }

    native_lat = None
    native_lon = None

    for file in files:

        print(
            f"\nProcessing: {file.name}"
        )

        (
            timestamp,
            lat_native,
            lon_native,
            variables
        ) = read_file(file)

        if native_lat is None:

            native_lat = lat_native
            native_lon = lon_native

            print(
                "\nNative geographic coverage:"
            )

            print(
                f"Latitude : "
                f"{lat_native.min():.3f}"
                f" -> "
                f"{lat_native.max():.3f}"
            )

            print(
                f"Longitude: "
                f"{lon_native.min():.3f}"
                f" -> "
                f"{lon_native.max():.3f}"
            )

        timestamps.append(
            timestamp
        )

        for variable in data:

            print(
                f"  Regridding {variable}..."
            )

            regridded = (
                regrid_to_common_grid(
                    variables[variable],
                    lat_native,
                    lon_native
                )
            )

            data[variable].append(
                regridded
            )

    # --------------------------------------------------------
    # Convert lists to arrays
    # --------------------------------------------------------

    for variable in data:

        data[variable] = np.stack(
            data[variable],
            axis=0
        ).astype(
            np.float32
        )

    timestamps = np.asarray(
        timestamps
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = xr.Dataset(

        {
            variable: (
                ("time", "y", "x"),
                values
            )

            for variable, values
            in data.items()
        },

        coords={

            "time": timestamps,

            "latitude": (
                "y",
                TARGET_LAT
            ),

            "longitude": (
                "x",
                TARGET_LON
            ),
        },

        attrs={

            "source": "INSAT-3D",

            "product":
                "3DIMG_L1C_ASIA_MER",

            "region":
                "Odisha prototype domain "
                "(20-21N, 86-88E)",

            "grid":
                "Common 27x27 geographic grid",

            "projection":
                "Mercator",

            "spatial_method":
                "Nearest native INSAT pixel "
                "after geographic coordinate "
                "conversion",

            "channels":
                "TIR1, TIR2, WV, VIS",
        }
    )

    return dataset


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "INSAT-3D -> COMMON ODisha GRID"
    )

    print("=" * 70)

    print("\nTarget grid:")

    print(
        f"Latitude : "
        f"{LAT_MIN} -> {LAT_MAX}"
    )

    print(
        f"Longitude: "
        f"{LON_MIN} -> {LON_MAX}"
    )

    print(
        f"Grid     : "
        f"{GRID_SIZE} x {GRID_SIZE}"
    )

    dataset = build_dataset()

    print("\nSaving:")

    print(
        OUTPUT_FILE
    )

    dataset.to_netcdf(
        OUTPUT_FILE,
        engine="netcdf4"
    )

    print(
        "\nSaved successfully."
    )

    print("\nDataset:")

    print(dataset)

    print("\nOutput size:")

    print(
        f"{OUTPUT_FILE.stat().st_size / (1024 ** 2):.2f} MB"
    )

    dataset.close()


if __name__ == "__main__":
    main()