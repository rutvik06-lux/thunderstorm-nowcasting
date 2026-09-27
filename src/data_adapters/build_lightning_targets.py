
from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multimodal"
    / "odisha_multimodal_20200501.nc"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "targets"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "lightning_targets_odisha_20200501.nc"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Any positive lightning density means lightning occurred
# during the corresponding 30-minute interval.
LIGHTNING_THRESHOLD = 0.0

CADENCE_MINUTES = 30


# ============================================================
# MAIN
# ============================================================

def build_targets():

    print("=" * 70)
    print("SIH26072YELLOW LIGHTNING TARGET BUILDER")
    print("=" * 70)

    print()
    print("Input:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load multimodal dataset
    # --------------------------------------------------------

    ds = xr.open_dataset(INPUT_FILE)

    try:

        required = [
            "lis_lightning_density",
            "lis_lightning_availability",
        ]

        for variable in required:

            if variable not in ds.data_vars:
                raise ValueError(
                    f"Required variable missing: {variable}"
                )

        lightning = (
            ds["lis_lightning_density"]
            .values
            .astype(np.float32)
        )

        availability = (
            ds["lis_lightning_availability"]
            .values
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if lightning.ndim != 3:
            raise ValueError(
                "Lightning density must have shape "
                "(time, y, x)"
            )

        if availability.shape != lightning.shape:
            raise ValueError(
                "Lightning availability shape does not "
                "match lightning density"
            )

        time = ds["time"].values
        latitude = ds["latitude"].values
        longitude = ds["longitude"].values

        print()
        print("Input dimensions:")
        print(f"  time = {len(time)}")
        print(f"  y    = {len(latitude)}")
        print(f"  x    = {len(longitude)}")

        # ----------------------------------------------------
        # Verify grid
        # ----------------------------------------------------

        if len(latitude) != 27 or len(longitude) != 27:
            raise ValueError(
                "Expected 27x27 model grid."
            )

        # ----------------------------------------------------
        # Build LIS observation mask
        # ----------------------------------------------------
        #
        # availability = 1:
        #   LIS coverage exists for this model-grid cell
        #
        # availability = 0:
        #   LIS did not provide an observation there
        #
        # Missing LIS coverage MUST NOT become a negative
        # lightning label.
        # ----------------------------------------------------

        observed = (
            np.isfinite(availability)
            & (availability > 0)
        )

        # ----------------------------------------------------
        # Detect lightning
        # ----------------------------------------------------

        lightning_detected = (
            np.isfinite(lightning)
            & (lightning > LIGHTNING_THRESHOLD)
        )

        # ----------------------------------------------------
        # Create target
        # ----------------------------------------------------
        #
        # Target semantics:
        #
        #   1 = lightning occurred during the corresponding
        #       30-minute interval
        #
        #   0 = LIS observed the cell during the interval
        #       and no lightning was detected
        #
        #   NaN = LIS observation unavailable
        #
        # Example:
        #
        #   time = 05:00
        #
        #   target represents:
        #
        #   05:00 <= lightning time < 05:30 UTC
        #
        # ----------------------------------------------------

        target = np.full(
            lightning.shape,
            np.nan,
            dtype=np.float32,
        )

        # Observed + lightning detected = 1
        target[
            observed & lightning_detected
        ] = 1.0

        # Observed + no lightning = 0
        target[
            observed & (~lightning_detected)
        ] = 0.0

        # ----------------------------------------------------
        # Target availability
        # ----------------------------------------------------

        target_availability = observed.astype(
            np.uint8
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        valid = np.isfinite(target)

        positive_count = int(
            np.count_nonzero(
                target[valid] == 1.0
            )
        )

        negative_count = int(
            np.count_nonzero(
                target[valid] == 0.0
            )
        )

        unavailable_count = int(
            np.count_nonzero(~valid)
        )

        observed_cells = int(
            np.count_nonzero(observed)
        )

        print()
        print("TARGET STATISTICS")
        print("-" * 70)

        print(
            f"Observed cells : {observed_cells}"
        )

        print(
            f"Positive cells : {positive_count}"
        )

        print(
            f"Negative cells : {negative_count}"
        )

        print(
            f"Unavailable    : {unavailable_count}"
        )

        if valid.sum() > 0:

            positive_ratio = (
                positive_count / int(valid.sum())
            )

            print(
                f"Positive ratio : {positive_ratio:.6f}"
            )

        # ----------------------------------------------------
        # Per-time diagnostic
        # ----------------------------------------------------

        print()
        print("PER-TIME TARGET SUMMARY")
        print("-" * 70)

        for i, timestamp in enumerate(time):

            frame_target = target[i]

            frame_valid = np.isfinite(
                frame_target
            )

            frame_positive = int(
                np.count_nonzero(
                    frame_target[frame_valid] == 1.0
                )
            )

            frame_negative = int(
                np.count_nonzero(
                    frame_target[frame_valid] == 0.0
                )
            )

            frame_available = int(
                np.count_nonzero(
                    observed[i]
                )
            )

            print(
                f"{timestamp} | "
                f"available={frame_available} | "
                f"positive={frame_positive} | "
                f"negative={frame_negative}"
            )

        # ----------------------------------------------------
        # Create output Dataset
        # ----------------------------------------------------

        target_ds = xr.Dataset(

            data_vars={

                "lightning_target": (
                    ("time", "y", "x"),
                    target,
                ),

                "target_availability": (
                    ("time", "y", "x"),
                    target_availability,
                ),

            },

            coords={

                "time": time,

                "y": np.arange(
                    len(latitude)
                ),

                "x": np.arange(
                    len(longitude)
                ),

                "latitude": (
                    "y",
                    latitude,
                ),

                "longitude": (
                    "x",
                    longitude,
                ),

            },

        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        target_ds.attrs.update(

            {

                "project":
                    "SIH26072YELLOW",

                "target_type":
                    "Lightning occurrence",

                "target_source":
                    "NASA ISS LIS",

                "target_definition":
                    "1 = lightning detected during the "
                    "corresponding 30-minute interval, "
                    "0 = LIS-observed interval with no "
                    "detected lightning, "
                    "NaN = LIS observation unavailable",

                "lightning_threshold":
                    str(LIGHTNING_THRESHOLD),

                "grid":
                    "27x27",

                "cadence":
                    "30 minutes",

                "target_time_semantics":
                    "Each target timestamp represents "
                    "the 30-minute interval beginning at "
                    "that timestamp. For example, 05:00 "
                    "represents 05:00-05:30 UTC.",

                "target_availability_definition":
                    "1 = LIS coverage available for the "
                    "model-grid cell during the target "
                    "interval; 0 = LIS observation "
                    "unavailable.",

                "missing_observation_policy":
                    "Unavailable LIS observations are "
                    "masked and are not treated as negative "
                    "lightning labels.",

                "native_lis_resolution":
                    "Approximately 0.5 degree viewtime cells",

                "model_grid_note":
                    "Lightning targets are represented on "
                    "the 27x27 model grid. This does not "
                    "imply native LIS resolution at 27x27.",

                "region":
                    "Odisha prototype domain",

                "domain":
                    "20N-21N, 86E-88E",

                "event":
                    "2020-05-01",

                "scientific_note":
                    "ISS-LIS is an orbital lightning sensor "
                    "with intermittent spatial-temporal "
                    "coverage. Zero lightning outside "
                    "available LIS observation coverage "
                    "must be treated as unknown.",

            }

        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        encoding = {

            "lightning_target": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            },

            "target_availability": {
                "zlib": True,
                "complevel": 4,
                "dtype": "uint8",
            },

        }

        target_ds.to_netcdf(
            OUTPUT_FILE,
            engine="netcdf4",
            encoding=encoding,
        )

        print()
        print("=" * 70)
        print("SUCCESS")
        print("=" * 70)

        print()
        print(
            "Lightning target dataset created:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print(target_ds)

    finally:

        ds.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    build_targets()
