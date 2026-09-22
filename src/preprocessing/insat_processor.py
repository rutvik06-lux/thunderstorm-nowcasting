
from pathlib import Path
import h5py
import numpy as np
from datetime import datetime, timedelta, timezone


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(r"data\raw\insat")
OUTPUT_DIR = Path(r"data\processed\insat")

# Mumbai region determined from actual INSAT geolocation
ROW_START = 885
ROW_END = 912

COL_START = 1153
COL_END = 1180


# Channels used in the first prototype
CHANNELS = {
    "MIR": ("IMG_MIR", "IMG_MIR_TEMP"),
    "TIR1": ("IMG_TIR1", "IMG_TIR1_TEMP"),
    "TIR2": ("IMG_TIR2", "IMG_TIR2_TEMP"),
    "WV": ("IMG_WV", "IMG_WV_TEMP"),
}


# ============================================================
# CONVERT MOSDAC TIMESTAMP
# ============================================================

def mosdac_time_to_datetime(minutes):
    """
    Convert MOSDAC time representation into UTC datetime.

    MOSDAC time:
        minutes since 2000-01-01 00:00:00 UTC
    """

    epoch = datetime(
        2000,
        1,
        1,
        tzinfo=timezone.utc
    )

    return epoch + timedelta(
        minutes=float(minutes)
    )


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(input_file: Path):

    print()
    print("=" * 70)
    print("PROCESSING:", input_file.name)
    print("=" * 70)

    with h5py.File(input_file, "r") as h:

        # ----------------------------------------------------
        # GEOLOCATION
        # ----------------------------------------------------

        lat = (
            h["Latitude"][:]
            .astype(np.float32)
            * 0.01
        )

        lon = (
            h["Longitude"][:]
            .astype(np.float32)
            * 0.01
        )

        crop_lat = lat[
            ROW_START:ROW_END,
            COL_START:COL_END
        ]

        crop_lon = lon[
            ROW_START:ROW_END,
            COL_START:COL_END
        ]

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp_value = float(
            h["time"][0]
        )

        timestamp_datetime = (
            mosdac_time_to_datetime(
                timestamp_value
            )
        )

        print()
        print(
            "Timestamp:",
            timestamp_datetime.isoformat()
        )

        # ----------------------------------------------------
        # EXTRACT CHANNELS
        # ----------------------------------------------------

        channel_arrays = []

        for name, (image_name, lookup_name) in CHANNELS.items():

            # Raw image
            image = h[image_name][0]

            # Lookup table for physical temperature
            lookup = h[lookup_name][:]

            # Mumbai crop
            counts = image[
                ROW_START:ROW_END,
                COL_START:COL_END
            ]

            # Get fill value
            fill_value = h[image_name].attrs.get(
                "_FillValue",
                [1023]
            )[0]

            # ------------------------------------------------
            # Convert raw count → brightness temperature
            # ------------------------------------------------

            temperature = lookup[counts].astype(
                np.float32
            )

            # Mark invalid pixels as NaN
            temperature[
                counts == fill_value
            ] = np.nan

            channel_arrays.append(
                temperature
            )

            # ------------------------------------------------
            # Channel statistics
            # ------------------------------------------------

            valid_fraction = (
                np.isfinite(temperature).mean()
            )

            print(
                f"{name:5s} | "
                f"shape={temperature.shape} | "
                f"valid={valid_fraction * 100:.2f}% | "
                f"range={np.nanmin(temperature):.2f}"
                f"–{np.nanmax(temperature):.2f} K"
            )

        # ----------------------------------------------------
        # STACK CHANNELS
        # ----------------------------------------------------

        tensor = np.stack(
            channel_arrays,
            axis=-1
        )

        print()
        print("Tensor shape:", tensor.shape)
        print("Tensor dtype:", tensor.dtype)
        print("Tensor size :", tensor.nbytes, "bytes")

    # ========================================================
    # SAVE PROCESSED DATA
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = OUTPUT_DIR / (
        input_file.stem + ".npz"
    )

    np.savez_compressed(
        output_file,

        # ML tensor
        data=tensor,

        # Spatial coordinates
        latitude=crop_lat,
        longitude=crop_lon,

        # Original MOSDAC timestamp
        timestamp=timestamp_value,

        # Human-readable UTC timestamp
        timestamp_iso=timestamp_datetime.isoformat(),

        # Channel names
        channels=np.array(
            list(CHANNELS.keys())
        )
    )

    print()
    print("Saved:", output_file)
    print("=" * 70)


# ============================================================
# PROCESS ALL HDF5 FILES
# ============================================================

def main():

    files = sorted(
        INPUT_DIR.rglob("*.h5")
    )

    if not files:

        print("No HDF5 files found in:")
        print(INPUT_DIR)

        return

    print()
    print("INSAT PROCESSOR")
    print("=" * 70)
    print("Input files :", len(files))
    print("Input dir   :", INPUT_DIR)
    print("Output dir  :", OUTPUT_DIR)

    for file in files:

        try:

            process_file(file)

        except Exception as e:

            print()
            print("[ERROR]")
            print(file)
            print(e)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
