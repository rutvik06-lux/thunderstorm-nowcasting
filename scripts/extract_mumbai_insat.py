import h5py
import glob
import os
import numpy as np

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_DIR = r"..\data\raw\insat\2026\0315"
OUTPUT_DIR = r"..\data\processed\insat\mumbai"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# MUMBAI CROP
# ------------------------------------------------------------

ROW_START = 885
ROW_END = 912       # exclusive

COL_START = 1153
COL_END = 1180      # exclusive

CHANNELS = [
    "IMG_MIR",
    "IMG_TIR1",
    "IMG_TIR2",
    "IMG_WV",
]

# ------------------------------------------------------------
# FIND FILES
# ------------------------------------------------------------

files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "*.h5")
    )
)

print("=" * 70)
print("MUMBAI INSAT EXTRACTION")
print("=" * 70)

print(f"Input files : {len(files)}")
print(f"Crop        : {ROW_END - ROW_START} x {COL_END - COL_START}")
print(f"Rows        : {ROW_START}:{ROW_END}")
print(f"Columns     : {COL_START}:{COL_END}")
print()

# ------------------------------------------------------------
# PROCESS EACH FILE
# ------------------------------------------------------------

for file_path in files:

    filename = os.path.basename(file_path)

    print("-" * 70)
    print(f"Processing: {filename}")

    with h5py.File(file_path, "r") as f:

        data = {}

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        if "time" in f:
            time_value = f["time"][()]
        else:
            time_value = None

        # ----------------------------------------------------
        # GEOLOCATION
        # ----------------------------------------------------

        lat = (
            f["Latitude"][
                ROW_START:ROW_END,
                COL_START:COL_END
            ].astype(np.float32)
            * 0.01
        )

        lon = (
            f["Longitude"][
                ROW_START:ROW_END,
                COL_START:COL_END
            ].astype(np.float32)
            * 0.01
        )

        # Convert fill values to NaN
        lat[lat > 90] = np.nan
        lon[lon > 180] = np.nan

        data["latitude"] = lat
        data["longitude"] = lon

        # ----------------------------------------------------
        # CHANNELS
        # ----------------------------------------------------

        for channel in CHANNELS:

            if channel not in f:
                print(f"[WARNING] Missing channel: {channel}")
                continue

            dataset = f[channel]

            # Dataset shape is expected to be:
            # (1, H, W)

            values = dataset[
                0,
                ROW_START:ROW_END,
                COL_START:COL_END
            ]

            data[channel] = values

            print(
                f"{channel}: "
                f"shape={values.shape}, "
                f"dtype={values.dtype}"
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_name = (
            filename.replace(".h5", ".npz")
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        np.savez_compressed(
            output_path,
            **data
        )

        print(f"[SAVED] {output_path}")

print()
print("=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)