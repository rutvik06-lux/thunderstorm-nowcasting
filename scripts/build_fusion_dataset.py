from pathlib import Path
import numpy as np

INPUT_DIR = Path(r"data\processed\storm_features")
OUTPUT_DIR = Path(r"data\processed\fused_samples")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INSAT_FEATURES = [
    "MIR",
    "TIR1",
    "TIR2",
    "WV",
    "TIR1_TIR2",
    "TIR1_COOLING",
    "TIR2_COOLING",
    "TIR1_GRADIENT",
]

RADAR_FEATURES = [
    "RADAR_REFLECTIVITY",
    "RADAR_VELOCITY",
]

LIGHTNING_FEATURES = [
    "LIGHTNING_DENSITY",
    "LIGHTNING_RATE",
]

# We reserve these channels now.
# They will be filled when your teammates provide real data.

ALL_FEATURES = (
    INSAT_FEATURES
    + RADAR_FEATURES
    + LIGHTNING_FEATURES
)

files = sorted(
    INPUT_DIR.glob("*.npz")
)

if not files:
    raise FileNotFoundError(
        f"No INSAT feature files found in {INPUT_DIR}"
    )

print("=" * 70)
print("MULTI-SOURCE FUSION DATASET BUILDER")
print("=" * 70)

print("INSAT files:", len(files))
print("INSAT channels:", len(INSAT_FEATURES))
print("Reserved radar channels:", len(RADAR_FEATURES))
print("Reserved lightning channels:", len(LIGHTNING_FEATURES))
print("Total fusion channels:", len(ALL_FEATURES))

print()
print("FUSION CHANNEL ORDER")

for index, name in enumerate(ALL_FEATURES):
    print(f"{index:02d} : {name}")

print()

for file in files:

    data = np.load(file)

    insat = data["features"].astype(
        np.float32
    )

    height, width, channels = insat.shape

    if (
        height != 27
        or width != 27
        or channels != 8
    ):
        raise ValueError(
            f"Unexpected INSAT shape: {insat.shape}"
        )

    # Placeholder channels.
    #
    # NaN means:
    # "sensor data not available yet"
    #
    # We deliberately do NOT use zeros because
    # zero could be interpreted as a real measurement.

    radar = np.full(
        (27, 27, len(RADAR_FEATURES)),
        np.nan,
        dtype=np.float32
    )

    lightning = np.full(
        (27, 27, len(LIGHTNING_FEATURES)),
        np.nan,
        dtype=np.float32
    )

    fused = np.concatenate(
        [
            insat,
            radar,
            lightning
        ],
        axis=-1
    )

    # Availability mask.
    #
    # 1 = actual observation available
    # 0 = missing

    availability = np.concatenate(
        [
            np.ones(
                (27, 27, len(INSAT_FEATURES)),
                dtype=np.float32
            ),
            np.zeros(
                (27, 27, len(RADAR_FEATURES)),
                dtype=np.float32
            ),
            np.zeros(
                (27, 27, len(LIGHTNING_FEATURES)),
                dtype=np.float32
            )
        ],
        axis=-1
    )

    output_file = (
        OUTPUT_DIR /
        file.name
    )

    np.savez_compressed(
        output_file,
        features=fused,
        availability=availability,
        latitude=data["latitude"],
        longitude=data["longitude"],
        timestamp_iso=data["timestamp_iso"],
        feature_names=np.array(
            ALL_FEATURES
        )
    )

print()
print("Fusion files created:", len(files))

# Inspect first file
example = sorted(
    OUTPUT_DIR.glob("*.npz")
)[0]

data = np.load(example)

print()
print("EXAMPLE FUSION SAMPLE")
print("File:", example.name)
print("Feature shape:", data["features"].shape)
print("Availability shape:", data["availability"].shape)
print()
print("Available channels:",
      int(data["availability"].sum()),
      "/",
      data["availability"].size)

print()
print("=" * 70)
print("FUSION INTERFACE READY")
print("=" * 70)
