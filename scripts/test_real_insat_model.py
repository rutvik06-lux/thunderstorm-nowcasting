from pathlib import Path
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STORM_FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "storm_features"
)


# ---------------------------------------------------------------------
# MODEL IMPORT
# ---------------------------------------------------------------------

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.models.nowcasting_model import ThunderstormNowcaster


# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

TIME_STEPS = 4
INSAT_CHANNELS = 8
TOTAL_CHANNELS = 12

HEIGHT = 27
WIDTH = 27

HORIZONS = 4


# ---------------------------------------------------------------------
# LOAD ONE STORM FEATURE FRAME
# ---------------------------------------------------------------------

def load_frame(file_path):

    data = np.load(file_path)

    features = data["features"]

    if features.shape != (
        HEIGHT,
        WIDTH,
        INSAT_CHANNELS,
    ):
        raise ValueError(
            f"Unexpected feature shape: "
            f"{features.shape}"
        )

    timestamp = str(
        data["timestamp_iso"]
    )

    return features, timestamp


# ---------------------------------------------------------------------
# FIND CONTINUOUS 4-FRAME SEQUENCES
# ---------------------------------------------------------------------

def find_sequences():

    files = sorted(
        STORM_FEATURE_DIR.glob("*.npz")
    )

    frames = []

    for file in files:

        try:

            features, timestamp = load_frame(
                file
            )

            frames.append(
                {
                    "file": file,
                    "features": features,
                    "timestamp": timestamp,
                }
            )

        except Exception as error:

            print(
                "[SKIP]",
                file.name,
                error,
            )

    sequences = []

    for i in range(
        len(frames) - TIME_STEPS + 1
    ):

        window = frames[
            i:i + TIME_STEPS
        ]

        timestamps = [
            item["timestamp"]
            for item in window
        ]

        # Parse timestamps.
        from datetime import datetime

        parsed = [
            datetime.fromisoformat(
                ts.replace(
                    "Z",
                    "+00:00",
                )
            )
            for ts in timestamps
        ]

        continuous = True

        for j in range(1, len(parsed)):

            difference = (
                parsed[j]
                - parsed[j - 1]
            ).total_seconds() / 60

            if difference != 30:

                continuous = False
                break

        if continuous:

            sequences.append(
                window
            )

    return sequences


# ---------------------------------------------------------------------
# BUILD MODEL INPUT
# ---------------------------------------------------------------------

def build_model_input(sequence):

    # INSAT:
    #
    # [time, height, width, channels]
    #
    # becomes:
    #
    # [time, channels, height, width]

    insat = np.stack(
        [
            item["features"]
            for item in sequence
        ],
        axis=0,
    )

    insat = np.transpose(
        insat,
        (0, 3, 1, 2),
    )

    # Create full 12-channel fusion tensor.
    fusion = np.full(
        (
            TIME_STEPS,
            TOTAL_CHANNELS,
            HEIGHT,
            WIDTH,
        ),
        np.nan,
        dtype=np.float32,
    )

    # Insert real INSAT features.
    fusion[
        :,
        :INSAT_CHANNELS,
        :,
        :,
    ] = insat

    # Availability mask.
    availability = np.zeros_like(
        fusion,
        dtype=np.float32,
    )

    # INSAT is available.
    availability[
        :,
        :INSAT_CHANNELS,
        :,
        :,
    ] = np.isfinite(
        insat
    ).astype(
        np.float32
    )

    # Radar + lightning remain unavailable.
    #
    # Channels:
    #
    # 8  Radar reflectivity
    # 9  Radar velocity
    # 10 Lightning density
    # 11 Lightning rate

    return fusion, availability


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("REAL INSAT -> THUNDERSTORM MODEL TEST")
    print("=" * 70)

    print()
    print("Storm feature directory:")
    print(STORM_FEATURE_DIR)

    sequences = find_sequences()

    print()
    print("Continuous 4-frame sequences:",
          len(sequences))

    if not sequences:

        raise RuntimeError(
            "No continuous 4-frame sequence found."
        )

    # Use the first valid sequence.
    sequence = sequences[0]

    print()
    print("SELECTED SEQUENCE:")

    for i, item in enumerate(sequence):

        print(
            f"{i}: {item['timestamp']}"
        )

    # Build fusion input.
    fusion, availability = (
        build_model_input(sequence)
    )

    print()
    print("Fusion NumPy shape:",
          fusion.shape)

    print(
        "Availability shape:",
        availability.shape,
    )

    print(
        "INSAT available:",
        availability[
            :,
            :8
        ].mean() * 100,
        "%",
    )

    print(
        "Radar available:",
        availability[
            :,
            8:10
        ].mean() * 100,
        "%",
    )

    print(
        "Lightning available:",
        availability[
            :,
            10:12
        ].mean() * 100,
        "%",
    )

    # Add batch dimension.
    fusion_tensor = torch.from_numpy(
        fusion
    ).unsqueeze(0)

    availability_tensor = torch.from_numpy(
        availability
    ).unsqueeze(0)

    print()
    print("PyTorch input shape:",
          fusion_tensor.shape)

    print(
        "PyTorch mask shape:",
        availability_tensor.shape,
    )

    # Model.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    model = ThunderstormNowcaster(
        input_channels=TOTAL_CHANNELS,
        hidden_channels=64,
        horizons=HORIZONS,
    ).to(device)

    model.eval()

    fusion_tensor = fusion_tensor.to(
        device
    )

    availability_tensor = (
        availability_tensor.to(device)
    )

    # Forward pass.
    with torch.no_grad():

        prediction = model(
            fusion_tensor,
            availability_tensor,
        )

    print()
    print("MODEL OUTPUT")
    print("-" * 70)

    print(
        "Output shape:",
        prediction.shape,
    )

    print(
        "Output dtype:",
        prediction.dtype,
    )

    print(
        "Output range:",
        f"{prediction.min().item():.4f}",
        "->",
        f"{prediction.max().item():.4f}",
    )

    print()

    horizon_names = [
        "+30 minutes",
        "+60 minutes",
        "+90 minutes",
        "+120 minutes",
    ]

    for i, name in enumerate(
        horizon_names
    ):

        horizon = prediction[
            0,
            i
        ]

        print(
            f"{name:12s} | "
            f"min={horizon.min().item():.4f} | "
            f"max={horizon.max().item():.4f} | "
            f"mean={horizon.mean().item():.4f}"
        )

    # Safety checks.
    expected_shape = (
        1,
        HORIZONS,
        HEIGHT,
        WIDTH,
    )

    assert (
        prediction.shape
        == expected_shape
    )

    assert torch.isfinite(
        prediction
    ).all()

    print()
    print("=" * 70)
    print("REAL INSAT MODEL TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":

    main()