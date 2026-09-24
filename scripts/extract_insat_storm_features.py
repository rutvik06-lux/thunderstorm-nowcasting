from pathlib import Path
import numpy as np
from datetime import datetime

INPUT_DIR = Path(r"data\processed\insat_features")
OUTPUT_DIR = Path(r"data\processed\storm_features")

CHANNEL_INDEX = {
    "MIR": 0,
    "TIR1": 1,
    "TIR2": 2,
    "WV": 3,
}

EXPECTED_INTERVAL_MINUTES = 30


def parse_timestamp(timestamp):
    return datetime.fromisoformat(
        timestamp.replace("Z", "+00:00")
    )


def calculate_gradient(field):
    gy, gx = np.gradient(field)

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )

    return magnitude.astype(np.float32)


def process_block(block):

    results = []

    for i, current in enumerate(block):

        tensor = current["tensor"]

        mir = tensor[:, :, CHANNEL_INDEX["MIR"]]
        tir1 = tensor[:, :, CHANNEL_INDEX["TIR1"]]
        tir2 = tensor[:, :, CHANNEL_INDEX["TIR2"]]
        wv = tensor[:, :, CHANNEL_INDEX["WV"]]

        thermal_difference = tir1 - tir2

        if i == 0:
            tir1_cooling = np.zeros_like(tir1)
            tir2_cooling = np.zeros_like(tir2)
        else:

            previous = block[i - 1]["tensor"]

            previous_tir1 = previous[
                :, :, CHANNEL_INDEX["TIR1"]
            ]

            previous_tir2 = previous[
                :, :, CHANNEL_INDEX["TIR2"]
            ]

            tir1_cooling = previous_tir1 - tir1
            tir2_cooling = previous_tir2 - tir2

        tir1_gradient = calculate_gradient(tir1)

        features = np.stack(
            [
                mir,
                tir1,
                tir2,
                wv,
                thermal_difference,
                tir1_cooling,
                tir2_cooling,
                tir1_gradient,
            ],
            axis=-1
        ).astype(np.float32)

        results.append(
            {
                "features": features,
                "timestamp": current["timestamp"],
                "latitude": current["latitude"],
                "longitude": current["longitude"],
            }
        )

    return results


def load_frames():

    files = sorted(
        INPUT_DIR.glob("*.npz")
    )

    frames = []

    for file in files:

        data = np.load(file)

        frames.append(
            {
                "file": file,
                "tensor": data["raw_data"].astype(
                    np.float32
                ),
                "timestamp": str(
                    data["timestamp_iso"]
                ),
                "datetime": parse_timestamp(
                    str(data["timestamp_iso"])
                ),
                "latitude": data["latitude"],
                "longitude": data["longitude"],
            }
        )

    return frames


def split_blocks(frames):

    if not frames:
        return []

    blocks = []
    current = [frames[0]]

    for frame in frames[1:]:

        difference = (
            frame["datetime"]
            - current[-1]["datetime"]
        ).total_seconds() / 60

        if difference == EXPECTED_INTERVAL_MINUTES:
            current.append(frame)
        else:
            blocks.append(current)
            current = [frame]

    blocks.append(current)

    return blocks


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    frames = load_frames()
    blocks = split_blocks(frames)

    print("=" * 70)
    print("INSAT STORM FEATURE EXTRACTION")
    print("=" * 70)

    print("Total frames:", len(frames))
    print("Continuous blocks:", len(blocks))
    print()

    total = 0

    for block_index, block in enumerate(blocks):

        print(
            f"Block {block_index + 1}: "
            f"{len(block)} frames"
        )

        results = process_block(block)

        for result in results:

            timestamp = result["timestamp"]

            safe_timestamp = (
                timestamp
                .replace(":", "")
                .replace("+00:00", "Z")
                .replace("-", "")
                .replace("T", "_")
            )

            output_file = (
                OUTPUT_DIR /
                f"insat_features_{safe_timestamp}.npz"
            )

            np.savez_compressed(
                output_file,
                features=result["features"],
                latitude=result["latitude"],
                longitude=result["longitude"],
                timestamp_iso=result["timestamp"],
                feature_names=np.array(
                    [
                        "MIR",
                        "TIR1",
                        "TIR2",
                        "WV",
                        "TIR1_TIR2",
                        "TIR1_COOLING",
                        "TIR2_COOLING",
                        "TIR1_GRADIENT",
                    ]
                )
            )

            total += 1

    print()
    print("Generated feature frames:", total)

    if total:

        example = sorted(
            OUTPUT_DIR.glob("*.npz")
        )[0]

        data = np.load(example)

        features = data["features"]

        print()
        print("EXAMPLE:")
        print("File:", example.name)
        print("Shape:", features.shape)
        print("dtype:", features.dtype)

        print()
        print("FEATURE CHANNELS:")

        for i, name in enumerate(
            data["feature_names"]
        ):
            channel = features[:, :, i]

            print(
                f"{i}: {name} | "
                f"min={np.nanmin(channel):.3f} | "
                f"max={np.nanmax(channel):.3f}"
            )

    print()
    print("=" * 70)
    print("STORM FEATURE EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
