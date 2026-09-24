from pathlib import Path
import numpy as np
from datetime import datetime

INPUT_DIR = Path(r"data\processed\insat_features")
OUTPUT_DIR = Path(r"data\processed\temporal_samples")

INPUT_FRAMES = 4
TARGET_FRAMES = 1
EXPECTED_INTERVAL_MINUTES = 30


def parse_timestamp(timestamp):
    return datetime.fromisoformat(
        timestamp.replace("Z", "+00:00")
    )


def load_frames():

    files = sorted(INPUT_DIR.glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No feature files found in {INPUT_DIR}"
        )

    frames = []

    for file in files:

        data = np.load(file)

        frames.append({
            "file": file,
            "tensor": data["data"].astype(np.float32),
            "timestamp": str(data["timestamp_iso"]),
            "datetime": parse_timestamp(
                str(data["timestamp_iso"])
            )
        })

    return frames


def find_continuous_blocks(frames):

    blocks = []
    current_block = [frames[0]]

    for frame in frames[1:]:

        previous = current_block[-1]

        difference = (
            frame["datetime"] -
            previous["datetime"]
        ).total_seconds() / 60

        if difference == EXPECTED_INTERVAL_MINUTES:
            current_block.append(frame)

        else:
            blocks.append(current_block)
            current_block = [frame]

    blocks.append(current_block)

    return blocks


def create_samples(block):

    samples = []

    required_frames = INPUT_FRAMES + TARGET_FRAMES

    if len(block) < required_frames:
        return samples

    for i in range(
        len(block) - required_frames + 1
    ):

        input_frames = block[
            i:i + INPUT_FRAMES
        ]

        target_frame = block[
            i + INPUT_FRAMES
        ]

        x = np.stack(
            [
                frame["tensor"]
                for frame in input_frames
            ],
            axis=0
        )

        y = target_frame["tensor"]

        sample = {
            "input": x,
            "target": y,
            "input_start": input_frames[0]["timestamp"],
            "input_end": input_frames[-1]["timestamp"],
            "target_time": target_frame["timestamp"]
        }

        samples.append(sample)

    return samples


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    frames = load_frames()

    print("=" * 70)
    print("INSAT TEMPORAL SAMPLE BUILDER")
    print("=" * 70)

    print("Total frames:", len(frames))
    print()

    blocks = find_continuous_blocks(frames)

    print("CONTINUOUS BLOCKS")
    print("-" * 70)

    for i, block in enumerate(blocks):

        print(
            f"Block {i + 1}: "
            f"{block[0]['timestamp']} -> "
            f"{block[-1]['timestamp']} "
            f"({len(block)} frames)"
        )

    print()

    all_samples = []

    for block_index, block in enumerate(blocks):

        samples = create_samples(block)

        print(
            f"Block {block_index + 1}: "
            f"{len(samples)} temporal samples"
        )

        all_samples.extend(samples)

    print()
    print("TOTAL TEMPORAL SAMPLES:", len(all_samples))
    print()

    if not all_samples:
        print(
            "No complete temporal samples available."
        )
        return

    inputs = np.stack(
        [sample["input"] for sample in all_samples],
        axis=0
    )

    targets = np.stack(
        [sample["target"] for sample in all_samples],
        axis=0
    )

    print("MODEL INPUT")
    print("Shape :", inputs.shape)
    print("dtype :", inputs.dtype)

    print()
    print("MODEL TARGET")
    print("Shape :", targets.shape)
    print("dtype :", targets.dtype)

    output_file = (
        OUTPUT_DIR /
        "insat_temporal_samples.npz"
    )

    np.savez_compressed(
        output_file,
        inputs=inputs,
        targets=targets,
        input_start=np.array(
            [s["input_start"] for s in all_samples]
        ),
        input_end=np.array(
            [s["input_end"] for s in all_samples]
        ),
        target_time=np.array(
            [s["target_time"] for s in all_samples]
        ),
        channels=np.array(
            ["MIR", "TIR1", "TIR2", "WV"]
        )
    )

    print()
    print("Saved:", output_file)

    print()
    print("=" * 70)
    print("TEMPORAL SAMPLE BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
