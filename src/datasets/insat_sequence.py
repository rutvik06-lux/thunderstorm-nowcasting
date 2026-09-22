from pathlib import Path
import numpy as np
from datetime import datetime


INPUT_DIR = Path(r"data\processed\insat")


def load_insat_sequence():

    files = sorted(INPUT_DIR.glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No INSAT processed files found in {INPUT_DIR}"
        )

    frames = []
    timestamps = []

    for file in files:

        data = np.load(file)

        frame = data["data"]
        timestamp = str(data["timestamp_iso"])

        frames.append(frame)
        timestamps.append(timestamp)

    # Combine all frames
    # (T, H, W, C)
    sequence = np.stack(frames, axis=0)

    # Convert to PyTorch-style format
    # (T, C, H, W)
    sequence = np.transpose(sequence, (0, 3, 1, 2))

    # Check timestamps
    parsed_times = [
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        for ts in timestamps
    ]

    for i in range(1, len(parsed_times)):

        difference = (
            parsed_times[i] - parsed_times[i - 1]
        ).total_seconds() / 60

        if difference != 30:
            print(
                f"WARNING: {timestamps[i-1]} -> "
                f"{timestamps[i]} = {difference} minutes"
            )

    return sequence, timestamps


if __name__ == "__main__":

    print("=" * 70)
    print("INSAT TEMPORAL SEQUENCE LOADER")
    print("=" * 70)

    sequence, timestamps = load_insat_sequence()

    print(f"Number of frames : {sequence.shape[0]}")
    print(f"Channels         : {sequence.shape[1]}")
    print(f"Height           : {sequence.shape[2]}")
    print(f"Width            : {sequence.shape[3]}")

    print()
    print("Tensor shape     :", sequence.shape)
    print("Tensor dtype     :", sequence.dtype)
    print("Tensor size      :", sequence.nbytes, "bytes")

    print()
    print("TIMESTAMPS:")

    for i, timestamp in enumerate(timestamps):
        print(f"{i}: {timestamp}")

    print()
    print("CHANNELS:")
    print("0 = MIR")
    print("1 = TIR1")
    print("2 = TIR2")
    print("3 = WV")

    print("=" * 70)
