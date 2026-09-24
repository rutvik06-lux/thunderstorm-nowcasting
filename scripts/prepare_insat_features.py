from pathlib import Path
import numpy as np

INPUT_DIR = Path(r"data\processed\insat")
OUTPUT_DIR = Path(r"data\processed\insat_features")

CHANNELS = ["MIR", "TIR1", "TIR2", "WV"]

# Initial physical ranges for normalization.
# Values are deliberately broad so real observations are not clipped aggressively.
CHANNEL_RANGES = {
    "MIR":  (180.0, 330.0),
    "TIR1": (180.0, 330.0),
    "TIR2": (180.0, 330.0),
    "WV":   (180.0, 300.0),
}

def normalize_channel(data, minimum, maximum):
    data = np.asarray(data, dtype=np.float32)

    normalized = (data - minimum) / (maximum - minimum)

    # Keep the model input bounded.
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.astype(np.float32)


def process_file(input_file):
    data = np.load(input_file)

    tensor = data["data"].astype(np.float32)
    timestamp = str(data["timestamp_iso"])

    print()
    print("=" * 70)
    print("PROCESSING:", input_file.name)
    print("=" * 70)
    print("Original shape:", tensor.shape)
    print("Timestamp:", timestamp)

    normalized = np.empty_like(tensor, dtype=np.float32)

    for channel_index, channel_name in enumerate(CHANNELS):

        channel = tensor[:, :, channel_index]

        minimum, maximum = CHANNEL_RANGES[channel_name]

        normalized[:, :, channel_index] = normalize_channel(
            channel,
            minimum,
            maximum
        )

        print(
            f"{channel_name:5s} | "
            f"raw={np.nanmin(channel):.2f}–{np.nanmax(channel):.2f} K | "
            f"normalized={np.nanmin(normalized[:, :, channel_index]):.3f}–"
            f"{np.nanmax(normalized[:, :, channel_index]):.3f}"
        )

    output_file = OUTPUT_DIR / input_file.name

    np.savez_compressed(
        output_file,
        data=normalized,
        raw_data=tensor,
        latitude=data["latitude"],
        longitude=data["longitude"],
        timestamp=data["timestamp"],
        timestamp_iso=data["timestamp_iso"],
        channels=data["channels"]
    )

    print("Saved:", output_file)
    print("Output shape:", normalized.shape)
    print("Output dtype:", normalized.dtype)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.npz"))

    # Don't accidentally process files from diagnostics or other folders.
    files = [
        f for f in files
        if f.is_file()
    ]

    if not files:
        raise FileNotFoundError(
            f"No processed INSAT files found in {INPUT_DIR}"
        )

    print("=" * 70)
    print("INSAT FEATURE PREPARATION")
    print("=" * 70)
    print("Input files :", len(files))
    print("Input dir   :", INPUT_DIR)
    print("Output dir  :", OUTPUT_DIR)

    for file in files:
        try:
            process_file(file)
        except Exception as e:
            print("[ERROR]")
            print(file)
            print(e)

    print()
    print("=" * 70)
    print("FEATURE PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
