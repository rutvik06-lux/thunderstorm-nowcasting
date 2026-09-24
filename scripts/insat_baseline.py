from pathlib import Path
import numpy as np

INPUT_DIR = Path(r"data\processed\storm_features")
OUTPUT_DIR = Path(r"data\processed\baseline_predictions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.glob("*.npz"))

if len(files) < 3:
    raise RuntimeError("Need at least 3 feature frames.")


def load_frame(path):
    data = np.load(path)

    return {
        "features": data["features"].astype(np.float32),
        "latitude": data["latitude"],
        "longitude": data["longitude"],
        "timestamp": str(data["timestamp_iso"])
    }


frames = [load_frame(f) for f in files]

print("=" * 70)
print("INSAT BASELINE NOWCAST")
print("=" * 70)

print("Available frames:", len(frames))

# Use only consecutive 30-minute frames.
# We don't cross gaps in the observations.
for i in range(2, len(frames)):

    previous_time = frames[i - 1]["timestamp"]
    current_time = frames[i]["timestamp"]

    # Check the actual temporal continuity using filenames/timestamps.
    from datetime import datetime

    t_prev = datetime.fromisoformat(
        previous_time.replace("Z", "+00:00")
    )
    t_curr = datetime.fromisoformat(
        current_time.replace("Z", "+00:00")
    )

    difference = (
        t_curr - t_prev
    ).total_seconds() / 60

    if difference != 30:
        continue

    f0 = frames[i - 2]["features"]
    f1 = frames[i - 1]["features"]
    actual = frames[i]["features"]

    # TIR1 and TIR2 indices.
    tir1_0 = f0[:, :, 1]
    tir1_1 = f1[:, :, 1]

    tir2_0 = f0[:, :, 2]
    tir2_1 = f1[:, :, 2]

    # Estimate the next temperature field using the
    # most recent temporal change.
    predicted_tir1 = (
        tir1_1 + (tir1_1 - tir1_0)
    )

    predicted_tir2 = (
        tir2_1 + (tir2_1 - tir2_0)
    )

    actual_tir1 = actual[:, :, 1]
    actual_tir2 = actual[:, :, 2]

    # Prediction error.
    tir1_mae = np.mean(
        np.abs(predicted_tir1 - actual_tir1)
    )

    tir2_mae = np.mean(
        np.abs(predicted_tir2 - actual_tir2)
    )

    # Cooling trend:
    # positive = cloud-top temperature decreasing.
    tir1_cooling = tir1_1 - predicted_tir1
    tir2_cooling = tir2_1 - predicted_tir2

    # Stronger cooling = stronger potential convective development.
    cooling_score = (
        np.maximum(tir1_cooling, 0)
        +
        np.maximum(tir2_cooling, 0)
    ) / 2.0

    output_file = (
        OUTPUT_DIR /
        f"baseline_{frames[i]['timestamp'].replace(':','').replace('+00:00','Z')}.npz"
    )

    np.savez_compressed(
        output_file,
        predicted_tir1=predicted_tir1.astype(np.float32),
        predicted_tir2=predicted_tir2.astype(np.float32),
        actual_tir1=actual_tir1.astype(np.float32),
        actual_tir2=actual_tir2.astype(np.float32),
        cooling_score=cooling_score.astype(np.float32),
        latitude=frames[i]["latitude"],
        longitude=frames[i]["longitude"],
        timestamp=frames[i]["timestamp"],
        tir1_mae=np.float32(tir1_mae),
        tir2_mae=np.float32(tir2_mae)
    )

    print()
    print("Prediction:", frames[i]["timestamp"])
    print(
        f"TIR1 MAE: {tir1_mae:.4f} K"
    )
    print(
        f"TIR2 MAE: {tir2_mae:.4f} K"
    )
    print(
        "Cooling score range:",
        f"{cooling_score.min():.4f}",
        "->",
        f"{cooling_score.max():.4f}"
    )

print()
print("=" * 70)
print("BASELINE COMPLETE")
print("=" * 70)
