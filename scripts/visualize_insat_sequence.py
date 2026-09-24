from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = Path(r"data\processed\insat")
OUTPUT_DIR = Path(r"data\processed\insat\diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2026-03-15T19:30:00+00:00"
END   = "2026-03-15T23:30:00+00:00"

files = sorted(INPUT_DIR.glob("*.npz"))

selected = []

for file in files:
    data = np.load(file)
    timestamp = str(data["timestamp_iso"])

    if START <= timestamp <= END:
        selected.append((timestamp, file))

print("=" * 70)
print("INSAT SEQUENCE VISUALIZATION")
print("=" * 70)
print(f"Selected frames: {len(selected)}")

for i, (timestamp, file) in enumerate(selected):
    print(f"{i}: {timestamp} -> {file.name}")

if len(selected) != 9:
    raise RuntimeError(
        f"Expected 9 continuous frames, found {len(selected)}"
    )

channels = {
    "TIR1": 1,
    "TIR2": 2,
}

for channel_name, channel_index in channels.items():

    arrays = []
    timestamps = []

    for timestamp, file in selected:
        data = np.load(file)
        frame = data["data"][:, :, channel_index]

        arrays.append(frame)
        timestamps.append(timestamp)

    vmin = min(np.nanmin(x) for x in arrays)
    vmax = max(np.nanmax(x) for x in arrays)

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    axes = axes.ravel()

    for i, (frame, timestamp) in enumerate(zip(arrays, timestamps)):

        image = axes[i].imshow(
            frame,
            origin="upper",
            vmin=vmin,
            vmax=vmax
        )

        dt = datetime.fromisoformat(timestamp)
        label = dt.strftime("%H:%M UTC")

        axes[i].set_title(label)
        axes[i].set_xlabel("Grid X")
        axes[i].set_ylabel("Grid Y")

        fig.colorbar(image, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"INSAT {channel_name} - Mumbai Prototype Region\n"
        f"15 March 2026 | 19:30–23:30 UTC",
        fontsize=15
    )

    fig.tight_layout()

    output = OUTPUT_DIR / f"insat_{channel_name.lower()}_19h30_23h30.png"

    fig.savefig(output, dpi=150)
    plt.show()

    print()
    print(f"Saved: {output}")

print()
print("=" * 70)
print("DONE")
print("=" * 70)
