from pathlib import Path
import numpy as np

OUTPUT_DIR = Path(r"data\processed\targets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEIGHT = 27
WIDTH = 27

HORIZONS_MINUTES = [30, 60, 90, 120]

TARGET_TYPES = [
    "thunderstorm_probability",
    "radar_reflectivity",
    "lightning_density",
]

# Empty target template.
# NaN means ground truth is not available yet.
targets = np.full(
    (
        len(HORIZONS_MINUTES),
        HEIGHT,
        WIDTH
    ),
    np.nan,
    dtype=np.float32
)

output = OUTPUT_DIR / "target_schema.npz"

np.savez_compressed(
    output,
    targets=targets,
    horizons_minutes=np.array(
        HORIZONS_MINUTES,
        dtype=np.int32
    ),
    target_types=np.array(TARGET_TYPES),
    height=np.int32(HEIGHT),
    width=np.int32(WIDTH),
)

print("=" * 70)
print("THUNDERSTORM TARGET SCHEMA")
print("=" * 70)

print("Grid:", HEIGHT, "x", WIDTH)
print("Horizons:", HORIZONS_MINUTES)
print("Target shape:", targets.shape)
print("Target dtype:", targets.dtype)
print()
print("Target types:")
for target in TARGET_TYPES:
    print("-", target)

print()
print("Saved:", output)
print("=" * 70)
