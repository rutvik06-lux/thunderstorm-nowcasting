from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FILE = Path(
    "data/processed/insat/3SIMG_15MAR2026_1930_L1B_STD_V01R00.npz"
)

OUTPUT = Path(
    "data/processed/insat/diagnostics/mumbai_grid.png"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if not FILE.exists():
    raise FileNotFoundError(
        f"INSAT file not found: {FILE}"
    )

data = np.load(FILE)

lat = data["latitude"]
lon = data["longitude"]
tensor = data["data"]

print("=" * 70)
print("INSAT MUMBAI SPATIAL GRID")
print("=" * 70)

print("File:", FILE)
print("Grid shape:", tensor.shape)

print(
    "Latitude:",
    lat.min(),
    "to",
    lat.max()
)

print(
    "Longitude:",
    lon.min(),
    "to",
    lon.max()
)

print("Channels:", data["channels"])
print("Timestamp:", data["timestamp_iso"])

mumbai_lat = 19.0760
mumbai_lon = 72.8777

plt.figure(figsize=(10, 8))

for i in range(lat.shape[0]):
    plt.plot(
        lon[i, :],
        lat[i, :],
        linewidth=0.5
    )

for j in range(lon.shape[1]):
    plt.plot(
        lon[:, j],
        lat[:, j],
        linewidth=0.5
    )

plt.scatter(
    mumbai_lon,
    mumbai_lat,
    s=80,
    marker="x",
    label="Mumbai"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "INSAT-3D Mumbai 27 x 27 Spatial Grid\n"
    "15 March 2026 - 19:30 UTC"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print("Saved:", OUTPUT)
print("=" * 70)
