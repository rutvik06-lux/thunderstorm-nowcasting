from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FILE = Path(
    "data/processed/insat/3SIMG_14MAR2026_2200_L1B_STD_V01R00.npz"
)

data = np.load(FILE)

lat = data["latitude"]
lon = data["longitude"]
tensor = data["data"]

print("Grid shape:", tensor.shape)
print("Latitude:", lat.min(), "to", lat.max())
print("Longitude:", lon.min(), "to", lon.max())

# Plot the grid
plt.figure(figsize=(10, 8))

# Draw grid lines
for i in range(lat.shape[0]):
    plt.plot(lon[i, :], lat[i, :], linewidth=0.5)

for j in range(lon.shape[1]):
    plt.plot(lon[:, j], lat[:, j], linewidth=0.5)

# Mumbai location
mumbai_lat = 19.0760
mumbai_lon = 72.8777

plt.scatter(
    mumbai_lon,
    mumbai_lat,
    s=80,
    marker="x",
    label="Mumbai"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("INSAT-3D Mumbai 27 × 27 Spatial Grid")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()