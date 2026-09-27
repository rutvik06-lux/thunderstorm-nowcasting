from pathlib import Path
import numpy as np


GRID_HEIGHT = 27
GRID_WIDTH = 27

INSAT_FEATURE_DIR = Path("data/processed/insat_features")


def load_insat_grid():
    files = sorted(INSAT_FEATURE_DIR.glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No INSAT feature files found in {INSAT_FEATURE_DIR}"
        )

    data = np.load(files[0])

    latitude = data["latitude"].astype(np.float32)
    longitude = data["longitude"].astype(np.float32)

    if latitude.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(
            f"Latitude shape {latitude.shape} != {(GRID_HEIGHT, GRID_WIDTH)}"
        )

    if longitude.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(
            f"Longitude shape {longitude.shape} != {(GRID_HEIGHT, GRID_WIDTH)}"
        )

    if not np.isfinite(latitude).all():
        raise ValueError("Latitude contains invalid values")

    if not np.isfinite(longitude).all():
        raise ValueError("Longitude contains invalid values")

    return latitude, longitude


def get_grid_metadata():
    latitude, longitude = load_insat_grid()

    return {
        "height": GRID_HEIGHT,
        "width": GRID_WIDTH,
        "latitude": latitude,
        "longitude": longitude,
        "lat_min": float(latitude.min()),
        "lat_max": float(latitude.max()),
        "lon_min": float(longitude.min()),
        "lon_max": float(longitude.max()),
    }


if __name__ == "__main__":
    grid = get_grid_metadata()

    print("=" * 60)
    print("COMMON GRID")
    print("=" * 60)
    print(f"Shape      : {grid['height']} x {grid['width']}")
    print(f"Latitude   : {grid['lat_min']:.4f} -> {grid['lat_max']:.4f}")
    print(f"Longitude  : {grid['lon_min']:.4f} -> {grid['lon_max']:.4f}")
    print()
    print("Latitude array shape :", grid["latitude"].shape)
    print("Longitude array shape:", grid["longitude"].shape)
    print()
    print("Grid loaded successfully.")
