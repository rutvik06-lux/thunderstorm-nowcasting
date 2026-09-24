from pathlib import Path
import numpy as np


GRID_HEIGHT = 27
GRID_WIDTH = 27

# Fusion channel definitions
RADAR_REFLECTIVITY = 8
RADAR_VELOCITY = 9
LIGHTNING_DENSITY = 10
LIGHTNING_RATE = 11


def validate_grid(latitude, longitude):
    """
    Validate that latitude and longitude describe
    the expected 27x27 Mumbai grid.
    """

    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)

    if latitude.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(
            f"Latitude grid must be "
            f"{GRID_HEIGHT}x{GRID_WIDTH}, "
            f"got {latitude.shape}"
        )

    if longitude.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(
            f"Longitude grid must be "
            f"{GRID_HEIGHT}x{GRID_WIDTH}, "
            f"got {longitude.shape}"
        )

    if not np.isfinite(latitude).all():
        raise ValueError("Latitude grid contains invalid values.")

    if not np.isfinite(longitude).all():
        raise ValueError("Longitude grid contains invalid values.")

    return latitude.astype(np.float32), longitude.astype(np.float32)


def create_empty_fusion_frame():
    """
    Create an empty 12-channel fusion frame.

    Missing radar/lightning data are represented by:
        value      = NaN
        availability = 0

    This is intentional.
    Missing data must NOT be treated as zero weather measurements.
    """

    features = np.full(
        (12, GRID_HEIGHT, GRID_WIDTH),
        np.nan,
        dtype=np.float32,
    )

    availability = np.zeros(
        (12, GRID_HEIGHT, GRID_WIDTH),
        dtype=np.float32,
    )

    return features, availability


def insert_sensor(
    features,
    availability,
    channel,
    data,
):
    """
    Insert one sensor field into the fusion tensor.

    data:
        27x27 array

    channel:
        Fusion channel number.
    """

    data = np.asarray(data, dtype=np.float32)

    if data.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(
            f"Sensor data must have shape "
            f"{GRID_HEIGHT}x{GRID_WIDTH}, "
            f"got {data.shape}"
        )

    valid = np.isfinite(data)

    features[channel] = np.where(
        valid,
        data,
        np.nan,
    )

    availability[channel] = valid.astype(np.float32)

    return features, availability


def build_fusion_frame(
    radar_reflectivity=None,
    radar_velocity=None,
    lightning_density=None,
    lightning_rate=None,
):
    """
    Build the radar/lightning portion of the
    reliability-aware fusion interface.

    Returns:

        features
            Shape: (12, 27, 27)

        availability
            Shape: (12, 27, 27)
    """

    features, availability = create_empty_fusion_frame()

    if radar_reflectivity is not None:
        features, availability = insert_sensor(
            features,
            availability,
            RADAR_REFLECTIVITY,
            radar_reflectivity,
        )

    if radar_velocity is not None:
        features, availability = insert_sensor(
            features,
            availability,
            RADAR_VELOCITY,
            radar_velocity,
        )

    if lightning_density is not None:
        features, availability = insert_sensor(
            features,
            availability,
            LIGHTNING_DENSITY,
            lightning_density,
        )

    if lightning_rate is not None:
        features, availability = insert_sensor(
            features,
            availability,
            LIGHTNING_RATE,
            lightning_rate,
        )

    return features, availability


def load_npz_sensor_file(file_path):
    """
    Load a teammate sensor file stored as NPZ.

    Expected fields:

        data
        latitude
        longitude
        timestamp

    Returns a dictionary containing the data.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Sensor file not found: {file_path}"
        )

    data = np.load(file_path)

    required = [
        "data",
        "latitude",
        "longitude",
        "timestamp",
    ]

    for field in required:
        if field not in data:
            raise ValueError(
                f"Missing required field '{field}' "
                f"in {file_path.name}"
            )

    latitude, longitude = validate_grid(
        data["latitude"],
        data["longitude"],
    )

    sensor_data = np.asarray(
        data["data"],
        dtype=np.float32,
    )

    return {
        "data": sensor_data,
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": str(data["timestamp"]),
    }


def print_fusion_status(features, availability):
    """
    Print a human-readable sensor availability report.
    """

    names = [
        "MIR",
        "TIR1",
        "TIR2",
        "WV",
        "TIR1_TIR2",
        "TIR1_COOLING",
        "TIR2_COOLING",
        "TIR1_GRADIENT",
        "RADAR_REFLECTIVITY",
        "RADAR_VELOCITY",
        "LIGHTNING_DENSITY",
        "LIGHTNING_RATE",
    ]

    print()
    print("=" * 70)
    print("FUSION SENSOR STATUS")
    print("=" * 70)

    for channel, name in enumerate(names):

        available_fraction = (
            availability[channel].mean() * 100
        )

        print(
            f"{channel:2d} | "
            f"{name:22s} | "
            f"available={available_fraction:6.2f}%"
        )

    print("=" * 70)


def test_adapter():

    print("=" * 70)
    print("WEATHER DATA ADAPTER TEST")
    print("=" * 70)

    # Empty fusion frame.
    features, availability = build_fusion_frame()

    print("Empty fusion shape       :", features.shape)
    print("Empty availability shape :", availability.shape)

    assert features.shape == (12, 27, 27)
    assert availability.shape == (12, 27, 27)

    # Simulate one completely available radar field.
    radar = np.ones(
        (27, 27),
        dtype=np.float32,
    ) * 35.0

    # Simulate one completely available lightning field.
    lightning = np.ones(
        (27, 27),
        dtype=np.float32
    ) * 4.0

    features, availability = build_fusion_frame(
        radar_reflectivity=radar,
        lightning_density=lightning,
    )

    print()
    print("Radar inserted            : YES")
    print("Lightning inserted        : YES")

    print_fusion_status(
        features,
        availability,
    )

    assert np.allclose(
        features[RADAR_REFLECTIVITY],
        35.0,
    )

    assert np.allclose(
        features[LIGHTNING_DENSITY],
        4.0,
    )

    assert np.all(
        availability[RADAR_REFLECTIVITY] == 1
    )

    assert np.all(
        availability[LIGHTNING_DENSITY] == 1
    )

    # Verify unavailable sensors remain unavailable.
    assert np.all(
        availability[RADAR_VELOCITY] == 0
    )

    assert np.all(
        availability[LIGHTNING_RATE] == 0
    )

    assert np.isnan(
        features[RADAR_VELOCITY]
    ).all()

    assert np.isnan(
        features[LIGHTNING_RATE]
    ).all()

    print()
    print("Missing sensors preserved : YES")
    print("Availability masks valid  : YES")
    print()
    print("ADAPTER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_adapter()