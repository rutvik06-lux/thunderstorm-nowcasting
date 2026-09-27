# temporal_multimodal_nowcast.py
# Temporal multimodal observation-derived nowcast.
# Uses only observations available at or before the analysis time.
# Output values are Convective Nowcast Scores, NOT calibrated probabilities.

from pathlib import Path
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MULTIMODAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multimodal"
    / "odisha_multimodal_20200501.nc"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "temporal_multimodal_nowcast_20200501_0430.nc"
)

ANALYSIS_INDEX = 1  # 04:30 UTC
HORIZONS = 4        # +30, +60, +90, +120


def normalize(field):
    """Min-max normalize a field safely."""
    field = np.asarray(field, dtype=np.float32)

    finite = np.isfinite(field)

    if not np.any(finite):
        return np.zeros_like(field, dtype=np.float32)

    values = field[finite]
    lo = np.percentile(values, 2)
    hi = np.percentile(values, 98)

    if hi <= lo:
        return np.zeros_like(field, dtype=np.float32)

    result = (field - lo) / (hi - lo)
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def spatial_gradient(field):
    """Return normalized spatial gradient magnitude."""
    gy, gx = np.gradient(field)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return normalize(magnitude)


def safe_get(ds, name, index):
    """Read a time slice and replace invalid values."""
    if name not in ds:
        raise KeyError(f"Missing required variable: {name}")

    value = ds[name].isel(time=index).values.astype(np.float32)
    return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def build_score(ds, current_idx, previous_idx=None):
    """
    Build a multimodal convective score using observations available
    at the current analysis time and optionally the immediately
    preceding observation.

    Predictive modalities:
      INSAT + IMERG + ERA5

    LIS is intentionally excluded from prediction.
    """

    # ------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------

    insat_availability = safe_get(
        ds, "insat_availability", current_idx
    )

    imerg_availability = safe_get(
        ds, "imerg_availability", current_idx
    )

    era5_availability = safe_get(
        ds, "era5_availability", current_idx
    )

    # ------------------------------------------------------------
    # INSAT
    # ------------------------------------------------------------

    tir1 = safe_get(ds, "insat_tir1", current_idx)
    tir2 = safe_get(ds, "insat_tir2", current_idx)
    wv = safe_get(ds, "insat_wv", current_idx)

    tir1_n = normalize(tir1)
    tir2_n = normalize(tir2)
    wv_n = normalize(wv)

    # Colder cloud tops -> stronger convective signal.
    cloud_coldness = 1.0 - tir1_n

    if previous_idx is not None:
        previous_tir1 = safe_get(ds, "insat_tir1", previous_idx)

        # Positive value means current TIR1 became colder.
        cooling = normalize(previous_tir1 - tir1)
    else:
        cooling = np.zeros_like(tir1, dtype=np.float32)

    satellite_evolution = (
        0.45 * cooling
        + 0.30 * cloud_coldness
        + 0.15 * (1.0 - tir2_n)
        + 0.10 * wv_n
    )

    # ------------------------------------------------------------
    # IMERG
    # ------------------------------------------------------------

    precipitation = safe_get(
        ds, "imerg_precipitation", current_idx
    )

    precipitation_n = normalize(precipitation)
    precipitation_gradient = spatial_gradient(precipitation)

    if previous_idx is not None:
        previous_precipitation = safe_get(
            ds, "imerg_precipitation", previous_idx
        )

        precipitation_growth = normalize(
            precipitation - previous_precipitation
        )
    else:
        precipitation_growth = np.zeros_like(
            precipitation,
            dtype=np.float32
        )

    precipitation_score = (
        0.50 * precipitation_n
        + 0.30 * precipitation_growth
        + 0.20 * precipitation_gradient
    )

    # ------------------------------------------------------------
    # ERA5 environment
    # ------------------------------------------------------------

    cape = normalize(
        safe_get(ds, "era5_cape", current_idx)
    )

    t2m = safe_get(ds, "era5_t2m", current_idx)
    d2m = safe_get(ds, "era5_d2m", current_idx)

    moisture = normalize(
        -(t2m - d2m)
    )

    u10 = safe_get(ds, "era5_u10", current_idx)
    v10 = safe_get(ds, "era5_v10", current_idx)

    wind_speed = normalize(
        np.sqrt(u10 ** 2 + v10 ** 2)
    )

    tcc = normalize(
        safe_get(ds, "era5_tcc", current_idx)
    )

    environmental_score = (
        0.45 * cape
        + 0.30 * moisture
        + 0.15 * wind_speed
        + 0.10 * tcc
    )

    # ------------------------------------------------------------
    # Multimodal fusion
    # ------------------------------------------------------------

    risk = (
        0.40 * environmental_score
        + 0.35 * satellite_evolution
        + 0.25 * precipitation_score
    )

    # ------------------------------------------------------------
    # Reliability-aware fusion
    # ------------------------------------------------------------

    confidence = (
        0.40 * insat_availability
        + 0.35 * imerg_availability
        + 0.25 * era5_availability
    )

    score = risk * confidence

    return (
        np.clip(score, 0.0, 1.0).astype(np.float32),
        confidence.astype(np.float32),
    )


def temporal_nowcast():
    """Generate +30/+60/+90/+120 minute nowcast scores."""

    if not MULTIMODAL_PATH.exists():
        raise FileNotFoundError(
            f"Multimodal dataset not found:\n{MULTIMODAL_PATH}"
        )

    ds = xr.open_dataset(MULTIMODAL_PATH)

    times = ds.time.values

    analysis_time = times[ANALYSIS_INDEX]

    print("=" * 70)
    print("TEMPORAL MULTIMODAL NOWCAST")
    print("=" * 70)
    print(f"Dataset: {MULTIMODAL_PATH}")
    print(f"Analysis time: {analysis_time}")
    print()

    # ------------------------------------------------------------
    # Current score
    # ------------------------------------------------------------

    current_score, confidence = build_score(
        ds,
        current_idx=ANALYSIS_INDEX,
        previous_idx=ANALYSIS_INDEX - 1,
    )

    # ------------------------------------------------------------
    # Generate four future horizons.
    #
    # Because we do not have future observations at inference time,
    # the current convective field is evolved using persistence with
    # horizon-dependent decay.
    # ------------------------------------------------------------

    forecast_maps = []

    horizon_decay = [
        1.00,  # +30
        0.90,  # +60
        0.78,  # +90
        0.65,  # +120
    ]

    for decay in horizon_decay:
        forecast = np.clip(
            current_score * decay,
            0.0,
            1.0,
        ).astype(np.float32)

        forecast_maps.append(forecast)

    forecast_maps = np.stack(forecast_maps, axis=0)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    y = np.arange(current_score.shape[0])
    x = np.arange(current_score.shape[1])

    horizon_minutes = np.array(
        [30, 60, 90, 120],
        dtype=np.int32,
    )

    output = xr.Dataset(
        {
            "nowcast_score": (
                ("horizon", "y", "x"),
                forecast_maps,
            ),
            "analysis_score": (
                ("y", "x"),
                current_score,
            ),
            "data_confidence": (
                ("y", "x"),
                confidence,
            ),
        },
        coords={
            "horizon": horizon_minutes,
            "y": y,
            "x": x,
        },
        attrs={
            "description": (
                "Temporal multimodal observation-derived "
                "thunderstorm nowcast"
            ),
            "analysis_time": str(analysis_time),
            "horizons": "+30,+60,+90,+120 minutes",
            "predictive_modalities": (
                "INSAT, IMERG, ERA5"
            ),
            "validation_modality": "NASA ISS LIS",
            "probability": (
                "No. Values are convective nowcast scores, "
                "not calibrated probabilities."
            ),
            "leakage_control": (
                "Prediction uses only observations available "
                "at or before analysis time."
            ),
            "temporal_strategy": (
                "Persistence-based horizon evolution; "
                "learned temporal model is a future stage."
            ),
        },
    )

    output.to_netcdf(OUTPUT_PATH)

    print("OUTPUT")
    print("-" * 70)
    print(f"Saved: {OUTPUT_PATH}")
    print()

    for i, horizon in enumerate(horizon_minutes):
        field = forecast_maps[i]

        print(
            f"+{horizon:3d} min | "
            f"min={field.min():.4f} | "
            f"max={field.max():.4f} | "
            f"mean={field.mean():.4f}"
        )

    print()
    print("TEMPORAL MULTIMODAL NOWCAST COMPLETE.")
    print("=" * 70)

    ds.close()


if __name__ == "__main__":
    temporal_nowcast()
