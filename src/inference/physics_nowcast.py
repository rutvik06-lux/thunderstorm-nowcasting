from pathlib import Path
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MULTIMODAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multimodal"
    / "odisha_multimodal_20200501.nc"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "targets"
    / "lightning_targets_odisha_20200501.nc"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

ANALYSIS_TIME = np.datetime64("2020-05-01T04:30:00")
TARGET_START = np.datetime64("2020-05-01T05:00:00")
TARGET_END = np.datetime64("2020-05-01T05:30:00")


# ============================================================
# HELPERS
# ============================================================

def normalize_field(field, valid_mask=None):
    """
    Robust 0-1 normalization using percentiles.

    This is NOT probability calibration.
    It only puts different physical variables onto a
    comparable numerical scale.
    """

    field = np.asarray(field, dtype=np.float32)

    if valid_mask is None:
        valid_mask = np.isfinite(field)

    valid = field[valid_mask]

    if valid.size == 0:
        return np.zeros_like(field, dtype=np.float32)

    low = np.nanpercentile(valid, 5)
    high = np.nanpercentile(valid, 95)

    if high <= low:
        return np.zeros_like(field, dtype=np.float32)

    result = (field - low) / (high - low)
    result = np.clip(result, 0.0, 1.0)

    result[~np.isfinite(result)] = 0.0

    return result.astype(np.float32)


def normalize_inverted(field, valid_mask=None):
    """
    Normalize a field where LOWER values correspond to
    stronger convective signal.

    Example:
        lower cloud-top temperature
        -> stronger deep convection signal
    """

    normalized = normalize_field(field, valid_mask)

    return 1.0 - normalized


def safe_get(ds, name):
    """
    Retrieve a variable from xarray dataset.
    """

    if name not in ds:
        raise KeyError(
            f"Required variable '{name}' was not found.\n"
            f"Available variables:\n{list(ds.data_vars)}"
        )

    return ds[name]


def spatial_gradient_magnitude(field):
    """
    Simple spatial gradient magnitude.
    """

    field = np.asarray(field, dtype=np.float32)

    gy, gx = np.gradient(field)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    return magnitude.astype(np.float32)


def temporal_difference(current, previous):
    """
    Current minus previous frame.
    """

    return (
        np.asarray(current, dtype=np.float32)
        - np.asarray(previous, dtype=np.float32)
    )


# ============================================================
# MAIN BASELINE
# ============================================================

def build_nowcast():

    print("=" * 70)
    print("PHYSICS / OBSERVATION-DERIVED THUNDERSTORM NOWCAST")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nMultimodal dataset:")
    print(MULTIMODAL_PATH)

    if not MULTIMODAL_PATH.exists():
        raise FileNotFoundError(
            f"\nMultimodal dataset not found:\n{MULTIMODAL_PATH}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    ds = xr.open_dataset(MULTIMODAL_PATH)

    print("\nDataset loaded.")
    print(f"Dimensions: {dict(ds.sizes)}")

    print("\nAvailable variables:")
    for variable in ds.data_vars:
        print(f"  - {variable}")

    # --------------------------------------------------------
    # FIND ANALYSIS FRAMES
    # --------------------------------------------------------

    times = ds.time.values

    if ANALYSIS_TIME not in times:
        raise ValueError(
            f"\nAnalysis time {ANALYSIS_TIME} is not available.\n"
            f"Available times:\n{times}"
        )

    analysis_index = int(np.where(times == ANALYSIS_TIME)[0][0])

    if analysis_index == 0:
        raise ValueError(
            "Need a previous frame to calculate temporal evolution."
        )

    previous_time = times[analysis_index - 1]

    print("\nAnalysis configuration:")
    print(f"  Previous frame : {previous_time}")
    print(f"  Analysis frame : {ANALYSIS_TIME}")
    print(f"  Target interval: {TARGET_START} -> {TARGET_END}")

    # --------------------------------------------------------
    # LOAD CURRENT / PREVIOUS DATA
    # --------------------------------------------------------

    current = ds.isel(time=analysis_index)
    previous = ds.isel(time=analysis_index - 1)

    # ========================================================
    # 1. ATMOSPHERIC INSTABILITY
    # ========================================================

    cape = safe_get(current, "era5_cape").values.astype(np.float32)

    cape_signal = normalize_field(cape)

    print("\nCAPE:")
    print(f"  min = {np.nanmin(cape):.2f}")
    print(f"  max = {np.nanmax(cape):.2f}")
    print(f"  mean = {np.nanmean(cape):.2f}")

    # ========================================================
    # 2. MOISTURE
    # ========================================================

    t2m = safe_get(current, "era5_t2m").values.astype(np.float32)
    d2m = safe_get(current, "era5_d2m").values.astype(np.float32)

    dewpoint_depression = t2m - d2m

    # Smaller dewpoint depression means moister air.
    moisture_signal = normalize_inverted(dewpoint_depression)

    print("\nMoisture:")
    print(
        f"  dewpoint depression = "
        f"{np.nanmin(dewpoint_depression):.2f} -> "
        f"{np.nanmax(dewpoint_depression):.2f} K"
    )

    # ========================================================
    # 3. SATELLITE CLOUD-TOP TEMPERATURE
    # ========================================================

    tir1 = safe_get(current, "insat_tir1").values.astype(np.float32)
    tir1_previous = safe_get(
        previous, "insat_tir1"
    ).values.astype(np.float32)

    tir_cooling = temporal_difference(
        tir1,
        tir1_previous,
    )

    # Negative change = cooling cloud tops.
    # Convert this into a positive convection signal.
    cooling_signal = normalize_inverted(tir_cooling)

    # Remove the interpretation that warm/noisy pixels are
    # automatically strong convection.
    cooling_signal = np.clip(
        -normalize_field(tir_cooling),
        0.0,
        1.0,
    )

    print("\nINSAT TIR1:")
    print(
        f"  current = "
        f"{np.nanmin(tir1):.2f} -> {np.nanmax(tir1):.2f}"
    )

    print(
        f"  cooling = "
        f"{np.nanmin(tir_cooling):.2f} -> "
        f"{np.nanmax(tir_cooling):.2f}"
    )

    # ========================================================
    # 4. PRECIPITATION
    # ========================================================

    precipitation = safe_get(
        current,
        "imerg_precipitation",
    ).values.astype(np.float32)

    precipitation_previous = safe_get(
        previous,
        "imerg_precipitation",
    ).values.astype(np.float32)

    precipitation_change = temporal_difference(
        precipitation,
        precipitation_previous,
    )

    precipitation_signal = normalize_field(precipitation)

    precipitation_growth_signal = normalize_field(
        np.maximum(precipitation_change, 0.0)
    )

    # ========================================================
    # 5. SPATIAL CONVECTIVE STRUCTURE
    # ========================================================

    precipitation_gradient = spatial_gradient_magnitude(
        precipitation
    )

    gradient_signal = normalize_field(
        precipitation_gradient
    )

    # ========================================================
    # 6. WIND ENVIRONMENT
    # ========================================================

    u10 = safe_get(current, "era5_u10").values.astype(np.float32)
    v10 = safe_get(current, "era5_v10").values.astype(np.float32)

    wind_speed = np.sqrt(
        u10 ** 2 +
        v10 ** 2
    )

    wind_signal = normalize_field(wind_speed)

    # ========================================================
    # 7. RELIABILITY / AVAILABILITY MASKS
    # ========================================================

    # INSAT availability
    insat_availability = (
        safe_get(current, "insat_availability")
        .values.astype(np.float32)
    )

    # IMERG availability
    imerg_availability = (
        safe_get(current, "imerg_availability")
        .values.astype(np.float32)
    )

    # ERA5 availability
    era5_availability = (
        safe_get(current, "era5_availability")
        .values.astype(np.float32)
    )

    # ========================================================
    # 8. BUILD ENVIRONMENT SCORE
    # ========================================================

    environment_score = (
        0.50 * cape_signal
        +
        0.30 * moisture_signal
        +
        0.20 * wind_signal
    )

    # ========================================================
    # 9. BUILD INITIATION SCORE
    # ========================================================

    initiation_score = (
        0.45 * cooling_signal
        +
        0.35 * precipitation_growth_signal
        +
        0.20 * gradient_signal
    )

    # ========================================================
    # 10. CURRENT CONVECTION SCORE
    # ========================================================

    current_convection = (
        0.60 * precipitation_signal
        +
        0.40 * gradient_signal
    )

    # ========================================================
    # 11. MULTIMODAL FUSION
    # ========================================================

    raw_score = (
        0.40 * environment_score
        +
        0.40 * initiation_score
        +
        0.20 * current_convection
    )

    # ========================================================
    # 12. RELIABILITY MASK
    # ========================================================

    # Each source contributes to reliability.
    #
    # This is NOT a probability.
    #
    # It tells us how well-supported the score is by
    # available observations.

    source_reliability = (
        0.40 * insat_availability
        +
        0.35 * imerg_availability
        +
        0.25 * era5_availability
    )

    source_reliability = np.clip(
        source_reliability,
        0.0,
        1.0,
    )

    # ========================================================
    # 13. RELIABILITY-AWARE SCORE
    # ========================================================

    nowcast_score = raw_score * source_reliability

    nowcast_score = np.clip(
        nowcast_score,
        0.0,
        1.0,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("NOWCAST RESULTS")
    print("=" * 70)

    print(
        f"\nRaw convective score:"
        f" {np.nanmin(raw_score):.3f} -> "
        f"{np.nanmax(raw_score):.3f}"
    )

    print(
        f"Reliability:"
        f" {np.nanmin(source_reliability):.3f} -> "
        f"{np.nanmax(source_reliability):.3f}"
    )

    print(
        f"Final nowcast score:"
        f" {np.nanmin(nowcast_score):.3f} -> "
        f"{np.nanmax(nowcast_score):.3f}"
    )

    max_index = np.unravel_index(
        np.nanargmax(nowcast_score),
        nowcast_score.shape,
    )

    print(
        f"\nMaximum-score grid cell:"
        f" row={max_index[0]}, col={max_index[1]}"
    )

    print(
        f"Maximum score:"
        f" {nowcast_score[max_index]:.3f}"
    )

    # ========================================================
    # LOAD TARGET
    # ========================================================

    target = None

    if TARGET_PATH.exists():

        target_ds = xr.open_dataset(TARGET_PATH)

        target_times = target_ds.time.values

        print("\nTarget dataset loaded.")

        matching = np.where(
            target_times == TARGET_START
        )[0]

        if matching.size > 0:

            target_index = int(matching[0])

            target = target_ds["lightning_target"].isel(
                time=target_index
            ).values.astype(np.float32)

            target_availability = (
                target_ds["target_availability"]
                .isel(time=target_index)
                .values.astype(np.float32)
            )

            target = np.nan_to_num(
                target,
                nan=0.0,
            )

            target_availability = np.nan_to_num(
                target_availability,
                nan=0.0,
            )

            observed_target_pixels = (
                target_availability > 0
            )

            positive_pixels = (
                (target > 0)
                &
                observed_target_pixels
            )

            print(
                f"Target positive pixels:"
                f" {int(positive_pixels.sum())}"
            )

            print(
                f"Observed target pixels:"
                f" {int(observed_target_pixels.sum())}"
            )

        target_ds.close()

    # ========================================================
    # SAVE NUMERICAL RESULT
    # ========================================================

    output_nc = (
        OUTPUT_DIR
        / "physics_nowcast_20200501_0430.nc"
    )

    result = xr.Dataset(
        data_vars={
            "nowcast_score": (
                ("latitude", "longitude"),
                nowcast_score,
            ),
            "environment_score": (
                ("latitude", "longitude"),
                environment_score,
            ),
            "initiation_score": (
                ("latitude", "longitude"),
                initiation_score,
            ),
            "current_convection": (
                ("latitude", "longitude"),
                current_convection,
            ),
            "source_reliability": (
                ("latitude", "longitude"),
                source_reliability,
            ),
        },
        coords={
            "latitude": ds.latitude.values,
            "longitude": ds.longitude.values,
        },
        attrs={
            "title": (
                "Observation-derived convective nowcast baseline"
            ),
            "analysis_time": str(ANALYSIS_TIME),
            "target_interval": (
                f"{TARGET_START} to {TARGET_END}"
            ),
            "interpretation": (
                "Uncalibrated convective nowcast score, "
                "not probability."
            ),
            "data_type": "Real observations",
        },
    )

    result.to_netcdf(output_nc)

    print(f"\nSaved numerical result:")
    print(output_nc)

    # ========================================================
    # VISUALIZATION
    # ========================================================

    latitude = ds.latitude.values
    longitude = ds.longitude.values

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10),
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    im0 = axes[0, 0].imshow(
        environment_score,
        origin="upper",
        aspect="auto",
    )

    axes[0, 0].set_title(
        "Environmental Potential"
    )

    plt.colorbar(
        im0,
        ax=axes[0, 0],
        fraction=0.046,
        pad=0.04,
    )

    # --------------------------------------------------------
    # Initiation
    # --------------------------------------------------------

    im1 = axes[0, 1].imshow(
        initiation_score,
        origin="upper",
        aspect="auto",
    )

    axes[0, 1].set_title(
        "Convective Initiation Signal"
    )

    plt.colorbar(
        im1,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04,
    )

    # --------------------------------------------------------
    # Current convection
    # --------------------------------------------------------

    im2 = axes[0, 2].imshow(
        current_convection,
        origin="upper",
        aspect="auto",
    )

    axes[0, 2].set_title(
        "Current Convection"
    )

    plt.colorbar(
        im2,
        ax=axes[0, 2],
        fraction=0.046,
        pad=0.04,
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    im3 = axes[1, 0].imshow(
        source_reliability,
        origin="upper",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    axes[1, 0].set_title(
        "Observation Reliability"
    )

    plt.colorbar(
        im3,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.04,
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    im4 = axes[1, 1].imshow(
        nowcast_score,
        origin="upper",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    axes[1, 1].set_title(
        "Final Convective Nowcast Score"
    )

    plt.colorbar(
        im4,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.04,
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if target is not None:

        im5 = axes[1, 2].imshow(
            target,
            origin="upper",
            aspect="auto",
            vmin=0,
            vmax=1,
        )

        axes[1, 2].set_title(
            "Observed LIS Lightning Target"
        )

        plt.colorbar(
            im5,
            ax=axes[1, 2],
            fraction=0.046,
            pad=0.04,
        )

        # Overlay target on score
        axes[1, 1].contour(
            target > 0,
            levels=[0.5],
            linewidths=1.5,
        )

    else:

        axes[1, 2].text(
            0.5,
            0.5,
            "Target unavailable",
            ha="center",
            va="center",
        )

        axes[1, 2].set_title(
            "Observed LIS Lightning Target"
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    for ax in axes.flat:
        ax.set_xlabel("Longitude grid")
        ax.set_ylabel("Latitude grid")

    fig.suptitle(
        "May 1, 2020 — Observation-Derived Thunderstorm Nowcast\n"
        "Analysis: 04:30 UTC → Target: 05:00–05:30 UTC",
        fontsize=16,
    )

    plt.tight_layout()

    output_png = (
        OUTPUT_DIR
        / "physics_nowcast_20200501_0430.png"
    )

    plt.savefig(
        output_png,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved visualization:")
    print(output_png)

    ds.close()

    print("\n" + "=" * 70)
    print("PHYSICS NOWCAST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    build_nowcast()