
from pathlib import Path
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter


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
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

ANALYSIS_TIME = np.datetime64("2020-05-01T04:30:00")
PREVIOUS_TIME = np.datetime64("2020-05-01T04:00:00")

TARGET_START = np.datetime64("2020-05-01T05:00:00")
TARGET_END = np.datetime64("2020-05-01T05:30:00")

GRID_SIZE = 27

# Motion search range in grid cells.
# One grid cell is roughly 7-8 km for this prototype.
MAX_SHIFT = 8

# If correlation is too weak, don't trust the estimated motion.
MIN_CORRELATION = 0.10


# ============================================================
# HELPERS
# ============================================================

def normalize_field(field):
    """
    Robust min-max normalization.

    Percentile clipping prevents one extreme pixel from dominating
    the whole field.
    """

    field = np.asarray(field, dtype=np.float32)

    valid = np.isfinite(field)

    if not np.any(valid):
        return np.zeros_like(field, dtype=np.float32)

    values = field[valid]

    low = np.percentile(values, 5)
    high = np.percentile(values, 95)

    if high <= low:
        return np.zeros_like(field, dtype=np.float32)

    result = (field - low) / (high - low)
    result = np.clip(result, 0.0, 1.0)

    result[~valid] = 0.0

    return result.astype(np.float32)


def temporal_growth(current, previous):
    """
    Positive temporal increase.

    Used for precipitation growth and satellite cooling.
    """

    current = np.asarray(current, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32)

    growth = current - previous

    growth[~np.isfinite(growth)] = 0.0

    return growth


def shift_field(field, dy, dx):
    """
    Shift a 2-D field without wrapping values around the boundary.

    Positive dy = downward/southward grid movement.
    Positive dx = right/eastward grid movement.
    """

    h, w = field.shape

    output = np.zeros_like(field)

    source_y_start = max(0, -dy)
    source_y_end = min(h, h - dy)

    source_x_start = max(0, -dx)
    source_x_end = min(w, w - dx)

    target_y_start = max(0, dy)
    target_y_end = target_y_start + (source_y_end - source_y_start)

    target_x_start = max(0, dx)
    target_x_end = target_x_start + (source_x_end - source_x_start)

    if (
        source_y_end > source_y_start
        and source_x_end > source_x_start
        and target_y_end > target_y_start
        and target_x_end > target_x_start
    ):
        output[
            target_y_start:target_y_end,
            target_x_start:target_x_end
        ] = field[
            source_y_start:source_y_end,
            source_x_start:source_x_end
        ]

    return output


def estimate_motion(previous_field, current_field):
    """
    Estimate integer grid-cell motion using 2-D cross-correlation.

    The dynamic convection field at 04:00 is compared with the
    dynamic convection field at 04:30.

    Returns:
        dy, dx, correlation_score
    """

    previous = previous_field - np.mean(previous_field)
    current = current_field - np.mean(current_field)

    previous_std = np.std(previous)
    current_std = np.std(current)

    if previous_std < 1e-8 or current_std < 1e-8:
        return 0, 0, 0.0

    previous = previous / previous_std
    current = current / current_std

    correlation = correlate2d(
        current,
        previous,
        mode="full",
        boundary="fill",
        fillvalue=0
    )

    center_y = previous.shape[0] - 1
    center_x = previous.shape[1] - 1

    y_min = max(0, center_y - MAX_SHIFT)
    y_max = min(correlation.shape[0], center_y + MAX_SHIFT + 1)

    x_min = max(0, center_x - MAX_SHIFT)
    x_max = min(correlation.shape[1], center_x + MAX_SHIFT + 1)

    search_area = correlation[y_min:y_max, x_min:x_max]

    peak_index = np.unravel_index(
        np.argmax(search_area),
        search_area.shape
    )

    peak_y = y_min + peak_index[0]
    peak_x = x_min + peak_index[1]

    dy = peak_y - center_y
    dx = peak_x - center_x

    # Normalize correlation to make the diagnostic easier to interpret.
    denom = np.sqrt(
        np.sum(previous ** 2) *
        np.sum(current ** 2)
    )

    if denom > 0:
        score = float(correlation[peak_y, peak_x] / denom)
    else:
        score = 0.0

    return int(dy), int(dx), score


# ============================================================
# BUILD DYNAMIC CONVECTIVE FIELD
# ============================================================

def build_dynamic_field(ds, time):
    """
    Build a dynamic convection field using only information
    available at the specified timestamp.

    This deliberately excludes CAPE, pressure and other slowly
    varying environmental fields from motion estimation.

    Dynamic components:
        1. INSAT TIR cooling
        2. IMERG precipitation
        3. IMERG precipitation growth
        4. precipitation spatial gradient
    """

    times = ds.time.values

    current_idx = int(np.argmin(np.abs(times - time)))

    if current_idx == 0:
        previous_idx = 0
    else:
        previous_idx = current_idx - 1

    current = ds.isel(time=current_idx)
    previous = ds.isel(time=previous_idx)

    # --------------------------------------------------------
    # INSAT cloud-top cooling
    # --------------------------------------------------------

    tir_current = current["insat_tir1"].values.astype(np.float32)
    tir_previous = previous["insat_tir1"].values.astype(np.float32)

    # A decrease in brightness temperature means cooling.
    cooling = tir_previous - tir_current

    cooling = normalize_field(cooling)

    # --------------------------------------------------------
    # IMERG precipitation
    # --------------------------------------------------------

    precip_current = current["imerg_precipitation"].values.astype(
        np.float32
    )

    precip_previous = previous["imerg_precipitation"].values.astype(
        np.float32
    )

    precip_current = np.nan_to_num(
        precip_current,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    precip_previous = np.nan_to_num(
        precip_previous,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    precip = normalize_field(precip_current)

    precip_growth = temporal_growth(
        precip_current,
        precip_previous
    )

    precip_growth = normalize_field(
        np.maximum(precip_growth, 0.0)
    )

    # --------------------------------------------------------
    # Spatial precipitation gradient
    # --------------------------------------------------------

    grad_y, grad_x = np.gradient(precip_current)

    precip_gradient = np.sqrt(
        grad_y ** 2 +
        grad_x ** 2
    )

    precip_gradient = normalize_field(
        precip_gradient
    )

    # --------------------------------------------------------
    # Combine dynamic signals
    # --------------------------------------------------------

    dynamic_field = (
        0.35 * cooling
        + 0.30 * precip
        + 0.20 * precip_growth
        + 0.15 * precip_gradient
    )

    dynamic_field = np.nan_to_num(
        dynamic_field,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Smooth tiny pixel-scale noise.
    dynamic_field = gaussian_filter(
        dynamic_field,
        sigma=0.8
    )

    dynamic_field = normalize_field(
        dynamic_field
    )

    return dynamic_field


# ============================================================
# CREATE +30 MINUTE NOWCAST
# ============================================================

def build_motion_nowcast(ds):
    """
    Estimate motion from 04:00 -> 04:30 and advect the 04:30
    dynamic convective field forward by 30 minutes.
    """

    previous_field = build_dynamic_field(
        ds,
        PREVIOUS_TIME
    )

    current_field = build_dynamic_field(
        ds,
        ANALYSIS_TIME
    )

    dy, dx, correlation = estimate_motion(
        previous_field,
        current_field
    )

    print()
    print("=" * 70)
    print("MOTION NOWCAST")
    print("=" * 70)

    print(f"Previous time : {PREVIOUS_TIME}")
    print(f"Analysis time : {ANALYSIS_TIME}")
    print(f"Target window : {TARGET_START} -> {TARGET_END}")

    print()
    print("Estimated grid motion:")
    print(f"  dy = {dy} cells")
    print(f"  dx = {dx} cells")

    print()
    print(f"Motion correlation score: {correlation:.3f}")

    if correlation < MIN_CORRELATION:
        print(
            "WARNING: correlation is weak. "
            "Motion estimate may be unreliable."
        )

        dy = 0
        dx = 0

        print("Fallback: zero-motion persistence.")

    forecast = shift_field(
        current_field,
        dy,
        dx
    )

    forecast = normalize_field(
        forecast
    )

    return (
        previous_field,
        current_field,
        forecast,
        dy,
        dx,
        correlation
    )


# ============================================================
# VALIDATION AGAINST LIS
# ============================================================

def validate_nowcast(
    forecast,
    target_ds
):
    """
    Compare the motion nowcast with the actual lightning target
    for 05:00-05:30.

    Metrics:
        - maximum forecast location
        - lightning centroid
        - distance between them
        - lightning recall in top 10% forecast cells
    """

    target_times = target_ds.time.values

    target_idx = int(
        np.argmin(
            np.abs(
                target_times - TARGET_START
            )
        )
    )

    # The actual target variable in the NetCDF is
    # "lightning_target".
    target = target_ds["lightning_target"].isel(
        time=target_idx
    ).values

    target_available = target_ds[
        "target_availability"
    ].isel(
        time=target_idx
    ).values

    target = np.asarray(target)
    target_available = np.asarray(
        target_available
    ).astype(bool)

    valid_lightning = (
        (target > 0.5)
        & target_available
    )

    lightning_positions = np.argwhere(
        valid_lightning
    )

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        f"Target time: {target_times[target_idx]}"
    )

    print(
        f"Actual positive cells: "
        f"{len(lightning_positions)}"
    )

    if len(lightning_positions) == 0:
        print("No observed lightning cells.")
        return {}

    # --------------------------------------------------------
    # Maximum forecast cell
    # --------------------------------------------------------

    forecast_max_position = np.unravel_index(
        np.argmax(forecast),
        forecast.shape
    )

    # --------------------------------------------------------
    # Actual lightning centroid
    # --------------------------------------------------------

    lightning_centroid = (
        lightning_positions[:, 0].mean(),
        lightning_positions[:, 1].mean()
    )

    distance = np.sqrt(
        (
            forecast_max_position[0]
            - lightning_centroid[0]
        ) ** 2
        +
        (
            forecast_max_position[1]
            - lightning_centroid[1]
        ) ** 2
    )

    # --------------------------------------------------------
    # Top 10%
    # --------------------------------------------------------

    valid_forecast = forecast[
        target_available
    ]

    threshold = np.percentile(
        valid_forecast,
        90
    )

    top_area = (
        (forecast >= threshold)
        & target_available
    )

    lightning_in_top = np.sum(
        valid_lightning
        & top_area
    )

    recall = (
        lightning_in_top
        / len(lightning_positions)
    )

    print()
    print(
        f"Forecast maximum: "
        f"row={forecast_max_position[0]}, "
        f"col={forecast_max_position[1]}"
    )

    print(
        f"Lightning centroid: "
        f"row={lightning_centroid[0]:.2f}, "
        f"col={lightning_centroid[1]:.2f}"
    )

    print(
        f"Grid-cell distance to centroid: "
        f"{distance:.3f}"
    )

    print(
        f"Top 10% threshold: "
        f"{threshold:.4f}"
    )

    print(
        f"Lightning cells inside top 10%: "
        f"{lightning_in_top}"
    )

    print(
        f"Lightning recall @ top 10%: "
        f"{recall:.3f}"
    )

    return {
        "forecast_max_row": forecast_max_position[0],
        "forecast_max_col": forecast_max_position[1],
        "lightning_centroid_row": lightning_centroid[0],
        "lightning_centroid_col": lightning_centroid[1],
        "centroid_distance": float(distance),
        "top10_threshold": float(threshold),
        "top10_lightning_cells": int(lightning_in_top),
        "top10_recall": float(recall),
    }


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(
    previous_field,
    current_field,
    forecast,
    dy,
    dx,
    correlation,
    validation
):
    output_nc = (
        OUTPUT_DIR
        / "motion_nowcast_20200501_0430.nc"
    )

    output_png = (
        OUTPUT_DIR
        / "motion_nowcast_20200501_0430.png"
    )

    y = np.arange(GRID_SIZE)
    x = np.arange(GRID_SIZE)

    ds_out = xr.Dataset(
        {
            "dynamic_field_0400": (
                ("y", "x"),
                previous_field
            ),

            "dynamic_field_0430": (
                ("y", "x"),
                current_field
            ),

            "motion_nowcast_0500": (
                ("y", "x"),
                forecast
            ),
        },
        coords={
            "y": y,
            "x": x,
        },
        attrs={
            "description":
                "Observation-derived motion nowcast baseline",

            "analysis_time":
                str(ANALYSIS_TIME),

            "target_window":
                f"{TARGET_START} to {TARGET_END}",

            "motion_dy_cells":
                int(dy),

            "motion_dx_cells":
                int(dx),

            "motion_correlation":
                float(correlation),

            "method":
                "2D cross-correlation followed by integer-grid advection",

            "probability":
                "No. Values are a convective nowcast score, not calibrated probability.",

            "leakage_control":
                "Forecast uses only 04:00 and 04:30 observations."
        }
    )

    ds_out.to_netcdf(
        output_nc
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    im0 = axes[0].imshow(
        previous_field,
        origin="lower"
    )

    axes[0].set_title(
        "Dynamic convection\n04:00 UTC"
    )

    axes[0].set_xlabel("Grid X")
    axes[0].set_ylabel("Grid Y")

    fig.colorbar(
        im0,
        ax=axes[0],
        fraction=0.046
    )

    im1 = axes[1].imshow(
        current_field,
        origin="lower"
    )

    axes[1].set_title(
        "Dynamic convection\n04:30 UTC"
    )

    axes[1].set_xlabel("Grid X")
    axes[1].set_ylabel("Grid Y")

    fig.colorbar(
        im1,
        ax=axes[1],
        fraction=0.046
    )

    im2 = axes[2].imshow(
        forecast,
        origin="lower"
    )

    axes[2].set_title(
        "Motion nowcast\n05:00 UTC"
    )

    axes[2].set_xlabel("Grid X")
    axes[2].set_ylabel("Grid Y")

    fig.colorbar(
        im2,
        ax=axes[2],
        fraction=0.046
    )

    fig.suptitle(
        "Motion-Aware Thunderstorm Nowcasting — Odisha Case Study",
        fontsize=14
    )

    fig.tight_layout()

    fig.savefig(
        output_png,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_nc, output_png


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MOTION-AWARE THUNDERSTORM NOWCASTING")
    print("=" * 70)

    print()
    print("Loading multimodal dataset:")
    print(MULTIMODAL_PATH)

    ds = xr.open_dataset(
        MULTIMODAL_PATH
    )

    print()
    print("Dataset loaded.")

    print("Dimensions:")
    for name, size in ds.sizes.items():
        print(f"  {name}: {size}")

    print(
        f"Time frames: "
        f"{ds.sizes['time']}"
    )

    # --------------------------------------------------------
    # Build nowcast
    # --------------------------------------------------------

    (
        previous_field,
        current_field,
        forecast,
        dy,
        dx,
        correlation
    ) = build_motion_nowcast(
        ds
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print()
    print("Loading lightning target:")
    print(TARGET_PATH)

    target_ds = xr.open_dataset(
        TARGET_PATH
    )

    validation = validate_nowcast(
        forecast,
        target_ds
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_nc, output_png = save_output(
        previous_field,
        current_field,
        forecast,
        dy,
        dx,
        correlation,
        validation
    )

    print()
    print("=" * 70)
    print("OUTPUTS")
    print("=" * 70)

    print()
    print(f"NetCDF : {output_nc}")
    print(f"Figure : {output_png}")

    print()
    print("MOTION NOWCAST COMPLETE.")
    print("=" * 70)

    ds.close()
    target_ds.close()


if __name__ == "__main__":
    main()

