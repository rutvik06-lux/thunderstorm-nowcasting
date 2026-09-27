from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "targets"
    / "lightning_targets_odisha_20200501.nc"
)

PHYSICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "physics_nowcast_20200501_0430.nc"
)

MOTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "motion_nowcast_20200501_0430.nc"
)

MULTIMODAL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "multimodal_inference_20200501_0430.nc"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_TIME = np.datetime64(
    "2020-05-01T05:00:00"
)


# ============================================================
# HELPERS
# ============================================================

def percentile_threshold(
    forecast,
    available,
    percentile=90,
):
    values = forecast[available]

    if values.size == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            percentile
        )
    )


def evaluate_forecast(
    name,
    forecast,
    target,
    target_available,
):
    """
    Evaluate one forecast against observed lightning.

    Metrics:
        - forecast maximum location
        - lightning centroid
        - centroid distance
        - top 10% recall
        - top 20% recall
        - forecast mean/max
    """

    forecast = np.asarray(
        forecast,
        dtype=np.float32
    )

    target = np.asarray(
        target,
        dtype=np.float32
    )

    target_available = np.asarray(
        target_available
    ).astype(bool)

    valid_forecast = (
        np.isfinite(forecast) &
        target_available
    )

    if not np.any(valid_forecast):
        raise ValueError(
            f"{name}: no valid forecast cells."
        )

    forecast_for_max = np.where(
        valid_forecast,
        forecast,
        -np.inf
    )

    max_row, max_col = np.unravel_index(
        np.argmax(forecast_for_max),
        forecast.shape
    )

    lightning_mask = (
        (target > 0) &
        target_available
    )

    lightning_rows, lightning_cols = np.where(
        lightning_mask
    )

    lightning_cells = len(
        lightning_rows
    )

    if lightning_cells > 0:

        centroid_row = float(
            lightning_rows.mean()
        )

        centroid_col = float(
            lightning_cols.mean()
        )

        centroid_distance = float(
            np.sqrt(
                (max_row - centroid_row) ** 2 +
                (max_col - centroid_col) ** 2
            )
        )

    else:

        centroid_row = np.nan
        centroid_col = np.nan
        centroid_distance = np.nan

    # --------------------------------------------------------
    # TOP 10%
    # --------------------------------------------------------

    threshold_10 = percentile_threshold(
        forecast,
        valid_forecast,
        90
    )

    top10_mask = (
        forecast >= threshold_10
    ) & valid_forecast

    lightning_in_top10 = int(
        np.sum(
            lightning_mask &
            top10_mask
        )
    )

    recall10 = (
        lightning_in_top10 /
        lightning_cells
        if lightning_cells > 0
        else np.nan
    )

    # --------------------------------------------------------
    # TOP 20%
    # --------------------------------------------------------

    threshold_20 = percentile_threshold(
        forecast,
        valid_forecast,
        80
    )

    top20_mask = (
        forecast >= threshold_20
    ) & valid_forecast

    lightning_in_top20 = int(
        np.sum(
            lightning_mask &
            top20_mask
        )
    )

    recall20 = (
        lightning_in_top20 /
        lightning_cells
        if lightning_cells > 0
        else np.nan
    )

    return {
        "name": name,

        "max_row": int(max_row),
        "max_col": int(max_col),

        "centroid_row": centroid_row,
        "centroid_col": centroid_col,

        "centroid_distance": centroid_distance,

        "lightning_cells": int(
            lightning_cells
        ),

        "top10_threshold": threshold_10,

        "lightning_in_top10": (
            lightning_in_top10
        ),

        "recall_top10": recall10,

        "top20_threshold": threshold_20,

        "lightning_in_top20": (
            lightning_in_top20
        ),

        "recall_top20": recall20,

        "forecast_mean": float(
            np.nanmean(
                forecast[
                    valid_forecast
                ]
            )
        ),

        "forecast_max": float(
            np.nanmax(
                forecast[
                    valid_forecast
                ]
            )
        ),
    }


# ============================================================
# LOAD TARGET
# ============================================================

def load_target():

    if not TARGET_PATH.exists():
        raise FileNotFoundError(
            f"Target file not found:\n"
            f"{TARGET_PATH}"
        )

    ds = xr.open_dataset(
        TARGET_PATH
    )

    times = ds["time"].values

    matches = np.where(
        times == TARGET_TIME
    )[0]

    if len(matches) == 0:
        raise ValueError(
            f"Target time not found: "
            f"{TARGET_TIME}"
        )

    idx = int(matches[0])

    target = ds[
        "lightning_target"
    ].isel(
        time=idx
    ).values

    target_available = ds[
        "target_availability"
    ].isel(
        time=idx
    ).values

    print()
    print(
        f"Target time: {TARGET_TIME}"
    )

    print(
        f"Target variable: lightning_target"
    )

    print(
        f"Observed target cells: "
        f"{int(np.sum(target_available > 0))}"
    )

    print(
        f"Lightning cells: "
        f"{int(np.sum(target > 0))}"
    )

    ds.close()

    return target, target_available


# ============================================================
# LOAD FORECASTS
# ============================================================

def load_forecasts():

    datasets = {}

    # --------------------------------------------------------
    # Physics
    # --------------------------------------------------------

    physics_ds = xr.open_dataset(
        PHYSICS_PATH
    )

    physics = physics_ds[
        "nowcast_score"
    ].values

    datasets["Physics"] = physics

    # --------------------------------------------------------
    # Motion
    # --------------------------------------------------------

    motion_ds = xr.open_dataset(
        MOTION_PATH
    )

    motion = motion_ds[
        "motion_nowcast_0500"
    ].values

    datasets["Motion"] = motion

    # --------------------------------------------------------
    # Multimodal
    # --------------------------------------------------------

    multimodal_ds = xr.open_dataset(
        MULTIMODAL_PATH
    )

    multimodal = multimodal_ds[
        "multimodal_nowcast"
    ].values

    datasets["Multimodal"] = multimodal

    physics_ds.close()
    motion_ds.close()
    multimodal_ds.close()

    return datasets


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print()
    print("=" * 90)
    print("NOWCAST COMPARISON")
    print("=" * 90)

    print()

    header = (
        f"{'Method':<15}"
        f"{'Max Cell':<14}"
        f"{'Centroid Dist':<17}"
        f"{'Top10 Recall':<16}"
        f"{'Top20 Recall':<16}"
        f"{'Mean':<10}"
        f"{'Max':<10}"
    )

    print(header)
    print("-" * 90)

    for result in results:

        max_cell = (
            f"({result['max_row']},"
            f"{result['max_col']})"
        )

        print(
            f"{result['name']:<15}"
            f"{max_cell:<14}"
            f"{result['centroid_distance']:<17.3f}"
            f"{result['recall_top10']:<16.3f}"
            f"{result['recall_top20']:<16.3f}"
            f"{result['forecast_mean']:<10.3f}"
            f"{result['forecast_max']:<10.3f}"
        )

    print()
    print(
        "Recall = fraction of observed lightning "
        "grid cells inside the forecast's highest-risk region."
    )

    print(
        "Scores are normalized risk scores, "
        "not calibrated probabilities."
    )

    print()
    print(
        "IMPORTANT: this is a single-event prototype "
        "comparison, not a general performance evaluation."
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results):

    output_path = (
        OUTPUT_DIR /
        "nowcast_comparison_20200501_0430.csv"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "method,"
            "max_row,"
            "max_col,"
            "centroid_distance,"
            "lightning_cells,"
            "top10_threshold,"
            "lightning_in_top10,"
            "recall_top10,"
            "top20_threshold,"
            "lightning_in_top20,"
            "recall_top20,"
            "forecast_mean,"
            "forecast_max\n"
        )

        for r in results:

            f.write(
                f"{r['name']},"
                f"{r['max_row']},"
                f"{r['max_col']},"
                f"{r['centroid_distance']},"
                f"{r['lightning_cells']},"
                f"{r['top10_threshold']},"
                f"{r['lightning_in_top10']},"
                f"{r['recall_top10']},"
                f"{r['top20_threshold']},"
                f"{r['lightning_in_top20']},"
                f"{r['recall_top20']},"
                f"{r['forecast_mean']},"
                f"{r['forecast_max']}\n"
            )

    return output_path


# ============================================================
# VISUAL COMPARISON
# ============================================================

def save_comparison_figure(
    forecasts,
    target,
    target_available,
):

    figure_path = (
        OUTPUT_DIR /
        "nowcast_comparison_20200501_0430.png"
    )

    lightning_mask = (
        (target > 0) &
        (target_available > 0)
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10)
    )

    panels = [
        (
            forecasts["Physics"],
            "Physics Baseline"
        ),
        (
            forecasts["Motion"],
            "Motion Baseline"
        ),
        (
            forecasts["Multimodal"],
            "Multimodal Nowcast"
        ),
    ]

    for ax, (field, title) in zip(
        axes.ravel(),
        panels
    ):

        image = ax.imshow(
            field,
            origin="upper",
            vmin=0,
            vmax=1,
            aspect="auto",
        )

        rows, cols = np.where(
            lightning_mask
        )

        ax.scatter(
            cols,
            rows,
            marker="x",
            s=35,
            linewidths=1.5,
            label="Observed LIS lightning"
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "Grid X"
        )

        ax.set_ylabel(
            "Grid Y"
        )

        ax.legend(
            loc="upper right"
        )

        plt.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Risk score"
        )

    # --------------------------------------------------------
    # Actual lightning
    # --------------------------------------------------------

    ax = axes.ravel()[3]

    rows, cols = np.where(
        lightning_mask
    )

    ax.scatter(
        cols,
        rows,
        s=45,
        marker="x",
        linewidths=1.8,
    )

    ax.set_title(
        "Observed LIS Lightning Target"
    )

    ax.set_xlabel(
        "Grid X"
    )

    ax.set_ylabel(
        "Grid Y"
    )

    ax.set_xlim(
        -0.5,
        target.shape[1] - 0.5
    )

    ax.set_ylim(
        target.shape[0] - 0.5,
        -0.5
    )

    ax.grid(
        alpha=0.25
    )

    fig.suptitle(
        "Nowcast Baseline Comparison\n"
        "Analysis: 04:30 UTC | "
        "Target: 05:00–05:30 UTC",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return figure_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print("AUTOMATIC NOWCAST VALIDATION")
    print("=" * 90)

    target, target_available = load_target()

    forecasts = load_forecasts()

    results = []

    for name, forecast in forecasts.items():

        result = evaluate_forecast(
            name,
            forecast,
            target,
            target_available,
        )

        results.append(
            result
        )

    print_results(
        results
    )

    csv_path = save_csv(
        results
    )

    figure_path = save_comparison_figure(
        forecasts,
        target,
        target_available,
    )

    print()
    print("=" * 90)
    print("VALIDATION OUTPUTS")
    print("=" * 90)

    print()
    print(
        f"CSV    : {csv_path}"
    )

    print(
        f"Figure : {figure_path}"
    )

    print()
    print(
        "NOWCAST VALIDATION COMPLETE."
    )


if __name__ == "__main__":
    main()