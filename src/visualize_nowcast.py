from pathlib import Path
import sys

# ============================================================
# PROJECT ROOT / PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt

from src.datasets.lightning_dataset import LightningNowcastingDataset
from src.models.nowcasting_model import ThunderstormNowcaster
"""
SIH26072YELLOW
==============================================================

REAL MULTIMODAL NOWCAST VISUALIZATION

Prototype event
---------------
Region      : Odisha
Domain      : 20N - 21N, 86E - 88E
Grid        : 27 x 27
Cadence     : 30 minutes
Event       : 01 May 2020
Input       : 04:00 + 04:30 UTC
Forecast    : +30 / +60 / +90 / +120 minutes

This script:

1. Loads the REAL multimodal dataset.
2. Loads the REAL LIS lightning targets.
3. Loads the existing ThunderstormNowcaster.
4. Runs one real-data forward pass.
5. Converts logits to probabilities.
6. Visualizes the four forecast horizons.
7. Shows the observed LIS target for each available horizon.
8. Saves the visualization to outputs/.

IMPORTANT
---------
The current prototype contains only one observed LIS target
interval. Therefore the model output is a pipeline demonstration,
NOT a statistically validated forecast.
"""


from pathlib import Path

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt

from src.datasets.lightning_dataset import LightningNowcastingDataset
from src.models.nowcasting_model import ThunderstormNowcaster


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "real_multimodal_nowcast_20200501.png"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

HISTORY = 2
HORIZONS = 4

INPUT_CHANNELS = 18
HIDDEN_CHANNELS = 64


HORIZON_LABELS = [
    "+30 min",
    "+60 min",
    "+90 min",
    "+120 min",
]


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check_file(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )

    print(f"[OK] {name}")
    print(f"     {path}")


def print_tensor_statistics(name, tensor):
    values = tensor.detach().cpu().numpy()

    print(
        f"{name}: "
        f"shape={values.shape} "
        f"min={np.nanmin(values):.6f} "
        f"max={np.nanmax(values):.6f} "
        f"mean={np.nanmean(values):.6f}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_real_sample():
    section("1. LOADING REAL MULTIMODAL DATA")

    check_file(
        MULTIMODAL_PATH,
        "Multimodal dataset",
    )

    check_file(
        TARGET_PATH,
        "Lightning target dataset",
    )

    dataset = LightningNowcastingDataset(
        str(MULTIMODAL_PATH),
        str(TARGET_PATH),
        history=HISTORY,
        horizons=HORIZONS,
    )

    print()
    print("Dataset length:", len(dataset))

    if len(dataset) == 0:
        raise ValueError(
            "Dataset contains no samples."
        )

    sample = dataset[0]

    required_keys = [
        "x",
        "availability",
        "target",
        "target_availability",
    ]

    missing = [
        key
        for key in required_keys
        if key not in sample
    ]

    if missing:
        raise ValueError(
            f"Dataset sample is missing keys: {missing}"
        )

    x = sample["x"]
    availability = sample["availability"]
    target = sample["target"]
    target_availability = sample["target_availability"]

    print()
    print("Sample keys:")
    print(" ", list(sample.keys()))

    print()
    print("Input:")
    print(" ", x.shape)

    print("Input availability:")
    print(" ", availability.shape)

    print("Target:")
    print(" ", target.shape)

    print("Target availability:")
    print(" ", target_availability.shape)

    return (
        dataset,
        sample,
    )


# ============================================================
# LOAD COORDINATES
# ============================================================

def load_coordinates():
    section("2. LOADING GRID COORDINATES")

    ds = xr.open_dataset(
        MULTIMODAL_PATH
    )

    try:
        latitude = ds["latitude"].values
        longitude = ds["longitude"].values
        times = ds["time"].values

    finally:
        ds.close()

    latitude = np.asarray(
        latitude,
        dtype=np.float32,
    )

    longitude = np.asarray(
        longitude,
        dtype=np.float32,
    )

    times = np.asarray(
        times,
        dtype="datetime64[ns]",
    )

    if latitude.ndim != 1:
        raise ValueError(
            "Expected 1-D latitude coordinate."
        )

    if longitude.ndim != 1:
        raise ValueError(
            "Expected 1-D longitude coordinate."
        )

    if len(latitude) != 27:
        raise ValueError(
            f"Expected 27 latitude points, got {len(latitude)}"
        )

    if len(longitude) != 27:
        raise ValueError(
            f"Expected 27 longitude points, got {len(longitude)}"
        )

    print(
        "Latitude:",
        f"{latitude[0]:.4f}",
        "->",
        f"{latitude[-1]:.4f}",
    )

    print(
        "Longitude:",
        f"{longitude[0]:.4f}",
        "->",
        f"{longitude[-1]:.4f}",
    )

    return (
        latitude,
        longitude,
        times,
    )


# ============================================================
# RUN MODEL
# ============================================================

def run_model(sample):
    section("3. RUNNING MULTIMODAL NOWCASTER")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    x = sample["x"]
    availability = sample["availability"]

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    x = x.unsqueeze(0).to(device)

    availability = (
        availability
        .unsqueeze(0)
        .to(device)
    )

    print()
    print("Model input:")
    print(" ", x.shape)

    print("Model availability:")
    print(" ", availability.shape)

    # --------------------------------------------------------
    # Create existing model
    # --------------------------------------------------------

    model = ThunderstormNowcaster(
        input_channels=INPUT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        horizons=HORIZONS,
    ).to(device)

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        "Model parameters:",
        parameter_count,
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            x,
            availability,
        )

        probabilities = torch.sigmoid(
            logits
        )

    print()
    print(
        "Logits shape:",
        logits.shape,
    )

    print(
        "Probability shape:",
        probabilities.shape,
    )

    print(
        "Probability range:",
        f"{probabilities.min().item():.6f}",
        "->",
        f"{probabilities.max().item():.6f}",
    )

    if not torch.isfinite(logits).all():
        raise ValueError(
            "Model produced non-finite logits."
        )

    if not torch.isfinite(probabilities).all():
        raise ValueError(
            "Model produced non-finite probabilities."
        )

    return (
        logits.cpu(),
        probabilities.cpu(),
    )


# ============================================================
# LOAD TARGETS
# ============================================================

def load_targets(sample):
    section("4. LOADING OBSERVED LIGHTNING TARGETS")

    target = sample["target"].detach().cpu()

    target_availability = (
        sample["target_availability"]
        .detach()
        .cpu()
    )

    print(
        "Target shape:",
        target.shape,
    )

    print(
        "Target availability shape:",
        target_availability.shape,
    )

    print()

    for horizon_index in range(HORIZONS):

        observed = (
            target_availability[
                horizon_index
            ].numpy()
            > 0
        )

        target_frame = (
            target[
                horizon_index
            ].numpy()
        )

        observed_pixels = int(
            np.count_nonzero(observed)
        )

        positive_pixels = int(
            np.count_nonzero(
                target_frame[observed] > 0
            )
        )

        print(
            f"{HORIZON_LABELS[horizon_index]:8s} | "
            f"observed={observed_pixels:4d} | "
            f"positive={positive_pixels:4d}"
        )

    return (
        target.numpy(),
        target_availability.numpy(),
    )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_visualization(
    probabilities,
    target,
    target_availability,
    latitude,
    longitude,
    times,
):
    section("5. CREATING NOWCAST VISUALIZATION")

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    if probabilities.shape != (
        HORIZONS,
        27,
        27,
    ):
        raise ValueError(
            "Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    # --------------------------------------------------------
    # Find analysis time
    #
    # history=2 means the final input frame is the
    # second frame of the sample.
    # --------------------------------------------------------

    analysis_time_index = HISTORY - 1

    if analysis_time_index >= len(times):
        raise ValueError(
            "Analysis time index is outside "
            "the available time coordinate."
        )

    analysis_time = times[
        analysis_time_index
    ]

    print()
    print(
        "Analysis time:",
        analysis_time,
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # First row = model forecasts
    # Second row = observed targets
    # --------------------------------------------------------

    forecast_axes = axes[0]
    target_axes = axes[1]

    lat_min = float(
        np.min(latitude)
    )

    lat_max = float(
        np.max(latitude)
    )

    lon_min = float(
        np.min(longitude)
    )

    lon_max = float(
        np.max(longitude)
    )

    extent = [
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    ]

    # --------------------------------------------------------
    # Plot each horizon
    # --------------------------------------------------------

    for horizon_index in range(HORIZONS):

        probability = probabilities[
            horizon_index
        ]

        observed_mask = (
            target_availability[
                horizon_index
            ]
            > 0
        )

        observed_target = target[
            horizon_index
        ].copy()

        # Hide unavailable target pixels.
        observed_target[
            ~observed_mask
        ] = np.nan

        # ----------------------------------------------------
        # Forecast
        # ----------------------------------------------------

        ax = forecast_axes[
            horizon_index
        ]

        image = ax.imshow(
            probability,
            origin="upper",
            extent=extent,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title(
            f"{HORIZON_LABELS[horizon_index]} "
            f"forecast"
        )

        ax.set_xlabel(
            "Longitude (°E)"
        )

        ax.set_ylabel(
            "Latitude (°N)"
        )

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Lightning probability",
        )

        # ----------------------------------------------------
        # Observed target
        # ----------------------------------------------------

        ax = target_axes[
            horizon_index
        ]

        # Use NaN where LIS was unavailable.
        # This prevents unavailable regions from being
        # visually interpreted as zero lightning.
        target_image = ax.imshow(
            observed_target,
            origin="upper",
            extent=extent,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title(
            f"{HORIZON_LABELS[horizon_index]} "
            f"observed LIS target"
        )

        ax.set_xlabel(
            "Longitude (°E)"
        )

        ax.set_ylabel(
            "Latitude (°N)"
        )

        fig.colorbar(
            target_image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Lightning occurrence",
        )

        # ----------------------------------------------------
        # Observability annotation
        # ----------------------------------------------------

        observed_pixels = int(
            np.count_nonzero(
                observed_mask
            )
        )

        positive_pixels = int(
            np.count_nonzero(
                observed_target > 0
            )
        )

        if observed_pixels == 0:

            ax.text(
                0.5,
                0.5,
                "LIS unavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "alpha": 0.85,
                },
            )

        else:

            ax.text(
                0.02,
                0.98,
                (
                    f"Observed cells: {observed_pixels}\n"
                    f"Positive cells: {positive_pixels}"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "alpha": 0.85,
                },
            )

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        (
            "SIH26072YELLOW — Real Multimodal "
            "Lightning Nowcast Prototype\n"
            f"Odisha | Analysis time: "
            f"{analysis_time}"
        ),
        fontsize=16,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("Visualization saved:")
    print(OUTPUT_FILE)

    return OUTPUT_FILE


# ============================================================
# PRINT FORECAST SUMMARY
# ============================================================

def print_forecast_summary(
    probabilities,
    target,
    target_availability,
):
    section("6. FORECAST SUMMARY")

    for horizon_index in range(HORIZONS):

        probability = probabilities[
            horizon_index
        ]

        observed = (
            target_availability[
                horizon_index
            ]
            > 0
        )

        observed_target = target[
            horizon_index
        ]

        max_probability = float(
            np.max(probability)
        )

        mean_probability = float(
            np.mean(probability)
        )

        if np.any(observed):

            observed_probability = (
                probability[observed]
            )

            observed_mean_probability = float(
                np.mean(
                    observed_probability
                )
            )

            positive = (
                observed_target[observed]
                > 0
            )

            positive_count = int(
                np.count_nonzero(
                    positive
                )
            )

        else:

            observed_mean_probability = (
                float("nan")
            )

            positive_count = 0

        print()
        print(
            HORIZON_LABELS[horizon_index]
        )

        print(
            "  max probability       :",
            f"{max_probability:.6f}",
        )

        print(
            "  domain mean           :",
            f"{mean_probability:.6f}",
        )

        if np.isfinite(
            observed_mean_probability
        ):

            print(
                "  observed-region mean  :",
                f"{observed_mean_probability:.6f}",
            )

        else:

            print(
                "  observed-region mean  :",
                "N/A — LIS unavailable",
            )

        print(
            "  observed positive cells:",
            positive_count,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    section("SIH26072YELLOW")

    print(
        "REAL MULTIMODAL NOWCAST VISUALIZATION"
    )

    print()
    print("Region     : Odisha")
    print("Domain     : 20N - 21N, 86E - 88E")
    print("Grid       : 27 x 27")
    print("Cadence    : 30 minutes")
    print("History    : 2 frames")
    print("Horizons   : +30/+60/+90/+120 minutes")
    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Current model is being visualized "
        "without a trained checkpoint."
    )
    print(
        "The probability maps demonstrate "
        "pipeline operation, not forecast skill."
    )

    # --------------------------------------------------------
    # Load sample
    # --------------------------------------------------------

    _, sample = load_real_sample()

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    latitude, longitude, times = (
        load_coordinates()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    _, probabilities = run_model(
        sample
    )

    probabilities = (
        probabilities[0]
        .numpy()
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    target, target_availability = (
        load_targets(sample)
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_forecast_summary(
        probabilities,
        target,
        target_availability,
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    output = create_visualization(
        probabilities,
        target,
        target_availability,
        latitude,
        longitude,
        times,
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    section("SUCCESS")

    print(
        "REAL MULTIMODAL NOWCAST VISUALIZATION CREATED"
    )

    print()
    print(
        "Output:"
    )

    print(output)

    print()
    print(
        "Pipeline verified:"
    )

    print(
        "REAL INSAT + IMERG + ERA5 + LIS"
    )

    print(
        "        ↓"
    )

    print(
        "18-channel input"
    )

    print(
        "        ↓"
    )

    print(
        "2-frame temporal history"
    )

    print(
        "        ↓"
    )

    print(
        "ThunderstormNowcaster"
    )

    print(
        "        ↓"
    )

    print(
        "4 forecast probability maps"
    )

    print(
        "        ↓"
    )

    print(
        "Observed LIS target comparison"
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print()
        print("=" * 78)
        print("VISUALIZATION FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise
