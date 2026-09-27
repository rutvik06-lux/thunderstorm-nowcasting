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

OUTPUT_DIR = PROJECT_ROOT / "outputs"

ANALYSIS_TIME = np.datetime64("2020-05-01T04:30:00")

EPS = 1e-8


# ============================================================
# HELPERS
# ============================================================

def minmax_normalize(field):
    """
    Normalize a 2D field to [0, 1].
    """
    field = np.asarray(field, dtype=np.float32)

    finite = np.isfinite(field)

    if not np.any(finite):
        return np.zeros_like(field, dtype=np.float32)

    vmin = np.nanmin(field)
    vmax = np.nanmax(field)

    if abs(vmax - vmin) < EPS:
        return np.zeros_like(field, dtype=np.float32)

    result = (field - vmin) / (vmax - vmin)

    return np.clip(
        np.nan_to_num(result, nan=0.0),
        0.0,
        1.0,
    ).astype(np.float32)


def normalize_positive(field):
    """
    Normalize a non-negative physical field to [0, 1].
    """
    field = np.asarray(field, dtype=np.float32)

    field = np.nan_to_num(
        field,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    field = np.maximum(field, 0.0)

    maximum = np.max(field)

    if maximum < EPS:
        return np.zeros_like(field, dtype=np.float32)

    return np.clip(field / maximum, 0.0, 1.0).astype(np.float32)


def spatial_gradient_magnitude(field):
    """
    Compute normalized spatial gradient magnitude.
    """
    field = np.asarray(field, dtype=np.float32)

    gy, gx = np.gradient(field)

    magnitude = np.sqrt(
        gy ** 2 +
        gx ** 2
    )

    return minmax_normalize(magnitude)


def safe_mean(fields):
    """
    Mean of fields while ignoring NaN/Inf.
    """
    stacked = np.stack(fields, axis=0)

    return np.nanmean(
        stacked,
        axis=0
    ).astype(np.float32)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    if not MULTIMODAL_PATH.exists():
        raise FileNotFoundError(
            f"Multimodal dataset not found:\n{MULTIMODAL_PATH}"
        )

    ds = xr.open_dataset(MULTIMODAL_PATH)

    print()
    print("=" * 70)
    print("MULTIMODAL INFERENCE")
    print("=" * 70)

    print()
    print("Dataset:")
    print(MULTIMODAL_PATH)

    print()
    print("Dimensions:")
    for name, size in ds.sizes.items():
        print(f"  {name}: {size}")

    print()
    print("Available variables:")

    for variable in ds.data_vars:
        print(f"  {variable}")

    return ds


# ============================================================
# TIME SELECTION
# ============================================================

def get_time_index(ds, target_time):
    times = ds["time"].values

    matches = np.where(
        times == target_time
    )[0]

    if len(matches) == 0:
        raise ValueError(
            f"Analysis time {target_time} "
            f"not found in dataset.\n"
            f"Available times:\n{times}"
        )

    return int(matches[0])


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def build_features(ds, time_idx):
    """
    Build physically meaningful multimodal features.

    Predictive modalities:
        INSAT
        IMERG
        ERA5

    LIS is deliberately excluded because it is used
    independently for validation.
    """

    print()
    print("-" * 70)
    print("BUILDING MULTIMODAL FEATURES")
    print("-" * 70)

    current = ds.isel(time=time_idx)

    previous_idx = max(0, time_idx - 1)
    previous = ds.isel(time=previous_idx)

    # --------------------------------------------------------
    # INSAT
    # --------------------------------------------------------

    tir1 = current["insat_tir1"].values
    tir2 = current["insat_tir2"].values
    wv = current["insat_wv"].values
    vis = current["insat_vis"].values

    insat_availability = (
        current["insat_availability"].values
        > 0
    )

    # Cloud-top cooling:
    # positive value means TIR1 became colder.
    tir1_previous = previous["insat_tir1"].values

    cooling = (
        tir1_previous -
        tir1
    )

    cooling = minmax_normalize(cooling)

    tir1_norm = minmax_normalize(tir1)
    tir2_norm = minmax_normalize(tir2)
    wv_norm = minmax_normalize(wv)
    vis_norm = minmax_normalize(vis)

    # --------------------------------------------------------
    # IMERG
    # --------------------------------------------------------

    precip = current[
        "imerg_precipitation"
    ].values

    precip_previous = previous[
        "imerg_precipitation"
    ].values

    imerg_availability = (
        current[
            "imerg_availability"
        ].values
        > 0
    )

    precip_norm = normalize_positive(
        precip
    )

    precip_growth = (
        precip -
        precip_previous
    )

    precip_growth = minmax_normalize(
        precip_growth
    )

    precip_gradient = (
        spatial_gradient_magnitude(
            precip
        )
    )

    # --------------------------------------------------------
    # ERA5
    # --------------------------------------------------------

    u10 = current[
        "era5_u10"
    ].values

    v10 = current[
        "era5_v10"
    ].values

    d2m = current[
        "era5_d2m"
    ].values

    t2m = current[
        "era5_t2m"
    ].values

    msl = current[
        "era5_msl"
    ].values

    sp = current[
        "era5_sp"
    ].values

    tcc = current[
        "era5_tcc"
    ].values

    cape = current[
        "era5_cape"
    ].values

    era5_availability = (
        current[
            "era5_availability"
        ].values
        > 0
    )

    wind_speed = np.sqrt(
        u10 ** 2 +
        v10 ** 2
    )

    wind_norm = normalize_positive(
        wind_speed
    )

    # Moisture proxy:
    # smaller T2m - D2m means moister near-surface air.
    dewpoint_depression = (
        t2m -
        d2m
    )

    moisture = (
        -dewpoint_depression
    )

    moisture = minmax_normalize(
        moisture
    )

    cape_norm = normalize_positive(
        cape
    )

    tcc_norm = minmax_normalize(
        tcc
    )

    msl_norm = minmax_normalize(
        msl
    )

    sp_norm = minmax_normalize(
        sp
    )

    # --------------------------------------------------------
    # FEATURE COLLECTION
    # --------------------------------------------------------

    features = {
        "insat_tir1": tir1_norm,
        "insat_tir2": tir2_norm,
        "insat_wv": wv_norm,
        "insat_vis": vis_norm,
        "insat_cooling": cooling,

        "imerg_precipitation": precip_norm,
        "imerg_growth": precip_growth,
        "imerg_gradient": precip_gradient,

        "era5_wind": wind_norm,
        "era5_moisture": moisture,
        "era5_cape": cape_norm,
        "era5_tcc": tcc_norm,
        "era5_msl": msl_norm,
        "era5_sp": sp_norm,

        "insat_availability": (
            insat_availability
            .astype(np.float32)
        ),

        "imerg_availability": (
            imerg_availability
            .astype(np.float32)
        ),

        "era5_availability": (
            era5_availability
            .astype(np.float32)
        ),
    }

    print()
    print(
        f"Feature count: {len(features)}"
    )

    for name, field in features.items():
        print(
            f"  {name:25s}"
            f" min={np.nanmin(field):.4f}"
            f" max={np.nanmax(field):.4f}"
            f" mean={np.nanmean(field):.4f}"
        )

    return features


# ============================================================
# MULTIMODAL FUSION
# ============================================================

def build_multimodal_risk(features):
    """
    Observation-derived multimodal fusion.

    This is NOT a trained probability model.

    Output:
        normalized convective risk score [0,1]
    """

    print()
    print("-" * 70)
    print("MULTIMODAL FUSION")
    print("-" * 70)

    # --------------------------------------------------------
    # Environmental state
    # --------------------------------------------------------

    environmental = (
        0.45 * features["era5_cape"] +
        0.30 * features["era5_moisture"] +
        0.15 * features["era5_wind"] +
        0.10 * features["era5_tcc"]
    )

    environmental = minmax_normalize(
        environmental
    )

    # --------------------------------------------------------
    # Satellite convective evolution
    # --------------------------------------------------------

    satellite_evolution = (
        0.50 * features["insat_cooling"] +
        0.20 * features["insat_tir1"] +
        0.15 * features["insat_wv"] +
        0.15 * features["insat_tir2"]
    )

    satellite_evolution = minmax_normalize(
        satellite_evolution
    )

    # --------------------------------------------------------
    # Precipitation / active convection
    # --------------------------------------------------------

    precipitation_state = (
        0.55 * features[
            "imerg_precipitation"
        ] +
        0.25 * features[
            "imerg_growth"
        ] +
        0.20 * features[
            "imerg_gradient"
        ]
    )

    precipitation_state = minmax_normalize(
        precipitation_state
    )

    # --------------------------------------------------------
    # Multimodal fusion
    # --------------------------------------------------------

    risk = (
        0.40 * environmental +
        0.35 * satellite_evolution +
        0.25 * precipitation_state
    )

    risk = minmax_normalize(
        risk
    )

    return {
        "environmental": environmental,
        "satellite_evolution": satellite_evolution,
        "precipitation_state": precipitation_state,
        "multimodal_risk": risk,
    }


# ============================================================
# DATA CONFIDENCE
# ============================================================

def build_confidence(features):
    """
    Estimate input-data availability confidence.

    This is NOT forecast probability.
    """

    insat = features[
        "insat_availability"
    ]

    imerg = features[
        "imerg_availability"
    ]

    era5 = features[
        "era5_availability"
    ]

    confidence = (
        0.40 * insat +
        0.35 * imerg +
        0.25 * era5
    )

    return np.clip(
        confidence,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# FINAL NOWCAST
# ============================================================

def build_nowcast(
    multimodal_risk,
    confidence
):
    """
    Combine multimodal risk with data confidence.
    """

    nowcast = (
        multimodal_risk *
        confidence
    )

    return np.clip(
        nowcast,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# OUTPUT
# ============================================================

def save_output(
    ds,
    features,
    fusion,
    confidence,
    nowcast,
    time_idx,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = str(
        ds["time"].values[time_idx]
    )[:16].replace(
        "-",
        ""
    ).replace(
        ":",
        ""
    ).replace(
        "T",
        "_"
    )

    output_path = (
        OUTPUT_DIR /
        f"multimodal_inference_{timestamp}.nc"
    )

    output = xr.Dataset(
        {
            "environmental_score": (
                ("y", "x"),
                fusion["environmental"]
            ),

            "satellite_evolution_score": (
                ("y", "x"),
                fusion[
                    "satellite_evolution"
                ]
            ),

            "precipitation_score": (
                ("y", "x"),
                fusion[
                    "precipitation_state"
                ]
            ),

            "multimodal_risk": (
                ("y", "x"),
                fusion[
                    "multimodal_risk"
                ]
            ),

            "data_confidence": (
                ("y", "x"),
                confidence
            ),

            "multimodal_nowcast": (
                ("y", "x"),
                nowcast
            ),
        },
        coords={
            "y": ds["y"].values,
            "x": ds["x"].values,
            "latitude": (
                "y",
                ds["latitude"].values
            ),
            "longitude": (
                "x",
                ds["longitude"].values
            ),
        },
        attrs={
            "description":
                "Observation-derived multimodal "
                "thunderstorm nowcast",

            "analysis_time":
                str(ds["time"].values[time_idx]),

            "method":
                "INSAT + IMERG + ERA5 feature fusion",

            "probability":
                "No. Values are normalized "
                "risk scores, not calibrated probabilities.",

            "lightning_target":
                "LIS excluded from predictive input "
                "and reserved for independent validation.",

            "data_status":
                "Real observations. No synthetic "
                "predictive data.",

            "prototype_status":
                "Observation-derived baseline. "
                "Not a trained neural probability model.",
        },
    )

    output.to_netcdf(
        output_path
    )

    return output_path


# ============================================================
# VISUALIZATION
# ============================================================

def save_figure(
    ds,
    fusion,
    confidence,
    nowcast,
    time_idx,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_path = (
        OUTPUT_DIR /
        "multimodal_inference_20200501_0430.png"
    )

    latitude = ds[
        "latitude"
    ].values

    longitude = ds[
        "longitude"
    ].values

    extent = [
        float(longitude.min()),
        float(longitude.max()),
        float(latitude.min()),
        float(latitude.max()),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10)
    )

    panels = [
        (
            fusion["environmental"],
            "Environmental State"
        ),
        (
            fusion["satellite_evolution"],
            "Satellite Evolution"
        ),
        (
            fusion["precipitation_state"],
            "Precipitation / Convection"
        ),
        (
            nowcast,
            "Final Multimodal Nowcast"
        ),
    ]

    for ax, (field, title) in zip(
        axes.ravel(),
        panels
    ):
        image = ax.imshow(
            field,
            origin="upper",
            extent=extent,
            vmin=0,
            vmax=1,
            aspect="auto",
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "Longitude"
        )

        ax.set_ylabel(
            "Latitude"
        )

        plt.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Normalized score"
        )

    fig.suptitle(
        "Multimodal Thunderstorm Nowcast — "
        "2020-05-01 04:30 UTC\n"
        "Forecast target: 05:00–05:30 UTC",
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
# VALIDATION
# ============================================================

def print_summary(
    ds,
    fusion,
    confidence,
    nowcast,
):
    risk = fusion[
        "multimodal_risk"
    ]

    max_row, max_col = np.unravel_index(
        np.argmax(nowcast),
        nowcast.shape
    )

    latitude = ds[
        "latitude"
    ].values

    longitude = ds[
        "longitude"
    ].values

    print()
    print("=" * 70)
    print("MULTIMODAL INFERENCE SUMMARY")
    print("=" * 70)

    print(
        f"Analysis time: "
        f"{ANALYSIS_TIME}"
    )

    print(
        f"Risk range: "
        f"{risk.min():.3f} -> "
        f"{risk.max():.3f}"
    )

    print(
        f"Risk mean: "
        f"{risk.mean():.3f}"
    )

    print(
        f"Confidence range: "
        f"{confidence.min():.3f} -> "
        f"{confidence.max():.3f}"
    )

    print(
        f"Confidence mean: "
        f"{confidence.mean():.3f}"
    )

    print(
        f"Nowcast range: "
        f"{nowcast.min():.3f} -> "
        f"{nowcast.max():.3f}"
    )

    print(
        f"Nowcast mean: "
        f"{nowcast.mean():.3f}"
    )

    print()
    print(
        "Maximum nowcast cell:"
    )

    print(
        f"  row = {max_row}"
    )

    print(
        f"  col = {max_col}"
    )

    print(
        f"  latitude = "
        f"{latitude[max_row]:.4f}"
    )

    print(
        f"  longitude = "
        f"{longitude[max_col]:.4f}"
    )

    print(
        f"  score = "
        f"{nowcast[max_row, max_col]:.4f}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These values are normalized "
        "convective risk scores."
    )

    print(
        "They are NOT calibrated lightning "
        "probabilities."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    ds = load_dataset()

    time_idx = get_time_index(
        ds,
        ANALYSIS_TIME
    )

    print()
    print(
        f"Selected analysis time: "
        f"{ds['time'].values[time_idx]}"
    )

    features = build_features(
        ds,
        time_idx
    )

    fusion = build_multimodal_risk(
        features
    )

    confidence = build_confidence(
        features
    )

    nowcast = build_nowcast(
        fusion["multimodal_risk"],
        confidence
    )

    print_summary(
        ds,
        fusion,
        confidence,
        nowcast,
    )

    output_path = save_output(
        ds,
        features,
        fusion,
        confidence,
        nowcast,
        time_idx,
    )

    figure_path = save_figure(
        ds,
        fusion,
        confidence,
        nowcast,
        time_idx,
    )

    print()
    print("=" * 70)
    print("OUTPUTS")
    print("=" * 70)

    print()
    print(
        f"NetCDF : {output_path}"
    )

    print(
        f"Figure : {figure_path}"
    )

    print()
    print(
        "MULTIMODAL INFERENCE COMPLETE."
    )


if __name__ == "__main__":
    main()