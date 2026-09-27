from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"

INSAT_DIR = PROCESSED / "insat"
IMERG_DIR = PROCESSED / "imerg"
ERA5_DIR = PROCESSED / "era5"
LIS_DIR = PROCESSED / "lis"
MULTIMODAL_DIR = PROCESSED / "multimodal"
TARGET_DIR = PROCESSED / "targets"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Thunderstorm Nowcasting API",
    description="Backend API for the SIH26072YELLOW prototype.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def find_latest_file(directory: Path, pattern: str):
    files = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return files[0] if files else None


def clean_value(value):
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

    return value


def array_summary(array):
    arr = np.asarray(array, dtype=float)

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
        }

    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def find_lat_lon(ds):
    lat_name = None
    lon_name = None

    for name in ["lat", "latitude", "Latitude"]:
        if name in ds:
            lat_name = name
            break

    for name in ["lon", "longitude", "Longitude"]:
        if name in ds:
            lon_name = name
            break

    if lat_name is None or lon_name is None:
        raise HTTPException(
            status_code=500,
            detail="Latitude/longitude variables not found in dataset.",
        )

    return lat_name, lon_name


def json_grid(values):
    arr = np.asarray(values, dtype=float)

    return [
        [
            clean_value(value)
            for value in row
        ]
        for row in arr
    ]


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Thunderstorm Nowcasting API",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "thunderstorm-nowcasting-api",
    }


# ============================================================
# DATA STATUS
# ============================================================

@app.get("/api/data-status")
def data_status():

    def count_files(directory, pattern="*.nc"):
        if not directory.exists():
            return 0

        return len(list(directory.glob(pattern)))

    return {
        "insat": {
            "available": INSAT_DIR.exists(),
            "files": count_files(INSAT_DIR),
        },
        "imerg": {
            "available": IMERG_DIR.exists(),
            "files": count_files(IMERG_DIR),
        },
        "era5": {
            "available": ERA5_DIR.exists(),
            "files": count_files(ERA5_DIR),
        },
        "lis": {
            "available": LIS_DIR.exists(),
            "files": count_files(LIS_DIR),
        },
        "multimodal": {
            "available": MULTIMODAL_DIR.exists(),
            "files": count_files(MULTIMODAL_DIR),
        },
        "targets": {
            "available": TARGET_DIR.exists(),
            "files": count_files(TARGET_DIR),
        },
    }


# ============================================================
# INSAT
# ============================================================

@app.get("/api/insat/latest")
def insat_latest():

    if not INSAT_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail="INSAT processed directory not found.",
        )

    path = find_latest_file(
        INSAT_DIR,
        "*.nc",
    )

    if path is None:
        raise HTTPException(
            status_code=404,
            detail="No processed INSAT NetCDF file found.",
        )

    try:
        ds = xr.open_dataset(path)

        lat_name, lon_name = find_lat_lon(ds)

        lat = ds[lat_name].values
        lon = ds[lon_name].values

        variables = {}

        preferred_variables = [
            "IMG_TIR1",
            "IMG_TIR2",
            "IMG_WV",
            "IMG_VIS",
            "IMG_TIR1_TEMP",
            "IMG_TIR2_TEMP",
            "IMG_WV_TEMP",
            "IMG_VIS_ALBEDO",
        ]

        for name in preferred_variables:
            if name in ds:
                values = ds[name].values

                # If time exists, use latest frame.
                if values.ndim == 3:
                    values = values[-1]

                variables[name] = {
                    "summary": array_summary(values),
                    "grid": json_grid(values),
                }

        time_values = None

        if "time" in ds.coords:
            time_values = [
                str(value)
                for value in ds["time"].values
            ]
        elif "time" in ds:
            time_values = [
                str(value)
                for value in ds["time"].values
            ]

        # Pick TIR1 as the main frontend visualization.
        main_variable = None

        for candidate in [
            "IMG_TIR1",
            "IMG_TIR1_TEMP",
        ]:
            if candidate in variables:
                main_variable = candidate
                break

        if main_variable is None:
            if not variables:
                raise HTTPException(
                    status_code=500,
                    detail="No supported INSAT variables found.",
                )

            main_variable = next(iter(variables))

        main_grid = variables[main_variable]["grid"]

        return {
            "status": "ok",
            "source": "INSAT",
            "file": path.name,
            "timestamp": (
                time_values[-1]
                if time_values
                else None
            ),
            "latitude": np.asarray(lat).tolist(),
            "longitude": np.asarray(lon).tolist(),
            "variable": main_variable,
            "grid": main_grid,
            "summary": variables[main_variable]["summary"],
            "variables": variables,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read INSAT dataset: {exc}",
        )

    finally:
        try:
            ds.close()
        except Exception:
            pass


# ============================================================
# NOWCAST
# ============================================================

def load_nowcast():

    path = find_latest_file(
        OUTPUTS,
        "temporal_multimodal_nowcast_*.nc",
    )

    if path is None:
        raise HTTPException(
            status_code=404,
            detail="Temporal multimodal nowcast file not found.",
        )

    try:
        return path, xr.open_dataset(path)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to open nowcast dataset: {exc}",
        )


def find_score_variable(ds):

    candidates = [
        "nowcast",
        "score",
        "risk",
        "probability",
        "forecast",
        "prediction",
    ]

    for name in candidates:
        if name in ds.data_vars:
            return name

    # Fallback: first variable containing 4 horizons.
    for name, da in ds.data_vars.items():

        if da.ndim >= 3:
            return name

    return None


@app.get("/api/nowcast")
def nowcast():

    path, ds = load_nowcast()

    try:
        variable = find_score_variable(ds)

        if variable is None:
            raise HTTPException(
                status_code=500,
                detail="No nowcast score variable found.",
            )

        da = ds[variable]

        return {
            "status": "ok",
            "file": path.name,
            "variable": variable,
            "dimensions": list(da.dims),
            "shape": list(da.shape),
            "summary": array_summary(da.values),
        }

    finally:
        ds.close()


@app.get("/api/nowcast/{horizon}")
def nowcast_horizon(horizon: int):

    if horizon not in [30, 60, 90, 120]:
        raise HTTPException(
            status_code=400,
            detail="Horizon must be one of 30, 60, 90, 120.",
        )

    path, ds = load_nowcast()

    try:
        variable = find_score_variable(ds)

        if variable is None:
            raise HTTPException(
                status_code=500,
                detail="No nowcast score variable found.",
            )

        da = ds[variable]

        # Try horizon coordinate.
        selected = None

        for coord_name in [
            "horizon",
            "forecast_horizon",
            "lead_time",
            "lead",
        ]:

            if coord_name in da.coords:

                coord = da.coords[coord_name].values

                for index, value in enumerate(coord):

                    try:
                        numeric = float(value)

                        if numeric == horizon:
                            selected = da.isel(
                                {da.get_axis_num(coord_name): index}
                            )

                            break

                    except Exception:
                        pass

                if selected is not None:
                    break

        # Fallback: horizon order = 30/60/90/120.
        if selected is None:

            if da.ndim < 3:
                raise HTTPException(
                    status_code=500,
                    detail="Nowcast variable does not contain spatial forecast data.",
                )

            index = [30, 60, 90, 120].index(horizon)

            selected = da.isel(
                {da.dims[0]: index}
            )

        values = selected.values

        return {
            "status": "ok",
            "file": path.name,
            "horizon": horizon,
            "variable": variable,
            "dimensions": list(selected.dims),
            "shape": list(selected.shape),
            "grid": json_grid(values),
            "summary": array_summary(values),
        }

    finally:
        ds.close()


# ============================================================
# VALIDATION
# ============================================================

@app.get("/api/validation")
def validation():

    csv_path = find_latest_file(
        OUTPUTS,
        "nowcast_comparison_*.csv",
    )

    if csv_path is None:
        return {
            "status": "no_data",
            "message": "Validation comparison CSV not found.",
            "methods": [],
        }

    try:
        df = pd.read_csv(csv_path)

        return {
            "status": "ok",
            "file": csv_path.name,
            "methods": df.to_dict(
                orient="records"
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read validation CSV: {exc}",
        )


# ============================================================
# RUN
# ============================================================



# --- MULTIMODAL DATA ENDPOINT ---

@app.get("/api/multimodal/latest")
def multimodal_latest():
    import xarray as xr
    import numpy as np
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data" / "processed" / "multimodal" / "odisha_multimodal_20200501.nc"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Multimodal dataset not found: {path}"
        )

    ds = xr.open_dataset(path)

    def arr(name):
        return np.nan_to_num(
            ds[name].values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).tolist()

    def coord(name):
        return np.asarray(ds[name].values).astype(float).tolist()

    return {
        "dataset": path.name,
        "times": [
            str(t) for t in ds["time"].values
        ],
        "latitude": coord("latitude"),
        "longitude": coord("longitude"),

        "insat": {
            "tir1": arr("insat_tir1"),
            "tir2": arr("insat_tir2"),
            "wv": arr("insat_wv"),
            "vis": arr("insat_vis"),
            "availability": arr("insat_availability")
        },

        "imerg": {
            "precipitation": arr("imerg_precipitation"),
            "availability": arr("imerg_availability")
        },

        "era5": {
            "u10": arr("era5_u10"),
            "v10": arr("era5_v10"),
            "d2m": arr("era5_d2m"),
            "t2m": arr("era5_t2m"),
            "msl": arr("era5_msl"),
            "sp": arr("era5_sp"),
            "tcc": arr("era5_tcc"),
            "cape": arr("era5_cape"),
            "availability": arr("era5_availability")
        },

        "lis": {
            "lightning_density": arr("lis_lightning_density"),
            "availability": arr("lis_lightning_availability"),
            "observation_seconds": arr("lis_observation_seconds")
        },

        "grid": {
            "y": 27,
            "x": 27,
            "time": 6
        }
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

