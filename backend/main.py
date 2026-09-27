from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd
import xarray as xr

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Thunderstorm Nowcasting API",
    description="Prototype backend for multimodal thunderstorm and lightning nowcasting.",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

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

def find_latest_file(pattern: str):
    files = sorted(
        OUTPUTS.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return files[0] if files else None


def load_nowcast():
    path = find_latest_file(
        "temporal_multimodal_nowcast_*.nc"
    )

    if path is None:
        raise HTTPException(
            status_code=404,
            detail="No temporal multimodal nowcast output found.",
        )

    try:
        ds = xr.open_dataset(path)
        return path, ds

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not open nowcast file: {exc}",
        )


def find_score_variable(ds):

    preferred = [
        "nowcast",
        "risk",
        "score",
        "probability",
        "lightning_risk",
    ]

    for name in preferred:
        if name in ds.data_vars:
            return name

    for name in ds.data_vars:
        if np.issubdtype(
            ds[name].dtype,
            np.number,
        ):
            return name

    return None


def array_summary(values: Any):

    arr = np.asarray(values)

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


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "Thunderstorm Nowcasting API",
        "status": "running",
        "docs": "/docs",
        "endpoints": [
            "/api/health",
            "/api/data-status",
            "/api/nowcast",
            "/api/nowcast/{horizon}",
            "/api/validation",
        ],
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "service": "thunderstorm-nowcasting-api",
        "version": "1.0.0",
    }


# ============================================================
# DATA STATUS
# ============================================================

@app.get("/api/data-status")
def data_status():

    directories = {
        "insat": PROCESSED / "insat",
        "imerg": PROCESSED / "imerg",
        "era5": PROCESSED / "era5",
        "lis": PROCESSED / "lis",
        "multimodal": PROCESSED / "multimodal",
        "targets": PROCESSED / "targets",
        "training": PROCESSED / "training",
    }

    result = {}

    for name, directory in directories.items():

        exists = directory.exists()

        if exists:
            files = list(directory.rglob("*"))
            file_count = sum(
                1 for item in files if item.is_file()
            )
        else:
            file_count = 0

        result[name] = {
            "available": exists,
            "files": file_count,
        }

    return result


# ============================================================
# NOWCAST SUMMARY
# ============================================================

@app.get("/api/nowcast")
def nowcast():

    path, ds = load_nowcast()

    try:

        score_variable = find_score_variable(ds)

        if score_variable is None:
            raise HTTPException(
                status_code=500,
                detail="No numeric nowcast variable found.",
            )

        data = ds[score_variable]

        result = {
            "status": "ok",
            "file": path.name,
            "score_variable": score_variable,
            "dimensions": {
                name: int(size)
                for name, size in data.sizes.items()
            },
            "overall": array_summary(
                data.values
            ),
            "horizons": [],
        }

        if data.ndim >= 3:

            for index in range(
                data.shape[0]
            ):

                field = data.isel(
                    {data.dims[0]: index}
                )

                result["horizons"].append({
                    "index": index,
                    "summary": array_summary(
                        field.values
                    ),
                })

        return result

    finally:
        ds.close()


# ============================================================
# INDIVIDUAL HORIZON
# ============================================================

@app.get("/api/nowcast/{horizon}")
def nowcast_horizon(horizon: int):

    allowed_horizons = [
        30,
        60,
        90,
        120,
    ]

    if horizon not in allowed_horizons:

        raise HTTPException(
            status_code=400,
            detail="Horizon must be 30, 60, 90 or 120 minutes.",
        )

    path, ds = load_nowcast()

    try:

        score_variable = find_score_variable(ds)

        if score_variable is None:
            raise HTTPException(
                status_code=500,
                detail="No numeric nowcast variable found.",
            )

        data = ds[score_variable]

        if data.ndim < 3:
            raise HTTPException(
                status_code=500,
                detail="Unexpected nowcast dimensions.",
            )

        index = allowed_horizons.index(
            horizon
        )

        if index >= data.shape[0]:

            raise HTTPException(
                status_code=500,
                detail="Requested horizon is not available.",
            )

        field = data.isel(
            {data.dims[0]: index}
        )

        values = np.asarray(
            field.values,
            dtype=float,
        )

        grid = []

        for row in values:

            grid.append([
                float(value)
                if np.isfinite(value)
                else None
                for value in row
            ])

        latitude = None
        longitude = None

        for name in [
            "lat",
            "latitude",
        ]:

            if name in ds.coords:
                latitude = (
                    np.asarray(
                        ds[name].values
                    ).tolist()
                )
                break

        for name in [
            "lon",
            "longitude",
        ]:

            if name in ds.coords:
                longitude = (
                    np.asarray(
                        ds[name].values
                    ).tolist()
                )
                break

        return {
            "status": "ok",
            "horizon_minutes": horizon,
            "file": path.name,
            "score_variable": score_variable,
            "shape": list(values.shape),
            "summary": array_summary(values),
            "latitude": latitude,
            "longitude": longitude,
            "grid": grid,
        }

    finally:
        ds.close()


# ============================================================
# VALIDATION
# ============================================================

@app.get("/api/validation")
def validation():

    path = find_latest_file(
        "nowcast_comparison_*.csv"
    )

    if path is None:

        return {
            "status": "unavailable",
            "message": "No validation comparison CSV found.",
        }

    try:

        df = pd.read_csv(path)

        records = []

        for record in df.to_dict(
            orient="records"
        ):

            cleaned = {}

            for key, value in record.items():

                if isinstance(value, float):
                    if not math.isfinite(value):
                        value = None

                cleaned[key] = value

            records.append(cleaned)

        return {
            "status": "ok",
            "file": path.name,
            "results": records,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Could not read validation file: {exc}",
        )


# ============================================================
# LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
