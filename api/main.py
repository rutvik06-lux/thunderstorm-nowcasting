from pathlib import Path
from datetime import datetime
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INSAT_DIR = PROJECT_ROOT / "data" / "processed" / "insat"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Thunderstorm Nowcasting API",
    description="Backend API for the SIH thunderstorm nowcasting dashboard",
    version="0.1.0",
)


# Allow the React/Vite development server to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def get_insat_files():
    """Return available processed INSAT files sorted by filename."""

    files = sorted(INSAT_DIR.glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No processed INSAT files found in {INSAT_DIR}"
        )

    return files


def load_insat_file(file_path: Path):
    """Load one processed INSAT frame."""

    data = np.load(file_path)

    tensor = data["data"]
    latitude = data["latitude"]
    longitude = data["longitude"]

    timestamp = str(data["timestamp_iso"])

    channels = [
        str(channel)
        for channel in data["channels"]
    ]

    return {
        "tensor": tensor,
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": timestamp,
        "channels": channels,
    }


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Thunderstorm Nowcasting API",
        "status": "online",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    files = get_insat_files()

    return {
        "status": "healthy",
        "insat_files": len(files),
        "insat_directory": str(INSAT_DIR),
    }


# ============================================================
# INSAT DATA
# ============================================================

@app.get("/api/insat/latest")
def get_latest_insat():
    """Return the latest available processed INSAT frame."""

    files = get_insat_files()

    latest_file = files[-1]

    frame = load_insat_file(latest_file)

    tensor = frame["tensor"]

    return {
        "filename": latest_file.name,
        "timestamp": frame["timestamp"],
        "channels": frame["channels"],
        "shape": list(tensor.shape),
        "latitude": frame["latitude"].tolist(),
        "longitude": frame["longitude"].tolist(),
        "data": tensor.tolist(),
    }


@app.get("/api/insat/frames")
def get_insat_frames():
    """Return metadata for all available INSAT frames."""

    files = get_insat_files()

    frames = []

    for file_path in files:
        data = np.load(file_path)

        timestamp = str(data["timestamp_iso"])

        frames.append(
            {
                "filename": file_path.name,
                "timestamp": timestamp,
            }
        )

    return {
        "count": len(frames),
        "frames": frames,
    }


@app.get("/api/insat/{filename}")
def get_insat_frame(filename: str):
    """Return a specific processed INSAT frame."""

    file_path = INSAT_DIR / filename

    if not file_path.exists():
        return {
            "error": "INSAT file not found",
            "filename": filename,
        }

    frame = load_insat_file(file_path)

    tensor = frame["tensor"]

    return {
        "filename": file_path.name,
        "timestamp": frame["timestamp"],
        "channels": frame["channels"],
        "shape": list(tensor.shape),
        "latitude": frame["latitude"].tolist(),
        "longitude": frame["longitude"].tolist(),
        "data": tensor.tolist(),
    }