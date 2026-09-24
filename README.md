# ⚡ Thunderstorm Nowcasting System

### AI/ML-Based Nowcasting of Thunderstorms and Lightning Using Multi-Source Atmospheric Observations

> **SIH 2026 — Problem Statement SIH26072**
> Software | Disaster Management | Meteorological Intelligence

---

## 📌 Overview

This project is an AI/ML-based thunderstorm nowcasting system designed to predict the **spatial and temporal evolution of thunderstorms and lightning** using multiple atmospheric observation sources.

The system is being developed as a prototype for the **Mumbai region**, with the architecture designed to scale to **Maharashtra and eventually India**.

The core idea is to fuse satellite, radar, lightning, surface weather and numerical weather prediction data into a reliability-aware deep-learning pipeline.

### Target Forecast Horizons

* +30 minutes
* +60 minutes
* +90 minutes
* +120 minutes

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────────┐
                    │   ATMOSPHERIC DATA      │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
     INSAT Satellite         Radar Data          Lightning Data
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   DATA PREPROCESSING    │
                    │                         │
                    │ • Spatial alignment    │
                    │ • Temporal alignment   │
                    │ • Normalization        │
                    │ • Quality control       │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ RELIABILITY-AWARE       │
                    │ DATA FUSION             │
                    │                         │
                    │ Features + Availability │
                    │ Masks                   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      AI MODEL            │
                    │                         │
                    │ ConvLSTM / CNN          │
                    │ Spatiotemporal Learning │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ THUNDERSTORM FORECAST   │
                    │                         │
                    │ +30 / +60 / +90 / +120 │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ DASHBOARD + ALERTS      │
                    └─────────────────────────┘
```

---

# 🛰️ Current Data Sources

## INSAT-3SIMG

The current prototype uses real **INSAT-3SIMG Level-1B observations obtained through MOSDAC**.

Current channels:

| Channel | Description        |
| ------- | ------------------ |
| MIR     | Middle Infrared    |
| TIR1    | Thermal Infrared 1 |
| TIR2    | Thermal Infrared 2 |
| WV      | Water Vapour       |

### Current prototype grid

```text
27 × 27 spatial grid
```

The prototype extracts a regional subset of INSAT observations for the Mumbai-region experiment.

The data is processed into NumPy tensors for model development.

---

## Radar

Radar data is part of the target multi-source architecture.

Planned radar features include:

* Reflectivity
* Radial velocity
* Rainfall / precipitation indicators
* Echo-top information
* VIL
* Storm-cell information

**Current status:** Integration pending because suitable Mumbai-region radar data access is still being established.

---

## Lightning

Planned lightning features include:

* Lightning density
* Lightning rate
* Lightning trend
* Spatial clustering

**Current status:** Integration pending.

---

## AWS / Surface Observations

Planned variables include:

* Temperature
* Relative humidity
* Pressure
* Wind speed
* Wind direction

**Current status:** Integration pending.

---

## NWP / Reanalysis

Planned atmospheric variables include:

* CAPE
* CIN
* Wind shear
* Vertical velocity
* Temperature
* Humidity

ERA5/NWP data will provide environmental context for thunderstorm development.

---

# 🧠 AI/ML Pipeline

The model architecture is designed around **spatiotemporal deep learning**.

Current implementation includes:

### Reliability-Aware Fusion

The fusion architecture separates:

```text
Meteorological Feature
        +
Availability Mask
```

This prevents an unavailable sensor from being incorrectly interpreted as a zero measurement.

For example:

```text
Radar unavailable ≠ Radar reflectivity = 0
```

This allows the model to learn from incomplete multi-sensor observations.

---

## ConvLSTM

The prototype model uses a ConvLSTM-based architecture for learning spatial and temporal evolution.

Input:

```text
[Batch, Time, Channels, Height, Width]
```

Current fusion representation:

```text
12 feature channels
+
12 availability-mask channels
=
24 channels entering the fusion block
```

The architecture currently contains:

* Reliability-aware fusion
* CNN feature extraction
* ConvLSTM temporal modelling
* Decoder
* Multi-horizon output structure

---

# 🔬 INSAT Feature Engineering

Current satellite-derived features include:

```text
MIR
TIR1
TIR2
WV
TIR1 - TIR2
TIR1 Cooling
TIR2 Cooling
TIR1 Gradient
```

### Cooling Feature

The cooling feature represents temporal change in cloud-top temperature.

A positive cooling value indicates that the current observation is colder than the previous observation.

Rapid cloud-top cooling can be used as an atmospheric development indicator.

However, the current system **does not treat cooling alone as proof of a thunderstorm**. Radar/lightning observations and proper target labels are required for validated thunderstorm prediction.

---

# 🧪 Current Prototype

The current prototype successfully demonstrates:

* Real INSAT data ingestion
* HDF5 → NumPy preprocessing
* Spatial cropping
* 27×27 regional grid
* Multi-channel satellite representation
* Temporal sequence construction
* INSAT feature engineering
* Baseline temporal prediction
* Reliability-aware fusion architecture
* ConvLSTM model
* Real INSAT → model pipeline
* FastAPI backend
* React + TypeScript dashboard
* Interactive Leaflet map
* Grid-cell inspection
* Forecast horizon interface

---

# 🖥️ Dashboard

The dashboard provides an operational-style interface for visualizing the nowcasting system.

Current functionality:

### Data Layers

* INSAT Satellite
* TIR1 visualization
* TIR1-derived storm indicator

### Forecast Controls

```text
+30 MIN
+60 MIN
+90 MIN
+120 MIN
```

### Grid Inspection

Selecting a grid cell displays:

* TIR1
* TIR2
* MIR
* WV
* Grid coordinates

### Data Status

The dashboard currently reports availability of:

```text
INSAT       AVAILABLE
RADAR       OFFLINE
LIGHTNING   OFFLINE
AWS         OFFLINE
```

---

# ⚙️ Backend

The project uses **FastAPI** to serve processed INSAT observations to the dashboard.

Run the backend from the project root:

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Current endpoints include:

```text
GET /
GET /health
GET /api/insat/latest
GET /api/insat/frames
GET /api/insat/{filename}
```

---

# 🌐 Frontend

The dashboard is built using:

* React
* TypeScript
* Vite
* Leaflet
* React-Leaflet

Run the dashboard:

```powershell
cd dashboard
npm install
npm run dev
```

The development server will provide a local URL similar to:

```text
http://localhost:5174/
```

The frontend uses the Vite development proxy to communicate with FastAPI.

---

# 📁 Project Structure

```text
thunderstorm-nowcasting/
│
├── api/
│   └── main.py
│
├── dashboard/
│   ├── public/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── ...
│
├── scripts/
│   ├── prepare_insat_features.py
│   ├── build_insat_temporal_samples.py
│   ├── extract_insat_storm_features.py
│   ├── build_fusion_dataset.py
│   ├── create_target_schema.py
│   ├── train_insat_temporal_model.py
│   └── ...
│
├── src/
│   ├── data_adapters/
│   │   └── weather_data_adapter.py
│   ├── datasets/
│   │   └── insat_sequence.py
│   ├── models/
│   │   └── nowcasting_model.py
│   └── preprocessing/
│       └── insat_processor.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── mdapi/
│
├── .gitignore
└── README.md
```

> Large raw and processed meteorological datasets are intentionally excluded from Git.

---

# 📊 Model Development Status

| Component                           | Status        |
| ----------------------------------- | ------------- |
| INSAT ingestion                     | ✅ Implemented |
| INSAT preprocessing                 | ✅ Implemented |
| Spatial grid                        | ✅ Implemented |
| Temporal sequences                  | ✅ Implemented |
| Satellite feature engineering       | ✅ Implemented |
| Baseline model                      | ✅ Implemented |
| Reliability-aware fusion            | ✅ Implemented |
| ConvLSTM architecture               | ✅ Implemented |
| FastAPI backend                     | ✅ Implemented |
| React dashboard                     | ✅ Implemented |
| Radar integration                   | 🔄 Pending    |
| Lightning integration               | 🔄 Pending    |
| AWS integration                     | 🔄 Pending    |
| NWP integration                     | 🔄 Pending    |
| Validated thunderstorm labels       | 🔄 Pending    |
| Multi-horizon thunderstorm training | 🔄 Pending    |
| Probability calibration             | 🔄 Planned    |
| Brier score / ECE evaluation        | 🔄 Planned    |
| Missing-sensor robustness testing   | 🔄 Planned    |
| Real-time ingestion                 | 🔄 Planned    |
| Automated alerts                    | 🔄 Planned    |

---

# 🎯 Development Roadmap

## Phase 1 — Satellite Prototype

* [x] INSAT data acquisition
* [x] INSAT preprocessing
* [x] Regional grid generation
* [x] Temporal sequence generation
* [x] Satellite feature engineering
* [x] Baseline prediction

## Phase 2 — Multi-Sensor Fusion

* [ ] Radar integration
* [ ] Lightning integration
* [ ] AWS integration
* [ ] NWP/reanalysis integration
* [ ] Spatial/temporal alignment
* [ ] Missing-data masks

## Phase 3 — Thunderstorm Prediction

* [ ] Generate reliable thunderstorm target labels
* [ ] Train multi-horizon model
* [ ] +30 minute prediction
* [ ] +60 minute prediction
* [ ] +90 minute prediction
* [ ] +120 minute prediction

## Phase 4 — Trust & Evaluation

* [ ] Probability calibration
* [ ] Brier score
* [ ] Expected Calibration Error
* [ ] Reliability diagrams
* [ ] Sensor ablation experiments
* [ ] Missing-sensor robustness tests

## Phase 5 — Operational System

* [ ] Real-time data ingestion
* [ ] Automated inference
* [ ] Alert generation
* [ ] Confidence estimation
* [ ] Maharashtra-scale deployment
* [ ] India-scale architecture

---

# 🔐 Data & Security

Meteorological datasets and credentials are not stored in this repository.

The repository intentionally ignores:

```text
.venv/
node_modules/
data/raw/
data/processed/
*.h5
*.h5.part
*.npz
mdapi/config.json
logs/
model checkpoints
```

Users should configure their own data-access credentials locally.

---

# ⚠️ Current Limitations

The current prototype should **not yet be interpreted as an operational thunderstorm warning system**.

At the current stage:

1. INSAT is the primary real observation source.
2. Radar and lightning observations are not yet integrated.
3. Proper thunderstorm target labels are still being established.
4. The current TIR-based storm indicator is not a validated probability forecast.
5. The ConvLSTM architecture is under development.
6. Multi-horizon thunderstorm probability outputs require dedicated training and validation.

---

# 🚀 Long-Term Vision

The final system is intended to operate as a multi-source atmospheric intelligence platform:

```text
INSAT
   +
Radar
   +
Lightning
   +
AWS
   +
NWP
   ↓
Reliability-Aware Fusion
   ↓
Spatiotemporal AI
   ↓
Calibrated Thunderstorm Probability
   ↓
+30 / +60 / +90 / +120 min
   ↓
Interactive Risk Map
   +
Confidence
   +
Automated Alerts
```

The prototype begins with a limited Mumbai-region domain while maintaining an architecture suitable for expansion to Maharashtra and India.
