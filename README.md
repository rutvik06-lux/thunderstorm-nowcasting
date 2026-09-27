# SIH26072YELLOW — AI-Based Thunderstorm & Lightning Nowcasting

> **Smart India Hackathon 2026 — SIH26072YELLOW**
> **Problem Statement:** AIML-based Nowcasting of Thunderstorm and Lightning using Atmospheric Observation including Multiple Radars, Satellite, Lightning and Model Data.

---

## 1. Overview

This project develops a **multimodal AI-based thunderstorm and lightning nowcasting system** that combines atmospheric observations from multiple sources to estimate near-term thunderstorm/lightning risk.

The system is designed around four principles:

* **Multimodal fusion** — combine satellite, precipitation, atmospheric and lightning observations.
* **Reliability awareness** — explicitly model data availability and missing observations.
* **Spatiotemporal forecasting** — learn how atmospheric conditions evolve across space and time.
* **Trustworthy predictions** — provide confidence/reliability information alongside forecast outputs.

The current prototype has been validated end-to-end using a **real multimodal thunderstorm event** containing INSAT, IMERG, ERA5 and LIS observations.

---

# 2. Problem

Thunderstorm and lightning development can evolve rapidly and is difficult to capture using a single observation source.

Different data sources provide different information:

| Source | Information                                                                |
| ------ | -------------------------------------------------------------------------- |
| INSAT  | Cloud-top temperature, water vapour and satellite cloud structure          |
| IMERG  | Precipitation                                                              |
| ERA5   | Atmospheric environment such as CAPE, temperature, pressure and winds      |
| LIS    | Independent lightning observations for target generation and validation    |
| Radar  | Reflectivity, velocity, storm structure and other precipitation signatures |

The challenge is not simply combining these datasets, but handling:

* different spatial resolutions
* different temporal resolutions
* missing observations
* sensor availability
* heterogeneous physical variables
* rapidly changing storm structures
* limited overlapping historical events

---

# 3. Proposed Solution

The proposed system follows:

```text
Atmospheric Observations
        ↓
Data Harmonization
        ↓
Availability / Reliability Masks
        ↓
Multimodal Feature Fusion
        ↓
Spatiotemporal AI Model
        ↓
Thunderstorm / Lightning Risk
        ↓
Multiple Forecast Horizons
        ↓
Interactive Weather Intelligence Dashboard
```

The final system is intended to provide forecasts at short time horizons such as:

```text
+15 min
+30 min
+60 min
+90 min
+120 min
```

---

# 4. Current Prototype

The current prototype focuses on a real event over Odisha because this region currently provides an overlapping set of real observational datasets suitable for end-to-end prototyping.

Current multimodal dataset:

```text
data/processed/multimodal/odisha_multimodal_20200501.nc
```

Dimensions:

```text
time = 6
y = 27
x = 27
```

Observation times:

```text
04:00 UTC
04:30 UTC
05:00 UTC
05:30 UTC
06:00 UTC
06:30 UTC
```

The current prototype uses a:

```text
27 × 27 spatial grid
```

with multiple atmospheric variables.

---

# 5. Multimodal Inputs

## INSAT

Current satellite variables:

```text
TIR1
TIR2
Water Vapour
Visible
```

Additional availability information is retained.

These variables provide information about cloud-top structure and atmospheric/cloud evolution.

---

## IMERG

Current precipitation variables:

```text
Precipitation
Availability
```

IMERG provides precipitation information that complements satellite cloud observations.

---

## ERA5

Current atmospheric variables:

```text
u10
v10
d2m
t2m
msl
sp
tcc
cape
```

along with an availability mask.

These variables represent the surrounding atmospheric environment and convective potential.

---

## LIS

Lightning Imaging Sensor observations are used primarily for:

* lightning target generation
* independent validation
* target availability

LIS is **not included as a predictive input** to the current AI model because doing so could introduce target leakage.

---

# 6. Reliability-Aware Fusion

A central feature of the system is that each modality is accompanied by an availability/reliability signal.

Conceptually:

```text
Observation
     +
Availability Mask
     ↓
Reliability-Aware Feature
```

This allows the model to distinguish between:

```text
No signal observed
```

and:

```text
Sensor/data source unavailable
```

This is important because missing observations should not automatically be interpreted as evidence of no storm activity.

---

# 7. AI Model Architecture

The current model is implemented in:

```text
src/models/nowcasting_model.py
```

Architecture:

```text
Multimodal Input
      │
      ├── Observation channels
      │
      └── Availability channels
              │
              ▼
        Convolution Layers
              │
              ▼
          ConvLSTM
              │
              ▼
           Decoder
              │
              ▼
     Multi-Horizon Output
```

Current model:

```text
ThunderstormNowcaster
```

Trainable parameters:

```text
341,700
```

Current input shape:

```text
(B, T, 16, 27, 27)
```

Availability input:

```text
(B, T, 16, 27, 27)
```

Output:

```text
(B, 4, 27, 27)
```

The current prototype therefore produces four forecast horizons.

---

# 8. Predictive Channels

The current model uses 16 predictive channels:

```text
0  INSAT TIR1
1  INSAT TIR2
2  INSAT WV
3  INSAT VIS
4  INSAT availability

5  IMERG precipitation
6  IMERG availability

7  ERA5 u10
8  ERA5 v10
9  ERA5 d2m
10 ERA5 t2m
11 ERA5 msl
12 ERA5 sp
13 ERA5 tcc
14 ERA5 CAPE
15 ERA5 availability
```

LIS lightning observations are excluded from predictive inference.

---

# 9. Current Model Status

The ConvLSTM architecture has been implemented and tested successfully.

However, the model is **not yet trained on a statistically sufficient multi-event dataset**.

Therefore:

> The current ConvLSTM output must not be interpreted as a trained or production-quality probability forecast.

The current dashboard uses a multimodal heuristic prototype while the supervised training corpus is being expanded.

---

# 10. Prototype Multimodal Heuristic

The current prototype combines:

### Environmental signal

```text
CAPE
Moisture
Wind
Total cloud cover
```

### Satellite signal

```text
Cloud-top cooling
TIR1
TIR2
Water vapour
```

### Precipitation signal

```text
Current precipitation
Precipitation growth
Spatial precipitation gradient
```

### Reliability

```text
INSAT availability
IMERG availability
ERA5 availability
```

The result is a normalized prototype thunderstorm-risk score.

These scores are **heuristic risk scores and not calibrated probabilities**.

---

# 11. Temporal Forecast Prototype

The prototype currently supports:

```text
+30 min
+60 min
+90 min
+120 min
```

Future evolution currently uses a persistence/decay mechanism.

Example decay factors:

```text
+30  → 1.00
+60  → 0.90
+90  → 0.78
+120 → 0.65
```

This demonstrates the intended multi-horizon interface.

The final system will replace this heuristic temporal evolution with a learned spatiotemporal forecasting model once sufficient training events are available.

---

# 12. Lightning Target

For the current real event:

```text
Analysis time:
2020-05-01 04:30 UTC
```

Target interval:

```text
2020-05-01 05:00–05:30 UTC
```

Target definition:

```text
1     = observed lightning
0     = observed but no lightning
NaN   = unavailable
```

The target availability mask is retained separately.

---

# 13. Real LIS Event

The raw LIS observation contains:

```text
207 flashes
5038 events
1567 groups
40747 viewtime records
```

Within the selected study region:

```text
117 flashes
```

The main lightning activity occurred around:

```text
05:18–05:20 UTC
```

The observed lightning target contains:

```text
55 positive grid cells
```

This real event is currently used for end-to-end prototype validation.

---

# 14. Baseline Methods

The prototype includes several baseline approaches.

## Physics-Informed Baseline

Uses environmental and observational indicators such as:

```text
CAPE
Moisture
Wind
Satellite cooling
Precipitation
Spatial gradients
Data reliability
```

## Motion Baseline

Uses previous observations to estimate storm evolution through spatial motion/correlation.

## Multimodal Heuristic

Combines:

```text
INSAT
IMERG
ERA5
Reliability
```

into a multimodal risk field.

These methods are currently intended for prototype comparison and system validation rather than final performance claims.

---

# 15. Important Validation Limitation

The current training corpus contains only:

```text
1 real event
1 valid supervised window
```

This is not sufficient for reliable training or statistical evaluation of a ConvLSTM.

Therefore, the project currently makes **no claim of generalizable learned-model performance**.

The next phase is to expand the overlapping historical dataset and create multiple independent thunderstorm events for:

* training
* validation
* testing
* ablation studies
* missing-sensor experiments
* calibration
* reliability evaluation

---

# 16. Data Availability

Current event inventory:

| Date       | INSAT | IMERG | ERA5 | LIS | Target | Multimodal |
| ---------- | ----: | ----: | ---: | --: | -----: | ---------: |
| 2020-04-30 |     — |     — |    — |   ✓ |      — |          — |
| 2020-05-01 |     ✓ |     ✓ |    ✓ |   ✓ |      ✓ |          ✓ |
| 2020-05-02 |     — |     ✓ |    — |   ✓ |      — |          — |
| 2020-05-03 |     — |     ✓ |    — |   ✓ |      — |          — |

The main current limitation is the availability of overlapping observations from all required modalities.

---

# 17. Dashboard

The frontend is built using:

```text
React
TypeScript
Vite
```

The dashboard is designed as a map-first weather intelligence interface.

Current concepts include:

```text
Interactive weather map
Layer controls
Observation timeline
Forecast horizons
Selected-cell information
Multimodal data status
Thunderstorm risk visualization
```

The dashboard reads time-dependent data using:

```text
[time][y][x]
```

rather than incorrectly treating the observations as only:

```text
[y][x]
```

This fixed an earlier issue where real data appeared as zeros in the UI.

---

# 18. Backend

Backend technologies:

```text
FastAPI
Python
xarray
NetCDF4
NumPy
SciPy
Pandas
h5py
```

Main backend:

```text
backend/main.py
```

Important API:

```text
GET /api/multimodal/latest
```

Production backend:

```text
https://thunderstorm-nowcasting-1.onrender.com
```

FastAPI documentation:

```text
https://thunderstorm-nowcasting-1.onrender.com/docs
```

---

# 19. Frontend Deployment

Frontend is deployed using:

```text
Vercel
```

Configuration:

```text
Framework: Vite
Root Directory: dashboard
Build Command: npm run build
Output Directory: dist
Install Command: npm install
```

Production API environment variable:

```text
VITE_API_BASE_URL=https://thunderstorm-nowcasting-1.onrender.com
```

The frontend uses:

```ts
fetch(`${import.meta.env.VITE_API_BASE_URL}/api/multimodal/latest`)
```

for the production API connection.

---

# 20. Backend Deployment

Backend is deployed using:

```text
Render
```

Configuration:

```text
Service: Web Service
Root Directory: .
Runtime: Python
Build Command: pip install -r requirements.txt
Start Command: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Render provides the runtime port through:

```text
$PORT
```

The local development port:

```text
8010
```

is only for the local development environment.

---

# 21. Repository Structure

```text
thunderstorm-nowcasting/
│
├── backend/
│   └── main.py
│
├── dashboard/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   └── main.tsx
│   ├── vite.config.ts
│   ├── package.json
│   └── ...
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── insat/
│       ├── imerg/
│       ├── era5/
│       ├── lis/
│       ├── multimodal/
│       ├── targets/
│       └── training/
│
├── src/
│   ├── datasets/
│   │   └── lightning_dataset.py
│   ├── models/
│   │   └── nowcasting_model.py
│   └── preprocessing/
│
├── mdapi/
│
├── outputs/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 22. Local Development

## Backend

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --port 8010
```

Backend:

```text
http://127.0.0.1:8010
```

API:

```text
http://127.0.0.1:8010/api/multimodal/latest
```

---

## Frontend

```powershell
cd dashboard
npm install
npm run dev
```

The Vite development proxy forwards:

```text
/api
```

to:

```text
http://127.0.0.1:8010
```

---

## Production Build

```powershell
cd dashboard
npm run build
```

---

# 23. Git Workflow

Check status:

```powershell
git status
```

Add changes:

```powershell
git add .
```

Commit:

```powershell
git commit -m "Describe change"
```

Push:

```powershell
git push origin main
```

Vercel and Render are connected to the `main` branch and can automatically redeploy after pushes.

---

# 24. Processed Data Git Policy

Most processed weather data is intentionally ignored:

```text
data/processed/
```

This prevents the repository from accidentally becoming very large.

The current prototype multimodal dataset is an exception because the deployed Render API needs access to it.

To force-add only that required file:

```powershell
git add -f data/processed/multimodal/odisha_multimodal_20200501.nc
```

Do not remove the general:

```text
data/processed/
```

ignore rule.

---

# 25. Current Deployment Blocker

The Render API currently reports:

```text
Multimodal dataset not found:
/opt/render/project/src/data/processed/multimodal/odisha_multimodal_20200501.nc
```

The reason is that the NetCDF file is ignored by Git under:

```text
data/processed/
```

The required fix is:

```powershell
git add -f data/processed/multimodal/odisha_multimodal_20200501.nc
git commit -m "Add multimodal prototype dataset"
git push origin main
```

After Render redeploys, verify:

```text
https://thunderstorm-nowcasting-1.onrender.com/api/multimodal/latest
```

The endpoint should then return the real multimodal dataset.

---

# 26. Next Development Roadmap

### Phase 1 — Deployment completion

```text
Add prototype NetCDF to Git
        ↓
Render redeployment
        ↓
Verify API
        ↓
Verify Vercel dashboard
```

### Phase 2 — Dataset pipeline

```text
Inspect target schema
        ↓
Fix lightning_dataset.py
        ↓
Build correct 16-channel training samples
        ↓
Apply target availability masks
```

### Phase 3 — Historical corpus

Expand to multiple independent thunderstorm events.

Target:

```text
Multiple events
Multiple atmospheric regimes
Multiple locations
Multiple sensor availability conditions
```

### Phase 4 — AI training

Train:

```text
ConvLSTM / U-Net / Attention-based model
```

using multimodal historical sequences.

### Phase 5 — Validation

Evaluate:

```text
CSI
F1
Precision
Recall
Brier Score
ECE
Reliability diagrams
Lead-time performance
Spatial displacement
Top-k lightning-cell recall
```

Evaluation should be performed on held-out events.

### Phase 6 — Reliability

Perform controlled missing-sensor experiments:

```text
INSAT missing
IMERG missing
ERA5 missing
Lightning unavailable
Multiple modalities unavailable
```

The model should degrade gracefully rather than treating missing data as zero signal.

### Phase 7 — Operational dashboard

Integrate:

```text
Radar
INSAT
Lightning
AWS
NWP
AI nowcast
Confidence
Alerts
```

and expand from the current prototype region toward:

```text
Mumbai / Maharashtra
        ↓
India
```

---

# 27. Planned Final Architecture

```text
                 ┌───────────────────────┐
                 │ Satellite / INSAT     │
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │ Radar Observations    │
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │ Lightning Observations│
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │ AWS / Ground Sensors   │
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │ ERA5 / NWP             │
                 └───────────┬───────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Data Harmonization       │
                │ Spatial + Temporal       │
                │ Alignment                │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Availability / Quality   │
                │ Masks                    │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Multimodal Fusion        │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Spatiotemporal AI        │
                │ ConvLSTM / U-Net /      │
                │ Attention               │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Multi-Horizon Nowcast    │
                │ +15 +30 +60 +90 +120   │
                └────────────┬────────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
          Thunderstorm Risk       Confidence
                  │                     │
                  └──────────┬──────────┘
                             ▼
                ┌─────────────────────────┐
                │ Operational Dashboard    │
                │ Map + Alerts + Timeline  │
                └─────────────────────────┘
```

---

# 28. Project Status

| Component                   | Status                         |
| --------------------------- | ------------------------------ |
| Real INSAT ingestion        | ✓                              |
| Real IMERG ingestion        | ✓                              |
| Real ERA5 ingestion         | ✓                              |
| Real LIS ingestion          | ✓                              |
| Multimodal NetCDF           | ✓                              |
| Reliability masks           | ✓                              |
| Lightning target concept    | ✓                              |
| Physics baseline            | ✓                              |
| Motion baseline             | ✓                              |
| Multimodal heuristic        | ✓                              |
| Temporal prototype          | ✓                              |
| ConvLSTM architecture       | ✓                              |
| ConvLSTM training           | Not yet                        |
| Multi-event corpus          | Not yet sufficient             |
| Frontend                    | ✓                              |
| FastAPI backend             | ✓                              |
| Vercel deployment           | ✓                              |
| Render deployment           | ✓                              |
| Render dataset availability | **Pending Git dataset upload** |
| Production AI inference     | Not yet                        |
| Maharashtra-scale model     | Future phase                   |

---

# 29. Scientific / Engineering Positioning

The project should be presented as a **reliability-aware multimodal nowcasting system**, rather than simply an AI weather dashboard.

The core contribution is the integration of:

```text
Heterogeneous atmospheric observations
        +
Explicit data availability
        +
Spatiotemporal modelling
        +
Independent lightning targets
        +
Confidence / reliability estimation
```

The system is designed to remain useful even when one or more observation sources are temporarily unavailable.

---

# 30. Current Honest Claim

The current prototype demonstrates that the complete pipeline can operate on a real multimodal thunderstorm event:

```text
Real observations
      ↓
Preprocessing
      ↓
Multimodal alignment
      ↓
Availability-aware fusion
      ↓
Lightning target
      ↓
Prototype nowcasting
      ↓
Interactive dashboard
      ↓
Cloud deployment
```

The next major scientific milestone is obtaining enough independent overlapping events to train and evaluate the learned spatiotemporal model statistically.

---

## 🚀 Live Prototype

**Prototype Dashboard:**  
https://thunderstorm-nowcasting-u6es-h3g7d4it5.vercel.app/

The live prototype demonstrates the current multimodal thunderstorm nowcasting dashboard, including the interactive map, observation timeline, multimodal atmospheric data, thunderstorm-risk visualization, and prototype forecast workflow.

> **Note:** This is the current prototype deployment. The learned ConvLSTM model is not yet trained on a statistically sufficient multi-event dataset, so the current risk/forecast visualization represents the prototype inference pipeline rather than a production-calibrated forecast.

| Live prototype | [Open Prototype](https://thunderstorm-nowcasting-u6es-h3g7d4it5.vercel.app/) |

## License

This project is developed as part of **Smart India Hackathon 2026**.

Dataset licensing and redistribution should follow the terms of the respective data providers.
