# Thunderstorm Nowcasting

A project focused on thunderstorm nowcasting using multiple meteorological and remote sensing inputs, including radar, INSAT satellite imagery, lightning observations, AWS data, and ERA5 reanalysis.

## Overview

This repository is structured to support a full weather-intelligence workflow:

- data collection and organization
- preprocessing and alignment across sources
- exploratory analysis and model experimentation
- API-based access to processed outputs
- dashboard visualization for monitoring thunderstorm activity

## Project Structure

```text
thunderstorm-nowcasting/
├── data/
│   ├── raw/
│   │   ├── radar/
│   │   ├── insat/
│   │   ├── lightning/
│   │   ├── aws/
│   │   └── era5/
│   ├── processed/
│   ├── aligned/
│   └── samples/
├── notebooks/
├── src/
├── configs/
├── api/
├── dashboard/
├── tools/
│   └── mdapi/
│       ├── mdapi.py
│       └── config.json
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
└── .venv/
```

## Data Sources

- Radar observations
- INSAT satellite data
- Lightning detection data
- AWS / station measurements
- ERA5 reanalysis fields

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the MDAPI Tool

```bash
python tools/mdapi/mdapi.py
```

## Workflow

1. Download raw data into the relevant folders under `data/raw/`.
2. Preprocess and align data into `data/aligned/` or `data/processed/`.
3. Use notebooks in `notebooks/` for experiments and analysis.
4. Expose processing results through the API or dashboard.
5. Evaluate and iterate on forecasts and monitoring outputs.

## Notes

This project is intended as a starter framework for thunderstorm nowcasting research and deployment. It can be expanded with model training pipelines, feature engineering, inference scripts, and production APIs.
