
"""
SIH26072YELLOW
==============================================================

REAL-DATA TRAINING CORPUS AUDIT

Purpose
-------
Audit the currently available real datasets before training.

The script checks:

    INSAT
    IMERG
    ERA5
    LIS

and determines:

    1. Available dates
    2. Number of frames/files
    3. Temporal coverage
    4. Common time overlap
    5. Potential 30-minute samples
    6. Whether each sample has the required
       input history and future forecast horizons

IMPORTANT
---------
This script DOES NOT:
    - modify datasets
    - download data
    - create synthetic data
    - train the model

It is purely an audit.
"""


from pathlib import Path
import re
from collections import defaultdict

import numpy as np
import xarray as xr


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# DATA DIRECTORIES
# ============================================================

RAW_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


INSAT_RAW_DIR = (
    RAW_DIR / "insat"
)

INSAT_PROCESSED_DIR = (
    PROCESSED_DIR / "insat"
)

IMERG_PROCESSED_DIR = (
    PROCESSED_DIR / "imerg"
)

ERA5_PROCESSED_DIR = (
    PROCESSED_DIR / "era5"
)

LIS_PROCESSED_DIR = (
    PROCESSED_DIR / "lis"
)

MULTIMODAL_DIR = (
    PROCESSED_DIR / "multimodal"
)

TARGET_DIR = (
    PROCESSED_DIR / "targets"
)


# ============================================================
# NOWCAST CONFIGURATION
# ============================================================

GRID_SIZE = 27

CADENCE_MINUTES = 30

HISTORY = 2

HORIZONS = 4


# ============================================================
# HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_row(label, value):

    print(
        f"{label:<30}: {value}"
    )


def safe_open_dataset(path):

    try:

        return xr.open_dataset(
            path
        )

    except Exception as exc:

        print(
            f"[WARNING] Could not open:\n"
            f"  {path}\n"
            f"  {type(exc).__name__}: {exc}"
        )

        return None


def normalize_time_array(values):

    result = []

    for value in values:

        try:

            timestamp = np.datetime64(
                value,
                "m"
            )

            result.append(
                timestamp
            )

        except Exception:

            continue

    return np.asarray(
        result,
        dtype="datetime64[m]"
    )


# ============================================================
# FIND RAW INSAT DATES
# ============================================================

def audit_raw_insat():

    section(
        "1. RAW INSAT INVENTORY"
    )

    if not INSAT_RAW_DIR.exists():

        print(
            "[WARNING] Raw INSAT directory does not exist:"
        )

        print(
            INSAT_RAW_DIR
        )

        return {}

    date_files = defaultdict(list)

    for file in INSAT_RAW_DIR.rglob("*"):

        if not file.is_file():

            continue

        name = file.name

        # ----------------------------------------------------
        # Look for filenames such as:
        #
        # 3SIMG_01MAY2020_0400_L1B_STD_V01R00.h5
        # ----------------------------------------------------

        match = re.search(
            r"3SIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})",
            name.upper()
        )

        if match is None:

            continue

        date_text = match.group(1)

        time_text = match.group(2)

        try:

            timestamp = np.datetime64(
                (
                    f"{date_text[:2]}-"
                    f"{date_text[2:5]}-"
                    f"{date_text[5:]}T"
                    f"{time_text[:2]}:"
                    f"{time_text[2:]}"
                )
            )

        except Exception:

            # Alternative parser below
            continue

        date_key = str(
            timestamp.astype(
                "datetime64[D]"
            )
        )

        date_files[
            date_key
        ].append(
            timestamp
        )

    if not date_files:

        print(
            "No recognizable raw INSAT files found."
        )

        return {}

    for date_key in sorted(
        date_files.keys()
    ):

        times = sorted(
            date_files[date_key]
        )

        print()
        print(
            f"Date: {date_key}"
        )

        print(
            f"  Files/frames : {len(times)}"
        )

        print(
            f"  First        : {times[0]}"
        )

        print(
            f"  Last         : {times[-1]}"
        )

    return date_files


# ============================================================
# AUDIT PROCESSED DATASET
# ============================================================

def audit_processed_dataset(
    directory,
    dataset_name,
):

    if not directory.exists():

        return None

    files = sorted(
        directory.glob("*.nc")
    )

    if not files:

        return None

    print()
    print(
        f"{dataset_name}:"
    )

    for path in files:

        ds = safe_open_dataset(
            path
        )

        if ds is None:

            continue

        try:

            if "time" in ds.coords:

                times = normalize_time_array(
                    ds["time"].values
                )

            elif "time" in ds:

                times = normalize_time_array(
                    ds["time"].values
                )

            else:

                times = np.asarray(
                    [],
                    dtype="datetime64[m]"
                )

            print()
            print(
                f"  File: {path.name}"
            )

            print(
                f"  Frames: {len(times)}"
            )

            if len(times) > 0:

                print(
                    f"  First : {times[0]}"
                )

                print(
                    f"  Last  : {times[-1]}"
                )

                if len(times) > 1:

                    deltas = (
                        np.diff(times)
                        .astype(
                            "timedelta64[m]"
                        )
                        .astype(int)
                    )

                    unique_deltas = sorted(
                        set(
                            deltas.tolist()
                        )
                    )

                    print(
                        "  Cadence:",
                        unique_deltas,
                        "minutes"
                    )

            print(
                "  Variables:",
                list(ds.data_vars)
            )

            print(
                "  Dimensions:",
                dict(ds.sizes)
            )

        finally:

            ds.close()

    return files


# ============================================================
# GET DATASET TIME COVERAGE
# ============================================================

def get_dataset_times(
    directory,
):

    if not directory.exists():

        return np.asarray(
            [],
            dtype="datetime64[m]"
        )

    all_times = []

    for path in sorted(
        directory.glob("*.nc")
    ):

        ds = safe_open_dataset(
            path
        )

        if ds is None:

            continue

        try:

            if "time" in ds.coords:

                times = normalize_time_array(
                    ds["time"].values
                )

            elif "time" in ds:

                times = normalize_time_array(
                    ds["time"].values
                )

            else:

                times = np.asarray(
                    [],
                    dtype="datetime64[m]"
                )

            all_times.extend(
                times.tolist()
            )

        finally:

            ds.close()

    if not all_times:

        return np.asarray(
            [],
            dtype="datetime64[m]"
        )

    return np.unique(
        np.asarray(
            all_times,
            dtype="datetime64[m]"
        )
    )


# ============================================================
# CHECK REGULAR 30-MINUTE GRID
# ============================================================

def build_common_30min_grid(
    start,
    end,
):

    if start > end:

        return np.asarray(
            [],
            dtype="datetime64[m]"
        )

    return np.arange(
        start,
        end + np.timedelta64(
            CADENCE_MINUTES,
            "m"
        ),
        np.timedelta64(
            CADENCE_MINUTES,
            "m"
        ),
        dtype="datetime64[m]"
    )


# ============================================================
# CALCULATE POTENTIAL SAMPLES
# ============================================================

def calculate_samples(
    common_times,
):

    required_steps = (
        HISTORY - 1
        + HORIZONS
    )

    samples = []

    # --------------------------------------------------------
    # For history=2 and horizons=4:
    #
    # input:
    #   t-30
    #   t
    #
    # target:
    #   t+30
    #   t+60
    #   t+90
    #   t+120
    #
    # Therefore we need:
    #
    #   t-30 through t+120
    #
    # = 5 intervals / 6 frames
    # --------------------------------------------------------

    for index in range(
        len(common_times)
    ):

        end_index = (
            index
            + required_steps
        )

        if end_index >= len(
            common_times
        ):

            break

        end_time = common_times[
            index
        ]

        target_times = common_times[
            index + HISTORY:
            end_index + 1
        ]

        if len(target_times) != HORIZONS:

            continue

        samples.append(
            (
                end_time,
                target_times.copy(),
            )
        )

    return samples


# ============================================================
# DATASET INTERSECTION
# ============================================================

def calculate_intersection(
    dataset_times,
):

    available = {
        name: set(
            times.tolist()
        )
        for name, times
        in dataset_times.items()
    }

    non_empty = [
        values
        for values in available.values()
        if values
    ]

    if not non_empty:

        return np.asarray(
            [],
            dtype="datetime64[m]"
        )

    intersection = set.intersection(
        *non_empty
    )

    if not intersection:

        return np.asarray(
            [],
            dtype="datetime64[m]"
        )

    return np.asarray(
        sorted(intersection),
        dtype="datetime64[m]"
    )


# ============================================================
# PRINT SAMPLE DETAILS
# ============================================================

def print_sample_details(
    samples,
    limit=20,
):

    print()

    if not samples:

        print(
            "No complete samples available."
        )

        return

    print(
        f"Showing up to {limit} potential samples:"
    )

    for index, (
        end_time,
        target_times,
    ) in enumerate(
        samples[:limit]
    ):

        print()

        print(
            f"Sample {index + 1}"
        )

        print(
            f"  Analysis time : {end_time}"
        )

        print(
            "  Input times   :"
        )

        input_start = (
            end_time
            - np.timedelta64(
                (HISTORY - 1)
                * CADENCE_MINUTES,
                "m",
            )
        )

        for step in range(
            HISTORY
        ):

            timestamp = (
                input_start
                + np.timedelta64(
                    step
                    * CADENCE_MINUTES,
                    "m",
                )
            )

            print(
                f"    {timestamp}"
            )

        print(
            "  Target times  :"
        )

        for target_index, timestamp in enumerate(
            target_times
        ):

            print(
                f"    {HORIZON_LABELS[target_index]}: "
                f"{timestamp}"
            )


# ============================================================
# HORIZON LABELS
# ============================================================

HORIZON_LABELS = [
    "+30 min",
    "+60 min",
    "+90 min",
    "+120 min",
]


# ============================================================
# MULTIMODAL DATASET AUDIT
# ============================================================

def audit_multimodal_files():

    section(
        "5. EXISTING MULTIMODAL DATASETS"
    )

    if not MULTIMODAL_DIR.exists():

        print(
            "No multimodal directory."
        )

        return

    files = sorted(
        MULTIMODAL_DIR.glob(
            "*.nc"
        )
    )

    if not files:

        print(
            "No multimodal NetCDF files found."
        )

        return

    for path in files:

        ds = safe_open_dataset(
            path
        )

        if ds is None:

            continue

        try:

            print()
            print(
                path.name
            )

            print(
                "  Dimensions:",
                dict(ds.sizes)
            )

            print(
                "  Variables:",
                list(ds.data_vars)
            )

            if "time" in ds.coords:

                times = normalize_time_array(
                    ds["time"].values
                )

                print(
                    "  Frames:",
                    len(times)
                )

                if len(times):

                    print(
                        "  First:",
                        times[0]
                    )

                    print(
                        "  Last:",
                        times[-1]
                    )

        finally:

            ds.close()


# ============================================================
# TARGET AUDIT
# ============================================================

def audit_target_files():

    section(
        "6. EXISTING LIGHTNING TARGET DATASETS"
    )

    if not TARGET_DIR.exists():

        print(
            "No target directory."
        )

        return

    files = sorted(
        TARGET_DIR.glob(
            "*.nc"
        )
    )

    if not files:

        print(
            "No target NetCDF files found."
        )

        return

    for path in files:

        ds = safe_open_dataset(
            path
        )

        if ds is None:

            continue

        try:

            print()
            print(
                path.name
            )

            print(
                "  Dimensions:",
                dict(ds.sizes)
            )

            print(
                "  Variables:",
                list(ds.data_vars)
            )

            if "time" in ds.coords:

                times = normalize_time_array(
                    ds["time"].values
                )

                print(
                    "  Frames:",
                    len(times)
                )

                if len(times):

                    print(
                        "  First:",
                        times[0]
                    )

                    print(
                        "  Last:",
                        times[-1]
                    )

        finally:

            ds.close()


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    section(
        "SIH26072YELLOW"
    )

    print(
        "REAL TRAINING CORPUS AUDIT"
    )

    print()

    print_row(
        "Project root",
        PROJECT_ROOT
    )

    print_row(
        "Grid",
        f"{GRID_SIZE} x {GRID_SIZE}"
    )

    print_row(
        "Cadence",
        f"{CADENCE_MINUTES} minutes"
    )

    print_row(
        "History",
        f"{HISTORY} frames"
    )

    print_row(
        "Forecast horizons",
        HORIZONS
    )

    print()

    print(
        "No files will be modified."
    )

    print(
        "No synthetic data will be created."
    )


    # ========================================================
    # RAW INSAT
    # ========================================================

    raw_insat = audit_raw_insat()


    # ========================================================
    # PROCESSED DATASET INVENTORY
    # ========================================================

    section(
        "2. PROCESSED DATASET INVENTORY"
    )

    audit_processed_dataset(
        INSAT_PROCESSED_DIR,
        "INSAT",
    )

    audit_processed_dataset(
        IMERG_PROCESSED_DIR,
        "IMERG",
    )

    audit_processed_dataset(
        ERA5_PROCESSED_DIR,
        "ERA5",
    )

    audit_processed_dataset(
        LIS_PROCESSED_DIR,
        "LIS",
    )


    # ========================================================
    # GET TIMES
    # ========================================================

    section(
        "3. TEMPORAL COVERAGE"
    )

    dataset_times = {

        "INSAT": get_dataset_times(
            INSAT_PROCESSED_DIR
        ),

        "IMERG": get_dataset_times(
            IMERG_PROCESSED_DIR
        ),

        "ERA5": get_dataset_times(
            ERA5_PROCESSED_DIR
        ),

        "LIS": get_dataset_times(
            LIS_PROCESSED_DIR
        ),
    }


    for name, times in dataset_times.items():

        print()

        print(
            f"{name}:"
        )

        print(
            f"  Unique frames: {len(times)}"
        )

        if len(times):

            print(
                f"  First: {times[0]}"
            )

            print(
                f"  Last : {times[-1]}"
            )

            if len(times) > 1:

                deltas = (
                    np.diff(times)
                    .astype(
                        "timedelta64[m]"
                    )
                    .astype(int)
                )

                print(
                    "  Cadence values:",
                    sorted(
                        set(
                            deltas.tolist()
                        )
                    ),
                    "minutes",
                )


    # ========================================================
    # COMMON INTERSECTION
    # ========================================================

    section(
        "4. COMMON MULTIMODAL TIME INTERSECTION"
    )

    common_times = (
        calculate_intersection(
            dataset_times
        )
    )

    print(
        "Common timestamps:",
        len(common_times)
    )

    if len(common_times):

        print(
            "First common time:",
            common_times[0]
        )

        print(
            "Last common time:",
            common_times[-1]
        )

        if len(common_times) > 1:

            deltas = (
                np.diff(common_times)
                .astype(
                    "timedelta64[m]"
                )
                .astype(int)
            )

            print(
                "Common cadence:",
                sorted(
                    set(
                        deltas.tolist()
                    )
                ),
                "minutes",
            )


    # ========================================================
    # POTENTIAL SAMPLES
    # ========================================================

    section(
        "7. POTENTIAL COMPLETE TRAINING SAMPLES"
    )

    if len(common_times) == 0:

        print(
            "No common timestamps."
        )

        samples = []

    else:

        samples = calculate_samples(
            common_times
        )

        print(
            "Required frames per sample:",
            HISTORY + HORIZONS,
        )

        print(
            "Potential complete samples:",
            len(samples),
        )

        if samples:

            print(
                "First analysis time:",
                samples[0][0]
            )

            print(
                "Last analysis time:",
                samples[-1][0]
            )

    print_sample_details(
        samples
    )


    # ========================================================
    # EXISTING MULTIMODAL / TARGET FILES
    # ========================================================

    audit_multimodal_files()

    audit_target_files()


    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    section(
        "8. AUDIT INTERPRETATION"
    )

    if len(samples) == 0:

        print(
            "RESULT: No complete multimodal training "
            "samples are currently available."
        )

        print()

        print(
            "Next action:"
        )

        print(
            "Acquire/process additional real event periods."
        )

    elif len(samples) == 1:

        print(
            "RESULT: Only one complete sample is available."
        )

        print()

        print(
            "This is insufficient for meaningful neural "
            "network training."
        )

        print()

        print(
            "Next action:"
        )

        print(
            "Expand the real event corpus."
        )

    else:

        print(
            "RESULT:",
            len(samples),
            "complete temporal samples are available."
        )

        print()

        print(
            "These samples can be inspected further before "
            "training."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "A timestamp existing in all four datasets does "
        "not automatically guarantee that every modality "
        "has valid observations at every grid cell."
    )

    print()

    print(
        "The existing availability masks must remain part "
        "of the training pipeline."
    )

    print()

    print(
        "The final training corpus should be split by "
        "event/time period rather than randomly mixing "
        "neighboring frames between train and validation."
    )

    section(
        "AUDIT COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print()
        print("=" * 78)
        print("AUDIT FAILED")
        print("=" * 78)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

