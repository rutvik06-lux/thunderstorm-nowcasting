from pathlib import Path
import re
import numpy as np
import xarray as xr


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MULTIMODAL_DIR = PROJECT_ROOT / "data" / "processed" / "multimodal"
TARGET_DIR = PROJECT_ROOT / "data" / "processed" / "targets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "training"

OUTPUT_PATH = OUTPUT_DIR / "training_corpus.nc"

HISTORY = 2
HORIZONS = 4
CADENCE_MINUTES = 30

GRID_Y = 27
GRID_X = 27
CHANNELS = 18


CHANNEL_NAMES = [
    "insat_tir1",
    "insat_tir2",
    "insat_wv",
    "insat_vis",
    "insat_availability",
    "imerg_precipitation",
    "imerg_availability",
    "era5_u10",
    "era5_v10",
    "era5_d2m",
    "era5_t2m",
    "era5_msl",
    "era5_sp",
    "era5_tcc",
    "era5_cape",
    "era5_availability",
    "lis_lightning_density",
    "lis_lightning_availability",
]


REQUIRED_MULTIMODAL = set(CHANNEL_NAMES)

REQUIRED_TARGET = {
    "lightning_target",
    "target_availability",
}


# ================================================================
# HELPERS
# ================================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def normalize_time(value):
    return np.datetime64(value, "ns")


def event_key(path):
    """
    Convert filenames such as:

        odisha_multimodal_20200501.nc
        lightning_targets_odisha_20200501.nc

    into the same matching key.
    """

    name = path.stem.lower()

    name = name.replace(
        "lightning_targets_",
        "",
    )

    name = name.replace(
        "multimodal_",
        "",
    )

    name = name.replace(
        "_multimodal",
        "",
    )

    name = re.sub(
        r"^targets_",
        "",
        name,
    )

    return name


def find_target(multimodal_path):
    key = event_key(multimodal_path)

    candidates = sorted(
        TARGET_DIR.glob("*.nc")
    )

    for candidate in candidates:
        if event_key(candidate) == key:
            return candidate

    # Date fallback
    match = re.search(
        r"\d{8}",
        multimodal_path.stem,
    )

    if match:
        date = match.group(0)

        date_matches = [
            candidate
            for candidate in candidates
            if date in candidate.stem
        ]

        if len(date_matches) == 1:
            return date_matches[0]

    return None


def validate_dataset(ds, required, name):
    missing = sorted(
        required - set(ds.data_vars)
    )

    if missing:
        raise ValueError(
            f"{name} missing variables: "
            + ", ".join(missing)
        )

    if "time" not in ds.dims:
        raise ValueError(
            f"{name} has no time dimension."
        )

    if ds.sizes.get("y") != GRID_Y:
        raise ValueError(
            f"{name}: expected y={GRID_Y}, "
            f"got {ds.sizes.get('y')}"
        )

    if ds.sizes.get("x") != GRID_X:
        raise ValueError(
            f"{name}: expected x={GRID_X}, "
            f"got {ds.sizes.get('x')}"
        )


def build_input_tensor(ds):
    """
    Build:

        [time, channel, y, x]
    """

    arrays = []

    for name in CHANNEL_NAMES:

        values = (
            ds[name]
            .values
            .astype(np.float32)
        )

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        arrays.append(values)

    return np.stack(
        arrays,
        axis=1,
    ).astype(np.float32)


def build_target_arrays(target_ds):

    target = (
        target_ds["lightning_target"]
        .values
        .astype(np.float32)
    )

    availability = (
        target_ds["target_availability"]
        .values
        .astype(np.float32)
    )

    target = np.nan_to_num(
        target,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    availability = np.nan_to_num(
        availability,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    availability = (
        availability > 0.5
    ).astype(np.float32)

    return target, availability


def make_lookup(times):
    return {
        normalize_time(time): index
        for index, time in enumerate(times)
    }


# ================================================================
# TEMPORAL WINDOW DISCOVERY
# ================================================================

def find_windows(multimodal_times, target_times):

    multimodal_lookup = make_lookup(
        multimodal_times
    )

    target_lookup = make_lookup(
        target_times
    )

    windows = []

    for analysis_time in multimodal_times:

        analysis_time = normalize_time(
            analysis_time
        )

        # --------------------------------------------------------
        # Input history
        #
        # HISTORY=2 means:
        #
        # analysis-30
        # analysis
        # --------------------------------------------------------

        input_times = []

        valid_input = True

        for offset in range(
            HISTORY - 1,
            -1,
            -1,
        ):

            timestamp = (
                analysis_time
                - np.timedelta64(
                    offset * CADENCE_MINUTES,
                    "m",
                )
            )

            if timestamp not in multimodal_lookup:
                valid_input = False
                break

            input_times.append(
                timestamp
            )

        if not valid_input:
            continue

        # --------------------------------------------------------
        # Future target horizons
        #
        # +30
        # +60
        # +90
        # +120
        # --------------------------------------------------------

        target_times_for_window = []

        valid_target = True

        for horizon in range(
            1,
            HORIZONS + 1,
        ):

            timestamp = (
                analysis_time
                + np.timedelta64(
                    horizon * CADENCE_MINUTES,
                    "m",
                )
            )

            if timestamp not in target_lookup:
                valid_target = False
                break

            target_times_for_window.append(
                timestamp
            )

        if not valid_target:
            continue

        windows.append(
            {
                "analysis_time": analysis_time,
                "input_indices": [
                    multimodal_lookup[t]
                    for t in input_times
                ],
                "target_indices": [
                    target_lookup[t]
                    for t in target_times_for_window
                ],
            }
        )

    return windows


# ================================================================
# EVENT PROCESSING
# ================================================================

def process_event(multimodal_path):

    target_path = find_target(
        multimodal_path
    )

    print()
    print("-" * 70)
    print(
        f"Event: {multimodal_path.name}"
    )
    print("-" * 70)

    if target_path is None:

        print(
            "Target: NOT FOUND"
        )

        return []

    print(
        f"Target: {target_path.name}"
    )

    with xr.open_dataset(
        multimodal_path
    ) as multimodal_ds, xr.open_dataset(
        target_path
    ) as target_ds:

        validate_dataset(
            multimodal_ds,
            REQUIRED_MULTIMODAL,
            "Multimodal dataset",
        )

        validate_dataset(
            target_ds,
            REQUIRED_TARGET,
            "Target dataset",
        )

        multimodal_times = (
            multimodal_ds.time.values
        )

        target_times = (
            target_ds.time.values
        )

        print(
            f"Multimodal frames : "
            f"{len(multimodal_times)}"
        )

        print(
            f"Target frames     : "
            f"{len(target_times)}"
        )

        input_tensor = build_input_tensor(
            multimodal_ds
        )

        target, target_availability = (
            build_target_arrays(
                target_ds
            )
        )

        windows = find_windows(
            multimodal_times,
            target_times,
        )

        print(
            f"Candidate windows : "
            f"{len(windows)}"
        )

        valid_samples = []

        for window in windows:

            target_indices = (
                window["target_indices"]
            )

            observed_cells = 0
            positive_cells = 0

            for index in target_indices:

                mask = (
                    target_availability[index]
                    > 0.5
                )

                observed_cells += int(
                    np.count_nonzero(mask)
                )

                positive_cells += int(
                    np.count_nonzero(
                        (
                            target[index] > 0.5
                        )
                        & mask
                    )
                )

            # No observed target anywhere means
            # there is nothing valid to train against.
            if observed_cells == 0:
                continue

            input_indices = (
                window["input_indices"]
            )

            sample_input = input_tensor[
                input_indices
            ]

            sample_availability = np.stack(
                [
                    input_tensor[
                        index,
                        4,
                        :,
                        :,
                    ],
                    input_tensor[
                        index,
                        6,
                        :,
                        :,
                    ],
                    input_tensor[
                        index,
                        15,
                        :,
                        :,
                    ],
                ],
                axis=0,
            )

            # ----------------------------------------------------
            # IMPORTANT
            #
            # The Dataset/model expects availability with the
            # SAME 18-channel shape as X.
            #
            # Therefore create a full mask:
            #
            #   physical channels -> modality availability
            #
            # Other channels inherit the corresponding modality
            # availability.
            # ----------------------------------------------------

            full_availability = np.zeros_like(
                sample_input,
                dtype=np.float32,
            )

            # INSAT channels 0-4
            full_availability[
                :,
                0:5,
                :,
                :,
            ] = sample_availability[
                0:1,
                :,
                :,
            ][:, None, :, :]

            # IMERG channels 5-6
            full_availability[
                :,
                5:7,
                :,
                :,
            ] = sample_availability[
                1:2,
                :,
                :,
            ][:, None, :, :]

            # ERA5 channels 7-15
            full_availability[
                :,
                7:16,
                :,
                :,
            ] = sample_availability[
                2:3,
                :,
                :,
            ][:, None, :, :]

            # LIS channels 16-17
            #
            # LIS is retained in the input structure but its
            # predictive availability is preserved separately.
            #
            # It is NOT used as the target.
            lis_availability = np.nan_to_num(
                input_tensor[
                    input_indices,
                    17,
                    :,
                    :,
                ],
                nan=0.0,
            )

            full_availability[
                :,
                16:18,
                :,
                :,
            ] = lis_availability[
                :,
                None,
                :,
                :,
            ]

            sample_target = target[
                target_indices
            ]

            sample_target_availability = (
                target_availability[
                    target_indices
                ]
            )

            valid_samples.append(
                {
                    "analysis_time": (
                        window["analysis_time"]
                    ),
                    "input_data": sample_input,
                    "availability": (
                        full_availability
                    ),
                    "target": sample_target,
                    "target_availability": (
                        sample_target_availability
                    ),
                    "observed_cells": (
                        observed_cells
                    ),
                    "positive_cells": (
                        positive_cells
                    ),
                    "event": multimodal_path.stem,
                }
            )

        print(
            f"Valid supervised : "
            f"{len(valid_samples)}"
        )

        return valid_samples


# ================================================================
# WRITE CORPUS
# ================================================================

def write_corpus(samples):

    if not samples:
        return None

    sample_count = len(samples)

    input_data = np.stack(
        [
            sample["input_data"]
            for sample in samples
        ],
        axis=0,
    ).astype(np.float32)

    availability = np.stack(
        [
            sample["availability"]
            for sample in samples
        ],
        axis=0,
    ).astype(np.float32)

    target = np.stack(
        [
            sample["target"]
            for sample in samples
        ],
        axis=0,
    ).astype(np.float32)

    target_availability = np.stack(
        [
            sample["target_availability"]
            for sample in samples
        ],
        axis=0,
    ).astype(np.float32)

    analysis_times = np.array(
        [
            sample["analysis_time"]
            for sample in samples
        ],
        dtype="datetime64[ns]",
    )

    observed_cells = np.array(
        [
            sample["observed_cells"]
            for sample in samples
        ],
        dtype=np.int32,
    )

    positive_cells = np.array(
        [
            sample["positive_cells"]
            for sample in samples
        ],
        dtype=np.int32,
    )

    # ------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------

    expected_input_shape = (
        sample_count,
        HISTORY,
        CHANNELS,
        GRID_Y,
        GRID_X,
    )

    expected_target_shape = (
        sample_count,
        HORIZONS,
        GRID_Y,
        GRID_X,
    )

    if input_data.shape != expected_input_shape:
        raise ValueError(
            f"Input shape {input_data.shape} "
            f"!= expected {expected_input_shape}"
        )

    if availability.shape != expected_input_shape:
        raise ValueError(
            f"Availability shape "
            f"{availability.shape} "
            f"!= expected "
            f"{expected_input_shape}"
        )

    if target.shape != expected_target_shape:
        raise ValueError(
            f"Target shape {target.shape} "
            f"!= expected {expected_target_shape}"
        )

    if target_availability.shape != expected_target_shape:
        raise ValueError(
            f"Target availability shape "
            f"{target_availability.shape} "
            f"!= expected "
            f"{expected_target_shape}"
        )

    # ------------------------------------------------------------
    # xarray dataset
    #
    # IMPORTANT:
    # input_data is used as the variable name.
    # x remains ONLY the spatial coordinate.
    # ------------------------------------------------------------

    dataset = xr.Dataset(
        data_vars={
            "input_data": (
                (
                    "sample",
                    "history",
                    "channel",
                    "y",
                    "x",
                ),
                input_data,
            ),

            "availability": (
                (
                    "sample",
                    "history",
                    "channel",
                    "y",
                    "x",
                ),
                availability,
            ),

            "target": (
                (
                    "sample",
                    "horizon",
                    "y",
                    "x",
                ),
                target,
            ),

            "target_availability": (
                (
                    "sample",
                    "horizon",
                    "y",
                    "x",
                ),
                target_availability,
            ),

            "observed_target_cells": (
                ("sample",),
                observed_cells,
            ),

            "positive_target_cells": (
                ("sample",),
                positive_cells,
            ),
        },

        coords={
            "sample": np.arange(
                sample_count
            ),

            "history": np.arange(
                HISTORY
            ),

            "channel": CHANNEL_NAMES,

            "horizon": np.array(
                [30, 60, 90, 120],
                dtype=np.int32,
            ),

            "y": np.arange(
                GRID_Y
            ),

            "x": np.arange(
                GRID_X
            ),

            "analysis_time": (
                "sample",
                analysis_times,
            ),
        },

        attrs={
            "description": (
                "Real-data multimodal thunderstorm "
                "nowcasting training corpus"
            ),

            "data_policy": (
                "Real observations only. "
                "No synthetic weather or lightning data."
            ),

            "history_frames": HISTORY,

            "forecast_horizons_minutes": (
                "30,60,90,120"
            ),

            "cadence_minutes": CADENCE_MINUTES,

            "grid_size": (
                f"{GRID_Y}x{GRID_X}"
            ),

            "channel_count": CHANNELS,

            "channel_order": ",".join(
                CHANNEL_NAMES
            ),

            "target_semantics": (
                "Lightning target with explicit "
                "target availability mask."
            ),

            "lis_semantics": (
                "LIS is sparse/orbital. "
                "Non-observation is not interpreted "
                "as zero lightning."
            ),

            "leakage_control": (
                "Input history ends at analysis time. "
                "Future observations are not used as inputs."
            ),

            "training_warning": (
                "Training should use event-level splits "
                "and sufficient independent events."
            ),
        },
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoding = {
        "input_data": {
            "zlib": True,
            "complevel": 4,
        },

        "availability": {
            "zlib": True,
            "complevel": 4,
        },

        "target": {
            "zlib": True,
            "complevel": 4,
        },

        "target_availability": {
            "zlib": True,
            "complevel": 4,
        },
    }

    dataset.to_netcdf(
        OUTPUT_PATH,
        encoding=encoding,
    )

    dataset.close()

    return {
        "samples": sample_count,
        "input_shape": input_data.shape,
        "availability_shape": availability.shape,
        "target_shape": target.shape,
        "target_availability_shape": (
            target_availability.shape
        ),
        "observed_cells": int(
            observed_cells.sum()
        ),
        "positive_cells": int(
            positive_cells.sum()
        ),
    }


# ================================================================
# AUDIT
# ================================================================

def audit(samples, event_results):

    header(
        "TRAINING CORPUS AUDIT"
    )

    event_count = len(
        event_results
    )

    events_with_samples = sum(
        1
        for result in event_results
        if result["sample_count"] > 0
    )

    sample_count = len(
        samples
    )

    observed = sum(
        sample["observed_cells"]
        for sample in samples
    )

    positive = sum(
        sample["positive_cells"]
        for sample in samples
    )

    print(
        f"Events discovered       : "
        f"{event_count}"
    )

    print(
        f"Events with samples     : "
        f"{events_with_samples}"
    )

    print(
        f"Valid supervised windows: "
        f"{sample_count}"
    )

    print(
        f"Observed target cells   : "
        f"{observed}"
    )

    print(
        f"Positive target cells   : "
        f"{positive}"
    )

    print()
    print(
        "Event breakdown:"
    )

    for result in event_results:

        print(
            f"  {result['event']:<45} "
            f"samples={result['sample_count']:>3}"
        )

    print()

    if sample_count == 0:

        print(
            "RESULT: No valid supervised training "
            "windows were found."
        )

    elif events_with_samples < 2:

        print(
            "RESULT: Corpus exists, but fewer than "
            "two independent events are available."
        )

        print(
            "Do NOT train/validate the ConvLSTM yet."
        )

    else:

        print(
            "RESULT: Multiple independent events "
            "are available for further corpus analysis."
        )

    print()
    print(
        "Rules:"
    )

    print(
        "  1. Preserve target_availability during training."
    )

    print(
        "  2. Never interpret LIS non-observation as zero lightning."
    )

    print(
        "  3. Split train/validation by event or time period."
    )

    print(
        "  4. Do not randomly split neighboring frames from one storm."
    )

    print(
        "  5. Do not create synthetic weather/lightning samples."
    )


# ================================================================
# MAIN
# ================================================================

def main():

    header(
        "REAL-DATA TRAINING CORPUS BUILDER"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Multimodal   : {MULTIMODAL_DIR}"
    )

    print(
        f"Targets      : {TARGET_DIR}"
    )

    print(
        f"Output       : {OUTPUT_PATH}"
    )

    print()
    print(
        f"History      : {HISTORY}"
    )

    print(
        f"Horizons     : +30/+60/+90/+120"
    )

    print(
        f"Grid         : {GRID_Y}x{GRID_X}"
    )

    print(
        f"Channels     : {CHANNELS}"
    )

    multimodal_files = sorted(
        MULTIMODAL_DIR.glob("*.nc")
    )

    print()
    print(
        f"Multimodal datasets found: "
        f"{len(multimodal_files)}"
    )

    if not multimodal_files:

        raise RuntimeError(
            "No multimodal NetCDF datasets found."
        )

    all_samples = []
    event_results = []

    for path in multimodal_files:

        try:

            samples = process_event(
                path
            )

            all_samples.extend(
                samples
            )

            event_results.append(
                {
                    "event": path.stem,
                    "sample_count": len(
                        samples
                    ),
                }
            )

        except Exception as exc:

            print()
            print(
                f"ERROR: {path.name}"
            )

            print(
                f"  {exc}"
            )

            event_results.append(
                {
                    "event": path.stem,
                    "sample_count": 0,
                }
            )

    audit(
        all_samples,
        event_results,
    )

    if not all_samples:

        print()
        print(
            "No NetCDF corpus was written."
        )

        return

    header(
        "WRITING TRAINING CORPUS"
    )

    summary = write_corpus(
        all_samples
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print(
        f"Input shape             : "
        f"{summary['input_shape']}"
    )

    print(
        f"Availability shape      : "
        f"{summary['availability_shape']}"
    )

    print(
        f"Target shape            : "
        f"{summary['target_shape']}"
    )

    print(
        f"Target availability     : "
        f"{summary['target_availability_shape']}"
    )

    print()
    print(
        f"Observed target cells   : "
        f"{summary['observed_cells']}"
    )

    print(
        f"Positive target cells   : "
        f"{summary['positive_cells']}"
    )

    header(
        "TRAINING CORPUS BUILD COMPLETE"
    )


if __name__ == "__main__":
    main()
