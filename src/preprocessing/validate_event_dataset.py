from pathlib import Path
import numpy as np
import xarray as xr

MULTIMODAL = Path(
    "data/processed/multimodal/odisha_multimodal_20200501.nc"
)
TARGETS = Path(
    "data/processed/targets/lightning_targets_odisha_20200501.nc"
)

for path in (MULTIMODAL, TARGETS):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

with xr.open_dataset(MULTIMODAL) as ds, xr.open_dataset(TARGETS) as target_ds:
    print("\n" + "=" * 72)
    print("MULTIMODAL DATASET")
    print("=" * 72)
    print(ds)

    print("\nMultimodal timestamps:")
    print(ds.time.values)

    print("\nVariables and valid-value counts:")
    for name, da in ds.data_vars.items():
        values = da.values
        finite = np.isfinite(values)
        print(
            f"{name:30s} "
            f"shape={str(values.shape):18s} "
            f"valid={finite.sum():8d} "
            f"missing={(~finite).sum():8d}"
        )

    print("\n" + "=" * 72)
    print("LIGHTNING TARGETS")
    print("=" * 72)
    print(target_ds)

    print("\nTarget timestamps:")
    print(target_ds.time.values)

    target_name = "lightning_target"
    if target_name not in target_ds:
        raise KeyError(
            f"Expected variable 'target'. Found: {list(target_ds.data_vars)}"
        )

    target = target_ds[target_name].values
    valid = np.isfinite(target)

    print("\nTarget summary:")
    print("  Shape:", target.shape)
    print("  Observed cells:", int(valid.sum()))
    print("  Positive cells:", int(np.nansum(target == 1)))
    print("  Negative cells:", int(np.nansum(target == 0)))
    print("  Unknown cells:", int((~valid).sum()))

    if "target_availability" in target_ds:
        availability = target_ds["target_availability"].values
        print(
            "  Availability values:",
            np.unique(availability[ np.isfinite(availability) ])
        )

    print("\n" + "=" * 72)
    print("TIME ALIGNMENT")
    print("=" * 72)

    input_times = np.asarray(ds.time.values).astype("datetime64[ns]")
    target_times = np.asarray(target_ds.time.values).astype("datetime64[ns]")

    common = np.intersect1d(input_times, target_times)

    print("Multimodal frames:", len(input_times))
    print("Target frames:", len(target_times))
    print("Matching timestamps:", len(common))
    print("Matching times:", common)

    if len(common) == 0:
        print("\nWARNING: No matching timestamps. Do not train or evaluate yet.")
    elif len(common) != len(input_times):
        print("\nWARNING: Not every input frame has a matching target.")
    else:
        print("\nAll multimodal timestamps have matching target timestamps.")
