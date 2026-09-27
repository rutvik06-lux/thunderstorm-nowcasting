from pathlib import Path
import numpy as np
import xarray as xr

MULTIMODAL = Path(
    "data/processed/multimodal/odisha_multimodal_20200501.nc"
)

TARGETS = Path(
    "data/processed/targets/lightning_targets_odisha_20200501.nc"
)

with xr.open_dataset(MULTIMODAL) as ds, xr.open_dataset(TARGETS) as td:

    print("\n" + "=" * 72)
    print("PHYSICAL VALUE CHECK")
    print("=" * 72)

    variables = [
        "insat_tir1",
        "insat_tir2",
        "insat_wv",
        "insat_vis",
        "imerg_precipitation",
        "era5_u10",
        "era5_v10",
        "era5_d2m",
        "era5_t2m",
        "era5_msl",
        "era5_sp",
        "era5_tcc",
        "era5_cape",
        "lis_lightning_density",
    ]

    for name in variables:

        values = ds[name].values
        valid = values[np.isfinite(values)]

        print(
            f"{name:28s}"
            f" min={valid.min():12.4f}"
            f" max={valid.max():12.4f}"
            f" mean={valid.mean():12.4f}"
        )

    print("\n" + "=" * 72)
    print("LIGHTNING BY TIMESTAMP")
    print("=" * 72)

    target = td["lightning_target"].values
    availability = td["target_availability"].values

    for i, timestamp in enumerate(td.time.values):

        observed = availability[i] == 1
        positive = (target[i] == 1) & observed
        negative = (target[i] == 0) & observed

        print(
            str(timestamp),
            "| observed:",
            int(observed.sum()),
            "| positive:",
            int(positive.sum()),
            "| negative:",
            int(negative.sum())
        )

    print("\n" + "=" * 72)
    print("LIGHTNING PEAK")
    print("=" * 72)

    density = ds["lis_lightning_density"].values

    max_index = np.unravel_index(
        np.nanargmax(density),
        density.shape
    )

    t, y, x = max_index

    print("Maximum lightning density:")
    print("  value     :", float(density[t, y, x]))
    print("  timestamp :", ds.time.values[t])
    print("  grid y    :", y)
    print("  grid x    :", x)
    print("  latitude  :", float(ds.latitude.values[y]))
    print("  longitude :", float(ds.longitude.values[x]))

    print("\n" + "=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)
