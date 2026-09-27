from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class LightningNowcastingDataset(Dataset):
    """
    PyTorch Dataset for multimodal lightning nowcasting.

    Input:
        data/processed/multimodal/odisha_multimodal_20200501.nc

    Target:
        data/processed/targets/lightning_targets_odisha_20200501.nc

    Input shape per sample:
        [T, C, H, W]

    Target shape:
        [4, H, W]

    Target horizons:
        +30, +60, +90, +120 minutes
    """

    CHANNELS = [
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

    AVAILABILITY_CHANNELS = [
        "insat_availability",
        "insat_availability",
        "insat_availability",
        "insat_availability",
        "insat_availability",

        "imerg_availability",
        "imerg_availability",

        "era5_availability",
        "era5_availability",
        "era5_availability",
        "era5_availability",
        "era5_availability",
        "era5_availability",
        "era5_availability",
        "era5_availability",
        "era5_availability",

        "lis_lightning_availability",
        "lis_lightning_availability",
    ]

    def __init__(
        self,
        multimodal_path,
        target_path,
        history=2,
        horizons=4,
    ):
        self.multimodal_path = Path(multimodal_path)
        self.target_path = Path(target_path)

        self.history = history
        self.horizons = horizons

        if self.horizons != 4:
            raise ValueError(
                "This prototype Dataset currently expects 4 horizons."
            )

        if not self.multimodal_path.exists():
            raise FileNotFoundError(
                f"Multimodal dataset not found: {self.multimodal_path}"
            )

        if not self.target_path.exists():
            raise FileNotFoundError(
                f"Target dataset not found: {self.target_path}"
            )

        self.ds = xr.open_dataset(self.multimodal_path)
        self.targets = xr.open_dataset(self.target_path)

        self._validate()

        self.samples = self._build_sample_indices()

        if len(self.samples) == 0:
            raise ValueError(
                "No valid samples were found. "
                f"history={self.history}, horizons={self.horizons}, "
                f"time_steps={self.ds.sizes['time']}"
            )

    def _validate(self):
        """Validate variables, dimensions, and timestamps."""

        missing_channels = [
            name for name in self.CHANNELS
            if name not in self.ds
        ]

        if missing_channels:
            raise ValueError(
                "Missing variables from multimodal dataset: "
                + ", ".join(missing_channels)
            )

        if "lightning_target" not in self.targets:
            raise ValueError(
                "Target variable 'lightning_target' not found."
            )

        if "target_availability" not in self.targets:
            raise ValueError(
                "Target variable 'target_availability' not found."
            )

        if not np.array_equal(
            self.ds.time.values,
            self.targets.time.values,
        ):
            raise ValueError(
                "Multimodal and target timestamps do not match."
            )

        if (
            self.ds.sizes["y"] != self.targets.sizes["y"]
            or self.ds.sizes["x"] != self.targets.sizes["x"]
        ):
            raise ValueError(
                "Multimodal and target grid dimensions do not match."
            )

    def _build_sample_indices(self):
        """
        Build forecast origins.

        A sample requires:

            history frames before/including t
            +
            4 future target frames

        Example with history=2:

            input: 04:00, 04:30
            target: 04:30, 05:00, 05:30, 06:00

        The target at each horizon is only usable where
        target_availability == 1.
        """

        n_times = self.ds.sizes["time"]

        samples = []

        for end_time in range(
            self.history - 1,
            n_times - self.horizons,
        ):
            target_start = end_time + 1
            target_end = target_start + self.horizons

            samples.append(
                (
                    end_time,
                    target_start,
                    target_end,
                )
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        end_time, target_start, target_end = self.samples[index]

        input_start = end_time - self.history + 1
        input_end = end_time + 1

        input_frames = []

        availability_frames = []

        for t in range(input_start, input_end):

            data_channels = []

            availability_channels = []

            for variable, availability_variable in zip(
                self.CHANNELS,
                self.AVAILABILITY_CHANNELS,
            ):

                values = self.ds[variable].isel(
                    time=t
                ).values.astype(np.float32)

                availability = self.ds[
                    availability_variable
                ].isel(time=t).values.astype(np.float32)

                values = np.nan_to_num(
                    values,
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

                data_channels.append(values)
                availability_channels.append(availability)

            input_frames.append(
                np.stack(data_channels, axis=0)
            )

            availability_frames.append(
                np.stack(availability_channels, axis=0)
            )

        x = np.stack(
            input_frames,
            axis=0,
        )

        availability = np.stack(
            availability_frames,
            axis=0,
        )

        target = self.targets[
            "lightning_target"
        ].isel(
            time=slice(target_start, target_end)
        ).values.astype(np.float32)

        target_availability = self.targets[
            "target_availability"
        ].isel(
            time=slice(target_start, target_end)
        ).values.astype(np.float32)

        target = np.nan_to_num(
            target,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        target_availability = np.nan_to_num(
            target_availability,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        x = torch.from_numpy(x)
        availability = torch.from_numpy(availability)
        target = torch.from_numpy(target)
        target_availability = torch.from_numpy(
            target_availability
        )

        return {
            "x": x,
            "availability": availability,
            "target": target,
            "target_availability": target_availability,
        }

    def close(self):
        self.ds.close()
        self.targets.close()


def test_dataset():

    print("=" * 72)
    print("LIGHTNING NOWCASTING DATASET TEST")
    print("=" * 72)

    dataset = LightningNowcastingDataset(
        multimodal_path=(
            "data/processed/multimodal/"
            "odisha_multimodal_20200501.nc"
        ),
        target_path=(
            "data/processed/targets/"
            "lightning_targets_odisha_20200501.nc"
        ),
        history=2,
        horizons=4,
    )

    print("Dataset samples :", len(dataset))
    print("Input channels  :", len(dataset.CHANNELS))
    print("History frames  :", dataset.history)
    print("Forecast hours  :", dataset.horizons)

    sample = dataset[0]

    print()
    print("Sample contents:")
    print("  x shape                 :", sample["x"].shape)
    print(
        "  availability shape      :",
        sample["availability"].shape,
    )
    print(
        "  target shape            :",
        sample["target"].shape,
    )
    print(
        "  target availability    :",
        sample["target_availability"].shape,
    )

    print()
    print("Expected:")
    print("  x                 : [2, 18, 27, 27]")
    print("  availability      : [2, 18, 27, 27]")
    print("  target            : [4, 27, 27]")
    print("  target availability: [4, 27, 27]")

    assert sample["x"].shape == (2, 18, 27, 27)
    assert sample["availability"].shape == (2, 18, 27, 27)
    assert sample["target"].shape == (4, 27, 27)
    assert sample["target_availability"].shape == (
        4,
        27,
        27,
    )

    assert torch.isfinite(sample["x"]).all()
    assert torch.isfinite(sample["availability"]).all()
    assert torch.isfinite(sample["target"]).all()
    assert torch.isfinite(
        sample["target_availability"]
    ).all()

    print()
    print("Input finite        : PASS")
    print("Availability finite : PASS")
    print("Target finite       : PASS")

    print()
    print("Target availability:")
    for h in range(4):
        observed = (
            sample["target_availability"][h] == 1
        ).sum().item()

        positive = (
            (sample["target"][h] == 1)
            & (sample["target_availability"][h] == 1)
        ).sum().item()

        print(
            f"  +{(h + 1) * 30:3d} min"
            f" | observed={observed:4d}"
            f" | positive={positive:4d}"
        )

    dataset.close()

    print()
    print("=" * 72)
    print("DATASET TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    test_dataset()
