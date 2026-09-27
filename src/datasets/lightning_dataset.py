from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


PREDICTIVE_CHANNELS = [
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
]


class LightningNowcastingDataset(Dataset):

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

        if not self.multimodal_path.exists():
            raise FileNotFoundError(
                f"Multimodal dataset not found: {self.multimodal_path}"
            )

        if not self.target_path.exists():
            raise FileNotFoundError(
                f"Target dataset not found: {self.target_path}"
            )

        self.ds = xr.open_dataset(self.multimodal_path)
        self.target_ds = xr.open_dataset(self.target_path)

        self._validate_inputs()

        self.samples = self._build_samples()

    def _validate_inputs(self):

        missing = [
            name
            for name in PREDICTIVE_CHANNELS
            if name not in self.ds
        ]

        if missing:
            raise ValueError(
                "Missing predictive channels:\n"
                + "\n".join(missing)
            )

        if "lightning_density" not in self.target_ds:
            raise ValueError(
                "Target dataset must contain lightning_density"
            )

        if "lightning_availability" not in self.target_ds:
            raise ValueError(
                "Target dataset must contain lightning_availability"
            )

        if "time" not in self.ds:
            raise ValueError("Multimodal dataset has no time coordinate")

        if "time" not in self.target_ds:
            raise ValueError("Target dataset has no time coordinate")

    def _build_samples(self):

        n_times = len(self.ds.time)

        samples = []

        # A sample consists of:
        #
        # history frames
        # followed by
        # horizons target frames.
        #
        # Example:
        #
        # history=2
        # horizons=4
        #
        # input:
        #   t0, t1
        #
        # target:
        #   t2, t3, t4, t5
        #
        # Only windows that have all required frames are considered.

        for start in range(
            0,
            n_times - self.history - self.horizons + 1,
        ):

            input_indices = list(
                range(
                    start,
                    start + self.history,
                )
            )

            target_indices = list(
                range(
                    start + self.history,
                    start + self.history + self.horizons,
                )
            )

            samples.append(
                {
                    "input_indices": input_indices,
                    "target_indices": target_indices,
                }
            )

        return samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        input_indices = sample["input_indices"]
        target_indices = sample["target_indices"]

        # ---------------------------------------------------------
        # INPUT
        # ---------------------------------------------------------

        input_frames = []

        availability_frames = []

        for time_index in input_indices:

            channels = []

            channel_availability = []

            for name in PREDICTIVE_CHANNELS:

                values = self.ds[name].isel(
                    time=time_index
                ).values

                values = np.asarray(
                    values,
                    dtype=np.float32,
                )

                values = np.nan_to_num(
                    values,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                channels.append(values)

                # Availability is represented explicitly.
                #
                # For actual availability channels:
                # use the supplied availability field.
                #
                # For ordinary data channels:
                # finite values are considered available.

                if name.endswith("_availability"):

                    available = (
                        np.isfinite(values)
                        & (values > 0)
                    ).astype(np.float32)

                else:

                    available = np.isfinite(
                        self.ds[name]
                        .isel(time=time_index)
                        .values
                    ).astype(np.float32)

                channel_availability.append(
                    available
                )

            frame = np.stack(
                channels,
                axis=0,
            )

            frame_availability = np.stack(
                channel_availability,
                axis=0,
            )

            input_frames.append(frame)
            availability_frames.append(
                frame_availability
            )

        x = np.stack(
            input_frames,
            axis=0,
        )

        availability = np.stack(
            availability_frames,
            axis=0,
        )

        # ---------------------------------------------------------
        # TARGET
        # ---------------------------------------------------------

        target_frames = []
        target_availability_frames = []

        for time_index in target_indices:

            target = np.asarray(
                self.target_ds[
                    "lightning_density"
                ].isel(
                    time=time_index
                ).values,
                dtype=np.float32,
            )

            target_available = np.asarray(
                self.target_ds[
                    "lightning_availability"
                ].isel(
                    time=time_index
                ).values,
                dtype=np.float32,
            )

            target = np.nan_to_num(
                target,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            target_available = np.nan_to_num(
                target_available,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            target_frames.append(target)
            target_availability_frames.append(
                target_available
            )

        target = np.stack(
            target_frames,
            axis=0,
        )

        target_availability = np.stack(
            target_availability_frames,
            axis=0,
        )

        # ---------------------------------------------------------
        # TENSORS
        # ---------------------------------------------------------

        x = torch.tensor(
            x,
            dtype=torch.float32,
        )

        availability = torch.tensor(
            availability,
            dtype=torch.float32,
        )

        target = torch.tensor(
            target,
            dtype=torch.float32,
        )

        target_availability = torch.tensor(
            target_availability,
            dtype=torch.float32,
        )

        # ---------------------------------------------------------
        # SAFETY CHECKS
        # ---------------------------------------------------------

        expected_x = (
            self.history,
            len(PREDICTIVE_CHANNELS),
            x.shape[-2],
            x.shape[-1],
        )

        expected_target = (
            self.horizons,
            target.shape[-2],
            target.shape[-1],
        )

        if tuple(x.shape) != expected_x:

            raise RuntimeError(
                f"Unexpected input shape: {tuple(x.shape)} "
                f"expected {expected_x}"
            )

        if tuple(availability.shape) != expected_x:

            raise RuntimeError(
                "Unexpected availability shape: "
                f"{tuple(availability.shape)}"
            )

        if tuple(target.shape) != expected_target:

            raise RuntimeError(
                f"Unexpected target shape: {tuple(target.shape)} "
                f"expected {expected_target}"
            )

        if tuple(target_availability.shape) != expected_target:

            raise RuntimeError(
                "Unexpected target availability shape: "
                f"{tuple(target_availability.shape)}"
            )

        return {
            "x": x,
            "availability": availability,
            "target": target,
            "target_availability": target_availability,
        }


if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[2]

    multimodal_path = (
        ROOT
        / "data"
        / "processed"
        / "multimodal"
        / "odisha_multimodal_20200501.nc"
    )

    target_path = (
        ROOT
        / "data"
        / "processed"
        / "targets"
        / "lightning_targets_odisha_20200501.nc"
    )

    print("=" * 70)
    print("16-CHANNEL LIGHTNING NOWCASTING DATASET TEST")
    print("=" * 70)

    print(f"Multimodal file: {multimodal_path}")
    print(f"Target file:     {target_path}")
    print()

    dataset = LightningNowcastingDataset(
        multimodal_path=multimodal_path,
        target_path=target_path,
        history=2,
        horizons=4,
    )

    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]

    print()
    print("Keys:")
    print(list(sample.keys()))

    print()
    print(f"X shape:                    {sample['x'].shape}")
    print(
        f"Input availability shape: {sample['availability'].shape}"
    )
    print(f"Target shape:               {sample['target'].shape}")
    print(
        "Target availability shape: "
        f"{sample['target_availability'].shape}"
    )

    print()
    print(f"X dtype:      {sample['x'].dtype}")
    print(f"Target dtype: {sample['target'].dtype}")

    print()
    print(
        "Predictive channels:",
        len(PREDICTIVE_CHANNELS),
    )

    print(
        "LIS channels in X: 0"
    )

    print(
        "Target positive pixels:",
        int(
            (
                sample["target"]
                * sample["target_availability"]
                > 0
            ).sum()
        ),
    )

    print()
    print("PREDICTIVE CHANNELS")
    for i, name in enumerate(PREDICTIVE_CHANNELS):
        print(f"{i:02d}: {name}")

    print()
    print("DATASET TEST PASSED")
    print("=" * 70)

