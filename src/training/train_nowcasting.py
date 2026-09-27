from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.lightning_dataset import LightningNowcastingDataset
from src.models.nowcasting_model import ThunderstormNowcaster


MULTIMODAL_PATH = Path(
    "data/processed/multimodal/"
    "odisha_multimodal_20200501.nc"
)

TARGET_PATH = Path(
    "data/processed/targets/"
    "lightning_targets_odisha_20200501.nc"
)


def masked_bce_loss(
    logits,
    target,
    target_availability,
):
    """
    Binary cross-entropy loss that ignores target pixels
    where lightning observations are unavailable.

    logits:
        [B, 4, H, W]

    target:
        [B, 4, H, W]

    target_availability:
        [B, 4, H, W]
    """

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )

    mask = target_availability

    valid_loss = loss * mask

    valid_pixels = mask.sum()

    if valid_pixels == 0:
        raise ValueError(
            "No valid target pixels are available "
            "for this batch."
        )

    return valid_loss.sum() / valid_pixels


def main():

    print("=" * 72)
    print("REAL MULTIMODAL NOWCASTING TRAINING SMOKE TEST")
    print("=" * 72)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    dataset = LightningNowcastingDataset(
        multimodal_path=MULTIMODAL_PATH,
        target_path=TARGET_PATH,
        history=2,
        horizons=4,
    )

    print("Dataset samples:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    model = ThunderstormNowcaster(
        input_channels=18,
        hidden_channels=64,
        horizons=4,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    model.train()

    print()
    print("Model parameters:", sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ))

    print()
    print("-" * 72)
    print("TRAINING")
    print("-" * 72)

    for epoch in range(1, 6):

        epoch_loss = 0.0

        for batch in loader:

            x = batch["x"].to(device)
            availability = batch[
                "availability"
            ].to(device)

            target = batch["target"].to(device)
            target_availability = batch[
                "target_availability"
            ].to(device)

            optimizer.zero_grad()

            logits = model(
                x,
                availability,
            )

            loss = masked_bce_loss(
                logits,
                target,
                target_availability,
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"Epoch {epoch:02d}"
            f" | loss={epoch_loss:.6f}"
        )

    print()
    print("-" * 72)
    print("INFERENCE CHECK")
    print("-" * 72)

    model.eval()

    with torch.no_grad():

        batch = next(iter(loader))

        x = batch["x"].to(device)
        availability = batch[
            "availability"
        ].to(device)

        logits = model(
            x,
            availability,
        )

        probabilities = torch.sigmoid(
            logits
        )

    print("Input shape        :", x.shape)
    print("Logits shape       :", logits.shape)
    print(
        "Probability shape  :",
        probabilities.shape,
    )

    print(
        "Probability range  :",
        f"{probabilities.min().item():.6f}"
        f" -> "
        f"{probabilities.max().item():.6f}",
    )

    print()
    print("Forecast horizons:")

    for horizon in range(4):

        probability = probabilities[
            0,
            horizon,
        ]

        print(
            f"  +{(horizon + 1) * 30:3d} min"
            f" | min={probability.min().item():.4f}"
            f" | max={probability.max().item():.4f}"
            f" | mean={probability.mean().item():.4f}"
        )

    assert torch.isfinite(logits).all()
    assert torch.isfinite(probabilities).all()

    assert probabilities.shape == (
        1,
        4,
        27,
        27,
    )

    print()
    print("=" * 72)
    print("END-TO-END SMOKE TEST PASSED")
    print("=" * 72)

    dataset.close()


if __name__ == "__main__":
    main()
