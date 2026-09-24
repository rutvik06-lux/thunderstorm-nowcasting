from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.nowcasting_model import ThunderstormNowcaster


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "temporal_samples" / "insat_temporal_samples.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-3


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

def load_dataset():

    print("=" * 70)
    print("LOADING INSAT TEMPORAL DATASET")
    print("=" * 70)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    data = np.load(DATA_FILE)

    inputs = data["inputs"].astype(np.float32)
    targets = data["targets"].astype(np.float32)

    print("Original input shape :", inputs.shape)
    print("Original target shape:", targets.shape)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Saved dataset:
    # inputs  = (samples, time, height, width, channels)
    # targets = (samples, height, width, channels)
    #
    # PyTorch model expects:
    # inputs  = (samples, time, channels, height, width)
    # targets = (samples, channels, height, width)
    # --------------------------------------------------------

    inputs = np.transpose(
        inputs,
        (0, 1, 4, 2, 3)
    )

    targets = np.transpose(
        targets,
        (0, 3, 1, 2)
    )

    print()
    print("Converted input shape :", inputs.shape)
    print("Converted target shape:", targets.shape)

    print("Input dtype :", inputs.dtype)
    print("Target dtype:", targets.dtype)

    return inputs, targets


# ------------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------

def create_dataloaders(inputs, targets):

    print()
    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    total_samples = len(inputs)

    if total_samples < 2:
        raise ValueError(
            "Need at least 2 samples for training and validation."
        )

    # Chronological split:
    # first samples -> training
    # last sample   -> validation

    validation_samples = 1
    training_samples = total_samples - validation_samples

    train_inputs = inputs[:training_samples]
    train_targets = targets[:training_samples]

    val_inputs = inputs[training_samples:]
    val_targets = targets[training_samples:]

    print(f"Training samples  : {len(train_inputs)}")
    print(f"Validation samples: {len(val_inputs)}")

    train_dataset = TensorDataset(
        torch.from_numpy(train_inputs),
        torch.from_numpy(train_targets)
    )

    val_dataset = TensorDataset(
        torch.from_numpy(val_inputs),
        torch.from_numpy(val_targets)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    return train_loader, val_loader


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

def create_model():

    model = ThunderstormNowcaster(
        input_channels=4,
        hidden_channels=64,
        horizons=4
    )

    model = model.to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("MODEL PARAMETERS:", parameter_count)

    return model


# ------------------------------------------------------------
# TRAIN ONE EPOCH
# ------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    for batch_inputs, batch_targets in loader:

        batch_inputs = batch_inputs.to(DEVICE)
        batch_targets = batch_targets.to(DEVICE)

        # ----------------------------------------------------
        # INSAT-only availability mask
        #
        # All 4 INSAT channels are available.
        # ----------------------------------------------------

        availability = torch.ones_like(
            batch_inputs
        )

        optimizer.zero_grad()

        predictions = model(
            batch_inputs,
            availability
        )

        loss = criterion(
            predictions,
            batch_targets
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

def evaluate_model(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    total_mae = 0.0

    with torch.no_grad():

        for batch_inputs, batch_targets in loader:

            batch_inputs = batch_inputs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            availability = torch.ones_like(
                batch_inputs
            )

            predictions = model(
                batch_inputs,
                availability
            )

            loss = criterion(
                predictions,
                batch_targets
            )

            # MAE in normalized space
            mae = torch.mean(
                torch.abs(
                    predictions - batch_targets
                )
            )

            total_loss += loss.item()
            total_mae += mae.item()

    average_loss = total_loss / len(loader)
    average_mae = total_mae / len(loader)

    return average_loss, average_mae


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("INSAT TEMPORAL NOWCAST MODEL")
    print("=" * 70)

    print("Device:", DEVICE)

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    inputs, targets = load_dataset()

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train_loader, val_loader = create_dataloaders(
        inputs,
        targets
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_mae = evaluate_model(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE: {val_mae:.6f}"
        )

    # --------------------------------------------------------
    # FINAL PREDICTION CHECK
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL PREDICTION CHECK")
    print("=" * 70)

    model.eval()

    sample_input = torch.from_numpy(
        inputs[-1:]
    ).to(DEVICE)

    sample_availability = torch.ones_like(
        sample_input
    )

    with torch.no_grad():

        prediction = model(
            sample_input,
            sample_availability
        )

    print("Input shape :", sample_input.shape)
    print("Output shape:", prediction.shape)

    print(
        "Output range:",
        float(prediction.min()),
        "->",
        float(prediction.max())
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = model_dir / "insat_temporal_nowcaster.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_channels": 4,
            "hidden_channels": 64,
            "horizons": 4,
            "grid_height": 27,
            "grid_width": 27,
        },
        model_path
    )

    print()
    print("Model saved:", model_path)

    print()
    print("=" * 70)
    print("INSAT TEMPORAL MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()