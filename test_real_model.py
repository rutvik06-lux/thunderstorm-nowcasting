import torch

from src.datasets.lightning_dataset import LightningNowcastingDataset
from src.models.nowcasting_model import ThunderstormNowcaster


MULTIMODAL_PATH = (
    r"data\processed\multimodal"
    r"\odisha_multimodal_20200501.nc"
)

TARGET_PATH = (
    r"data\processed\targets"
    r"\lightning_targets_odisha_20200501.nc"
)


print("=" * 72)
print("REAL DATA → MULTIMODAL MODEL SMOKE TEST")
print("=" * 72)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)
print()


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

dataset = LightningNowcastingDataset(
    MULTIMODAL_PATH,
    TARGET_PATH,
    history=2,
    horizons=4,
)

print("Dataset length:", len(dataset))


sample = dataset[0]

print("Dataset keys:", list(sample.keys()))
print()


# ------------------------------------------------------------
# 2. EXTRACT TENSORS
# ------------------------------------------------------------

x = sample["x"]
availability = sample["availability"]

target = sample["target"]
target_availability = sample["target_availability"]


print("X shape:")
print(" ", x.shape)

print("Input availability shape:")
print(" ", availability.shape)

print("Target shape:")
print(" ", target.shape)

print("Target availability shape:")
print(" ", target_availability.shape)

print()


# ------------------------------------------------------------
# 3. ADD BATCH DIMENSION
# ------------------------------------------------------------

x = x.unsqueeze(0).to(device)

availability = availability.unsqueeze(0).to(device)

target = target.unsqueeze(0).to(device)

target_availability = (
    target_availability
    .unsqueeze(0)
    .to(device)
)


print("Model X:")
print(" ", x.shape)

print("Model availability:")
print(" ", availability.shape)

print()


# ------------------------------------------------------------
# 4. CREATE MODEL
# ------------------------------------------------------------

model = ThunderstormNowcaster(
    input_channels=18,
    hidden_channels=64,
    horizons=4,
).to(device)

model.eval()

print(
    "Model parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
    ),
)

print()


# ------------------------------------------------------------
# 5. FORWARD PASS
# ------------------------------------------------------------

with torch.no_grad():

    logits = model(
        x,
        availability,
    )

    probabilities = torch.sigmoid(
        logits
    )


# ------------------------------------------------------------
# 6. VALIDATE OUTPUT
# ------------------------------------------------------------

print("Logits shape:")
print(" ", logits.shape)

print("Probability shape:")
print(" ", probabilities.shape)

print()

print(
    "Probability range:",
    f"{probabilities.min().item():.6f}",
    "->",
    f"{probabilities.max().item():.6f}",
)

print()

print(
    "Target positive pixels:",
    int(target.sum().item())
)

print(
    "Observed target pixels:",
    int(target_availability.sum().item())
)

print()


# ------------------------------------------------------------
# 7. ASSERTIONS
# ------------------------------------------------------------

assert x.shape == (
    1,
    2,
    18,
    27,
    27,
)

assert availability.shape == (
    1,
    2,
    18,
    27,
    27,
)

assert target.shape == (
    1,
    4,
    27,
    27,
)

assert target_availability.shape == (
    1,
    4,
    27,
    27,
)

assert logits.shape == (
    1,
    4,
    27,
    27,
)

assert probabilities.shape == (
    1,
    4,
    27,
    27,
)

assert torch.isfinite(logits).all()

assert torch.isfinite(probabilities).all()

assert probabilities.min().item() >= 0.0

assert probabilities.max().item() <= 1.0


print("Input shape       : PASS")
print("Availability shape: PASS")
print("Target shape      : PASS")
print("Output shape      : PASS")
print("Logits finite     : PASS")
print("Probabilities     : PASS")
print()
print("=" * 72)
print("REAL DATA MODEL SMOKE TEST PASSED")
print("=" * 72)