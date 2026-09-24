from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_DIR = Path(r"data\processed\baseline_predictions")
OUTPUT_DIR = Path(r"data\processed\baseline_predictions\diagnostics")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(
    BASELINE_DIR.glob("baseline_*.npz")
)

if not files:
    raise FileNotFoundError(
        "No baseline prediction files found."
    )

# Select the strongest cooling-score case.
best_file = None
best_score = -np.inf

for file in files:
    data = np.load(file)
    score = float(np.nanmax(data["cooling_score"]))

    if score > best_score:
        best_score = score
        best_file = file

data = np.load(best_file)

actual = data["actual_tir1"]
predicted = data["predicted_tir1"]
error = np.abs(predicted - actual)
cooling = data["cooling_score"]

timestamp = str(data["timestamp"])

print("=" * 70)
print("INSAT BASELINE DIAGNOSTIC")
print("=" * 70)
print("Selected file:", best_file.name)
print("Timestamp:", timestamp)
print("Maximum cooling score:", best_score)
print("Mean absolute error:", np.mean(error))
print("Maximum absolute error:", np.max(error))

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

images = []

images.append(
    axes[0, 0].imshow(actual)
)

axes[0, 0].set_title(
    "Actual TIR1"
)

images.append(
    axes[0, 1].imshow(predicted)
)

axes[0, 1].set_title(
    "Baseline Predicted TIR1"
)

images.append(
    axes[1, 0].imshow(error)
)

axes[1, 0].set_title(
    "Absolute Prediction Error"
)

images.append(
    axes[1, 1].imshow(cooling)
)

axes[1, 1].set_title(
    "Cooling Score"
)

for ax, image in zip(
    axes.ravel(),
    images
):
    plt.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

fig.suptitle(
    f"INSAT Baseline Diagnostic - {timestamp}",
    fontsize=14
)

fig.tight_layout()

output = (
    OUTPUT_DIR /
    "baseline_diagnostic.png"
)

fig.savefig(
    output,
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

print()
print("Saved:", output)
print("=" * 70)
