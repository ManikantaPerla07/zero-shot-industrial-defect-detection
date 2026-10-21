import csv
csv.field_size_limit(10**7)

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_ROOT = ROOT / "data" / "raw" / "mvtec_anomaly_detection" / "bottle"
TEST_ROOT = DATA_ROOT / "test"

CHECKPOINT = (
    ROOT
    / "outputs"
    / "patchcore_bottle"
    / "Patchcore"
    / "MVTecAD"
    / "bottle"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

OUTPUT_DIR = ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "patchcore_bottle_results.csv"
JSON_PATH = OUTPUT_DIR / "metrics.json"


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def get_scalar(value):
    """Convert a tensor or number into a Python float."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    return float(np.asarray(value).reshape(-1)[0])


def flatten_predictions(items):
    """Handle predictions returned as nested lists."""
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from flatten_predictions(item)
        else:
            yield item


# --------------------------------------------------
# 3. Check required files
# --------------------------------------------------

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT}"
    )

if not TEST_ROOT.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_ROOT}"
    )


# --------------------------------------------------
# 4. Load the trained PatchCore model
# --------------------------------------------------

print("Loading trained PatchCore model...")

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
    precision="float32",
)

engine = Engine(
    accelerator="cpu",
    devices=1,
    default_root_dir=str(ROOT / "results" / "evaluation"),
)

all_records = []
pixel_scores = []
pixel_labels = []


# --------------------------------------------------
# 5. Evaluate every test category
# --------------------------------------------------

categories = sorted(
    folder for folder in TEST_ROOT.iterdir()
    if folder.is_dir()
)

for category_dir in categories:
    category = category_dir.name

    image_files = sorted(
        path for path in category_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )

    if not image_files:
        continue

    print(
        f"\nEvaluating {category}: "
        f"{len(image_files)} images"
    )

    dataset = PredictDataset(
        path=category_dir,
        image_size=(256, 256),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT),
    )

    if predictions is None:
        raise RuntimeError(
            f"No predictions returned for category: {category}"
        )

    category_predictions = list(flatten_predictions(predictions))

    if len(category_predictions) != len(image_files):
        raise RuntimeError(
            f"Expected {len(image_files)} predictions for "
            f"{category}, but received {len(category_predictions)}. "
            "Please share this message so we can fix the evaluation."
        )

    for index, prediction in enumerate(category_predictions):
        image_path_value = getattr(prediction, "image_path", None)

        if isinstance(image_path_value, (list, tuple)):
            image_path_value = (
                image_path_value[0] if image_path_value else None
            )

        image_path = (
            Path(str(image_path_value))
            if image_path_value is not None
            else image_files[index]
        )

        # MVTec's "good" category contains normal images.
        actual_label = int(category != "good")

        score = get_scalar(prediction.pred_score)
        predicted_label = int(get_scalar(prediction.pred_label))

        all_records.append({
            "image": image_path.name,
            "category": category,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "anomaly_score": score,
        })

        # Collect pixel-level data for anomaly localization.
        anomaly_map = prediction.anomaly_map

        if torch.is_tensor(anomaly_map):
            anomaly_map = anomaly_map.detach().cpu().numpy()

        anomaly_map = np.asarray(anomaly_map)

        while anomaly_map.ndim > 2:
            anomaly_map = anomaly_map[0]

        if anomaly_map.ndim != 2:
            print(
                f"Warning: Skipping pixel metrics for "
                f"{image_path.name}; unexpected map shape "
                f"{anomaly_map.shape}"
            )
            continue

        if category == "good":
            mask = np.zeros(anomaly_map.shape, dtype=np.uint8)
        else:
            mask_path = (
                DATA_ROOT
                / "ground_truth"
                / category
                / f"{image_path.stem}_mask.png"
            )

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if mask is None:
                print(f"Warning: Missing ground-truth mask: {mask_path}")
                continue

            mask = cv2.resize(
                mask,
                (anomaly_map.shape[1], anomaly_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        mask = (mask > 0).astype(np.uint8)

        pixel_scores.append(
            anomaly_map.astype(np.float32).ravel()
        )
        pixel_labels.append(mask.ravel())


# --------------------------------------------------
# 6. Calculate evaluation metrics
# --------------------------------------------------

if not all_records:
    raise RuntimeError("No images were evaluated.")

y_true = np.array(
    [record["actual_label"] for record in all_records]
)

y_pred = np.array(
    [record["predicted_label"] for record in all_records]
)

y_scores = np.array(
    [record["anomaly_score"] for record in all_records]
)

metrics = {
    "images_evaluated": len(all_records),
    "image_accuracy": float(accuracy_score(y_true, y_pred)),
    "image_precision": float(
        precision_score(y_true, y_pred, zero_division=0)
    ),
    "image_recall": float(
        recall_score(y_true, y_pred, zero_division=0)
    ),
    "image_f1_score": float(
        f1_score(y_true, y_pred, zero_division=0)
    ),
    "image_roc_auc": float(roc_auc_score(y_true, y_scores)),
}

if pixel_scores and pixel_labels:
    combined_pixel_scores = np.concatenate(pixel_scores)
    combined_pixel_labels = np.concatenate(pixel_labels)

    if np.unique(combined_pixel_labels).size == 2:
        metrics["pixel_roc_auc"] = float(
            roc_auc_score(
                combined_pixel_labels,
                combined_pixel_scores,
            )
        )


# --------------------------------------------------
# 7. Save the results
# --------------------------------------------------

with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "category",
            "actual_label",
            "predicted_label",
            "anomaly_score",
        ],
    )

    writer.writeheader()
    writer.writerows(all_records)

with JSON_PATH.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\n" + "=" * 45)
print("EVALUATION COMPLETED")
print("=" * 45)

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print(f"\nPer-image results saved to: {CSV_PATH}")
print(f"Metrics saved to: {JSON_PATH}")