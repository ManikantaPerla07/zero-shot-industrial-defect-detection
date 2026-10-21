import csv
csv.field_size_limit(10**7)

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

TEST_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "mvtec_anomaly_detection"
    / "bottle"
    / "test"
)

OUTPUT_DIR = ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "clip_bottle_results.csv"
JSON_PATH = OUTPUT_DIR / "clip_metrics.json"

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# 2. Zero-shot text descriptions
# --------------------------------------------------

NORMAL_PROMPTS = [
    "a photo of a normal bottle",
    "a photo of an intact bottle with no defects",
    "a photo of an undamaged industrial bottle",
    "a photo of a bottle without cracks or damage",
    "a close-up photo of a perfect bottle",
]

DEFECT_PROMPTS = [
    "a photo of a defective bottle",
    "a photo of a damaged industrial bottle",
    "a photo of a bottle with a crack or broken area",
    "a close-up photo of a bottle with a visible defect",
    "a photo of a bottle with contamination or damage",
]

PROMPTS = NORMAL_PROMPTS + DEFECT_PROMPTS


# --------------------------------------------------
# 3. Load the pretrained CLIP model
# --------------------------------------------------

if not TEST_ROOT.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_ROOT}"
    )

print(f"Loading CLIP model: {MODEL_NAME}")
print(f"Using device: {DEVICE}")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model = CLIPModel.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.eval()

print("CLIP model loaded successfully.")


# --------------------------------------------------
# 4. Collect test images
# --------------------------------------------------

image_items = []

for category_dir in sorted(TEST_ROOT.iterdir()):
    if not category_dir.is_dir():
        continue

    category = category_dir.name

    for image_path in sorted(category_dir.iterdir()):
        if image_path.suffix.lower() not in {
            ".png", ".jpg", ".jpeg", ".bmp"
        }:
            continue

        # MVTec AD labels: "good" is normal;
        # other bottle test categories contain defects.
        actual_label = int(category != "good")

        image_items.append({
            "path": image_path,
            "category": category,
            "actual_label": actual_label,
        })

if not image_items:
    raise RuntimeError("No test images were found.")

print(f"Found {len(image_items)} test images.")


# --------------------------------------------------
# 5. Perform zero-shot classification
# --------------------------------------------------

records = []

with torch.inference_mode():
    for start in range(0, len(image_items), BATCH_SIZE):
        batch_items = image_items[start:start + BATCH_SIZE]

        images = []

        for item in batch_items:
            with Image.open(item["path"]) as image:
                images.append(image.convert("RGB"))

        inputs = processor(
            text=PROMPTS,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        outputs = model(**inputs)

        # Each image gets similarity scores against
        # all normal and defective text descriptions.
        similarities = outputs.logits_per_image

        normal_scores = similarities[
            :, :len(NORMAL_PROMPTS)
        ].mean(dim=1)

        defect_scores = similarities[
            :, len(NORMAL_PROMPTS):
        ].mean(dim=1)

        class_scores = torch.stack(
            [normal_scores, defect_scores],
            dim=1,
        )

        # Relative scores across the two classes.
        relative_scores = torch.softmax(
            class_scores,
            dim=1,
        )

        predictions = relative_scores.argmax(dim=1)

        for index, item in enumerate(batch_items):
            records.append({
                "image": item["path"].name,
                "category": item["category"],
                "actual_label": item["actual_label"],
                "predicted_label": int(
                    predictions[index].item()
                ),
                "normal_similarity": float(
                    normal_scores[index].item()
                ),
                "defect_similarity": float(
                    defect_scores[index].item()
                ),
                "relative_defect_score": float(
                    relative_scores[index, 1].item()
                ),
            })

        completed = min(start + BATCH_SIZE, len(image_items))
        print(f"Processed {completed}/{len(image_items)} images")


# --------------------------------------------------
# 6. Calculate classification metrics
# --------------------------------------------------

y_true = np.array([
    row["actual_label"] for row in records
])

y_pred = np.array([
    row["predicted_label"] for row in records
])

y_scores = np.array([
    row["relative_defect_score"] for row in records
])

metrics = {
    "model": MODEL_NAME,
    "evaluation_type": "zero-shot image-level classification",
    "images_evaluated": len(records),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(
        precision_score(y_true, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_true, y_pred, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_true, y_pred, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_true, y_scores)),
}


# --------------------------------------------------
# 7. Save predictions and metrics
# --------------------------------------------------

fieldnames = [
    "image",
    "category",
    "actual_label",
    "predicted_label",
    "normal_similarity",
    "defect_similarity",
    "relative_defect_score",
]

with CSV_PATH.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

with JSON_PATH.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\n" + "=" * 45)
print("CLIP ZERO-SHOT EVALUATION COMPLETED")
print("=" * 45)

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print(f"\nImage results saved to: {CSV_PATH}")
print(f"Metrics saved to: {JSON_PATH}")

print(
    "\nNote: Relative defect scores are not calibrated "
    "probabilities. CLIP is evaluated here for image-level "
    "classification, not pixel-level defect segmentation."
)