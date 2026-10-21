import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs" / "evaluation"
DOCS = ROOT / "docs"

patchcore_path = RESULTS / "metrics.json"
clip_path = RESULTS / "clip_metrics.json"

if not patchcore_path.exists() or not clip_path.exists():
    raise FileNotFoundError(
        "Run evaluate_model.py and compare_clip.py first."
    )

with patchcore_path.open(encoding="utf-8") as file:
    patchcore = json.load(file)

with clip_path.open(encoding="utf-8") as file:
    clip = json.load(file)

comparison = [
    {
        "Model": "PatchCore",
        "Images": patchcore["images_evaluated"],
        "Accuracy": patchcore["image_accuracy"],
        "Precision": patchcore["image_precision"],
        "Recall": patchcore["image_recall"],
        "F1 Score": patchcore["image_f1_score"],
        "Image ROC-AUC": patchcore["image_roc_auc"],
        "Pixel ROC-AUC": patchcore.get("pixel_roc_auc", ""),
    },
    {
        "Model": "CLIP Zero-Shot",
        "Images": clip["images_evaluated"],
        "Accuracy": clip["accuracy"],
        "Precision": clip["precision"],
        "Recall": clip["recall"],
        "F1 Score": clip["f1_score"],
        "Image ROC-AUC": clip["roc_auc"],
        "Pixel ROC-AUC": "Not evaluated",
    },
]

RESULTS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

csv_path = RESULTS / "model_comparison.csv"
report_path = DOCS / "model_comparison.md"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(comparison[0].keys()),
    )
    writer.writeheader()
    writer.writerows(comparison)

headers = list(comparison[0].keys())

lines = [
    "# Industrial Defect Detection: Model Comparison",
    "",
    "Both models were evaluated on the same 83 MVTec AD bottle test images.",
    "",
    "| " + " | ".join(headers) + " |",
    "|" + "|".join(["---"] * len(headers)) + "|",
]

for row in comparison:
    values = []
    for header in headers:
        value = row[header]
        if isinstance(value, float):
            values.append(f"{value:.4f}")
        else:
            values.append(str(value))
    lines.append("| " + " | ".join(values) + " |")

lines.extend([
    "",
    "## Interpretation and limitations",
    "",
    "- PatchCore provides image-level anomaly detection and pixel-level localization.",
    "- CLIP was evaluated for zero-shot image-level classification using text prompts.",
    "- The current CLIP decision rule classified every test image as defective. "
    "Its decision threshold requires further investigation.",
    "- CLIP's pixel-level localization was not evaluated.",
    "- Results apply to this bottle test set and do not establish performance "
    "on other product categories.",
    "- The CLIP score is a relative text-similarity-derived score, not a "
    "calibrated probability.",
])

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Model comparison saved successfully.")
print(f"CSV report: {csv_path}")
print(f"Markdown report: {report_path}")
print("\nComparison:")

for row in comparison:
    print(f"\n{row['Model']}")
    for key, value in row.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")