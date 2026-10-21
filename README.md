# Zero-Shot Industrial Defect Detection Using Foundation Vision Models

> Academic prototype: PatchCore-based industrial anomaly detection with a separate CLIP zero-shot image-classification comparison.

[![Live Demo](https://img.shields.io/badge/Live-Demo-orange?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/Manikantaperla/zero-shot-industrial-defect-detection) [![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![Project Report](https://img.shields.io/badge/Project-Report-green?style=flat)](docs/PROJECT_REPORT.md) [![Model Comparison](https://img.shields.io/badge/Model-Comparison-purple?style=flat)](docs/model_comparison.md)

[Live Demo](https://huggingface.co/spaces/Manikantaperla/zero-shot-industrial-defect-detection) · [Project Report](docs/PROJECT_REPORT.md) · [Model Comparison](docs/model_comparison.md)

## Overview

This project explores visual quality inspection for industrial bottles. The primary application uses PatchCore to flag abnormal images and visualize suspicious regions with an anomaly heatmap. A separate experiment evaluates CLIP using normal/defect text prompts for zero-shot image-level classification.

**Important scope distinction:** PatchCore is the deployed anomaly-detection model and uses normal bottle training images. CLIP is a zero-shot comparison experiment; it is not the heatmap/localization model in the current app.

## Live Application

**Try it:** https://huggingface.co/spaces/Manikantaperla/zero-shot-industrial-defect-detection

The public Gradio app is hosted on Hugging Face Spaces with ZeroGPU. The application has been manually tested with one defective and one normal bottle example:

- Defective example: `Potential defect detected`, anomaly score `0.9515`.
- Normal example: `Likely normal`, anomaly score `0.2298`.

Scores are uncalibrated model scores, not probabilities. The heatmap is min-max normalized separately for each image, so its warm colours represent relatively higher values within that image and should not be compared directly across images.

## Key Features

- Upload an image through a Gradio interface.
- Get PatchCore's image-level anomaly score and prediction.
- Visualize the anomaly map as a heatmap overlay.
- Evaluate image-level metrics and pixel-level ROC-AUC for PatchCore.
- Compare PatchCore with CLIP zero-shot image-level classification.
- Save per-image predictions, metric JSON files, and a Markdown comparison report.

## Methodology

### PatchCore

PatchCore builds a memory bank of representative features from normal training images. During inference, feature distances identify patches that differ from the normal reference. This implementation uses:

- Backbone: `wide_resnet50_2`
- Feature layers: `layer2` and `layer3`
- Coreset sampling ratio: `0.1`
- Nearest neighbours: `9`
- Precision: FP32

### CLIP zero-shot comparison

The experiment uses `openai/clip-vit-base-patch32` and text-prompt groups describing normal and defective bottles. It compares mean image-text similarity scores to generate an image-level label and relative defect score. No bottle-specific CLIP fine-tuning was performed.

## Dataset

The experiment uses only the bottle category from the [MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). The category contains defect-free training images, normal and defective test images, and ground-truth masks for defects.

| Split / category | Images | Label |
|---|---:|---|
| `train/good` | 209 | Normal training images |
| `test/good` | 20 | Normal test images |
| `test/broken_large` | 20 | Defective |
| `test/broken_small` | 22 | Defective |
| `test/contamination` | 21 | Defective |
| **Total test set** | **83** | **20 normal, 63 defective** |

The dataset is not included in this repository. Follow MVTec AD's licence terms; the dataset is provided under CC BY-NC-SA 4.0 and is not for commercial use.

## Initial Evaluation Results

Both scripts evaluated the same 83 bottle test images. These are project-script results for this subset, not official benchmark results.

| Metric | PatchCore | CLIP zero-shot |
|---|---:|---:|
| Test images | 83 | 83 |
| Accuracy | **98.80%** | 75.90% |
| Precision | **100.00%** | 75.90% |
| Recall | 98.41% | **100.00%** |
| F1-score | **99.20%** | 86.30% |
| Image ROC-AUC | **100.00%** | 93.57% |
| Pixel ROC-AUC | **98.56%** | Not evaluated |

### Interpretation and limitations

- At the current Anomalib prediction threshold, PatchCore classified 82 of 83 test images correctly: 62 of 63 defects and all 20 normal bottles.
- PatchCore's reported image ROC-AUC of 1.0000 reflects score ranking on this small subset; it does not mean every threshold produces perfect classification.
- CLIP classified all 83 images as defective. This explains its 100% recall and 75.90% accuracy, with 20 false positives. Prompt and decision-rule refinement is required.
- Pixel ROC-AUC is an aggregate over resized anomaly maps and masks; it is not a per-category score.
- Only the bottle category was evaluated. Performance on other products and real manufacturing conditions has not been validated.

## Technology Stack

- Python, PyTorch, torchvision
- Anomalib / PatchCore
- Hugging Face Transformers / CLIP
- OpenCV, NumPy, Pillow
- scikit-learn
- Gradio
- Hugging Face Spaces ZeroGPU for public inference

## Project Structure

```text
zero-shot-industrial-defect-detection/
├── app.py
├── train_patchcore.py
├── evaluate_model.py
├── compare_clip.py
├── compare_results.py
├── configs/config.yaml
├── docs/
│   ├── PROJECT_REPORT.md
│   └── model_comparison.md
├── outputs/evaluation/
│   ├── patchcore_bottle_results.csv
│   ├── clip_bottle_results.csv
│   ├── metrics.json
│   ├── clip_metrics.json
│   └── model_comparison.csv
├── requirements.txt
└── src/utils/
```

The trained checkpoint is stored with the Hugging Face Space via Git LFS, not in this GitHub source repository. The dataset and the local virtual environment are also excluded from Git.

## Run Locally

Tested locally with Python 3.11.

```powershell
git clone https://github.com/ManikantaPerla07/zero-shot-industrial-defect-detection.git
cd zero-shot-industrial-defect-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio, usually `http://127.0.0.1:7860`.

To reproduce evaluation, make the bottle dataset and trained checkpoint available at the expected paths, then run:

```powershell
python evaluate_model.py
python compare_clip.py
python compare_results.py
```

Outputs are written to `outputs/evaluation/`. Because dependency versions are not fully pinned, a fresh environment may resolve different compatible package versions.

## Limitations and Future Work

- Validate additional product categories and acquisition conditions.
- Tune CLIP prompt templates and evaluate a decision threshold on a separate validation split.
- Report per-defect-category results and threshold-independent vs threshold-dependent metrics separately.
- Investigate zero-shot anomaly-localization methods for a more direct localization comparison.
- Pin package versions and add automated regression tests.

## References

1. Bergmann et al., “MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_Detection_CVPR_2019_paper.html
2. Roth et al., “Towards Total Recall in Industrial Anomaly Detection,” CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html
3. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML 2021. https://proceedings.mlr.press/v139/radford21a.html
4. Anomalib PatchCore documentation. https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/models/image/patchcore.html

## Project Status

The current-scope prototype has been implemented, evaluated on the bottle subset, documented, published on GitHub, and deployed as a public Hugging Face Space. Results should be presented with the category and evaluation limitations stated above.