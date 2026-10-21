# Zero-Shot Industrial Defect Detection Using Foundation Vision Models

An industrial visual inspection project that detects anomalies in bottle images using **PatchCore** and compares its performance with **CLIP zero-shot classification**. The application generates anomaly heatmaps to help visualize potentially defective regions.

## Overview

Manual quality inspection can be time-consuming and inconsistent. This project explores pretrained vision models for identifying abnormal patterns in industrial products.

The current implementation focuses on the bottle category of the MVTec Anomaly Detection (MVTec AD) dataset. It uses PatchCore for anomaly detection and localization, alongside CLIP for zero-shot image-level classification.

## Key Features

- Upload an industrial product image through a Gradio web interface.
- Detect potential anomalies using a pretrained-feature-based PatchCore model.
- Generate anomaly heatmaps to visualize suspicious regions.
- Display an image-level anomaly score and predicted result.
- Compare PatchCore against CLIP zero-shot image classification.
- Evaluate model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- Save per-image predictions and evaluation reports.

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Anomaly Detection:** Anomalib, PatchCore
- **Foundation Vision Model:** CLIP (`openai/clip-vit-base-patch32`)
- **Image Processing:** OpenCV, Pillow, NumPy
- **Evaluation:** scikit-learn
- **Web Interface:** Gradio
- **Configuration:** YAML

## System Architecture

```text
MVTec AD Bottle Dataset
          |
          v
  Image Preprocessing
          |
          +-----------------------------+
          |                             |
          v                             v
  PatchCore Model                 CLIP Model
  Anomaly Detection               Zero-Shot Classification
          |                             |
          v                             v
  Anomaly Score                   Image-Level Score
  Anomaly Heatmap                 Normal/Defective Label
          |                             |
          +--------------+--------------+
                         |
                         v
             Evaluation and Comparison
                         |
                         v
              Results and Web Interface
```

## Dataset

This project uses the **bottle category** of the MVTec Anomaly Detection (MVTec AD) dataset.

The dataset includes normal training images, normal test images, defective test images, and ground-truth masks for defective regions.

The local dataset directory is:

```text
data/raw/mvtec_anomaly_detection/bottle/
├── train/
│   └── good/
├── test/
│   ├── good/
│   ├── broken_large/
│   ├── broken_small/
│   └── contamination/
└── ground_truth/
```

Only normal bottle images are used to build the PatchCore memory bank. Test images are used for evaluation.

Dataset source: [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad)

Please follow the dataset's applicable licence and terms when downloading or redistributing it.

## Models

### 1. PatchCore

PatchCore is the primary anomaly-detection model in the current implementation. It uses pretrained Wide ResNet-50-2 features from `layer2` and `layer3` to identify deviations from normal training examples.

Current configuration:

- Backbone: `wide_resnet50_2`
- Feature layers: `layer2`, `layer3`
- Coreset sampling ratio: `0.1`
- Number of nearest neighbors: `9`
- Precision: FP32

The model generates an image-level anomaly score and an anomaly map for localization.

### 2. CLIP Zero-Shot Classification

CLIP compares image features with text descriptions of normal and defective bottles without training on the bottle test images.

The current experiment uses the pretrained `openai/clip-vit-base-patch32` model and multiple text prompts for both classes.

**Important distinction:** CLIP is currently evaluated for image-level classification only. Pixel-level defect localization is implemented and evaluated for PatchCore, not for this CLIP experiment.

## Initial Evaluation Results

Both models were evaluated on the same 83 bottle test images.

| Metric | PatchCore | CLIP Zero-Shot |
|---|---:|---:|
| Test images | 83 | 83 |
| Accuracy | 98.80% | 75.90% |
| Precision | 100.00% | 75.90% |
| Recall | 98.41% | 100.00% |
| F1-score | 99.20% | 86.30% |
| Image ROC-AUC | 100.00% | 93.57% |
| Pixel ROC-AUC | 98.56% | Not evaluated |

These are initial results from the project's local evaluation scripts, not official benchmark results or evidence of generalization to other industrial product categories.

### Interpretation

- PatchCore performed strongly on this particular bottle test set.
- CLIP obtained an image ROC-AUC of 93.57%, indicating useful ranking performance in this evaluation.
- The current CLIP classification rule predicted every test image as defective. Its decision threshold and prompt design require further investigation.
- Pixel-level ROC-AUC was calculated for PatchCore using anomaly maps and available ground-truth masks.
- The anomaly score is not a calibrated probability of a defect.

## Project Structure

```text
zero-shot-industrial-defect-detection/
├── app.py
├── evaluate_model.py
├── compare_clip.py
├── compare_results.py
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── model_comparison.md
├── outputs/
│   ├── evaluation/
│   │   ├── metrics.json
│   │   ├── clip_metrics.json
│   │   ├── patchcore_bottle_results.csv
│   │   ├── clip_bottle_results.csv
│   │   └── model_comparison.csv
│   └── patchcore_bottle/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── inference/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── tests/
├── requirements.txt
└── README.md
```

The trained checkpoint and dataset are stored locally. They must be made available separately if someone is to reproduce inference from a fresh clone.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd zero-shot-industrial-defect-detection
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the actual repository URL after publishing the project.

### 2. Create a virtual environment

Python 3.11 is the version used during local development.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The dependency file will be finalized as part of project packaging.

### 4. Make the dataset and checkpoint available

Place the bottle dataset in:

```text
data/raw/mvtec_anomaly_detection/bottle/
```

Place the trained PatchCore checkpoint at the path expected by the application:

```text
outputs/patchcore_bottle/Patchcore/MVTecAD/bottle/v0/weights/lightning/model.ckpt
```

The checkpoint is required to run PatchCore inference without retraining.

## Run the Application

Start the Gradio application:

```bash
python app.py
```

Open the local URL printed by Gradio, normally:

```text
http://127.0.0.1:7860
```

Upload a bottle image and select **Analyze Image** to view the prediction, anomaly score, and heatmap.

## Run Evaluation

Evaluate PatchCore on the bottle test dataset:

```bash
python evaluate_model.py
```

Run CLIP zero-shot classification:

```bash
python compare_clip.py
```

Generate the combined comparison report:

```bash
python compare_results.py
```

Evaluation files are written to `outputs/evaluation/`. The Markdown comparison report is saved to `docs/model_comparison.md`.

## Current Limitations

- Evaluation is restricted to the bottle category and 83 test images.
- The current PatchCore checkpoint and inference pipeline have been tested locally on CPU.
- CLIP uses image-level classification in the current implementation; it does not produce the pixel-level defect heatmaps used by PatchCore.
- The CLIP classification threshold and text prompts need further validation.
- Results may differ on other product types, acquisition conditions, or industrial datasets.
- The current metrics are preliminary and should be checked against a reproducible, clearly documented evaluation protocol before being reported as benchmark claims.
- Public deployment requires the model checkpoint, dependencies, and dataset-access arrangements to be handled appropriately.

## Future Improvements

- Investigate CLIP prompt design and classification thresholds.
- Improve model evaluation and add per-defect-category analysis.
- Add more industrial product categories.
- Improve the user interface and result visualization.
- Add automated tests and reproducible environment setup.
- Publish the application using a suitable hosting platform.

## Project Status

The PatchCore inference pipeline, Gradio interface, CLIP comparison experiment, and initial evaluation scripts have been implemented. Documentation, reproducibility checks, repository packaging, and deployment remain in progress.

## Acknowledgements

- [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [Anomalib](https://github.com/open-edge-platform/anomalib)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Gradio](https://www.gradio.app/)

## 🚀 Live Demo

**Try the deployed application:** [Industrial Defect Detection on Hugging Face](https://huggingface.co/spaces/Manikantaperla/zero-shot-industrial-defect-detection)

Upload a bottle image to view the PatchCore anomaly score, predicted result, and heatmap overlay.

**Scope:** The current model has been evaluated on the bottle category of the MVTec AD dataset. Other product categories have not been validated.
