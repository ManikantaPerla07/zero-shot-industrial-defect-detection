# Project Report: Zero-Shot Industrial Defect Detection Using Foundation Vision Models

**Subtitle:** PatchCore anomaly detection with a CLIP zero-shot comparison  
**Course/project:** CSE471  
**Prepared by:** Manikanta Perla  
**Programme:** B.Tech Computer Science & Engineering (AI & ML), Batch 2027  
**Institution:** Lovely Professional University  
**Date:** 21 October 2026

## Abstract

This project implements a bottle-focused industrial anomaly-detection prototype using PatchCore with pretrained Wide ResNet-50-2 features. The model is trained using defect-free bottle images from the MVTec Anomaly Detection (MVTec AD) dataset and produces an image-level anomaly score plus a pixel-wise anomaly map. A Gradio web application enables image upload, prediction, and anomaly heatmap visualization. A separate experiment evaluates OpenAI CLIP as a zero-shot image-level classifier using prompts for normal and defective bottles.

On the 83-image bottle test subset, the project's evaluation scripts reported 98.80% image accuracy and 98.56% pixel ROC-AUC for PatchCore. CLIP reported 75.90% accuracy and 93.57% image ROC-AUC, but its current decision rule classified all 83 images as defective. These are preliminary, category-specific project results—not official benchmark scores.

## 1. Problem Statement and Objectives

Industrial quality inspection must identify abnormal products while highlighting suspicious regions for review. The project aims to build a working visual inspection aid, evaluate a normal-only anomaly-detection baseline, compare it with zero-shot image-text classification, and make the result accessible through a public demo.

Objectives:

- Train a PatchCore baseline from normal bottle images only.
- Produce an image-level anomaly score and pixel-wise anomaly map.
- Build and deploy an interactive Gradio application.
- Compare PatchCore and CLIP on the same test images.
- Document results, limitations, and reproducibility requirements.

## 2. Dataset

The project uses only the bottle category from [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). The dataset contains defect-free training images, normal and anomalous test images, and pixel-precise masks for defects [1].

| Split | Images | Meaning |
|---|---:|---|
| `train/good` | 209 | Normal training bottles |
| `test/good` | 20 | Normal test bottles |
| `test/broken_large` | 20 | Defective bottles |
| `test/broken_small` | 22 | Defective bottles |
| `test/contamination` | 21 | Defective bottles |
| **Total test set** | **83** | **20 normal, 63 defective** |

The dataset is not included in the public GitHub repository. MVTec AD is released under CC BY-NC-SA 4.0, which restricts commercial use; any redistribution or deployment must respect the dataset terms.

## 3. Methodology

### 3.1 PatchCore

PatchCore compares patch features from test images against a representative memory bank of normal patch features [2, 4]. Configuration used in this project:

- Backbone: `wide_resnet50_2`
- Feature layers: `layer2`, `layer3`
- Coreset sampling ratio: `0.1`
- Nearest neighbors: `9`
- Precision: FP32

At inference, PatchCore produces an anomaly score and anomaly map. The app renders a JET-colour heatmap blended with the input image.

### 3.2 CLIP zero-shot comparison

CLIP aligns images and natural-language descriptions in a shared representation space [3]. This experiment uses `openai/clip-vit-base-patch32`, grouped prompts for normal and defective bottle descriptions, and relative image-text similarity scores to predict a class. CLIP is evaluated only for image-level classification in this implementation; it does not produce the app's pixel-level heatmap.

### 3.3 Application and Deployment

The Gradio interface displays the uploaded image, heatmap overlay, predicted label, and anomaly score. The public application is hosted on [Hugging Face Spaces](https://huggingface.co/spaces/Manikantaperla/zero-shot-industrial-defect-detection) using ZeroGPU. The trained checkpoint is stored in the Space repository through Git LFS; the dataset is not stored in the GitHub source repository.

## 4. Evaluation Protocol

Both model scripts evaluated the same 83 test images. Image-level ground-truth labels were assigned by category: `good` is normal, and `broken_large`, `broken_small`, and `contamination` are defective. PatchCore pixel ROC-AUC was computed from anomaly maps and the available ground-truth masks; the normal images use all-zero masks.

## 5. Results

| Metric | PatchCore | CLIP zero-shot |
|---|---:|---:|
| Test images | 83 | 83 |
| Accuracy | **98.80%** | 75.90% |
| Precision | **100.00%** | 75.90% |
| Recall | 98.41% | **100.00%** |
| F1-score | **99.20%** | 86.30% |
| Image ROC-AUC | **100.00%** | 93.57% |
| Pixel ROC-AUC | **98.56%** | Not evaluated |

### Interpretation

- Under the current Anomalib prediction threshold, PatchCore classified 82 of 83 images correctly: 62 of 63 defective bottles and all 20 normal bottles.
- The image ROC-AUC of 1.0000 refers to score ranking on this subset. It does not imply that every threshold yields perfect classification.
- CLIP predicted every image as defective. This produced 100% recall, 20 false positives among 20 normal bottles, and 75.90% accuracy. Prompt design and the decision rule need refinement.
- Pixel ROC-AUC is aggregated across resized anomaly maps and masks, not reported separately for each defect type.

These measurements are preliminary project-script results on one category. They should not be described as official MVTec benchmark scores or evidence of generalization to other product types.

## 6. Live Application Checks

The deployed Space was manually tested with two examples:

- Defective bottle: displayed `Potential defect detected`, score `0.9515`.
- Normal bottle: displayed `Likely normal`, score `0.2298`.

The live app returns the original image and heatmap overlay for both examples. Heatmaps are min-max normalized per image; warm colours show higher relative anomaly values within that image and are not calibrated probabilities.

## 7. Limitations

- Evaluation covers only the bottle category and 83 test images.
- PatchCore is not fully zero-shot in this implementation; it uses defect-free normal images to build its memory bank.
- CLIP currently predicts all test images as defective under the current prompt-based rule.
- Scores are not calibrated probabilities.
- Other product categories and real factory conditions have not been validated.
- Fresh environments may resolve different dependency versions because the requirements list is not a fully pinned lock file.

## 8. Conclusion and Future Work

The project delivers a working PatchCore anomaly-detection application with heatmap visualization, initial evaluation outputs, a CLIP zero-shot comparison, and a public deployment. The results are promising for the bottle subset, while the CLIP all-defect behaviour identifies a clear area for improvement.

Next steps include validating more MVTec categories, calibrating CLIP prompts and thresholds on a separate validation split, reporting per-category metrics, adding automated tests, and pinning dependencies.

## References

1. P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR, 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_Detection_CVPR_2019_paper.html
2. K. Roth et al., “Towards Total Recall in Industrial Anomaly Detection,” CVPR, 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html
3. A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML, PMLR 139, 2021. https://proceedings.mlr.press/v139/radford21a.html
4. Anomalib documentation, “PatchCore.” https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/models/image/patchcore.html
5. MVTec Software GmbH, “MVTec AD—Industrial Anomaly Detection Dataset.” https://www.mvtec.com/research-teaching/datasets/mvtec-ad