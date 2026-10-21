# Industrial Defect Detection: Model Comparison

Both models were evaluated on the same 83 MVTec AD bottle test images.

| Model | Images | Accuracy | Precision | Recall | F1 Score | Image ROC-AUC | Pixel ROC-AUC |
|---|---|---|---|---|---|---|---|
| PatchCore | 83 | 0.9880 | 1.0000 | 0.9841 | 0.9920 | 1.0000 | 0.9856 |
| CLIP Zero-Shot | 83 | 0.7590 | 0.7590 | 1.0000 | 0.8630 | 0.9357 | Not evaluated |

## Interpretation and limitations

- PatchCore provides image-level anomaly detection and pixel-level localization.
- CLIP was evaluated for zero-shot image-level classification using text prompts.
- The current CLIP decision rule classified every test image as defective. Its decision threshold requires further investigation.
- CLIP's pixel-level localization was not evaluated.
- Results apply to this bottle test set and do not establish performance on other product categories.
- The CLIP score is a relative text-similarity-derived score, not a calibrated probability.
