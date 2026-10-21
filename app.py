from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from anomalib.engine import Engine
from anomalib.models import Patchcore


CHECKPOINT = Path(
    "outputs/patchcore_bottle/Patchcore/MVTecAD/bottle/v0/"
    "weights/lightning/model.ckpt"
)

if not CHECKPOINT.is_file():
    raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT}")

print("Loading PatchCore model...")

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
    default_root_dir="results/app_inference",
)

print("Model configuration ready.")


def analyze_image(image_path):
    if not image_path:
        return None, None, "Please upload an image first."

    try:
        original = Image.open(image_path).convert("RGB")

        predictions = engine.predict(
            model=model,
            data_path=str(image_path),
            ckpt_path=str(CHECKPOINT),
        )

        if not predictions:
            return original, None, "No prediction was returned."

        prediction = predictions[0]

        score = float(prediction.pred_score.reshape(-1)[0].item())
        is_defective = bool(prediction.pred_label.reshape(-1)[0].item())

        anomaly_map = (
            prediction.anomaly_map
            .detach()
            .cpu()
            .squeeze()
            .numpy()
        )

        anomaly_map = np.nan_to_num(anomaly_map)

        normalized = cv2.normalize(
            anomaly_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        heatmap_bgr = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET,
        )

        heatmap_rgb = cv2.cvtColor(
            heatmap_bgr,
            cv2.COLOR_BGR2RGB,
        )

        heatmap_image = Image.fromarray(heatmap_rgb).resize(
            original.size,
            Image.Resampling.BILINEAR,
        )

        overlay = Image.blend(
            original,
            heatmap_image,
            alpha=0.45,
        )

        status = (
            "Potential defect detected"
            if is_defective
            else "Likely normal"
        )

        report = (
            f"### Result: {status}\n\n"
            f"**Anomaly score:** `{score:.4f}`\n\n"
            "**Model:** PatchCore with a pretrained "
            "Wide ResNet-50-2 feature extractor\n\n"
            "**Heatmap:** Brighter colors indicate higher relative "
            "anomaly scores in this image.\n\n"
            "*The anomaly score is not a calibrated probability. "
            "This prototype has been evaluated on the MVTec AD "
            "bottle category; predictions on other product types "
            "are not validated.*"
        )

        return original, overlay, report

    except Exception as exc:
        return None, None, f"Prediction failed: {type(exc).__name__}: {exc}"


with gr.Blocks(title="Industrial Defect Detection") as demo:
    gr.Markdown(
        """
        # Zero-Shot Industrial Defect Detection
        Upload an industrial product image to inspect potential anomalies.
        The current prototype uses a PatchCore baseline trained on
        normal MVTec AD bottle images.
        """
    )

    image_input = gr.Image(
        type="filepath",
        label="Upload an industrial image",
    )

    analyze_button = gr.Button(
        "Analyze Image",
        variant="primary",
    )

    with gr.Row():
        original_output = gr.Image(label="Original Image")
        heatmap_output = gr.Image(label="Anomaly Heatmap Overlay")

    report_output = gr.Markdown(label="Analysis Result")

    analyze_button.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[original_output, heatmap_output, report_output],
    )


if __name__ == "__main__":
    demo.launch()
