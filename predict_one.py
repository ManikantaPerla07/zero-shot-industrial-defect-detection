from pathlib import Path
from anomalib.engine import Engine
from anomalib.models import Patchcore
import torch

checkpoint = Path(
    "outputs/patchcore_bottle/Patchcore/MVTecAD/bottle/v0/"
    "weights/lightning/model.ckpt"
)

image = Path(
    "data/raw/mvtec_anomaly_detection/bottle/"
    "test/broken_large/000.png"
)

if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

if not image.is_file():
    raise FileNotFoundError(f"Test image not found: {image}")

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
    precision="float32",
)

engine = Engine(accelerator="cpu", devices=1)

print("Loading saved model and predicting one image...")

predictions = engine.predict(
    model=model,
    data_path=str(image),
    ckpt_path=str(checkpoint),
)

if not predictions:
    raise RuntimeError("No predictions were returned.")

prediction = predictions[0]

print("Prediction: OK")
print("Image:", image.name)
print("Prediction type:", type(prediction).__name__)

for field in ("pred_score", "pred_label", "anomaly_map"):
    value = getattr(prediction, field, None)
    if value is not None:
        if torch.is_tensor(value):
            print(f"{field}: shape={tuple(value.shape)}")
            if field != "anomaly_map":
                print(f"{field} value: {value.detach().cpu().tolist()}")
        else:
            print(f"{field}: {value}")

print("Saved-checkpoint inference: OK")
