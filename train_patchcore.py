from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

print("1. Loading bottle dataset...")

datamodule = MVTecAD(
    root="data/raw/mvtec_anomaly_detection",
    category="bottle",
    train_batch_size=4,
    eval_batch_size=4,
    num_workers=0,
)

print("2. Creating PatchCore model...")

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
    precision="float32",
)

engine = Engine(
    default_root_dir="outputs/patchcore_bottle",
    accelerator="cpu",
    devices=1,
    max_epochs=1,
)

print("3. Building normal-image feature memory...")
print("This may take several minutes on CPU.")

engine.fit(
    model=model,
    datamodule=datamodule,
)

print("4. Evaluating on test images...")
results = engine.test(
    model=model,
    datamodule=datamodule,
)

print("\nPatchCore training and evaluation finished.")
print(results)
