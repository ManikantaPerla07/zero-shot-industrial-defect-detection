from pathlib import Path
from huggingface_hub import snapshot_download

root = Path("data/raw/mvtec_anomaly_detection")
root.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="foersben/mvtec-ad",
    repo_type="dataset",
    local_dir=str(root),
    allow_patterns=[
        "bottle/**",
        "license.txt",
        "readme.txt",
    ],
)

print("Bottle training images:",
      (root / "bottle" / "train" / "good").is_dir())
print("Bottle test images:",
      (root / "bottle" / "test").is_dir())
print("Dataset download step finished.")
