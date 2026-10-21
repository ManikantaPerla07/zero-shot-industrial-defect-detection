from pathlib import Path
from anomalib.data.datamodules.image.mvtecad import DOWNLOAD_INFO

root = Path("data/raw/mvtec_anomaly_detection")

print("Dataset URL:", DOWNLOAD_INFO.url)
print("Dataset folder exists:", root.exists())

if root.exists():
    print("\nFiles in dataset folder:")
    for item in root.iterdir():
        if item.is_file():
            print(f"{item.name} - {item.stat().st_size:,} bytes")
        else:
            print(f"{item.name}/ (folder)")
