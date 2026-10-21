from pathlib import Path

root = Path("data/raw/mvtec_anomaly_detection/bottle")
extensions = {".png", ".jpg", ".jpeg", ".bmp"}

def count_images(folder):
    if not folder.exists():
        return 0
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

print("Bottle dataset:", root.exists())
print("Normal training images:", count_images(root / "train" / "good"))

print("\nTest images by category:")
test_root = root / "test"
if test_root.exists():
    for folder in sorted(test_root.iterdir()):
        if folder.is_dir():
            print(f"  {folder.name}: {count_images(folder)}")

print("\nGround-truth masks:", count_images(root / "ground_truth"))
print("\nDataset verification complete.")
