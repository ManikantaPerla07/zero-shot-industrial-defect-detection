import csv
csv.field_size_limit(10**7)

from anomalib.data import MVTecAD

data = MVTecAD(
    root="data/raw/mvtec_anomaly_detection",
    category="bottle",
    train_batch_size=4,
    eval_batch_size=4,
)

data.prepare_data()
data.setup("fit")
data.setup("test")

print("Anomalib dataset loading: OK")
print("Category: bottle")
print("Training and test setup: complete")
