from ultralytics import YOLO
from pathlib import Path

# Load dataset configuration
yaml_path = "dataset/processed/PPE/data_balanced.yaml"

# Load YOLO model
model = YOLO("yolov8n.pt")



print("Checking dataset configuration...")

from ultralytics.data.utils import check_det_dataset

dataset = check_det_dataset(yaml_path)

print("\n===== YOLO DATASET PATH CHECK =====")

for split in ["train", "val", "test"]:
    split_path = Path(dataset[split])
    print(f"{split}: {split_path}")
    print(f"Exists: {split_path.exists()}")

print("\nDataset YAML loaded successfully!")