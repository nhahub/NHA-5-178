
from pathlib import Path
from collections import Counter, defaultdict
import yaml

# Dataset paths
DATASET_DIR = Path("dataset/processed/PPE")
DATA_YAML = DATASET_DIR / "data.yaml"

# Load class names
with open(DATA_YAML, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

class_names = config["names"]

if isinstance(class_names, dict):
    class_names = {
        int(key): value for key, value in class_names.items()
    }

# Analyze training labels only
labels_dir = DATASET_DIR / "train" / "labels"

object_counts = Counter()
image_counts = Counter()
empty_labels = 0
total_images = 0

for label_file in labels_dir.glob("*.txt"):
    total_images += 1
    image_classes = set()

    with open(label_file, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        empty_labels += 1

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        if class_id not in class_names:
            continue

        object_counts[class_id] += 1
        image_classes.add(class_id)

    for class_id in image_classes:
        image_counts[class_id] += 1

print("\n========== CLASS BALANCE ANALYSIS ==========")
print(f"Total training images: {total_images}")
print(f"Images with empty labels: {empty_labels}")

print("\nClass distribution:")
print("-" * 65)
print(f"{'Class':<15}{'Objects':<15}{'Images':<15}{'Object %':<15}")
print("-" * 65)

total_objects = sum(object_counts.values())

for class_id, name in class_names.items():
    objects = object_counts[class_id]
    images = image_counts[class_id]
    percentage = (objects / total_objects * 100) if total_objects else 0

    print(f"{name:<15}{objects:<15}{images:<15}{percentage:.2f}%")

print("-" * 65)
print(f"Total objects: {total_objects}")

print("\nAnalysis completed.")
