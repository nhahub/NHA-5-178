
from pathlib import Path
from collections import Counter
import yaml

# Dataset paths
DATASET_DIR = Path("dataset/processed/PPE")
DATA_YAML = DATASET_DIR / "data.yaml"
LABELS_DIR = DATASET_DIR / "train" / "labels"

# Load class names
with open(DATA_YAML, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

class_names = config["names"]

if isinstance(class_names, dict):
    class_names = {
        int(key): value for key, value in class_names.items()
    }

# Count object sizes
size_counts = Counter()
class_size_counts = {
    class_id: Counter() for class_id in class_names
}

total_objects = 0

for label_file in LABELS_DIR.glob("*.txt"):
    with open(label_file, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        if class_id not in class_names:
            continue

        width = float(parts[3])
        height = float(parts[4])

        # Normalized bounding box area
        area = width * height

        if area < 0.01:
            size = "Small"
        elif area < 0.05:
            size = "Medium"
        else:
            size = "Large"

        size_counts[size] += 1
        class_size_counts[class_id][size] += 1
        total_objects += 1

print("\n========== OBJECT SIZE ANALYSIS ==========")
print(f"Total objects analyzed: {total_objects}")

print("\nOverall size distribution:")
for size in ["Small", "Medium", "Large"]:
    count = size_counts[size]
    percentage = count / total_objects * 100 if total_objects else 0
    print(f"{size:<10}: {count:<6} ({percentage:.2f}%)")

print("\nSize distribution per class:")
print("-" * 60)
print(f"{'Class':<15}{'Small':<12}{'Medium':<12}{'Large':<12}")
print("-" * 60)

for class_id, name in class_names.items():
    counts = class_size_counts[class_id]
    print(
        f"{name:<15}"
        f"{counts['Small']:<12}"
        f"{counts['Medium']:<12}"
        f"{counts['Large']:<12}"
    )

print("-" * 60)
print("\nAnalysis completed.")
