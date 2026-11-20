
from pathlib import Path
from collections import Counter
import yaml

DATASET_DIR = Path("dataset/processed/PPE")

with open(DATASET_DIR / "data.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

names = config["names"]
if isinstance(names, dict):
    names = {int(k): v for k, v in names.items()}

labels_dir = DATASET_DIR / "train" / "labels"

class_image_counts = Counter()
class_object_counts = Counter()
combination_counts = Counter()

for label_file in labels_dir.glob("*.txt"):
    image_classes = set()

    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        if class_id in names:
            image_classes.add(class_id)
            class_object_counts[class_id] += 1

    for class_id in image_classes:
        class_image_counts[class_id] += 1

    if image_classes:
        combination = " + ".join(
            names[class_id] for class_id in sorted(image_classes)
        )
        combination_counts[combination] += 1

print("\n===== IMAGES PER CLASS =====")
for class_id, name in names.items():
    print(f"{name}: {class_image_counts[class_id]} images")

print("\n===== CLASS COMBINATIONS =====")
for combination, count in combination_counts.most_common():
    print(f"{combination}: {count} images")

print("\n===== TOTAL OBJECTS =====")
for class_id, name in names.items():
    print(f"{name}: {class_object_counts[class_id]} objects")

print("\nAnalysis completed.")
