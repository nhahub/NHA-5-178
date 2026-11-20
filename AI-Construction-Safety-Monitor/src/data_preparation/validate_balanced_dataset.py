
from pathlib import Path

DATASET_DIR = Path("dataset/processed/PPE/train_balanced")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

image_extensions = {".jpg", ".jpeg", ".png"}

images = [
    p for p in IMAGES_DIR.iterdir()
    if p.suffix.lower() in image_extensions
]

labels = list(LABELS_DIR.glob("*.txt"))

image_stems = {p.stem for p in images}
label_stems = {p.stem for p in labels}

missing_labels = image_stems - label_stems
missing_images = label_stems - image_stems

invalid_labels = []

for label_path in labels:
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()

        if len(parts) != 5:
            invalid_labels.append((label_path.name, line_number))
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            if class_id not in range(4):
                raise ValueError()

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                raise ValueError()

            if w <= 0 or h <= 0:
                raise ValueError()

        except ValueError:
            invalid_labels.append((label_path.name, line_number))

print("===== BALANCED DATASET VALIDATION =====")
print(f"Images: {len(images)}")
print(f"Labels: {len(labels)}")
print(f"Missing labels: {len(missing_labels)}")
print(f"Missing images: {len(missing_images)}")
print(f"Invalid label lines: {len(invalid_labels)}")

if (
    len(images) == 1800
    and len(labels) == 1800
    and not missing_labels
    and not missing_images
    and not invalid_labels
):
    print("\nVALIDATION PASSED!")
else:
    print("\nVALIDATION NEEDS ATTENTION!")
