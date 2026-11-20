
from pathlib import Path
import cv2
import random

# =========================
# 1. Paths
# =========================

DATASET_DIR = Path("dataset/processed/PPE")

SOURCE_IMAGES = DATASET_DIR / "train" / "images"
SOURCE_LABELS = DATASET_DIR / "train" / "labels"

OUTPUT_IMAGES = DATASET_DIR / "train_balanced" / "images"
OUTPUT_LABELS = DATASET_DIR / "train_balanced" / "labels"

# =========================
# 2. Settings
# =========================

TARGET_CLASSES = {1, 2}  # NoHelmet, NoVest

# Maximum number of augmented images
MAX_AUGMENTED_IMAGES = 200

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(42)

# =========================
# 3. Create output folders
# =========================

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

# Clear previous generated files
for folder in [OUTPUT_IMAGES, OUTPUT_LABELS]:
    for file in folder.iterdir():
        if file.is_file():
            file.unlink()

# =========================
# 4. Copy original images
# =========================

original_count = 0
candidates = []

for image_path in SOURCE_IMAGES.iterdir():

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    label_text = label_path.read_text(encoding="utf-8")

    cv2.imwrite(
        str(OUTPUT_IMAGES / image_path.name),
        image
    )

    (OUTPUT_LABELS / label_path.name).write_text(
        label_text,
        encoding="utf-8"
    )

    original_count += 1

    lines = label_text.splitlines()

    class_ids = set()

    for line in lines:
        parts = line.split()

        if len(parts) == 5:
            class_ids.add(int(parts[0]))

    if class_ids & TARGET_CLASSES:
        candidates.append(image_path)

# =========================
# 5. Select limited images
# =========================

random.shuffle(candidates)

selected_images = candidates[:MAX_AUGMENTED_IMAGES]

# =========================
# 6. Horizontal Flip
# =========================

augmented_count = 0

for image_path in selected_images:

    label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    flipped_image = cv2.flip(image, 1)

    flipped_labels = []

    for line in lines:

        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        new_x_center = 1.0 - x_center

        flipped_labels.append(
            f"{class_id} {new_x_center:.6f} "
            f"{y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    output_name = f"{image_path.stem}_flip{image_path.suffix}"

    success = cv2.imwrite(
        str(OUTPUT_IMAGES / output_name),
        flipped_image
    )

    if not success:
        continue

    output_label = OUTPUT_LABELS / f"{image_path.stem}_flip.txt"

    output_label.write_text(
        "\n".join(flipped_labels) + "\n",
        encoding="utf-8"
    )

    augmented_count += 1

# =========================
# 7. Final report
# =========================

print("\nDataset preparation completed.")

print(f"Original images copied: {original_count}")
print(f"Candidate images: {len(candidates)}")
print(f"Selected images: {len(selected_images)}")
print(f"Augmented images created: {augmented_count}")

print(f"Total training images: {original_count + augmented_count}")

print(f"Output directory: {DATASET_DIR / 'train_balanced'}")
