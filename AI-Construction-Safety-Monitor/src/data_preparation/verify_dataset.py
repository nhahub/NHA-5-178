
from pathlib import Path
from collections import Counter

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "dataset" / "raw" / "PPE"

CLASS_NAMES = {
    0: "Helmet",
    1: "NoHelmet",
    2: "NoVest",
    3: "Vest"
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("VERIFYING DATASET LOCATION")
print("Dataset path:", DATASET_ROOT)

total_images = 0

for split in ["train", "valid"]:
    images_dir = DATASET_ROOT / split / "images"
    labels_dir = DATASET_ROOT / split / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing folders in {split}")

    images = {
        p.stem for p in images_dir.iterdir()
        if p.suffix.lower() in EXTENSIONS
    }
    labels = {p.stem for p in labels_dir.glob("*.txt")}

    counts = Counter()
    invalid = 0

    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()

            try:
                if len(parts) != 5:
                    raise ValueError

                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                if class_id not in CLASS_NAMES:
                    raise ValueError
                if not all(0 <= v <= 1 for v in (x, y, w, h)):
                    raise ValueError
                if w <= 0 or h <= 0:
                    raise ValueError

                counts[class_id] += 1

            except ValueError:
                invalid += 1

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Missing labels:", len(images - labels))
    print("Labels without images:", len(labels - images))
    print("Invalid annotations:", invalid)

    for class_id, name in CLASS_NAMES.items():
        print(f"{name}: {counts[class_id]}")

    total_images += len(images)

print("\nTOTAL IMAGES:", total_images)
print("VERIFICATION FINISHED")
