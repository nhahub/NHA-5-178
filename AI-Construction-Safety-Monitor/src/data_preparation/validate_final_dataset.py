
from pathlib import Path
from collections import Counter
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "processed" / "PPE"

SPLITS = ["train", "valid", "test"]

CLASS_NAMES = {
    0: "Helmet",
    1: "NoHelmet",
    2: "NoVest",
    3: "Vest"
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_split(split):
    images_dir = DATASET_ROOT / split / "images"
    labels_dir = DATASET_ROOT / split / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing folders in {split}")

    images = {
        p.stem for p in images_dir.iterdir()
        if p.suffix.lower() in EXTENSIONS
    }
    labels = {p.stem for p in labels_dir.glob("*.txt")}

    missing_labels = images - labels
    missing_images = labels - images

    class_counts = Counter()
    invalid_lines = 0
    empty_labels = 0

    for label_file in labels_dir.glob("*.txt"):
        content = label_file.read_text(encoding="utf-8").strip()

        if not content:
            empty_labels += 1
            continue

        for line in content.splitlines():
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

                class_counts[class_id] += 1

            except ValueError:
                invalid_lines += 1

    print(f"\n--- {split.upper()} ---")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Missing labels:", len(missing_labels))
    print("Labels without images:", len(missing_images))
    print("Empty labels:", empty_labels)
    print("Invalid annotations:", invalid_lines)

    print("Class distribution:")
    for class_id, name in CLASS_NAMES.items():
        print(f"  {name}: {class_counts[class_id]}")

    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "empty_labels": empty_labels,
        "invalid_annotations": invalid_lines
    }


if __name__ == "__main__":
    print("=" * 50)
    print("FINAL PPE DATASET VALIDATION")
    print("=" * 50)

    config_path = DATASET_ROOT / "data.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\nDATA.YAML:")
    print("Classes:", config["names"])
    print("Train:", config["train"])
    print("Validation:", config["val"])
    print("Test:", config["test"])

    results = {}

    for split in SPLITS:
        results[split] = validate_split(split)

    total_images = sum(r["images"] for r in results.values())

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)

    print("Total images:", total_images)

    all_valid = all(
        r["images"] == r["labels"]
        and r["missing_labels"] == 0
        and r["missing_images"] == 0
        and r["empty_labels"] == 0
        and r["invalid_annotations"] == 0
        for r in results.values()
    )

    print("Dataset integrity:", "PASSED" if all_valid else "FAILED")
