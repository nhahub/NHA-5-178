
from pathlib import Path
from collections import Counter
import csv

DATASET_DIR = Path("dataset/processed/PPE")
LABELS_DIR = DATASET_DIR / "valid" / "labels"

MISSED_CSV = Path(
    "runs/PPE/small_object_analysis/missed_small_objects.csv"
)

OUTPUT_DIR = Path("runs/PPE/annotation_audit")

CLASS_NAMES = {
    0: "Helmet",
    1: "NoHelmet",
    2: "NoVest",
    3: "Vest"
}


def main():

    if not MISSED_CSV.exists():
        raise FileNotFoundError(MISSED_CSV)

    if not LABELS_DIR.exists():
        raise FileNotFoundError(LABELS_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missed_by_image = Counter()

    with open(MISSED_CSV, encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            missed_by_image[row["image"]] += 1

    report = []

    for image_name, missed_count in missed_by_image.items():

        label_path = LABELS_DIR / f"{Path(image_name).stem}.txt"

        class_counts = Counter()
        total_objects = 0
        small_objects = 0

        if label_path.exists():

            with open(label_path, encoding="utf-8") as file:

                for line in file:

                    parts = line.split()

                    if len(parts) != 5:
                        continue

                    cls = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])

                    class_counts[cls] += 1
                    total_objects += 1

                    if w * h < 0.01:
                        small_objects += 1

        report.append({
            "image": image_name,
            "total_GT_objects": total_objects,
            "small_GT_objects": small_objects,
            "missed_small_objects": missed_count,
            "Helmet": class_counts[0],
            "NoHelmet": class_counts[1],
            "NoVest": class_counts[2],
            "Vest": class_counts[3],
            "label_file_exists": label_path.exists()
        })

    report.sort(
        key=lambda item: item["missed_small_objects"],
        reverse=True
    )

    output_path = OUTPUT_DIR / "annotation_audit.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=report[0].keys()
        )

        writer.writeheader()
        writer.writerows(report)

    print("\n========== ANNOTATION AUDIT ==========")
    print(f"Images reviewed: {len(report)}")

    for item in report:

        print(
            f"\n{item['image']}"
            f"\n  Total GT: {item['total_GT_objects']}"
            f"\n  Small GT: {item['small_GT_objects']}"
            f"\n  Missed small: {item['missed_small_objects']}"
            f"\n  Classes: Helmet={item['Helmet']}, "
            f"NoHelmet={item['NoHelmet']}, "
            f"NoVest={item['NoVest']}, "
            f"Vest={item['Vest']}"
        )

    print(f"\nReport saved to: {output_path}")
    print("\nAUDIT COMPLETED!")


if __name__ == "__main__":
    main()
