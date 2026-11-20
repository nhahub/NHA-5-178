
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import cv2
import csv

MODEL_PATH = Path(
    r"C:\Users\Hp\Documents\GitHub\NHA-5-178"
    r"\runs\detect\runs\PPE\balanced_30_epochs\weights\best.pt"
)

DATASET_DIR = Path("dataset/processed/PPE")
IMAGES_DIR = DATASET_DIR / "valid" / "images"
LABELS_DIR = DATASET_DIR / "valid" / "labels"

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.50
SMALL_AREA_THRESHOLD = 0.01

OUTPUT_DIR = Path("runs/PPE/small_object_analysis")


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )
    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    return intersection / union if union else 0


def read_small_labels(label_path):

    objects = []

    if not label_path.exists():
        return objects

    with open(label_path, encoding="utf-8") as file:

        for line in file:

            parts = line.split()

            if len(parts) != 5:
                continue

            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            if w * h < SMALL_AREA_THRESHOLD:

                objects.append({
                    "class": cls,
                    "box": [
                        xc - w / 2,
                        yc - h / 2,
                        xc + w / 2,
                        yc + h / 2
                    ]
                })

    return objects


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        raise FileNotFoundError("Validation folders not found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    missed_objects = []
    missed_by_class = defaultdict(int)

    total_small = 0
    detected_small = 0

    for image_path in sorted(IMAGES_DIR.iterdir()):

        if image_path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png"
        }:
            continue

        label_path = LABELS_DIR / f"{image_path.stem}.txt"

        ground_truth = read_small_labels(label_path)

        if not ground_truth:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False
        )[0]

        predictions = []

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            predictions.append({
                "class": int(box.cls[0]),
                "box": [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height
                ]
            })

        candidates = []

        for gi, gt in enumerate(ground_truth):

            for pi, pred in enumerate(predictions):

                if gt["class"] != pred["class"]:
                    continue

                overlap = calculate_iou(
                    gt["box"], pred["box"]
                )

                if overlap >= IOU_THRESHOLD:
                    candidates.append((overlap, gi, pi))

        candidates.sort(reverse=True)

        used_gt = set()
        used_pred = set()

        for overlap, gi, pi in candidates:

            if gi in used_gt or pi in used_pred:
                continue

            used_gt.add(gi)
            used_pred.add(pi)

        total_small += len(ground_truth)
        detected_small += len(used_gt)

        for gi, gt in enumerate(ground_truth):

            if gi not in used_gt:

                cls = gt["class"]
                missed_by_class[cls] += 1

                missed_objects.append({
                    "image": image_path.name,
                    "class": model.names[cls],
                    "class_id": cls,
                    "confidence_threshold": CONFIDENCE,
                    "ground_truth_area": round(
                        (gt["box"][2] - gt["box"][0]) *
                        (gt["box"][3] - gt["box"][1]), 6
                    )
                })

    csv_path = OUTPUT_DIR / "missed_small_objects.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:

        fieldnames = [
            "image",
            "class",
            "class_id",
            "confidence_threshold",
            "ground_truth_area"
        ]

        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(missed_objects)

    print("\n========== MISSED SMALL OBJECTS ==========")

    print(f"Total small GT objects: {total_small}")
    print(f"Detected small objects: {detected_small}")
    print(f"Missed small objects: {len(missed_objects)}")

    recall = detected_small / total_small if total_small else 0

    print(f"Small Object Recall: {recall:.4f}")

    print("\nMissed objects per class:")

    for cls, count in sorted(missed_by_class.items()):
        print(f"{model.names[cls]}: {count}")

    unique_images = len({
        item["image"] for item in missed_objects
    })

    print(f"\nImages containing missed objects: {unique_images}")
    print(f"CSV saved to: {csv_path}")
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
