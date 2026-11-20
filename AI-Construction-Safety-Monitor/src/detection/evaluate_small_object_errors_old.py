
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import cv2

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
MEDIUM_AREA_THRESHOLD = 0.05


def iou(box1, box2):
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


def read_labels(path):
    objects = []

    if not path.exists():
        return objects

    with open(path, encoding="utf-8") as file:
        for line in file:
            parts = line.split()

            if len(parts) != 5:
                continue

            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            area = w * h

            if area < SMALL_AREA_THRESHOLD:
                size = "Small"
            elif area < MEDIUM_AREA_THRESHOLD:
                size = "Medium"
            else:
                size = "Large"

            objects.append({
                "class": cls,
                "size": size,
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
        raise FileNotFoundError("Validation images or labels folder not found.")

    model = YOLO(str(MODEL_PATH))

    overall = defaultdict(int)
    size_stats = defaultdict(lambda: {"GT": 0, "TP": 0, "FN": 0})
    class_stats = defaultdict(
        lambda: {"TP": 0, "FP": 0, "FN": 0}
    )

    for image_path in sorted(IMAGES_DIR.iterdir()):

        if image_path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png"
        }:
            continue

        label_path = LABELS_DIR / f"{image_path.stem}.txt"
        ground_truth = read_labels(label_path)

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

        # Count all ground-truth objects by size
        for gt in ground_truth:
            size_stats[gt["size"]]["GT"] += 1

        # Match predictions against ALL ground-truth objects
        candidates = []

        for gi, gt in enumerate(ground_truth):
            for pi, pred in enumerate(predictions):

                if gt["class"] != pred["class"]:
                    continue

                overlap = iou(gt["box"], pred["box"])

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

            gt = ground_truth[gi]
            cls = gt["class"]

            overall["TP"] += 1
            class_stats[cls]["TP"] += 1

            size_stats[gt["size"]]["TP"] += 1

        # Unmatched ground-truth objects = FN
        for gi, gt in enumerate(ground_truth):

            if gi not in used_gt:
                cls = gt["class"]
                overall["FN"] += 1
                class_stats[cls]["FN"] += 1
                size_stats[gt["size"]]["FN"] += 1

        # Unmatched predictions = FP
        for pi, pred in enumerate(predictions):

            if pi not in used_pred:
                cls = pred["class"]
                overall["FP"] += 1
                class_stats[cls]["FP"] += 1

    precision = (
        overall["TP"] / (overall["TP"] + overall["FP"])
        if overall["TP"] + overall["FP"] else 0
    )

    recall = (
        overall["TP"] / (overall["TP"] + overall["FN"])
        if overall["TP"] + overall["FN"] else 0
    )

    small_recall = (
        size_stats["Small"]["TP"] / size_stats["Small"]["GT"]
        if size_stats["Small"]["GT"] else 0
    )

    print("\n========== CORRECTED DETECTION ERROR ANALYSIS ==========")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"IoU threshold: {IOU_THRESHOLD}")

    print("\n--- Overall Detection Metrics ---")
    print(f"True Positives: {overall['TP']}")
    print(f"False Positives: {overall['FP']}")
    print(f"False Negatives: {overall['FN']}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")

    print("\n--- Object Size Analysis ---")

    for size in ["Small", "Medium", "Large"]:
        stats = size_stats[size]

        size_recall = (
            stats["TP"] / stats["GT"]
            if stats["GT"] else 0
        )

        print(
            f"{size}: GT={stats['GT']}, "
            f"TP={stats['TP']}, "
            f"FN={stats['FN']}, "
            f"Recall={size_recall:.4f}"
        )

    print(f"\nSmall Object Recall: {small_recall:.4f}")

    print("\n--- Per-Class Detection Metrics ---")

    for cls, stats in sorted(class_stats.items()):
        print(
            f"{model.names[cls]}: "
            f"TP={stats['TP']}, "
            f"FP={stats['FP']}, "
            f"FN={stats['FN']}"
        )

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
