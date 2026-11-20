
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import cv2

# Model and dataset paths
MODEL_PATH = Path(
    r"C:\Users\Hp\Documents\GitHub\NHA-5-178"
    r"\runs\detect\runs\PPE\balanced_30_epochs\weights\best.pt"
)

DATASET_DIR = Path("dataset/processed/PPE")
IMAGES_DIR = DATASET_DIR / "valid" / "images"
LABELS_DIR = DATASET_DIR / "valid" / "labels"

IMAGE_SIZES = [320, 640]

# Experimental evaluation settings
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.50
SMALL_AREA_THRESHOLD = 0.01


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

    return intersection / union if union > 0 else 0


def load_small_objects(label_path):
    small_objects = []

    if not label_path.exists():
        return small_objects

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center, y_center, width, height = map(
                float, parts[1:]
            )

            if width * height >= SMALL_AREA_THRESHOLD:
                continue

            x1 = (x_center - width / 2)
            y1 = (y_center - height / 2)
            x2 = (x_center + width / 2)
            y2 = (y_center + height / 2)

            small_objects.append({
                "class_id": class_id,
                "box": [x1, y1, x2, y2],
                "matched": False
            })

    return small_objects


def evaluate_size(model, images, image_size):

    total_small = 0
    total_detected = 0

    class_counts = defaultdict(lambda: [0, 0])

    for image_path in images:

        label_path = LABELS_DIR / f"{image_path.stem}.txt"
        ground_truth = load_small_objects(label_path)

        total_small += len(ground_truth)

        for obj in ground_truth:
            class_counts[obj["class_id"]][0] += 1

        if not ground_truth:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: Cannot read {image_path.name}")
            continue

        height, width = image.shape[:2]

        result = model.predict(
            source=str(image_path),
            imgsz=image_size,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False
        )[0]

        predictions = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Normalize prediction coordinates
            pred_box = [
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height
            ]

            predictions.append({
                "class_id": class_id,
                "confidence": confidence,
                "box": pred_box
            })

        # Match predictions to small ground-truth objects
        candidates = []

        for gt_index, gt in enumerate(ground_truth):
            for pred_index, pred in enumerate(predictions):

                if gt["class_id"] != pred["class_id"]:
                    continue

                iou = calculate_iou(gt["box"], pred["box"])

                if iou >= IOU_THRESHOLD:
                    candidates.append(
                        (iou, gt_index, pred_index)
                    )

        candidates.sort(reverse=True)

        used_gt = set()
        used_predictions = set()

        for iou, gt_index, pred_index in candidates:

            if gt_index in used_gt:
                continue

            if pred_index in used_predictions:
                continue

            used_gt.add(gt_index)
            used_predictions.add(pred_index)

            gt = ground_truth[gt_index]

            gt["matched"] = True
            class_counts[gt["class_id"]][1] += 1
            total_detected += 1

    recall = (
        total_detected / total_small
        if total_small else 0
    )

    return total_small, total_detected, recall, class_counts


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Validation images not found: {IMAGES_DIR}"
        )

    images = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        raise ValueError("No validation images found!")

    print(f"Validation images: {len(images)}")
    print(f"Small object threshold: {SMALL_AREA_THRESHOLD * 100:.1f}%")
    print(f"Confidence threshold: {CONFIDENCE}")
    print(f"IoU threshold: {IOU_THRESHOLD}")

    model = YOLO(str(MODEL_PATH))

    for image_size in IMAGE_SIZES:

        print("\n================================")
        print(f"SMALL OBJECT EVALUATION: {image_size}")
        print("================================")

        total, detected, recall, class_counts = evaluate_size(
            model, images, image_size
        )

        print(f"Small ground-truth objects: {total}")
        print(f"Correctly detected objects: {detected}")
        print(f"Small Object Recall: {recall:.4f} ({recall * 100:.2f}%)")

        print("\nPer-class results:")

        class_names = model.names

        for class_id in sorted(class_counts):
            count, found = class_counts[class_id]

            class_recall = found / count if count else 0

            print(
                f"{class_names[class_id]}: "
                f"{found}/{count} "
                f"({class_recall * 100:.2f}%)"
            )

    print("\nSmall object comparison completed!")


if __name__ == "__main__":
    main()
