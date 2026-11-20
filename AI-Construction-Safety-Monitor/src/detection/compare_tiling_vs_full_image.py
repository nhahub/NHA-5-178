
from pathlib import Path
from ultralytics import YOLO
import cv2

# ==================== PATHS ====================

MODEL_PATH = Path(
    r"C:\Users\Hp\Documents\GitHub\NHA-5-178"
    r"\runs\detect\runs\PPE\balanced_30_epochs\weights\best.pt"
)

IMAGE_PATH = Path(
    "dataset/processed/PPE/valid/images/"
    "03295_jpg.rf.28082ec6b792bc923e0dcbe473c4551e.jpg"
)

LABEL_PATH = Path(
    "dataset/processed/PPE/valid/labels/"
    "03295_jpg.rf.28082ec6b792bc923e0dcbe473c4551e.txt"
)

OUTPUT_DIR = Path("runs/PPE/tiling_comparison")

# ==================== SETTINGS ====================

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.50
SMALL_AREA_THRESHOLD = 0.01

TILE_SIZE = 640
OVERLAP = 0.20


# ==================== IOU ====================

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


# ==================== GROUND TRUTH ====================

def read_ground_truth(path, image_width, image_height):

    objects = []

    with open(path, encoding="utf-8") as file:

        for line in file:

            parts = line.split()

            if len(parts) != 5:
                continue

            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            # Convert normalized YOLO coordinates to pixels
            box = [
                (xc - w / 2) * image_width,
                (yc - h / 2) * image_height,
                (xc + w / 2) * image_width,
                (yc + h / 2) * image_height
            ]

            objects.append({
                "class": cls,
                "box": box,
                "small": w * h < SMALL_AREA_THRESHOLD
            })

    return objects


# ==================== TILE POSITIONS ====================

def get_positions(length):

    if length <= TILE_SIZE:
        return [0]

    stride = int(TILE_SIZE * (1 - OVERLAP))

    positions = list(
        range(0, length - TILE_SIZE + 1, stride)
    )

    last_position = length - TILE_SIZE

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


# ==================== NMS ====================

def remove_duplicates(detections):

    detections.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    kept = []

    for detection in detections:

        duplicate = False

        for previous in kept:

            if detection["class"] != previous["class"]:
                continue

            overlap = calculate_iou(
                detection["box"],
                previous["box"]
            )

            if overlap >= 0.50:
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept


# ==================== FULL IMAGE PREDICTION ====================

def predict_full(model, image):

    result = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device="cpu",
        verbose=False
    )[0]

    predictions = []

    for box in result.boxes:

        predictions.append({
            "class": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "box": box.xyxy[0].tolist()
        })

    return predictions


# ==================== TILED PREDICTION ====================

def predict_tiled(model, image):

    height, width = image.shape[:2]

    detections = []

    tile_count = 0

    for y in get_positions(height):

        for x in get_positions(width):

            x2 = min(x + TILE_SIZE, width)
            y2 = min(y + TILE_SIZE, height)

            tile = image[y:y2, x:x2]

            tile_count += 1

            print(f"Processing tile {tile_count}")

            result = model.predict(
                source=tile,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device="cpu",
                verbose=False
            )[0]

            for box in result.boxes:

                x1, y1, bx2, by2 = box.xyxy[0].tolist()

                detections.append({
                    "class": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "box": [
                        x1 + x,
                        y1 + y,
                        bx2 + x,
                        by2 + y
                    ]
                })

    print(f"Total tiles: {tile_count}")

    return remove_duplicates(detections)


# ==================== EVALUATION ====================

def evaluate(ground_truth, predictions):

    candidates = []

    for gi, gt in enumerate(ground_truth):

        for pi, pred in enumerate(predictions):

            if gt["class"] != pred["class"]:
                continue

            overlap = calculate_iou(
                gt["box"],
                pred["box"]
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

    tp = len(used_gt)
    fp = len(predictions) - len(used_pred)
    fn = len(ground_truth) - len(used_gt)

    small_gt_indices = [
        i for i, gt in enumerate(ground_truth)
        if gt["small"]
    ]

    small_tp = sum(
        i in used_gt for i in small_gt_indices
    )

    small_fn = len(small_gt_indices) - small_tp

    precision = (
        tp / (tp + fp) if tp + fp else 0
    )

    recall = (
        tp / (tp + fn) if tp + fn else 0
    )

    small_recall = (
        small_tp / len(small_gt_indices)
        if small_gt_indices else 0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "Small GT": len(small_gt_indices),
        "Small TP": small_tp,
        "Small FN": small_fn,
        "Small Recall": small_recall
    }


# ==================== SAVE PREDICTIONS ====================

def save_prediction_image(image, predictions, model, output_path):

    output_image = image.copy()

    for detection in predictions:

        x1, y1, x2, y2 = map(
            int, detection["box"]
        )

        cls = detection["class"]
        confidence = detection["confidence"]

        label = (
            f"{model.names[cls]} {confidence:.2f}"
        )

        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output_image,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imwrite(str(output_path), output_image)


# ==================== PRINT RESULTS ====================

def print_results(name, results):

    print(f"\n--- {name} ---")

    for metric, value in results.items():

        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")

        else:
            print(f"{metric}: {value}")


# ==================== MAIN ====================

def main():

    for path in [MODEL_PATH, IMAGE_PATH, LABEL_PATH]:

        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError("Cannot read input image.")

    height, width = image.shape[:2]

    print(f"Original image: {width}x{height}")

    model = YOLO(str(MODEL_PATH))

    ground_truth = read_ground_truth(
        LABEL_PATH,
        width,
        height
    )

    print(f"Ground Truth Objects: {len(ground_truth)}")

    # Run both methods
    full_predictions = predict_full(model, image)

    tiled_predictions = predict_tiled(model, image)

    # Evaluate both methods
    full_results = evaluate(
        ground_truth,
        full_predictions
    )

    tiled_results = evaluate(
        ground_truth,
        tiled_predictions
    )

    # Save visual results
    save_prediction_image(
        image,
        full_predictions,
        model,
        OUTPUT_DIR / "full_image_prediction.jpg"
    )

    save_prediction_image(
        image,
        tiled_predictions,
        model,
        OUTPUT_DIR / "tiled_prediction.jpg"
    )

    # Print comparison
    print("\n========== FULL IMAGE VS TILING ==========")

    print_results("FULL IMAGE", full_results)

    print_results("IMAGE TILING", tiled_results)

    print("\n========== COMPARISON SUMMARY ==========")

    print(
        f"Full image detections: {len(full_predictions)}"
    )

    print(
        f"Tiled detections: {len(tiled_predictions)}"
    )

    print(
        f"Small Recall improvement: "
        f"{(tiled_results['Small Recall'] - full_results['Small Recall']) * 100:.2f} percentage points"
    )

    print(f"\nResults saved in: {OUTPUT_DIR}")

    print("\nCOMPARISON COMPLETED!")


if __name__ == "__main__":
    main()
