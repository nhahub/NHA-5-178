
from pathlib import Path
import cv2
from ultralytics import YOLO

# Model
MODEL_PATH = Path(
    r"C:\Users\Hp\Documents\GitHub\NHA-5-178"
    r"\runs\detect\runs\PPE\balanced_30_epochs\weights\best.pt"
)

# Original validation image
IMAGE_PATH = Path(
    "dataset/processed/PPE/valid/images/"
    "03295_jpg.rf.28082ec6b792bc923e0dcbe473c4551e.jpg"
)

# Settings
TILE_SIZE = 640
OVERLAP = 0.20
CONFIDENCE = 0.25
NMS_IOU = 0.50

OUTPUT_DIR = Path("runs/PPE/tiling_test")


def get_positions(length, tile_size, overlap):

    if length <= tile_size:
        return [0]

    stride = int(tile_size * (1 - overlap))

    positions = list(range(0, length - tile_size + 1, stride))

    last_position = length - tile_size

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


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


def class_aware_nms(detections):

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

            if overlap >= NMS_IOU:
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError("Could not read input image.")

    height, width = image.shape[:2]

    model = YOLO(str(MODEL_PATH))

    x_positions = get_positions(
        width, TILE_SIZE, OVERLAP
    )

    y_positions = get_positions(
        height, TILE_SIZE, OVERLAP
    )

    all_detections = []

    tile_count = 0

    for y in y_positions:

        for x in x_positions:

            x2 = min(x + TILE_SIZE, width)
            y2 = min(y + TILE_SIZE, height)

            tile = image[y:y2, x:x2]

            tile_count += 1

            print(
                f"Processing tile {tile_count}: "
                f"x={x}, y={y}, "
                f"width={tile.shape[1]}, "
                f"height={tile.shape[0]}"
            )

            result = model.predict(
                source=tile,
                imgsz=640,
                conf=CONFIDENCE,
                device="cpu",
                verbose=False
            )[0]

            for box in result.boxes:

                x1, y1, bx2, by2 = box.xyxy[0].tolist()

                # Convert tile coordinates to original image coordinates
                all_detections.append({
                    "class": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "box": [
                        x1 + x,
                        y1 + y,
                        bx2 + x,
                        by2 + y
                    ]
                })

    final_detections = class_aware_nms(all_detections)

    # Draw final detections
    output_image = image.copy()

    for detection in final_detections:

        x1, y1, x2, y2 = map(
            int, detection["box"]
        )

        cls = detection["class"]
        confidence = detection["confidence"]

        label = (
            f"{model.names[cls]} "
            f"{confidence:.2f}"
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

    output_path = OUTPUT_DIR / "tiled_prediction.jpg"

    cv2.imwrite(str(output_path), output_image)

    print("\n========== IMAGE TILING RESULTS ==========")
    print(f"Image size: {width}x{height}")
    print(f"Tiles processed: {tile_count}")
    print(f"Raw detections: {len(all_detections)}")
    print(f"Detections after NMS: {len(final_detections)}")

    print("\nPer-class detections:")

    class_counts = {}

    for detection in final_detections:

        cls = detection["class"]

        class_counts[cls] = class_counts.get(cls, 0) + 1

    for cls, count in sorted(class_counts.items()):
        print(f"{model.names[cls]}: {count}")

    print(f"\nSaved result: {output_path}")
    print("\nTILING TEST COMPLETED!")


if __name__ == "__main__":
    main()
