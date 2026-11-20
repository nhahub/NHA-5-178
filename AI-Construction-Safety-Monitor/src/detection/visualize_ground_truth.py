
from pathlib import Path
import cv2

IMAGE_PATH = Path(
    "dataset/processed/PPE/valid/images/"
    "03295_jpg.rf.28082ec6b792bc923e0dcbe473c4551e.jpg"
)

LABEL_PATH = Path(
    "dataset/processed/PPE/valid/labels/"
    "03295_jpg.rf.28082ec6b792bc923e0dcbe473c4551e.txt"
)

OUTPUT_DIR = Path("runs/PPE/tiling_comparison")


def main():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)

    if not LABEL_PATH.exists():
        raise FileNotFoundError(LABEL_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError("Could not read image.")

    height, width = image.shape[:2]

    class_names = {
        0: "Helmet",
        1: "NoHelmet",
        2: "NoVest",
        3: "Vest"
    }

    count = 0

    with open(LABEL_PATH, encoding="utf-8") as file:

        for line in file:

            parts = line.split()

            if len(parts) != 5:
                continue

            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            # Convert normalized YOLO coordinates to pixels
            x1 = int((xc - w / 2) * width)
            y1 = int((yc - h / 2) * height)
            x2 = int((xc + w / 2) * width)
            y2 = int((yc + h / 2) * height)

            color = (255, 0, 0)  # Blue in OpenCV

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = class_names.get(cls, str(cls))

            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            count += 1

    output_path = OUTPUT_DIR / "ground_truth_visualization.jpg"

    cv2.imwrite(str(output_path), image)

    print("\n========== GROUND TRUTH VISUALIZATION ==========")
    print(f"Image dimensions: {width}x{height}")
    print(f"Ground Truth Objects: {count}")
    print(f"Saved to: {output_path}")
    print("\nVisualization completed!")


if __name__ == "__main__":
    main()
