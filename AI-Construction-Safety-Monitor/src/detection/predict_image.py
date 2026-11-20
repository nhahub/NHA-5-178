
from ultralytics import YOLO
from pathlib import Path


def main():

    # 1. Load the trained model
    model_path = Path(
        r"C:\Users\Hp\Documents\GitHub\NHA-5-178"
        r"\runs\detect\runs\PPE\balanced_30_epochs\weights\best.pt"
    )

    model = YOLO(str(model_path))

    # 2. Select the new test image
    image_path = Path("test_images/construction.jpg")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 3. Run object detection
    results = model.predict(
        source=str(image_path),
        imgsz=320,
        conf=0.25,
        device="cpu",
        save=True,
        project="runs/PPE",
        name="new_image_prediction",
        exist_ok=False
    )

    # 4. Print detection results
    print("\n===== DETECTION RESULTS =====")

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            print(
                f"{class_name}: "
                f"Confidence = {confidence:.2f}"
            )

    print("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()
