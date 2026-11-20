
from ultralytics import YOLO
from pathlib import Path


def main():

    # 1. Model path
    model_path = Path(
        r"C:\Users\Hp\Documents\GitHub\NHA-5-178"
        r"\runs\detect\runs\PPE\balanced_30_epochs\weights\best.pt"
    )

    # 2. Input folder
    input_dir = Path("test_images")

    # 3. Output folder
    output_dir = Path("runs/PPE/batch_test")

    # Check model and images
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {input_dir}")

    image_extensions = {".jpg", ".jpeg", ".png"}

    images = [
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    if not images:
        raise ValueError("No images found in test_images folder!")

    print(f"Found {len(images)} images.")

    # 4. Load trained model
    model = YOLO(str(model_path))

    # 5. Run batch prediction
    results = model.predict(
        source=str(input_dir),
        imgsz=320,
        conf=0.25,
        device="cpu",
        save=True,
        project="runs/PPE",
        name="batch_test",
        exist_ok=True
    )

    # 6. Print results for every image
    print("\n========== BATCH DETECTION RESULTS ==========")

    for result in results:

        print(f"\nImage: {Path(result.path).name}")

        detection_count = 0

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            detection_count += 1

            print(
                f"  {class_name}: "
                f"Confidence = {confidence:.2f}"
            )

        print(f"Total detections: {detection_count}")

    print("\n============================================")
    print("BATCH PREDICTION COMPLETED!")
    print(f"Results saved in: {output_dir}")
    print("============================================")


if __name__ == "__main__":
    main()
