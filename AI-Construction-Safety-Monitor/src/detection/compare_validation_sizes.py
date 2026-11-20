
from ultralytics import YOLO
from pathlib import Path

# 1. Model path
model_path = Path(
    r"C:\Users\Hp\Documents\GitHub\NHA-5-178"
    r"\runs\detect\runs\PPE\balanced_30_epochs\weights\best.pt"
)

# 2. Dataset configuration
data_yaml = "dataset/processed/PPE/data_balanced.yaml"

# 3. Image sizes to compare
image_sizes = [320, 640]


def main():

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = YOLO(str(model_path))

    comparison_results = []

    for image_size in image_sizes:

        print("\n================================")
        print(f"VALIDATION WITH IMGSZ={image_size}")
        print("================================")

        results = model.val(
            data=data_yaml,
            split="val",
            imgsz=image_size,
            batch=4,
            device="cpu",
            workers=0,
            plots=False,
            project="runs/PPE",
            name=f"validation_imgsz_{image_size}",
            exist_ok=True
        )

        precision = results.box.mp
        recall = results.box.mr
        map50 = results.box.map50
        map50_95 = results.box.map

        comparison_results.append({
            "imgsz": image_size,
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50-95": map50_95
        })

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"mAP50: {map50:.4f}")
        print(f"mAP50-95: {map50_95:.4f}")

    print("\n========== FINAL COMPARISON ==========")

    for result in comparison_results:
        print(f"\nImage Size: {result['imgsz']}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"mAP50: {result['mAP50']:.4f}")
        print(f"mAP50-95: {result['mAP50-95']:.4f}")

    print("\nValidation comparison completed!")


if __name__ == "__main__":
    main()
