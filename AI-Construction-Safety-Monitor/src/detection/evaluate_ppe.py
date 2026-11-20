
from ultralytics import YOLO
from pathlib import Path


def main():

    # ==============================
    # 1. Load the best trained model
    # ==============================

    model_path = (
    r"C:\Users\Hp\Documents\GitHub\NHA-5-178"
    r"\runs\detect\runs\PPE\balanced_30_epochs\weights\best.pt"
)

    model = YOLO(model_path)

    # ==============================
    # 2. Evaluate on Test Dataset
    # ==============================

    results = model.val(
        data="dataset/processed/PPE/data_balanced.yaml",
        split="test",
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project="runs/PPE",
        name="test_evaluation",
        exist_ok=False
    )

    # ==============================
    # 3. Print Evaluation Results
    # ==============================

    print("\n================================")
    print("TEST EVALUATION COMPLETED!")
    print("================================")

    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    print("\nEvaluation results saved successfully.")


if __name__ == "__main__":
    main()
