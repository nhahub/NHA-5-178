
from ultralytics import YOLO
from pathlib import Path


def main():

    # ==============================
    # 1. Model
    # ==============================

    model = YOLO("yolov8n.pt")

    # ==============================
    # 2. Dataset
    # ==============================

    dataset_yaml = "dataset/processed/PPE/data_balanced.yaml"

    # ==============================
    # 3. Training Configuration
    # ==============================

    results = model.train(

        data=dataset_yaml,

        epochs=30,

        imgsz=320,

        batch=4,

        device="cpu",

        workers=0,

        project="runs/PPE",

        name="balanced_30_epochs",

        exist_ok=False,

        patience=10,

        save=True,

        plots=True,

        seed=42,

        verbose=True
    )

    # ==============================
    # 4. Training Completed
    # ==============================

    print("\n================================")
    print("BALANCED TRAINING COMPLETED!")
    print("================================")

    print("Model: YOLOv8n")
    print("Dataset: Balanced PPE Dataset")
    print("Epochs: 30")
    print("Image Size: 320")
    print("Batch Size: 4")
    print("Device: CPU")

    print("\nCheck the following folder for results:")
    print("runs/PPE/balanced_30_epochs")


if __name__ == "__main__":
    main()
