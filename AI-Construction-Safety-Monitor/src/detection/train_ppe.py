
from ultralytics import YOLO

def main():
    # Load a lightweight YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Start a short CPU training test
    results = model.train(
        data="dataset/processed/PPE/data.yaml",
        epochs=1,
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        project="runs/PPE",
        name="first_test",
        exist_ok=True
    )

    print("Training test completed successfully!")

if __name__ == "__main__":
    main()
