
from ultralytics import YOLO
from pathlib import Path
import csv

# Model path
model_path = Path(
    r"C:\Users\Hp\Documents\GitHub\NHA-5-178"
    r"\runs\detect\runs\PPE\balanced_30_epochs\weights\best.pt"
)

# Input folder
input_dir = Path("test_images")

# Load model
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

if not input_dir.exists():
    raise FileNotFoundError(f"Image folder not found: {input_dir}")

model = YOLO(str(model_path))

# Compare image sizes
image_sizes = [320, 640]

# Store results
all_results = []

for image_size in image_sizes:

    print(f"\n========== TESTING IMGSZ={image_size} ==========")

    results = model.predict(
        source=str(input_dir),
        imgsz=image_size,
        conf=0.25,
        device="cpu",
        save=True,
        project="runs/PPE",
        name=f"image_size_{image_size}",
        exist_ok=True
    )

    for result in results:

        image_name = Path(result.path).name

        class_counts = {}
        confidences = []

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            class_counts[class_name] = (
                class_counts.get(class_name, 0) + 1
            )

            confidences.append(confidence)

        all_results.append({
            "image": image_name,
            "imgsz": image_size,
            "total_detections": len(result.boxes),
            "Helmet": class_counts.get("Helmet", 0),
            "NoHelmet": class_counts.get("NoHelmet", 0),
            "NoVest": class_counts.get("NoVest", 0),
            "Vest": class_counts.get("Vest", 0),
            "average_confidence": (
                sum(confidences) / len(confidences)
                if confidences else 0
            )
        })

# Save comparison CSV
output_file = Path("runs/PPE/image_size_comparison.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=all_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(all_results)

print("\n================================")
print("IMAGE SIZE COMPARISON COMPLETED")
print(f"Results saved to: {output_file}")
print("================================")
