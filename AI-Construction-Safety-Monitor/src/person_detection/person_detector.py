"""Local, image-only person detection using a verified Ultralytics checkpoint."""
from pathlib import Path


class PersonDetector:
    def __init__(self, weights, confidence=0.25, image_size=640, device="cpu"):
        from ultralytics import YOLO

        weights = Path(weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Local person checkpoint not found: {weights}")
        self.model = YOLO(str(weights))
        person_ids = [int(k) for k, v in self.model.names.items() if v == "person"]
        if len(person_ids) != 1:
            raise ValueError("Checkpoint must contain exactly one person class")
        self.person_class = person_ids[0]
        self.confidence, self.image_size, self.device = confidence, image_size, device

    def detect(self, image):
        result = self.model.predict(source=image, classes=[self.person_class],
                                    conf=self.confidence, imgsz=self.image_size,
                                    device=self.device, verbose=False, save=False)[0]
        return [{"person_id": i, "box": [float(x) for x in box.xyxy[0].tolist()],
                 "confidence": float(box.conf[0])}
                for i, box in enumerate(result.boxes, 1)]
