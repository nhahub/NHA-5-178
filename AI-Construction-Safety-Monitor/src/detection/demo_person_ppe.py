"""Image-only, local-checkpoint demonstration. Run from project root."""
import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from person_detection.person_detector import PersonDetector
from monitoring.association import associate_ppe

EXPECTED = {0: "Helmet", 1: "NoHelmet", 2: "NoVest", 3: "Vest"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New directory under runs/PPE/step4_person_association")
    parser.add_argument("--person-weights", type=Path, default=PROJECT / "yolov8n.pt")
    parser.add_argument("--ppe-weights", type=Path, default=PROJECT.parent / "runs/detect/runs/PPE/balanced_30_epochs/weights/best.pt")
    args = parser.parse_args()
    root = (PROJECT / "runs/PPE/step4_person_association").resolve()
    out = args.output.resolve()
    if out == root or root not in out.parents:
        parser.error("Output must be a new subdirectory under runs/PPE/step4_person_association")
    for path in [args.person_weights, args.ppe_weights, *args.images]:
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    person_detector = PersonDetector(args.person_weights)
    ppe_model = YOLO(str(args.ppe_weights.resolve()))
    if ppe_model.names != EXPECTED:
        raise ValueError(f"Unexpected PPE class mapping: {ppe_model.names}")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"person_weights": str(args.person_weights.resolve()),
                "ppe_weights": str(args.ppe_weights.resolve()),
                "settings": {"person_imgsz": 640, "ppe_imgsz": 320, "detection_conf": 0.25,
                             "association_conf": 0.5, "region_overlap": 0.6, "ambiguity_margin": 0.15},
                "images": []}
    for number, path in enumerate(args.images, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        people = person_detector.detect(image)
        result = ppe_model.predict(source=image, imgsz=320, conf=0.25, device="cpu", verbose=False, save=False)[0]
        detections = [{"class_name": EXPECTED[int(b.cls[0])], "confidence": float(b.conf[0]),
                       "box": [float(x) for x in b.xyxy[0].tolist()]} for b in result.boxes]
        associations = associate_ppe(people, detections)
        visual = image.copy()
        for person in associations:
            x1, y1, x2, y2 = map(int, person["person_box"])
            cv2.rectangle(visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"P{person['person_id']} H:{person['helmet_status']} V:{person['vest_status']}"
            cv2.putText(visual, text, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1)
        name = f"{number:03d}_{path.stem}.jpg"
        if not cv2.imwrite(str(out / name), visual):
            raise OSError(f"Could not write {name}")
        manifest["images"].append({"source": str(path.resolve()), "visualization": name,
                                   "person_detections": people, "ppe_detections": detections,
                                   "associations": associations})
    (out / "results.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest['images'])} image results to {out}")


if __name__ == "__main__":
    main()
