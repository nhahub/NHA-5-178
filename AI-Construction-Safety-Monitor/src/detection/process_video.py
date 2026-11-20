"""Process a local video with the existing person/PPE image prototype."""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from monitoring.association import associate_ppe
from monitoring.video import process_capture
from person_detection.person_detector import PersonDetector

EXPECTED = {0: "Helmet", 1: "NoHelmet", 2: "NoVest", 3: "Vest"}


def run(video, out, target_fps, person_weights, ppe_weights):
    from ultralytics import YOLO

    report = {"input": str(video.resolve()), "status": "error", "errors": [], "frames": [],
              "settings": {"target_fps": target_fps, "person_imgsz": 640, "ppe_imgsz": 320,
                           "detector_confidence": 0.25, "association_confidence": 0.5},
              "weights": {"person": str(person_weights.resolve()), "ppe": str(ppe_weights.resolve())},
              "timestamp_method": "source frame index / reported FPS; approximate for variable-frame-rate video"}
    capture = writer = None
    try:
        if not video.is_file():
            raise FileNotFoundError(f"Video not found: {video}")
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("Target FPS must be finite and positive")
        for weights in (person_weights, ppe_weights):
            if not weights.is_file():
                raise FileNotFoundError(f"Local checkpoint not found: {weights}")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError("Cannot open or decode video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count_raw = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        count = int(count_raw) if math.isfinite(count_raw) and count_raw > 0 else 0
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video has invalid FPS or frame dimensions")
        output_fps = min(fps, target_fps)
        report["source"] = {"fps": fps, "width": width, "height": height, "reported_frames": count}
        report["output_video"] = {"path": "annotated.mp4", "fps": output_fps,
                                  "timing": "sampled frames at fixed output FPS; approximate for irregular sampling/errors"}
        person = PersonDetector(person_weights)
        ppe = YOLO(str(ppe_weights.resolve()))
        if ppe.names != EXPECTED:
            raise ValueError(f"Unexpected PPE class mapping: {ppe.names}")
        writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
        if not writer.isOpened():
            raise OSError("Cannot open output video writer")

        def infer(frame):
            people = person.detect(frame)
            result = ppe.predict(source=frame, imgsz=320, conf=0.25, device="cpu", verbose=False, save=False)[0]
            detections = [{"class_name": EXPECTED[int(b.cls[0])], "confidence": float(b.conf[0]),
                           "box": [float(x) for x in b.xyxy[0].tolist()]} for b in result.boxes]
            return associate_ppe(people, detections), detections

        result = process_capture(capture, writer, infer, fps, target_fps, count, (height, width))
        report.update(result)
        if not result["frames"]:
            report["errors"].append({"error": "No valid sampled frames could be processed"})
        successful = any(frame["status"] == "ok" for frame in result["frames"])
        report["status"] = "error" if not successful else ("partial" if report["errors"] else "complete")
    except Exception as exc:
        report["errors"].append({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--fps", type=float, default=2.0, help="Frames to process per source second (default 2)")
    parser.add_argument("--output", required=True, type=Path, help="New directory under runs/PPE/step5_video")
    parser.add_argument("--person-weights", type=Path, default=PROJECT / "yolov8n.pt")
    parser.add_argument("--ppe-weights", type=Path, default=PROJECT.parent / "runs/detect/runs/PPE/balanced_30_epochs/weights/best.pt")
    args = parser.parse_args()
    root = (PROJECT / "runs/PPE/step5_video").resolve()
    out = args.output.resolve()
    if out == root or root not in out.parents:
        parser.error("Output must be a new subdirectory under runs/PPE/step5_video")
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists: {out}")
    report = run(args.video, out, args.fps, args.person_weights, args.ppe_weights)
    print(f"Status: {report['status']}; results: {out / 'results.json'}")
    for error in report["errors"]:
        print(f"Error: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
