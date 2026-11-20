"""Frame sampling and processing for a file-based experimental video pipeline."""
import math
import time

import cv2


def sample_indices(source_fps, target_fps):
    """Yield whether each successive constant-rate source frame is sampled."""
    if not all(math.isfinite(x) and x > 0 for x in (source_fps, target_fps)):
        raise ValueError("Frame rates must be finite and positive")
    target_fps = min(source_fps, target_fps)
    index = 0
    next_time = 0.0
    while True:
        timestamp = index / source_fps
        selected = timestamp + 1e-9 >= next_time
        if selected:
            next_time = (math.floor((timestamp + 1e-9) * target_fps) + 1) / target_fps
        yield selected
        index += 1


def process_capture(capture, writer, infer, source_fps, target_fps, frame_count=0, expected_size=None):
    """Process decoded frames. Timestamps are index/FPS estimates, not VFR PTS."""
    selector = sample_indices(source_fps, target_fps)
    frames, errors = [], []
    index = 0
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            if frame_count and index < frame_count:
                errors.append({"frame_index": index, "timestamp_seconds": index/source_fps,
                               "error": "Read failed before reported frame count; video may be truncated"})
            elif not frame_count:
                errors.append({"frame_index": index, "timestamp_seconds": index/source_fps,
                               "error": "End of stream or read failure: frame count unavailable"})
            break
        selected = next(selector)
        timestamp = index / source_fps
        if (frame is None or not hasattr(frame, "shape") or frame.size == 0
                or len(frame.shape) != 3 or frame.shape[2] != 3
                or (expected_size is not None and frame.shape[:2] != expected_size)):
            errors.append({"frame_index": index, "timestamp_seconds": timestamp,
                           "error": "Decoder returned an invalid frame"})
            index += 1
            continue
        if selected:
            item = {"frame_index": index, "timestamp_seconds": timestamp,
                    "status": "ok", "people": None, "ppe_detections": None, "error": None}
            visual = frame.copy()
            try:
                people, detections = infer(frame)
                item["people"], item["ppe_detections"] = people, detections
                for person in people:
                    x1, y1, x2, y2 = map(int, person["person_box"])
                    cv2.rectangle(visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"P{person['person_id']} H:{person['helmet_status']} V:{person['vest_status']}"
                    cv2.putText(visual, text, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,0), 1)
            except Exception as exc:
                item.update(status="error", people=None, ppe_detections=None,
                            error=f"{type(exc).__name__}: {exc}")
                errors.append({"frame_index": index, "timestamp_seconds": timestamp, "error": item["error"]})
                cv2.putText(visual, "INFERENCE ERROR", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
            writer.write(visual)
            frames.append(item)
        index += 1
    return {"frames": frames, "errors": errors, "decoded_frames": index,
            "elapsed_seconds": time.perf_counter()-started}
