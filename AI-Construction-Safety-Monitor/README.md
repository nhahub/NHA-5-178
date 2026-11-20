
# AI Construction Safety Monitor

An AI-powered system for monitoring Personal Protective Equipment (PPE) compliance at construction sites using Computer Vision.

## Project Overview

The system aims to detect workers and identify whether they are wearing the required safety equipment.

## PPE Detection Classes

| Class ID | Class Name |
|---|---|
| 0 | Helmet |
| 1 | NoHelmet |
| 2 | NoVest |
| 3 | Vest |

## Dataset

**Dataset:** Personal Protective Equipment (PPE)

**Source:** Mendeley Data

https://data.mendeley.com/datasets/zkzghjvpn2/6

- Total images: 2,286
- Original classes: 4
- Annotation format: YOLO

Dataset images are maintained locally and are not included in this repository.

## Technology Stack

- Python
- YOLOv8n
- OpenCV
- PyTorch
- FastAPI
- MySQL
- React

## Project Structure

- `dataset/`: Dataset configuration and local data.
- `src/`: Computer Vision implementation.
- `models/`: Trained model weights.
- `notebooks/`: Experiments and analysis.
- `backend/`: Backend API.
- `frontend/`: Dashboard.
- `tests/`: Testing scripts.

## Implementation Plan

1. Dataset Preparation — **complete for prototype use**; annotation quality and scene leakage remain open.
2. YOLO Model Training — **baseline complete**; no retraining justified yet.
3. Model Evaluation — **complete for prototype progression**; label verification remains partial.
4. Person Detection and Association — **partial**; image prototype and synthetic tests exist, but PPE ownership needs independent human verification.
5. Video Processing — **partial**; local-file prototype, focused tests and one sample run exist; broader video validation remains pending.
6. Safety Rules — **partial**; experimental PPE incident rules and tests exist; event-level validation remains pending.
7. Backend and Database — not started.
8. Frontend Integration — not started.
9. Final System Integration — not started.

## Project Status

Dataset preparation, YOLOv8n training, and validation evaluation have been completed for prototype progression. Step 3 includes automated dataset checks and a reproducible validation baseline; human annotation verification remains partial, with suspected incomplete and potentially incorrect labels. Step 4 has an initial person detection and PPE association prototype. The current model is experimental and does not establish reliable worker-level compliance.

The [Steps 1–4 decision report](STEP_1_4_DECISION.md) describes the smallest outstanding review and the limits for beginning video processing.

## Person and PPE association prototype

Step 4 has an image-only prototype using the local COCO `yolov8n.pt` person class and the existing four-class PPE checkpoint. Head and torso regions are geometric approximations. Missing, low-confidence, conflicting or ambiguously assigned evidence can yield `UNKNOWN`; `MISSING` requires explicit associated NoHelmet or NoVest evidence. Thresholds are experimental and worker-level accuracy has not been validated.

From the project root, run the unit tests and an image demo using a new output directory:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_association.py' -v
.\.venv\Scripts\python.exe src/detection/demo_person_ppe.py test_images/construction.jpg --output runs/PPE/step4_person_association/demo_NEW_ID
```

The demo writes image results and `results.json` to the new directory and refuses to reuse an existing directory. Local checkpoints are required; their paths can be supplied using `--person-weights` and `--ppe-weights`. This is not video monitoring or an alerting system.

## Experimental local-video processing

The file-based runner samples source frames at a configurable rate and applies the same person detector, PPE detector and association logic. It writes `annotated.mp4` and `results.json` to a **new** directory under `runs/PPE/step5_video/`; failed frames contain errors rather than PPE decisions. Person IDs are per frame and do not track workers. No alerts are implemented.

```powershell
.\.venv\Scripts\python.exe src/detection/process_video.py path\to\sample.mp4 --fps 2 --output runs/PPE/step5_video/sample_NEW_ID
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_video.py' -v
```

`--fps` specifies samples per source second, capped at the source frame rate; it does not promise processing speed. Timestamps are estimated from frame index and reported FPS, so variable-frame-rate input may be inaccurate. The annotated video contains sampled frames at a fixed output FPS; its timing is approximate if sampling intervals vary or frames are dropped. The JSON records per-frame timestamps, detections, association evidence and errors. Check the final status and errors before interpreting results. Local weights may be supplied with `--person-weights` and `--ppe-weights`; output requires an available OpenCV MP4 codec. One sample clip has been processed; representative video validation is pending.

## Experimental PPE safety rules

Apply the rules as a separate step to an existing video `results.json`; this preserves the original detections and does not rerun inference:

```powershell
.\.venv\Scripts\python.exe src/safety_rules/evaluate_ppe_rules.py runs/PPE/step5_video/sample_01/results.json --output runs/PPE/step5_video/rules_NEW_ID
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_ppe_rules.py' -v
```

`incidents.json` contains unverified model-derived candidates, not confirmed violations. A candidate requires an associated `MISSING` status and explicit matching NoHelmet/NoVest evidence of at least 0.5 confidence; UNKNOWN, explanations signalling uncertainty and conflicting confident evidence suppress it. Consecutive sampled frames are matched only when their person boxes have a unique mutual overlap of IoU at least 0.5. Each PPE type is reported once during a contiguous missing-evidence episode. Frame errors, gaps, absent people, uncertainty or ambiguous matches reset continuity; the same person may then create a new candidate. Box overlap is not identity tracking and can incorrectly split or merge people, so event-level validation is still required.
