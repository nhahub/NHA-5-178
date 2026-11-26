# Step 3: model evaluation and label verification

Status: **PARTIAL**. Automated evaluation and structural checks are complete; label reliability cannot be certified until the visual decisions and boundary-box review are completed.

## Dataset integrity and annotation review

| Split | Images | Labels | Missing labels | Orphan labels | Invalid rows | Box edges outside image | Unreadable images |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1600 | 1600 | 0 | 0 | 0 | 40 | 0 |
| Validation | 343 | 343 | 0 | 0 | 0 | 11 | 0 |
| Test | 343 | 343 | 0 | 0 | 0 | 1 | 0 |

There were no empty label files. Invalid rows cover field count, class, finite normalized coordinates and positive width/height; edge violations use tolerance 1e-6. Edge violations can be minor rounding or genuine annotation defects; inspect them before deciding. Detailed file, line and label values are in `dataset_integrity.json`. Existing validator checks individual coordinates but not all box edges.

No exact cross-split image SHA-256 matches or cross-split perceptual-hash pairs within Hamming distance 5 were found. Perceptual hashes use grayscale 32x32 DCT and the 8x8 low-frequency median; they are a heuristic and are not proof of scene independence. The installed environment lacked ImageHash, so the original pHash checker was not run. Scene/video leakage remains unverified. The test split was inspected structurally but not evaluated or used for tuning.

Existing missed-object CSV: 63 rows over 40 images, class counts Helmet 23, NoHelmet 17, NoVest 11, Vest 12. These were generated historically at 640 and are candidates, not verified errors. The existing annotation audit counts labels and does not inspect visual completeness.

## Reproduced validation results

Both runs use the same existing checkpoint, validation split and Ultralytics 8.4.202, Python 3.10.0, PyTorch 2.14.1+cpu, OpenCV 5.0.0; CPU, batch 4, workers 0, validation confidence 0.001, NMS IoU 0.7. Staged copies avoid modifying source labels/cache. Checkpoint SHA-256: `949868ad6938a30ac1b380c25e0ceb6eebb17ece9fbf0a00953b453143503f23`.

| Image size | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 320 | 0.6243 | 0.6949 | 0.6815 | 0.4674 |
| 640 | 0.5584 | 0.6439 | 0.6158 | 0.4067 |

There are 829 validation instances. Per-class values and complete settings are in `evaluation.json`, `per_class_metrics.csv`, and `evaluation_640.json`. Precision and recall are Ultralytics operating-point outputs; they should not be equated with the fixed-threshold counts below. These metrics depend on current labels and are not proof of real-world performance.

The existing all-object error evaluator was run at 320, confidence 0.25, class-aware greedy IoU matching at 0.50. It reported TP 649, FP 581, FN 180; 192/296 small objects found (64.86%). Small means normalized ground-truth box area < 0.01. These fixed-threshold results are not mAP. Its matching is greedy, not a globally optimal assignment. `compare_small_objects.py` matches only small labels and can allocate predictions differently. Full output: `small_object_fixed_threshold.txt`.

## Visual review pack

`visual_review/` contains 19 image folders with source copies, annotated ground truth and newly generated 320/confidence 0.25 full-image predictions. One folder also contains the existing historical tiled image, whose generating settings/checkpoint were not independently verified. Box IDs G1... and P1... identify separate ground-truth and prediction boxes; they do not assert a match. `visual_review/review.csv` has 143 prefilled box rows and blank suspected-issue, decision and notes fields. Add rows for visible objects not represented by boxes. Record cross-references between G and P IDs in notes.

Selection is a convenience sample: initial missed rows from each class, highest-count missed images, and six images absent from the missed queue. It is not statistically representative; the six are not automatically ordinary, and low-light, occlusion and crowding coverage must be manually confirmed and expanded if absent. See `review_manifest.json`. In particular inspect 03295 for apparent unlabeled objects and 03178 for its out-of-bounds box; inspect all 11 validation boundary entries listed in `dataset_integrity.json`. Human review has not yet occurred.

Suggested decisions: correct label, missing label, wrong class, box adjustment, true detector miss, spurious/duplicate prediction, uncertain. Define visibility and class semantics before judging; do not change labels as part of this review. For each frame, first inspect original and ground truth, then predictions and, if relevant, the historical tile view. Record condition tags and ambiguous cases.

## Step 3 checklist

- DONE — Check train/validation/test image-label pairing, structural label values, image readability and box edges.
- DONE — Check exact image duplicates and pHash candidates across splits, within stated heuristic limitations.
- DONE — Validate current checkpoint at 320 and 640 on validation only and record aggregate/per-class values, settings, hashes and versions.
- DONE — Run existing all-object fixed-threshold error analysis after inspecting matching logic.
- DONE — Prepare a 19-image visual-review pack and blank human decision fields.
- PARTIAL — Representative coverage: targeted convenience sample exists; lighting, occlusion, crowding and ordinary-scene tags need confirmation and supplementation.
- PARTIAL — Label verification: human review of missed, apparent false-positive and 11 validation boundary cases is outstanding; source/scene leakage is not excluded.
- BLOCKED — Claim that validation annotations are reliable: requires human adjudication, documented class policy and follow-up on identified errors.

## Commands and next actions

From project root, open `visual_review/review.csv` and the relevant image folders with your preferred image viewer. Record human decisions; do not modify source data. Use `dataset_integrity.json` to locate all boundary violations. Agree the annotation policy, complete the sample and scene-category review, then decide whether any corrections need a separately versioned dataset and rerun evaluation against that version. Keep the test split untouched for tuning.

To reproduce the primary pipeline in a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -u runs/PPE/step3_evaluation/run_step3.py runs/PPE/step3_evaluation/review_NEW_UNIQUE_ID
```

The script refuses an existing output directory and stages validation files; it requires installed OpenCV, NumPy, PyTorch, Ultralytics, Pillow and PyYAML. It loads the existing checkpoint and performs inference only. The separate 640 script in this report directory refers to this run's staged validation; use a fresh output location for any new comparison to avoid overwrites.
