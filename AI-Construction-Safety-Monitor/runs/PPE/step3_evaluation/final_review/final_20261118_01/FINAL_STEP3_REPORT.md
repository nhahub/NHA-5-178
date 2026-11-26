# Final Step 3 report

**Decision: COMPLETE FOR PROTOTYPE PROGRESSION**, with unresolved annotation-quality limitations. The existing model is suitable as an experimental input for the next prototype stage, not for autonomous safety decisions or claims of reliable worker-level compliance.

| Area | Status | Evidence and limits |
|---|---|---|
| Model evaluation | DONE | Saved 320 and 640 validation runs provide aggregate and per-class results, settings, hashes and versions. No test-set evaluation was performed in this audit. |
| Automated dataset integrity | DONE | Saved whole-split integrity checks and the 52 reported box-boundary records were examined. These checks cannot establish semantic annotation quality. |
| Human annotation verification | PARTIAL | Visual triage was performed by an automated assistant, not a human annotator. The 19 images are a convenience sample, not a representative annotation audit; no labels have been verified comprehensively. |

## Boundary audit — all 52 records

Every saved boundary record was recalculated from its normalized label coordinates and original image dimensions. Results: **52 minor rounding**, **0 suspicious** and **0 clear invalid coordinates** under the geometry-only decision rule: maximum edge overflow at most 0.5 pixel = minor rounding; above 0.5 to 2 pixels = suspicious; above 2 pixels = clearly outside the image. All 52 overflow by approximately **0.005 pixel**, across 40 training, 11 validation and one test box. This is consistent with rounding at an image boundary; it is not proof of how the annotations were generated or that their class and object extent are correct. The thresholds are explicit review heuristics, not dataset annotation policy. Every label filename, line, image dimensions, overflow and classification is preserved in `boundary_classification.csv`. Original labels were not changed.

## Nineteen-image visual triage

All 19 original/ground-truth/prediction triplets and the existing CSV were inspected using the four contact sheets in this directory. The individual image observations and priorities are in `visual_case_findings.csv`. Box classes, ambiguity and distant objects cannot always be resolved from contact sheets; these are provisional observations, not human verification.

- **00031, 00048, 00308, 03220 and 03295:** visible crowds with comparatively sparse ground-truth boxes suggest incomplete labelling, especially for helmets. Unlabelled targets could be counted as false positives; verify the intended annotation scope before making a definitive claim.
- **00024 and 00025:** visible equipment on additional workers appears not to be labelled; this again depends on the annotation policy.
- **00269:** two NoHelmet labels appear to be over people wearing hard hats. This is a high-priority possible class error, requiring inspection at original resolution and confirmation of class policy.
- **03214:** the two ground-truth vest boxes have no visible 320 prediction in the saved rendering, consistent with missed detections in a difficult tunnel scene.
- **03178:** its edge overflow is only 0.005 pixel; distant workers remain difficult to detect. The historical tiling view for 03295 has unverified provenance and is not evidence of tiling efficacy.

The existing missed-object CSV contains 63 candidate misses across 40 images; the annotation-audit CSV is a count summary, not a semantic annotation audit. Neither proves completeness or error rates. The pack includes crowd and low-light/tunnel examples, but selection is not statistically representative and no quantitative prevalence may be inferred.

## Saved model evaluation verified

The current checkpoint SHA-256 matches both saved reports: `949868ad6938a30ac1b380c25e0ceb6eebb17ece9fbf0a00953b453143503f23`. The balanced YAML SHA-256 also matches the 320 report: `7dd7c60d4e48c561d4c7ed31ad380f95d327d6cb484679874953d48ac442f448`. Both saved evaluations use validation only, 343 images/829 labelled instances, CPU, batch 4, workers 0, confidence 0.001 and NMS IoU 0.7; versions: Python 3.10.0, Ultralytics 8.4.202, PyTorch 2.14.1+cpu, OpenCV 5.0.0. The 640 report references the same checkpoint hash and staged validation YAML. This audit checked saved results and hashes; it did not rerun inference.

| imgsz | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| 320 | 0.6243 | 0.6949 | 0.6815 | 0.4674 |
| 640 | 0.5584 | 0.6439 | 0.6158 | 0.4067 |

| Class | 320 precision | 320 recall | 320 mAP50 | 320 mAP50–95 | 640 mAP50 |
|---|---:|---:|---:|---:|---:|
| Helmet | 0.5496 | 0.6727 | 0.6217 | 0.4303 | 0.5822 |
| NoHelmet | 0.6523 | 0.7572 | 0.7380 | 0.5117 | 0.7007 |
| NoVest | 0.5071 | 0.5491 | 0.5254 | 0.3489 | 0.4541 |
| Vest | 0.7880 | 0.8003 | 0.8409 | 0.5786 | 0.7263 |

The 320 per-class CSV agrees with the 320 JSON. The fixed-threshold evaluator at 320, confidence 0.25 and class-aware greedy matching IoU 0.5 reports 649 TP, 581 FP, 180 FN and 192/296 small objects detected (64.86% recall); small means normalized area below 0.01. These results are not Ultralytics mAP or the same operating-point precision/recall, and imperfect labels affect both. The previously reported 640 small-object 233/296 is historical and not reproduced by this saved 320 result.

## Prototype suitability and limitations

The existing data, model, and validation results support **controlled prototype development**, using 320 as the documented baseline. NoVest is notably weak, and visual evidence suggests incomplete labels and possible class mistakes; metrics should be interpreted as relative to the existing annotations, not ground-truth certainty. Person association must represent ambiguous, occluded or missing PPE evidence as unknown; missing a detection does not prove a violation. Image-based evaluation does not establish video or event-level performance. Exact and perceptual-hash split checks did not flag cross-split matches, but related scenes and source-video leakage are not ruled out. The test split remains held out.

For traceability, prior artifacts are in `../../review_20261118_01/`: `dataset_integrity.json`, `evaluation.json`, `evaluation_640.json`, `per_class_metrics.csv`, `small_object_fixed_threshold.txt`, `review_manifest.json` and `visual_review/review.csv`. This report does not modify those artifacts, images, annotations, splits or model weights.
