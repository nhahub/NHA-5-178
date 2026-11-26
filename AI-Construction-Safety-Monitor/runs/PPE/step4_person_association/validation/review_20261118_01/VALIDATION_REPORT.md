# Step 4 provisional association validation

**Status: PARTIAL.** Four existing validation images were selected for a close-up single-person case, multiple workers, a crowded scene and an overlapping/occluded scene. Existing demo outputs were on different `test_images`, so the existing demo was run once on the four selected validation images. No model training or test-split evaluation was performed.

## Annotation provenance and limits

`person_annotations.csv` contains nine manually drawn **assistant-provisional**, not human-verified, person boxes and visible helmet/vest statuses. The annotator used visual image evidence, not the absence of model detections, to select `PRESENT`; unclear or obscured PPE was marked `UNKNOWN`. The selected boxes are non-exhaustive, especially in crowded images. Coordinates are approximate, and the annotation was not blinded to the saved prediction visualizations. Thus this is exploratory triage, not an independent gold standard, a complete scene-level detector benchmark, or evidence of reliable worker-level association. Human review fields are explicitly marked `no`. There are no independently annotated PPE-box-to-person ownership labels.

## Measurements on the selected boxes only

Person matching uses one-to-one greedy IoU >= 0.50. All **9/9 selected provisional person boxes** had a matching predicted person. This is coverage of selected visible examples, not dataset recall or precision; many people in the crowded images were not annotated, and unmatched detections cannot be counted as false positives.

Among the nine matched people, there are 18 helmet/vest decisions: **12/18 UNKNOWN**. Of the 10 decisions with provisionally known visible status, **5 agree** and **5 are UNKNOWN**; none contradicts the provisional known status. This is status agreement, not proof that the assigned PPE detection belongs to the correct individual. No provisionally known PRESENT item received MISSING (0/10 known-status decisions), but one **MISSING** decision occurred where the provisional visible status was UNKNOWN; it cannot be assessed as true or false without independent review. There are no provisional known-MISSING examples, so negative-class behavior is not validated. `matched_results.csv` gives per-person results, and `metrics.json` gives denominators and per-image counts.

Correct and incorrect **physical PPE-to-person assignment counts are not measurable** from this sample: it has no independent PPE detection ownership annotations. Confident status agreement alone cannot establish association correctness, especially where two workers wear the same equipment. False MISSING is likewise only assessable for provisionally known PRESENT statuses, and is not established for the unknown example.

## Existing tests and code

All eight existing synthetic tests passed. No demonstrated implementation bug was isolated from this limited sample, so no code or tests were changed. Geometry thresholds remain uncalibrated; overlaps, detector failures, occlusion and class errors can still lead to wrong or UNKNOWN outcomes.

## Next action

Have a human annotator independently verify/correct a small labelled set with person boxes, per-person visible PPE status and explicit PPE-box ownership, including confident no-helmet/no-vest examples and crowded/occluded cases. Then score person matching, physical assignments, UNKNOWNs and false MISSING separately. Freeze the selection and criteria before assessing reliability. Keep source validation labels and the held-out test split unchanged.

To reproduce these exploratory calculations from this directory's saved inference and annotations, run from project root:

```powershell
.\.venv\Scripts\python.exe runs/PPE/step4_person_association/validation/review_20261118_01/evaluate_annotations.py
```

This command regenerates `metrics.json` and `matched_results.csv` in this validation directory; preserve those files or use a copy if their current state must remain unchanged. Inference settings and raw predictions are in `inference/results.json`.
