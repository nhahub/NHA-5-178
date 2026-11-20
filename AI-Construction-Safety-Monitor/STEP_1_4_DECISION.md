# Steps 1–4: prototype progression decision

**Decision:** proceed with a controlled video-processing prototype using the existing checkpoint while completing a small independent association review. Step 4 remains partial. Do not treat displayed PPE statuses as verified violations or use them for autonomous safety decisions.

| Step | Status | Evidence and limitation |
|---|---|---|
| 1. Dataset preparation | Complete for prototype use | Train/validation/test splits and structural checks exist. Labels may be incomplete or incorrect; scene leakage is not ruled out. |
| 2. Training | Baseline complete | Local YOLOv8n 30-epoch checkpoint exists. This does not establish deployment readiness. |
| 3. Evaluation | Complete for prototype progression | Validation at 320: mAP50 0.6815 and mAP50–95 0.4674; NoVest mAP50 0.5254. Human label verification remains partial. See `runs/PPE/step3_evaluation/final_review/final_20261118_01/FINAL_STEP3_REPORT.md`. |
| 4. Person/PPE association | Partial | Person detector, conservative geometry, local inference and eight synthetic tests exist. The four-image exploratory sample has no human-verified PPE ownership labels. |

## Retraining decision

**Not justified now.** The model has identifiable limitations—weak NoVest performance and 192/296 small objects detected in a separate fixed-threshold 320 analysis—but evaluation labels may be incomplete and the 640 aggregate results are lower than 320. These findings do not identify retraining as the cause-specific remedy. First distinguish missing or wrong detector outputs from correct outputs assigned to the wrong person. Reconsider targeted data work and training only if independently reviewed examples show a recurring detector failure, rather than label noise or association error, and an agreed target cannot be met through the existing baseline.

## Minimum additional Step 4 verification

Use the **four existing validation images and saved inference**, not a new dataset or full annotation campaign. Independently check five selected person cases: `00016/A1` (single person, a `MISSING` decision against provisional `UNKNOWN` vest evidence), `00025/B2` and `00025/B3` (overlapping workers), `00783/C3` (occluded group), and `00031/D2` (crowded scene). The IDs, provisional boxes and prior status comparisons are in `runs/PPE/step4_person_association/validation/review_20261118_01/person_annotations.csv` and `matched_results.csv`. These annotations are assistant-provisional and were not blinded to predictions.

For each case, a human should inspect the original image first and record whether the selected person box is acceptable and whether the helmet and vest are visibly PRESENT, MISSING or UNKNOWN; use UNKNOWN whenever the relevant area is obscured. Then inspect `inference/results.json`: use the matched predicted person ID and its `associated_ppe` entries, including their boxes and detection IDs, to record for each assigned item whether it belongs to that person, is incorrect, or cannot be judged. The saved annotated person images help locate IDs; use the PPE coordinates and existing Step 3 prediction renderings where applicable to identify evidence. Record decisions separately from the original CSV and labels. A human should also identify one clearly visible NoHelmet or NoVest case among these images **if one exists**; otherwise mark negative-case coverage unverified rather than infer absence. This is a small diagnostic gate, not statistically representative validation.

Separate the outcomes: a visible item with no suitable detector box is a **detection** miss; a suitable box allocated to a different person is an **association** error; a correct box rejected as ambiguous or below threshold is an **abstention**; a person not detected is a **person-detection** miss. Confirm false `MISSING` only when human-visible evidence establishes PPE is present. The one `00016/A1` MISSING against provisional UNKNOWN is an unresolved decision, not a proven false alert.

If a repeatable association defect is demonstrated, add a regression test for that exact scene geometry and make the smallest local fix; do not change thresholds based on this tiny convenience sample alone. If none is demonstrated, retain the existing conservative behavior and document the results. Five cases cannot prove reliable worker-level performance, so retain Step 4 PARTIAL until representative independently labelled scenarios support the intended claims.

## Step 5 boundary and next action

**Step 5 may begin** as an experimental pipeline that opens a local video, processes frames with the existing detector/association interface, and reports truthful processing errors and UNKNOWN states. It must not imply a missing detection proves missing PPE, present prototype results as safety violations, or claim reliable real-time monitoring. Measure runtime and validate video behavior separately. Safety rules, alerts and production deployment remain later work.

**Exact next action:** independently adjudicate the five cases above, including PPE ownership, and record the decisions in a separate review file. In parallel, the team may begin the bounded Step 5 video-input prototype; no training, annotation edits or checkpoint changes are needed first.
