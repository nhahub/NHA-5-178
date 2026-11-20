# Step 5 video processing and safety rules: final quality review

**Handoff recommendation:** suitable for a controlled AI-team prototype handoff with the limitations below; not established as reliable for real-world PPE decisions or alerts. Keep all incidents as unverified candidates and preserve UNKNOWN. Passing software tests does not establish model, association or incident accuracy.

## What works and evidence

- [`src/detection/process_video.py`](src/detection/process_video.py) loads local checkpoints, samples video frames, applies person/PPE inference and association, and records detections, confidence and per-frame errors. [`src/monitoring/video.py`](src/monitoring/video.py) estimates timestamps as source frame index divided by reported FPS. Missing/unreadable video, invalid metadata, invalid frames and inference errors are surfaced; an inference failure records `people: null` rather than PPE statuses. A run with no successful frames is `error`; a run with recorded errors is `partial`.
- The saved [`sample_01/results.json`](runs/PPE/step5_video/sample_01/results.json) reports 206 decoded frames at 25 FPS, 17 sampled frames at requested 2 FPS and zero errors. Its saved [`annotated.mp4`](runs/PPE/step5_video/sample_01/annotated.mp4) was previously decoded successfully for all 17 frames at 2 FPS. The indices 0, 13, 25, 38 correspond to 0.00, 0.52, 1.00, 1.52 seconds under the stated constant-FPS approximation. The clip is one example, not a real-time benchmark.
- [`src/safety_rules/ppe_rules.py`](src/safety_rules/ppe_rules.py) requires associated `MISSING`, explicit NoHelmet/NoVest evidence at confidence >=0.5, no uncertainty explanation and no conflicting confident positive evidence. PRESENT, UNKNOWN, ambiguous/conflicting evidence and failed frames do not create incidents in the tested paths. Results remain separate from incident candidates.
- Saved [`incidents.json`](runs/PPE/step5_video/safety_rules_demo_01/incidents.json) contains one unverified vest candidate. The [visual review note](runs/PPE/step5_video/safety_rules_demo_01/visual_review.md) records the user's visual confirmation for the first frame only; the screenshot was not supplied with that documentation update. This is not a confirmed real-world violation or evidence of general event accuracy.

## Tests and fix

Before changes: full suite **20/20 passed**; focused video **5/5 passed** and safety rules **6/6 passed**. Three added regression tests then reproduced failures: a supposedly complete report with no frames was accepted; a frame marked failed inside an otherwise complete report still yielded complete status; and a frame with an indexed processing error could yield an incident and complete status.

The targeted fix in `src/safety_rules/ppe_rules.py` rejects a missing/empty frame list, marks rule evaluation partial for source errors or failed/invalid frames, and resets/skips a frame with a recorded error at that index. `tests/test_ppe_rules.py` adds regression tests for all three scenarios. After the fix: **23/23 full-suite tests passed**, focused video **5/5 passed**, focused rules **9/9 passed**. Re-evaluating the existing saved 17-frame JSON in memory produced the same single incident as the saved output; no prior artifact was overwritten. No model inference or training was run in this review.

## Limitations and handoff risks

- **Timing:** source timestamps use reported constant FPS, not per-frame presentation timestamps; VFR or incorrect metadata can be wrong. The annotated output uses fixed FPS and may drift on irregular sampling or dropped frames. Processing speed is not a real-time promise.
- **Failure handling:** OpenCV's writer is checked when opened, but writes are not verified per frame. A writer exception can prevent accumulated frame details from being copied into the outer report; abrupt process termination may prevent JSON creation. Frame-count metadata may understate truncation, and a missing count cannot distinguish EOF from a read error. These are remaining risks, not failures observed in the sample. Long videos accumulate results in memory.
- **UNKNOWN and ambiguity:** tested UNKNOWN, uncertainty explanations and inference failures are suppressed. This guarantee depends on honest upstream statuses/evidence and valid input; a confidently wrong detector class or wrong worker association can still produce MISSING and a false candidate. Human labels for representative crowded/occluded scenes and false-MISSING rates remain unavailable.
- **Continuity/deduplication:** frame-local `person_id` values are not identities. Rules use unique mutual adjacent-frame box IoU >=0.5 and one incident per contiguous PPE-specific missing-evidence episode. UNKNOWN, absence, errors, gaps or ambiguous matches break continuity, so duplicates can recur; wrong spatial matches can merge workers and suppress a distinct incident. No validated multi-object tracker or event-level accuracy exists.
- **Input/report validation:** the fix protects the tested empty/inconsistent cases, but the evaluator does not comprehensively validate arbitrary malformed JSON, box geometry, ordering or FPS values. Source processing can be partial and output candidates remain provisional. Do not use status `complete` as evidence of detection accuracy.

## Next validation and changed files

Before an alert or reliability claim, use a small independently labelled set of short clips containing compliant workers, visible missing helmet and vest, nearby/overlapping workers, occlusion, transitions and failures. Report person association errors, false MISSING on compliant worker-items, UNKNOWN rate, event false positives/misses, duplicates and cross-person suppression with denominators; agree acceptance thresholds before tuning. Test VFR and writer/decode failures on the intended hardware separately.

Files changed in this review: [`src/safety_rules/ppe_rules.py`](src/safety_rules/ppe_rules.py), [`tests/test_ppe_rules.py`](tests/test_ppe_rules.py), and this report. Existing dataset, splits, model weights, training configuration and baseline results were not changed.

**Video Processing: PARTIAL** — software paths and one clip tested; timing/failure edge cases and representative frame-level accuracy remain unverified.

**Safety Rules: PARTIAL** — branch behavior and saved-result replay tested; association and event-level accuracy and identity continuity remain unverified.
