# Consolidated visual review decisions

**Status: provisional assistant visual review; independent human adjudication is still pending for A1, B2, B3, C3, D2 and 00269.** The five-case [`review_sheet.csv`](../priority_cases_20261118_01/review_sheet.csv) remains blank. The observations below are from the original images and saved overlays/outputs, not completed human decisions. Prior assistant triage is also not independent ground truth. No dataset labels were changed.

The one separately documented **human-reported confirmation** is the first frame of the sample video: the user reported the right-side worker wearing a helmet without a vest and the left-side worker wearing both; see [`visual_review.md`](../../step5_video/safety_rules_demo_01/visual_review.md). The screenshot was not supplied in that documentation exchange. This is one frame, not validation of the five cases or a real-world violation.

## Five priority cases

Model assignments below describe what the saved pipeline did. The visual observations are this assistant's inspection, not human confirmation; exact ownership, box completeness and visible PPE semantics need independent verification. The provisional target-person boxes and saved predicted-person matches are in the [review package](../priority_cases_20261118_01/index.html).

| Case | Visual evidence and existing annotations | Saved detection and association | Decision / remaining uncertainty |
|---|---|---|---|
| **A1 / 00016** | Red helmet clearly visible; a substantial part of the torso is obscured by plans. Dataset has one Helmet label and no vest-related label. No vest is visible in the exposed upper shirt, but the obscured region prevents confirming the entire torso. | P1 received D1 Helmet 0.922 and D2 NoVest 0.615; output helmet PRESENT / vest MISSING. D2 is spatially assigned to the only person. | Helmet association appears plausible. Whether NoVest is correct, a detector error, or insufficiently observable cannot be decided solely from the exposed torso. Review the full-resolution image and agreed visibility policy. Do not record a false MISSING without evidence of a vest. |
| **B2 / 00025** | Middle worker visibly has a red helmet and high-visibility vest. Dataset labels two vests and one helmet across the scene. | P2 received D5 Helmet 0.279 and D4 Vest 0.320; both are below the 0.5 association threshold, so both statuses are UNKNOWN. | The saved boxes appear to belong to the target worker; their low confidence caused abstention, not a demonstrated wrong-person association. Human-review ownership and annotation coverage; do not call threshold-based UNKNOWN a model absence or a confirmed assignment error. |
| **B3 / 00025** | Foreground right worker visibly has helmet and high-visibility vest. The existing Helmet label appears to cover this worker; only two vest labels exist for three visibly vested workers. | P1 received D1 Helmet 0.742; vest UNKNOWN and no vest evidence assigned to P1. | Helmet assignment appears plausible. A vest appears to be missing from the dataset annotation for this worker, subject to full-resolution box/policy confirmation. The absence of assigned evidence might arise from detector failure, low score, NMS or association; inspect all saved detection boxes before assigning a cause. |
| **C3 / 00783** | The back-facing target visibly wears a yellow helmet; the torso is partly occluded and vest visibility is uncertain. Dataset labels include Helmet, NoHelmet and two NoVest boxes in the wider image. | P6 received D2 Helmet 0.502; helmet PRESENT / vest UNKNOWN. | Helmet assignment appears plausible but is close to the decision threshold. Verify person boundary and helmet ownership in the overlapping group; keep vest UNKNOWN if not visible. A carried helmet elsewhere must not be treated as worn PPE. |
| **D2 / 00031** | Foreground crouched worker visibly wears a red helmet; torso is partly obscured. Only three Helmet boxes are present in a crowded scene with additional visible helmets. | P1 received D7 Helmet 0.310; both statuses UNKNOWN because no sufficiently confident associated evidence exists. | D7 appears to cover the target helmet, but the association abstains at the configured threshold. Dataset coverage appears incomplete; independently check each visible helmet against the class/visibility policy. No wrong-person assignment is established. |

## Image 00269: possible class errors

Original-resolution [`00269 image`](../../step3_evaluation/review_20261118_01/visual_review/00269_jpg.rf.0df82c5293d7fe31962e1bcfa25a8a1c/original.jpg), [ground-truth overlay](../../step3_evaluation/review_20261118_01/visual_review/00269_jpg.rf.0df82c5293d7fe31962e1bcfa25a8a1c/ground_truth.jpg), and [saved 320 prediction](../../step3_evaluation/review_20261118_01/visual_review/00269_jpg.rf.0df82c5293d7fe31962e1bcfa25a8a1c/full_prediction.jpg) were inspected. Both workers visibly appear helmeted: left yellow, right blue. The saved prediction contains two Helmet detections around 0.66 confidence and one NoVest detection around 0.38. Predictions support investigation but are not independent ground truth.

The existing validation label file `dataset/processed/PPE/valid/labels/00269_jpg.rf.0df82c5293d7fe31962e1bcfa25a8a1c.txt` has **two class 1 (NoHelmet) rows** over those heads. This is strong visual evidence of a class-map/annotation inconsistency under the current class definitions, although independent human confirmation and a check of the dataset's box convention are still required.

**Exact proposed correction, conditional on that confirmation:** in a separately versioned annotation copy, change the first token from `1` to `0` on both rows, preserving coordinates initially; independently review box extents because the boxes cover head/face regions as well as helmets. The current rows are:

```text
1 0.46799498746867163 0.44387453874538746 0.051278195488721805 0.09645756457564576
1 0.6525563909774437 0.3781180811808118 0.05879699248120301 0.10992619926199261
```

Proposed class-only replacements would use the same rows with leading `0`. Do **not** apply them to the current split as part of this review. If the annotation policy excludes either partially visible head or prescribes helmet-only geometry, a human must decide the appropriate geometry or exclusion first. No exact new box is recommended from this screenshot without that decision.

## Error attribution and correction decisions

- **Confirmed from saved code/output:** B2's low-confidence detections and D2's low-confidence helmet lead to UNKNOWN under the configured threshold. That is a documented operating-point abstention, not proof of a detector miss or association bug.
- **Visually suggested label quality issues:** 00269's two NoHelmet classes, the missing third vest label in 00025, and missing helmet labels in crowded 00031. Only the 00269 class-only changes above are specific enough to propose; they still need independent confirmation. For 00025/00031, human verification and fresh coordinates are needed before proposing edits.
- **Possible detector/threshold issue:** B3 has a visibly vested worker but no vest assigned; establish whether a suitable detector box exists before attributing the result to detection or association. B2/D2 have visible PPE with lower-confidence detections. A1's NoVest is inconclusive due to occlusion.
- **Wrong-owner association:** no case in this review establishes a wrong-person assignment. Plausible visual assignment is not independently labelled ownership. C3 and the overlapping workers in 00025 are the most useful targeted ownership checks.

## Model and next experiment

**Retain the current checkpoint for the controlled prototype; targeted retraining is not yet justified.** A few suspected label errors, abstentions and one uncertain prediction do not establish whether training data, confidence thresholds or association geometry are the dominant problem. The five-case review is selected, small and not independent.

Smallest next experiment: have a human independently annotate these five case-person boxes and visible helmet/vest status, confirm/correct 00269, and label ownership of every saved PPE detection relevant to those workers. Add the fewest existing images needed to include at least one clearly observable NoHelmet and one NoVest if absent from the five; do not infer missing PPE from an absent detector box. Freeze the reviewed sample and policy before comparing the saved 320 predictions. Report separately: person matches / labelled persons at IoU >=0.5; correct, incorrect and unjudgeable PPE-to-person assignments / assessable assigned detections; UNKNOWN / person-item decisions; false MISSING / visibly compliant person-item instances; and detector TP/FP/FN by class at the existing thresholds. Report actual denominators and unassessable cases. This diagnostic sample cannot establish population-level accuracy; no numeric success target should be tuned on it.

**Exact next action:** complete the five blank human-review rows and independently confirm 00269's two class changes and PPE-box policy, saving the decisions separately from the original labels and prior outputs.
