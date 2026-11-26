# Annotation verification and preparation

**Decision: no independently confirmed dataset correction is ready to apply.** The separate [patch manifest](annotation_patch_manifest.json) contains two specific but unapproved class-change candidates and four review-only candidates. Nothing has been changed in any dataset, model, split or prior output.

## What the available policy establishes

- Local [`data.yaml`](../../../../dataset/processed/PPE/data.yaml) and `dataset/raw/PPE/data.yaml` establish class IDs: 0 Helmet, 1 NoHelmet, 2 NoVest, 3 Vest. Existing scripts interpret YOLO detection rows as `class x_center y_center width height` with normalized coordinates. This verifies representation and the project's operational class map, **not** the original annotator's semantic instructions.
- Local `dataset/raw/PPE/README.dataset.txt` identifies the Roboflow export and CC BY 4.0, while `README.roboflow.txt` describes the YOLO export. The referenced [Roboflow project](https://universe.roboflow.com/ying2001/20250731-ppe2286-m0ega) lists the four classes but provides no project description. The [Mendeley dataset record](https://data.mendeley.com/datasets/zkzghjvpn2/6) describes YOLO-format PPE data, but neither reviewed source specifies whether helmet boxes include face/head, minimum visibility, partial occlusion, or whether every visible worker must be labelled. The sources do not establish that the local export is byte-identical to Mendeley version 6.
- Existing 00269 boxes include head/face as well as helmet. Some existing NoVest boxes cover torso clothing, e.g. 00739. These examples show **observed practice**, not an authoritative or necessarily consistent rule. A responsible annotator/team must agree the meaning, box extent and visibility threshold before approving changes.

## Review evidence and proposed changes

The existing five-case `priority_cases_20261118_01/review_sheet.csv` and `simple_review_20261119_01/blank_decision_sheet.csv` are blank. A downloaded `C:/Users/Hp/Downloads/ppe_review_decisions.csv` was also inspected; it contains no selected decisions. The previous `human_decisions_20261119_01/HUMAN_REVIEW_DECISIONS.md` is explicitly assistant-provisional. The separately user-confirmed first frame of the sample video is **not** a labelled dataset image and is not a dataset correction.

| Image | Split and visually supported observation | Proposal / uncertainty |
|---|---|---|
| `00269_jpg.rf.0df82c5293d7fe31962e1bcfa25a8a1c.jpg` | **valid**. Original shows yellow and blue helmets; both existing rows are class 1 NoHelmet. | Two class-only `1 -> 0` candidates, exact existing/proposed rows and SHA-256 in manifest. Strong visual concern but **not confirmed corrections**: human confirmation and the head-versus-helmet extent policy are outstanding. Preserve coordinates only if that policy permits them. |
| `00016_jpg.rf.a5de42674ff1996a69f005db19136b4e.jpg` | **valid**. A helmeted person has a bare-looking shirt region; much of the torso is covered by plans. Existing label is Helmet only. | Possible missing NoVest (class 2); status uncertain. No box or class addition is approved. Do not infer absence from a detector output. |
| `00739_jpg.rf.58a11b096dede5f18d7fec19b5fb02e8.jpg` | **valid**. Existing rows already include two NoVest boxes; a further left-side person appears to wear ordinary clothing without a visible vest. Scene is dark. | Potential additional NoVest; image visibility, person scope and box geometry need review. No coordinates proposed. |
| `00025_jpg.rf.75084f3cd7ae2aff6d2c8714db534bde.jpg` | **valid**. Three workers visibly wear high-visibility vests, while the current file contains two Vest boxes for other workers. | Potential missing **Vest (class 3)** for foreground-right worker; this is not a NoVest candidate. Human must confirm annotation scope and draw a box. |
| `00031_jpg.rf.708f654c7f1b7e851d9aca1598b86e96.jpg` | **valid**. Crowded scene appears to have more visible helmets than its three Helmet labels. | Potential missing Helmet boxes; verify individual visibility and extents before proposing rows. |
| `00783_jpg.rf.b14d4d7f5bba1016d08a48b61d317eac.jpg` | **valid**. Two existing NoVest labels are present in an occluded group; assistant review has not established an additional missing NoVest. | No correction proposed; verify person/torso ownership and ambiguous visibility. |

All image/split references above were checked against `dataset/processed/PPE/split_groups.csv` and the actual validation files. **Train edits: none proposed. Test edits: none proposed or applied.** These are validation examples, so any accepted revision changes the benchmark: create a separately versioned validation copy, record original and new hashes plus reviewer decisions, and report which benchmark version every metric uses. Do not silently replace the held-out test split or compare scores across label versions as if they used the same ground truth.

## Next action

Agree and record the class meaning, object/torso box extent, completeness scope and occlusion/visibility policy. Then independently review the two 00269 rows and the potential NoVest cases 00016 and 00739 using the existing review images. Record the reviewer and decision in a new artifact; only after agreement prepare a separate versioned annotation copy and re-evaluate its affected validation images. Until then, keep every manifest item unapproved and retain the current original dataset.
