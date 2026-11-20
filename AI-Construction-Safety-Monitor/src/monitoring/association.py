"""Conservative spatial association; all boxes are image-pixel xyxy coordinates."""
from math import isfinite

PPE_TYPES = {"Helmet": ("helmet", "PRESENT"), "NoHelmet": ("helmet", "MISSING"),
             "Vest": ("vest", "PRESENT"), "NoVest": ("vest", "MISSING")}


def valid_box(box):
    return (len(box) == 4 and all(isfinite(float(v)) for v in box)
            and box[0] < box[2] and box[1] < box[3])


def region_for_person(box, kind):
    """Heuristic regions; intentionally conservative and not pose estimation."""
    x1, y1, x2, y2 = box
    height = y2 - y1
    if kind == "helmet":
        return [x1, y1, x2, y1 + 0.40 * height]
    if kind == "vest":
        return [x1, y1 + 0.20 * height, x2, y1 + 0.80 * height]
    raise ValueError(f"Unknown PPE region: {kind}")


def overlap_fraction(box, region):
    """Fraction of the PPE box that falls in the person's relevant region."""
    if not valid_box(box) or not valid_box(region):
        return 0.0
    x1, y1 = max(box[0], region[0]), max(box[1], region[1])
    x2, y2 = min(box[2], region[2]), min(box[3], region[3])
    return max(0, x2-x1) * max(0, y2-y1) / ((box[2]-box[0]) * (box[3]-box[1]))


def associate_ppe(people, detections, min_confidence=0.5, min_overlap=0.6,
                  ambiguity_margin=0.15):
    """Return one record per person. Input detections have class_name, box, confidence.

    A near-tied detection is assigned to nobody and marks all competing people
    unknown for that item. IDs are stable only within this single invocation.
    """
    if not 0 <= min_confidence <= 1 or not 0 <= min_overlap <= 1 or not 0 <= ambiguity_margin <= 1:
        raise ValueError("Thresholds must be between 0 and 1")
    records = []
    ids = set()
    for index, person in enumerate(people, 1):
        box = person["box"]
        if not valid_box(box):
            raise ValueError("Invalid person box")
        pid = person.get("person_id", index)
        if pid in ids:
            raise ValueError("Person IDs must be unique")
        ids.add(pid)
        confidence = float(person["confidence"])
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid person confidence")
        records.append({"person_id": pid, "person_box": list(box),
                        "person_confidence": confidence, "helmet_status": "UNKNOWN",
                        "vest_status": "UNKNOWN", "associated_ppe": [],
                        "explanations": {"helmet": [], "vest": []}})

    for index, detection in enumerate(detections, 1):
        name = detection["class_name"]
        if name not in PPE_TYPES:
            raise ValueError(f"Unsupported PPE class: {name}")
        if not valid_box(detection["box"]):
            raise ValueError("Invalid PPE box")
        confidence = float(detection["confidence"])
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid PPE confidence")
        kind, _ = PPE_TYPES[name]
        candidates = []
        for i, person in enumerate(people):
            region = region_for_person(person["box"], kind)
            score = overlap_fraction(detection["box"], region)
            if score > 0 and score >= min_overlap:
                candidates.append((score, i))
        candidates.sort(reverse=True)
        if not candidates:
            continue
        competitors = [c for c in candidates if candidates[0][0] - c[0] <= ambiguity_margin]
        if len(competitors) > 1:
            for _, i in competitors:
                records[i]["explanations"][kind].append(f"Detection {index} has ambiguous person association")
            continue
        score, i = candidates[0]
        records[i]["associated_ppe"].append({"detection_id": index, "class_name": name,
            "box": list(detection["box"]), "confidence": confidence, "region_overlap": score})
        if confidence < min_confidence:
            records[i]["explanations"][kind].append(f"Detection {index} is below the confidence threshold")

    for rec in records:
        for kind in ("helmet", "vest"):
            evidence = [d for d in rec["associated_ppe"] if PPE_TYPES[d["class_name"]][0] == kind
                        and d["confidence"] >= min_confidence]
            statuses = {PPE_TYPES[d["class_name"]][1] for d in evidence}
            if len(statuses) > 1:
                rec["explanations"][kind].append("Conflicting positive and negative PPE evidence")
            elif not evidence:
                rec["explanations"][kind].append("No sufficiently confident associated evidence")
            elif not rec["explanations"][kind]:
                rec[kind + "_status"] = statuses.pop()
    return records
