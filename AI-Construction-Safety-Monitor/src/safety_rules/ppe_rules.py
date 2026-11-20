"""Experimental per-frame PPE incidents; spatial matching is not tracking."""
import math


def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = x * y
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


class PPERules:
    """Deduplicate contiguous missing-evidence episodes via unique adjacent-frame IoU."""
    def __init__(self, match_iou=0.5, min_confidence=0.5):
        if not 0 < match_iou <= 1 or not 0 <= min_confidence <= 1:
            raise ValueError('Invalid rules thresholds')
        self.match_iou = match_iou
        self.min_confidence = min_confidence
        self.previous = []
        self.next_id = 1

    def reset(self):
        self.previous = []

    def process(self, frame):
        if frame.get('status') != 'ok' or not isinstance(frame.get('people'), list):
            self.reset()
            return []
        people = frame['people']
        candidates = [[j for j, old in enumerate(self.previous)
                       if iou(person['person_box'], old['box']) >= self.match_iou]
                      for person in people]
        used_by = {j: sum(j in c for c in candidates) for j in range(len(self.previous))}
        current, incidents = [], []
        for person, matches in zip(people, candidates):
            if len(matches) == 1 and used_by[matches[0]] == 1:
                old = self.previous[matches[0]]
                track_id, active = old['id'], old['active']
            else:
                track_id, active = self.next_id, set()
                self.next_id += 1
            new_active = set()
            for kind, negative, positive in [('helmet', 'NoHelmet', 'Helmet'), ('vest', 'NoVest', 'Vest')]:
                if person.get(kind + '_status') != 'MISSING' or person.get('explanations', {}).get(kind):
                    continue
                evidence = person.get('associated_ppe', [])
                negatives = [e for e in evidence if e.get('class_name') == negative
                             and isinstance(e.get('confidence'), (int, float))
                             and math.isfinite(e['confidence']) and self.min_confidence <= e['confidence'] <= 1]
                # Defensive check: conflicting sufficiently confident evidence invalidates an incident.
                positives = [e for e in evidence if e.get('class_name') == positive
                             and isinstance(e.get('confidence'), (int, float))
                             and math.isfinite(e['confidence']) and self.min_confidence <= e['confidence'] <= 1]
                if not negatives or positives:
                    continue
                new_active.add(kind)
                if kind not in active:
                    best = max(negatives, key=lambda e: e['confidence'])
                    incidents.append({'timestamp_seconds': frame['timestamp_seconds'],
                        'frame_index': frame['frame_index'], 'person_id': person['person_id'],
                        'dedup_person_id': track_id, 'person_box': person['person_box'],
                        'person_confidence': person['person_confidence'], 'ppe_type': kind,
                        'decision': 'MISSING', 'confidence': best['confidence'],
                        'evidence': negatives, 'review_status': 'unverified_model_output'})
            current.append({'id': track_id, 'box': person['person_box'], 'active': new_active})
        self.previous = current
        return incidents


def evaluate_report(report):
    """Evaluate saved video frames, resetting deduplication on errors and sampling gaps."""
    frames = report.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('A non-empty list of processed frames is required')
    rules = PPERules()
    incidents = []
    previous_time = None
    target = report.get('settings', {}).get('target_fps')
    source_fps = report.get('source', {}).get('fps')
    if not isinstance(target, (float, int)) or not isinstance(source_fps, (float, int)) or target <= 0 or source_fps <= 0:
        raise ValueError('Positive source and target FPS are required to detect gaps')
    interval = 1 / min(target, source_fps)
    error_indices = {e.get('frame_index') for e in report.get('errors', []) if e.get('frame_index') is not None}
    prev_index = None
    incomplete = bool(report.get('errors')) or report.get('status') != 'complete'
    for frame in frames:
        timestamp = frame['timestamp_seconds']
        index = frame['frame_index']
        if frame.get('status') != 'ok' or not isinstance(frame.get('people'), list) or index in error_indices:
            incomplete = True
            rules.reset()
            previous_time, prev_index = timestamp, index
            continue
        if previous_time is not None and (timestamp <= previous_time or timestamp - previous_time > interval + 1/source_fps + 1e-6
                 or any(prev_index < e <= index for e in error_indices)):
            rules.reset()
        incidents.extend(rules.process(frame))
        previous_time, prev_index = timestamp, index
    return {'source': report.get('input'), 'source_status': report.get('status'),
            'status': 'partial' if incomplete else 'complete',
            'method': 'Adjacent sampled frames only: unique mutual person box IoU >= 0.5; per-PPE contiguous MISSING episodes. Resets on uncertainty, absence, errors, gaps or ambiguous matches. IDs are not verified identities.',
            'warning': 'Experimental, unverified model-derived incidents; not confirmed violations.',
            'incidents': incidents}
