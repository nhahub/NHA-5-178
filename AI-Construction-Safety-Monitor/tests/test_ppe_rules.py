import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from safety_rules.ppe_rules import PPERules, evaluate_report


def person(status='MISSING', name='NoHelmet', confidence=.8, box=None, pid=1):
    return {'person_id': pid, 'person_box': box or [0, 0, 100, 200], 'person_confidence': .9,
            'helmet_status': status, 'vest_status': 'UNKNOWN',
            'associated_ppe': [{'class_name': name, 'confidence': confidence, 'box': [10, 0, 50, 40]}],
            'explanations': {'helmet': [], 'vest': []}}


def frame(index, people, status='ok'):
    return {'frame_index': index, 'timestamp_seconds': index * .5, 'status': status, 'people': people}


class RulesTests(unittest.TestCase):
    def test_present_unknown_and_missing_require_explicit_confident_negative(self):
        for status in ('PRESENT', 'UNKNOWN'):
            self.assertEqual(PPERules().process(frame(0, [person(status)])), [])
        self.assertEqual(PPERules().process(frame(0, [person(name='Helmet')])), [])
        self.assertEqual(PPERules().process(frame(0, [person(confidence=.49)])), [])
        event = PPERules().process(frame(0, [person()]))[0]
        self.assertEqual((event['ppe_type'], event['confidence']), ('helmet', .8))
        self.assertEqual(event['evidence'][0]['class_name'], 'NoHelmet')

    def test_ambiguous_and_conflicting_evidence(self):
        p = person()
        p['explanations']['helmet'] = ['ambiguous person association']
        self.assertEqual(PPERules().process(frame(0, [p])), [])
        p = person()
        p['associated_ppe'].append({'class_name': 'Helmet', 'confidence': .8})
        self.assertEqual(PPERules().process(frame(0, [p])), [])

    def test_consecutive_dedup_and_frame_person_id_swap(self):
        rules = PPERules()
        first = rules.process(frame(0, [person(pid=1)]))
        self.assertEqual(len(first), 1)
        self.assertEqual(rules.process(frame(1, [person(pid=2, box=[2, 0, 102, 200])])), [])
        self.assertEqual(rules.process(frame(2, [person(status='UNKNOWN')])), [])
        self.assertEqual(len(rules.process(frame(3, [person()]))), 1)

    def test_different_ppe_and_people(self):
        p = person()
        p['vest_status'] = 'MISSING'
        p['associated_ppe'].append({'class_name': 'NoVest', 'confidence': .7, 'box': [10,50,50,100]})
        q = person(pid=2, box=[200,0,300,200])
        rules = PPERules()
        self.assertEqual(len(rules.process(frame(0, [p, q]))), 3)
        self.assertEqual(rules.process(frame(1, [p, q])), [])

    def test_error_and_gap_reset(self):
        r = {'input': 'test', 'status': 'partial', 'settings': {'target_fps': 2}, 'source': {'fps': 25},
             'frames': [frame(0, [person()]), frame(1, None, 'error'), frame(2, [person()])]}
        self.assertEqual(len(evaluate_report(r)['incidents']), 2)
        r['frames'] = [frame(0, [person()]), frame(4, [person()])]
        self.assertEqual(len(evaluate_report(r)['incidents']), 2)

    def test_ambiguous_matching_does_not_merge_people(self):
        rules = PPERules()
        rules.process(frame(0, [person(pid=1), person(pid=2, box=[2,0,102,200])]))
        events = rules.process(frame(1, [person(pid=3, box=[1,0,101,200])]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['dedup_person_id'], 3)

    def test_empty_report_is_not_complete(self):
        report = {'status': 'complete', 'settings': {'target_fps': 2}, 'source': {'fps': 25}, 'frames': []}
        with self.assertRaises(ValueError):
            evaluate_report(report)

    def test_recorded_error_suppresses_corresponding_frame_and_marks_partial(self):
        report = {'status': 'complete', 'settings': {'target_fps': 2}, 'source': {'fps': 25},
                  'errors': [{'frame_index': 0, 'error': 'decode failed'}], 'frames': [frame(0, [person()])]}
        result = evaluate_report(report)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['incidents'], [])

    def test_failed_frame_in_complete_report_is_partial(self):
        report = {'status': 'complete', 'settings': {'target_fps': 2}, 'source': {'fps': 25},
                  'frames': [frame(0, None, 'error')]}
        result = evaluate_report(report)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['incidents'], [])


if __name__ == '__main__':
    unittest.main()
