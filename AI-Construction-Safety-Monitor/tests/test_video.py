import json
import sys
import tempfile
import unittest
from itertools import islice
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from monitoring.video import process_capture, sample_indices
from detection.process_video import run


class Capture:
    def __init__(self, frames):
        self.frames = iter(frames)

    def read(self):
        return next(self.frames, (False, None))


class Writer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)


class VideoTests(unittest.TestCase):
    def test_sampling_and_timestamps(self):
        self.assertEqual(list(islice(sample_indices(10, 2), 12)),
                         [True, False, False, False, False, True, False, False, False, False, True, False])
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        writer = Writer()
        result = process_capture(Capture([(True, frame)] * 12), writer, lambda _: ([], []), 10, 2, 12)
        self.assertEqual([x['frame_index'] for x in result['frames']], [0, 5, 10])
        self.assertEqual([x['timestamp_seconds'] for x in result['frames']], [0, .5, 1])
        self.assertEqual(len(writer.frames), 3)
        self.assertEqual(result['errors'], [])

    def test_non_integer_sampling_and_rate_cap(self):
        self.assertEqual(list(islice(sample_indices(5, 2), 10)),
                         [True, False, False, True, False, True, False, False, True, False])
        self.assertEqual(list(islice(sample_indices(2, 10), 3)), [True, True, True])
        with self.assertRaises(ValueError):
            next(sample_indices(0, 2))

    def test_errors_do_not_create_ppe_decisions(self):
        frame = np.zeros((50, 80, 3), dtype=np.uint8)
        def fail(_):
            raise RuntimeError('inference failed')
        result = process_capture(Capture([(True, frame), (True, None)]), Writer(), fail, 1, 1, 3)
        self.assertEqual(result['frames'][0]['status'], 'error')
        self.assertIsNone(result['frames'][0]['people'])
        self.assertEqual(len(result['errors']), 3)  # inference, invalid frame, premature read failure

    def test_unknown_is_preserved(self):
        frame = np.zeros((50, 80, 3), dtype=np.uint8)
        person = {'person_id': 1, 'person_box': [1, 1, 20, 40], 'helmet_status': 'UNKNOWN',
                  'vest_status': 'UNKNOWN', 'person_confidence': 0.9, 'associated_ppe': [], 'explanations': {}}
        result = process_capture(Capture([(True, frame)]), Writer(), lambda _: ([person], []), 1, 1, 1)
        self.assertEqual(result['frames'][0]['people'][0]['helmet_status'], 'UNKNOWN')
        self.assertEqual(result['frames'][0]['people'][0]['vest_status'], 'UNKNOWN')

    def test_missing_video_records_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            out = Path(tmp)
            report = run(out / 'absent.mp4', out, 2, out / 'person.pt', out / 'ppe.pt')
            self.assertEqual(report['status'], 'error')
            self.assertEqual(report['frames'], [])
            self.assertIn('Video not found', json.loads((out / 'results.json').read_text())['errors'][0]['error'])


if __name__ == '__main__':
    unittest.main()
