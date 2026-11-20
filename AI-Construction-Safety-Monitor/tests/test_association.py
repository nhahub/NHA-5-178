import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from monitoring.association import associate_ppe, overlap_fraction, region_for_person


def person(pid=1, box=None):
    return {"person_id": pid, "box": box or [0, 0, 100, 200], "confidence": .9}


def ppe(name, box, confidence=.9):
    return {"class_name": name, "box": box, "confidence": confidence}


class AssociationTests(unittest.TestCase):
    def test_geometry(self):
        self.assertEqual(region_for_person([0, 0, 100, 200], "helmet"), [0, 0, 100, 80])
        self.assertEqual(overlap_fraction([0, 0, 20, 20], [10, 0, 30, 20]), .5)
        self.assertEqual(overlap_fraction([0, 0, 0, 20], [0, 0, 30, 20]), 0)

    def test_single_person_and_explicit_missing(self):
        result = associate_ppe([person()], [ppe("Helmet", [20, 10, 60, 50]), ppe("NoVest", [20, 50, 80, 140])])[0]
        self.assertEqual((result["helmet_status"], result["vest_status"]), ("PRESENT", "MISSING"))
        self.assertEqual(len(result["associated_ppe"]), 2)
        self.assertEqual(result["associated_ppe"][0]["confidence"], .9)

    def test_missing_evidence_is_unknown(self):
        result = associate_ppe([person()], [])[0]
        self.assertEqual((result["helmet_status"], result["vest_status"]), ("UNKNOWN", "UNKNOWN"))
        self.assertTrue(result["explanations"]["helmet"])

    def test_distinct_people(self):
        people = [person(1), person(2, [200, 0, 300, 200])]
        result = associate_ppe(people, [ppe("NoHelmet", [220, 10, 260, 50])])
        self.assertEqual(result[0]["helmet_status"], "UNKNOWN")
        self.assertEqual(result[1]["helmet_status"], "MISSING")

    def test_overlapping_people_are_ambiguous(self):
        people = [person(1), person(2, [10, 0, 110, 200])]
        result = associate_ppe(people, [ppe("NoHelmet", [30, 10, 60, 50])])
        self.assertEqual([x["helmet_status"] for x in result], ["UNKNOWN", "UNKNOWN"])
        self.assertTrue(all(not x["associated_ppe"] for x in result))
        self.assertTrue(all("ambiguous" in x["explanations"]["helmet"][0] for x in result))

    def test_conflict_is_unknown(self):
        result = associate_ppe([person()], [ppe("Helmet", [20, 10, 60, 50]), ppe("NoHelmet", [20, 10, 60, 50])])[0]
        self.assertEqual(result["helmet_status"], "UNKNOWN")
        self.assertIn("Conflicting", result["explanations"]["helmet"][0])

    def test_low_confidence_is_unknown(self):
        result = associate_ppe([person()], [ppe("NoVest", [20, 50, 80, 140], .49)])[0]
        self.assertEqual(result["vest_status"], "UNKNOWN")
        self.assertEqual(result["associated_ppe"][0]["confidence"], .49)

    def test_wrong_region_and_invalid_boxes(self):
        result = associate_ppe([person()], [ppe("NoHelmet", [20, 120, 60, 160])])[0]
        self.assertEqual(result["helmet_status"], "UNKNOWN")
        with self.assertRaises(ValueError):
            associate_ppe([person(box=[1, 1, 0, 0])], [])

    def test_zero_overlap_threshold_must_not_assign_disjoint_detection(self):
        # The public API accepts 0, but zero overlap must not imply ownership.
        result = associate_ppe([person()], [ppe("NoHelmet", [200, 10, 240, 50])], min_overlap=0)[0]
        self.assertEqual(result["helmet_status"], "UNKNOWN")
        self.assertEqual(result["associated_ppe"], [])


if __name__ == "__main__":
    unittest.main()
