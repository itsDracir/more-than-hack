import os
import sys
import unittest

import numpy as np

from solution.labeling import label_frames
from solution.tests.synthetic import FPS, PITCH, synthetic_match

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def spell_f1(truth, labels, blocks):
    runs = [(first, last + 1) for _, first, last in blocks]
    gt = pm.label_codes(truth)
    pred = pm.fill_abstentions(pm.label_codes(labels), runs)
    return pm.spell_scores(gt, pred, runs)["mean_f1"]


class LabelFramesTest(unittest.TestCase):

    def setUp(self):
        self.features, self.events, self.blocks, self.truth = synthetic_match()

    def test_one_valid_label_per_frame_and_x_outside_the_halves(self):
        labels = label_frames(self.features, self.events, self.blocks, PITCH, FPS)
        self.assertEqual(len(labels), len(self.features))
        self.assertTrue(set(np.unique(labels)) <= {"H", "A", "D", "X"})
        _, first, last = self.blocks[0]
        self.assertTrue((labels[:first] == "X").all())
        self.assertTrue((labels[last + 1:] == "X").all())

    def test_recovers_the_possessions_of_a_synthetic_match(self):
        labels = label_frames(self.features, self.events, self.blocks, PITCH, FPS)
        self.assertGreaterEqual(spell_f1(self.truth, labels, self.blocks), 0.85)

    def test_works_without_smart_tagging(self):
        labels = label_frames(self.features, None, self.blocks, PITCH, FPS)
        self.assertGreaterEqual(spell_f1(self.truth, labels, self.blocks), 0.6)

    def test_missing_columns_are_named_in_the_error(self):
        broken = self.features.drop(columns=["home_nearest_m"])
        with self.assertRaises(ValueError) as caught:
            label_frames(broken, self.events, self.blocks, PITCH, FPS)
        self.assertIn("home_nearest_m", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
