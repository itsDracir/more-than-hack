import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import games
from solution.labeling import REQUIRED_COLUMNS
from solution.run import ROOT as REPO_ROOT
from solution.run import display_path, load_events, load_features, main, write_submission

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


class RunTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_written_file_passes_the_official_validation(self):
        labels = np.array(["X", "H", "H", "A", "D", "X"])
        path = write_submission(os.path.join(self.tmp, "team.csv"), labels)
        self.assertIsNone(pm.validate_submission(pm.read_label_csv(path), len(labels)))

    def test_reads_the_preprocessing_notebook_csv(self):
        # El notebook escriu els booleans com a True/False i els NaN com una cel·la buida.
        path = os.path.join(self.tmp, "features.csv")
        with open(path, "w") as handle:
            handle.write("frame,ball_visible,ball_x_norm,home_nearest_m\n"
                         "0,False,,\n1,True,0.5,1.25\n")
        features = load_features(path, n_frames=2)
        self.assertEqual(features["ball_visible"].dtype, bool)
        self.assertTrue(np.isnan(features.loc[0, "ball_x_norm"]))
        self.assertAlmostEqual(features.loc[1, "home_nearest_m"], 1.25)

    def test_display_path_is_relative_inside_the_repo(self):
        inside = os.path.join(REPO_ROOT, "evaluation", "results", "team.csv")
        self.assertEqual(display_path(inside), os.path.join("evaluation", "results", "team.csv"))

    @unittest.skipUnless(os.path.splitdrive(tempfile.gettempdir())[0].lower()
                         != os.path.splitdrive(REPO_ROOT)[0].lower(),
                         "needs the temp dir on another drive than the repo (Windows)")
    def test_display_path_survives_another_drive(self):
        other = os.path.join(self.tmp, "team.csv")
        self.assertEqual(display_path(other), os.path.abspath(other))

    def test_missing_feature_table_says_how_to_build_it(self):
        import contextlib
        import io
        missing = os.path.join(self.tmp, "nope.csv")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
            code = main(["--name", "t", "--features", missing, "--out", self.tmp])
        self.assertEqual(code, 2)
        self.assertIn("01_preprocessing.ipynb", printed.getvalue())
        self.assertIn("nope.csv", printed.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "t.csv")))

    def run_main(self, columns, events=None):
        """Executa main() amb una taula sintètica d'un frame per fila i retorna (codi, text)."""
        import contextlib
        import io
        n = games.source_frames("tactical")
        features = os.path.join(self.tmp, "features.csv")
        pd.DataFrame({"frame": np.arange(n), **{c: np.nan for c in columns}}).to_csv(
            features, index=False)
        args = ["--name", "t", "--features", features, "--out", self.tmp]
        if events is not None:
            args += ["--events", events]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
            code = main(args)
        return code, printed.getvalue()

    def test_features_without_team_speeds_say_to_rerun_the_notebook(self):
        # Les velocitats per equip les calcula el notebook en la mateixa lectura de l'ATD; run.py
        # ja no torna a llegir el tracking en brut.
        code, printed = self.run_main(["ball_x_norm", "ball_y_norm"])
        self.assertEqual(code, 2)
        self.assertIn("home_mean_speed_mps", printed)
        self.assertIn("01_preprocessing.ipynb", printed)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "t.csv")))

    def test_missing_set_piece_file_says_to_rerun_the_notebook(self):
        code, printed = self.run_main(REQUIRED_COLUMNS,
                                      events=os.path.join(self.tmp, "no_events.json"))
        self.assertEqual(code, 2)
        self.assertIn("no_events.json", printed)
        self.assertIn("01_preprocessing.ipynb", printed)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "t.csv")))

    def test_reads_the_set_piece_intervals_written_by_the_notebook(self):
        intervals = [{"code": "SET PIECES", "start_frame": 10, "end_frame": 135,
                      "tags": {"Type": "Kick Off (Start)"}}]
        path = os.path.join(self.tmp, "events.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(intervals, handle)
        self.assertEqual(load_events(path), intervals)

    def test_rejects_set_piece_intervals_with_missing_fields(self):
        path = os.path.join(self.tmp, "events.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([{"code": "SET PIECES", "start_frame": 10}], handle)
        with self.assertRaises(ValueError):
            load_events(path)

    def test_rejects_features_that_do_not_cover_every_frame(self):
        path = os.path.join(self.tmp, "features.csv")
        pd.DataFrame({"frame": [0, 2], "ball_x_norm": [0.1, 0.2]}).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            load_features(path, n_frames=3)


if __name__ == "__main__":
    unittest.main()
