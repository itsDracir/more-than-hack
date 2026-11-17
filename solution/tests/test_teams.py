import unittest

import numpy as np

from solution.teams import HOME, AWAY, NONE, segment_teams, team_evidence


def control(n, team):
    """Distàncies per a `n` frames on `team` té un jugador amb la pilota i l'altre és a 5 m."""
    near, far = np.full(n, 0.5), np.full(n, 5.0)
    return (near, far) if team == HOME else (far, near)


class TeamEvidenceTest(unittest.TestCase):

    def test_sign_follows_the_closer_team(self):
        home = np.array([0.5, 5.0, np.nan, 0.5, np.nan])
        away = np.array([5.0, 0.5, np.nan, np.nan, 0.5])
        evidence = team_evidence(home, away)
        self.assertGreater(evidence[0], 0)
        self.assertLess(evidence[1], 0)
        self.assertEqual(evidence[2], 0)             # sense pilota, sense indici
        self.assertGreater(evidence[3], 0)           # només hi ha locals seguits a prop de la pilota
        self.assertLess(evidence[4], 0)

    def test_nobody_near_the_ball_is_weak_evidence(self):
        near = team_evidence(np.array([0.5]), np.array([5.0]))[0]
        far = team_evidence(np.array([8.0]), np.array([12.0]))[0]
        self.assertGreater(near, 10 * far)


class SegmentTeamsTest(unittest.TestCase):

    def test_brief_opponent_touch_does_not_flip(self):
        home, away = control(300, HOME)
        home[150:153], away[150:153] = 5.0, 0.5     # un desviament
        codes = segment_teams(team_evidence(home, away), np.ones(300, dtype=bool))
        self.assertTrue((codes == HOME).all())

    def test_turnover_is_placed_at_the_first_touch_of_the_new_team(self):
        h1, a1 = control(100, HOME)
        gap = np.full(30, np.nan)                    # el tracker perd la pilota durant el duel
        h2, a2 = control(170, AWAY)
        home = np.concatenate([h1, gap, h2])
        away = np.concatenate([a1, gap, a2])
        codes = segment_teams(team_evidence(home, away), np.ones(300, dtype=bool))
        np.testing.assert_array_equal(codes, [HOME] * 130 + [AWAY] * 170)

    def test_dead_frames_and_evidence_free_runs_get_no_team(self):
        home, away = control(100, HOME)
        home[60:], away[60:] = np.nan, np.nan        # el segon tram en joc no veu mai la pilota
        alive = np.ones(100, dtype=bool)
        alive[40:50] = False
        codes = segment_teams(team_evidence(home, away), alive)
        self.assertTrue((codes[:40] == HOME).all())
        self.assertTrue((codes[40:50] == NONE).all())
        self.assertTrue((codes[50:] == HOME).all())  # del 50 al 59 encara es veu el local amb la pilota

        codes = segment_teams(team_evidence(np.full(10, np.nan), np.full(10, np.nan)),
                              np.ones(10, dtype=bool))
        self.assertTrue((codes == NONE).all())


if __name__ == "__main__":
    unittest.main()
