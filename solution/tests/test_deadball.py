import unittest

import numpy as np

from solution.deadball import (ball_moving, ball_speed_mps, dead_evidence, dead_mask,
                               detect_restart, speed_threshold)

PITCH = (105.0, 68.0)
FPS = 25


def moving_ball(n, start_moving, step=0.004):
    """Pilota aturada al punt central i després xutada al llarg de x a partir de `start_moving`."""
    x = np.full(n, 0.5)
    x[start_moving:] = 0.5 + step * np.arange(1, n - start_moving + 1)
    x = np.minimum(x, 0.9)                          # es queda dins del camp
    y = np.full(n, 0.5)
    return x, y


class SpeedThresholdTest(unittest.TestCase):

    def test_median_of_the_half_ignoring_nan(self):
        values = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
        self.assertAlmostEqual(speed_threshold(values), 2.0)

    def test_all_missing_gives_nan(self):
        self.assertTrue(np.isnan(speed_threshold(np.full(5, np.nan))))


class BallSpeedTest(unittest.TestCase):

    def test_speed_in_metres_per_second_and_nan_across_gaps(self):
        x = np.array([0.5, 0.5 + 1 / 105.0, np.nan, 0.6])
        y = np.full(4, 0.5)
        speed = ball_speed_mps(x, y, PITCH, FPS)
        self.assertTrue(np.isnan(speed[0]))
        self.assertAlmostEqual(speed[1], 25.0)        # 1 m en un frame a 25 fps
        self.assertTrue(np.isnan(speed[2]) and np.isnan(speed[3]))


class DetectRestartTest(unittest.TestCase):

    def test_finds_the_kick(self):
        x, y = moving_ball(200, start_moving=60)
        restart = detect_restart(ball_speed_mps(x, y, PITCH, FPS), 10, 150)
        self.assertIn(restart, (60, 61))

    def test_none_when_ball_never_moves(self):
        x, y = moving_ball(200, start_moving=200)
        self.assertIsNone(detect_restart(ball_speed_mps(x, y, PITCH, FPS), 10, 150))

    def test_single_frame_jitter_is_not_a_kick(self):
        x, y = moving_ball(200, start_moving=200)
        x[80] += 0.01                               # salt d'~1 m durant un sol frame
        self.assertIsNone(detect_restart(ball_speed_mps(x, y, PITCH, FPS), 10, 150))


class DeadEvidenceTest(unittest.TestCase):

    def test_slow_players_and_ball_out_point_to_dead(self):
        n = 100
        speed = np.full(n, 2.5)
        speed[:50] = 0.8
        x = np.full(n, 0.5)
        y = np.full(n, 0.5)
        x[70:75] = 1.03                              # pilota per sobre de la línia de fons
        evidence = dead_evidence(speed, 1.7, x, y, PITCH)
        self.assertTrue((evidence[:50] > 0).all())
        self.assertTrue((evidence[50:70] < 0).all())
        self.assertTrue((evidence[70:75] > 0).all())

    def test_missing_ball_alone_is_not_evidence(self):
        n = 10
        evidence = dead_evidence(np.full(n, np.nan), 1.7, np.full(n, np.nan),
                                 np.full(n, np.nan), PITCH)
        np.testing.assert_array_equal(evidence, np.zeros(n))


class DeadMaskTest(unittest.TestCase):

    def test_long_stoppage_found_and_short_dip_ignored(self):
        n = 3000
        speed = np.full(n, 2.6)
        speed[1000:1400] = 0.9                       # una aturada de 16 s
        speed[2000:2010] = 0.9                       # els jugadors afluixen 0,4 s: no és aturada
        nan = np.full(n, np.nan)
        dead = dead_mask(speed, nan, nan, blocks=[(1, 0, n - 1)], events=None, pitch=PITCH,
                         fps=FPS)
        self.assertTrue(dead[1060:1340].all())
        self.assertFalse(dead[:940].any())
        self.assertFalse(dead[1460:].any())

    def test_everything_before_the_kickoff_is_dead(self):
        n = 2000
        speed = np.full(n, 2.6)                      # jugadors escalfant ràpid: la velocitat diu joc
        speed[1500:1700] = 0.9                       # una aturada més tard, perquè "lent" tingui sentit
        x, y = moving_ball(n, start_moving=230)
        events = [{"code": "SET PIECES", "start_frame": 150, "end_frame": 275,
                   "tags": {"Type": "Kick Off (Start)"}}]
        dead = dead_mask(speed, x, y, blocks=[(1, 100, n - 1)], events=events, pitch=PITCH,
                         fps=FPS)
        self.assertTrue(dead[100:229].all())
        self.assertFalse(dead[235:1450].any())


class BallMovingTest(unittest.TestCase):

    def test_moving_needs_two_metres_in_a_second_inside_the_pitch(self):
        x, y = moving_ball(100, start_moving=50)
        moving = ball_moving(x, y, PITCH, FPS)
        self.assertFalse(moving[:50].any())
        self.assertTrue(moving[60:].all())

    def test_unseen_ball_is_not_moving(self):
        x = np.full(60, np.nan)
        self.assertFalse(ball_moving(x, x, PITCH, FPS).any())

    def test_ball_that_goes_and_comes_back_within_a_second_is_moving(self):
        # Una paret: 1,5 m d'anada i 1,5 m de tornada en mig segon. El desplaçament net és zero,
        # però la pilota ha recorregut 3 m.
        x = np.full(100, 0.5)
        step = 0.25 / PITCH[0]
        x[30:36] = 0.5 + step * np.arange(1, 7)
        x[36:42] = 0.5 + step * np.arange(5, -1, -1)
        y = np.full(100, 0.5)
        moving = ball_moving(x, y, PITCH, FPS)
        self.assertTrue(moving[42:56].all())
        self.assertFalse(moving[:30].any())

    def test_jitter_of_a_parked_ball_is_not_movement(self):
        # Pilota aturada que el tracker fa tremolar 3 cm amunt i avall cada frame.
        x = 0.5 + (0.015 / PITCH[0]) * (-1.0) ** np.arange(100)
        y = np.full(100, 0.5)
        self.assertFalse(ball_moving(x, y, PITCH, FPS).any())

    def test_single_frame_tracking_glitch_is_not_movement(self):
        # Un salt de 3 m durant un sol frame (75 m/s): error del tracker, no una pilota que es mou.
        x = np.full(100, 0.5)
        x[60] = 0.5 + 3.0 / PITCH[0]
        y = np.full(100, 0.5)
        self.assertFalse(ball_moving(x, y, PITCH, FPS).any())


class SlowPossessionTest(unittest.TestCase):
    """Troballa del partit real (diagnòstic agregat): la velocitat col·lectiva és unimodal, i el joc de
    possessió lent és tan lent com una aturada. El que els separa és la pilota -- passada d'un a
    l'altre en un cas, quieta o desapareguda en l'altre."""

    def test_ball_moving_in_the_pitch_points_to_play(self):
        n = 50
        moving = np.zeros(n, dtype=bool)
        moving[25:] = True
        evidence = dead_evidence(np.full(n, 0.8), 1.7, np.full(n, 0.5), np.full(n, 0.5), PITCH,
                                 moving=moving)
        self.assertTrue((evidence[:25] > 0).all())
        self.assertTrue((evidence[25:] < 0).all())

    def test_slow_build_up_with_the_ball_moving_is_play(self):
        n = 6000
        rng = np.random.default_rng(0)
        speed = rng.normal(2.0, 0.6, n).clip(0.2)
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
        build_up = slice(1000, 1750)                 # 30 s de passades lentes a la defensa
        speed[build_up] = rng.normal(1.0, 0.2, 750)
        x[build_up] = 0.3 + 0.08 * np.sin(2 * np.pi * np.arange(750) / 75)
        y[build_up] = 0.5
        speed[3000:3400] = rng.normal(0.9, 0.2, 400)  # aturada de 16 s: sense pilota, caminant
        dead = dead_mask(speed, x, y, blocks=[(1, 0, n - 1)], events=None, pitch=PITCH, fps=FPS)
        self.assertLess(dead[build_up].mean(), 0.1)
        self.assertGreater(dead[3050:3350].mean(), 0.9)


class KickoffWindowTest(unittest.TestCase):

    def test_undetected_first_half_kickoff_does_not_kill_the_half(self):
        # La primera part es puntua des del mateix xut inicial, així que la seva finestra de
        # Kick Off comença abans del bloc. Si no es troba el xut, abans del bloc no hi ha res a
        # forçar.
        n = 3000
        speed = np.full(n, 2.6)
        speed[2000:2300] = 0.9
        nan = np.full(n, np.nan)
        events = [{"code": "SET PIECES", "start_frame": 12, "end_frame": 137,
                   "tags": {"Type": "Kick Off (Start)"}}]
        dead = dead_mask(speed, nan, nan, blocks=[(1, 100, n - 1)], events=events, pitch=PITCH,
                         fps=FPS)
        self.assertFalse(dead[100:1900].any())


if __name__ == "__main__":
    unittest.main()
