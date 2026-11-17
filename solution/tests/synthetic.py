"""Un petit partit fals amb la resposta coneguda, per provar el pipeline de cap a peus sense les
dades.

Codifica les hipòtesis que fa el pipeline -- els jugadors afluixen quan la pilota és morta,
l'equip amb la pilota té un jugador a prop, el tracker perd la pilota a ràfegues -- així que
superar-lo demostra que el mecanisme funciona, no que les hipòtesis es compleixin al partit real.
"""

import numpy as np
import pandas as pd

PITCH = (105.0, 68.0)
FPS = 25
PAD = 100

PLAN = [("D", 300), ("H", 500), ("A", 350), ("D", 200), ("A", 600), ("H", 250), ("A", 400),
        ("D", 250), ("H", 700), ("D", 150), ("A", 450)]


def synthetic_match(seed=0):
    """(features, events, blocks, truth) per a una part sintètica."""
    rng = np.random.default_rng(seed)
    truth = np.array(["X"] * PAD + [lab for lab, n in PLAN for _ in range(n)] + ["X"] * PAD)
    n = len(truth)
    blocks = [(1, PAD, n - PAD - 1)]

    # Velocitat col·lectiva: corrent durant el joc, caminant durant una aturada.
    alive = (truth == "H") | (truth == "A")
    speed = np.where(alive, rng.normal(2.6, 0.4, n), rng.normal(0.9, 0.3, n)).clip(0, None)

    # El tracker perd la pilota a ràfegues: ~60% visible.
    visible = np.zeros(n, dtype=bool)
    state = True
    for t in range(n):
        state = rng.random() > 0.02 if state else rng.random() < 0.03
        visible[t] = state

    x = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, n)), 0.05, 0.95)
    y = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, n)), 0.05, 0.95)

    own = np.abs(rng.normal(0.8, 0.4, n))
    other = 2.5 + np.abs(rng.normal(0, 2.0, n))
    home = np.where(truth == "H", own, other)
    away = np.where(truth == "A", own, other)

    # Desviaments: l'altre equip toca la pilota durant tres frames, més o menys cada sis segons.
    for t in np.flatnonzero(alive & (rng.random(n) < 1 / 150.0)):
        team = truth[t]
        home[t:t + 3] = 3.0 if team == "H" else 0.4
        away[t:t + 3] = 0.4 if team == "H" else 3.0

    events = []
    start = PAD
    previous = None
    for index, (label, length) in enumerate(PLAN):
        stop = start + length
        if label == "D":
            home[start:stop] = away[start:stop] = 3.0
            if index:                                  # la pilota ha sortit per la línia de fons
                x[start:start + 15] = 1.02
                visible[start:start + 15] = True
            taker = PLAN[index + 1][0] if index + 1 < len(PLAN) else None
            if taker:                                  # represa: pilota quieta, llançador a sobre, xut
                x[stop - 60:stop] = x[stop - 61]
                y[stop - 60:stop] = y[stop - 61]
                visible[stop - 60:stop + 10] = True
                x[stop:stop + 10] = x[stop - 1] + 0.004 * np.arange(1, 11)
                y[stop:stop + 10] = y[stop - 1]
                near_home = taker == "H"
                home[stop - 30:stop] = 0.6 if near_home else 6.0
                away[stop - 30:stop] = 6.0 if near_home else 0.6
                events.append({"code": "SET PIECES", "start_frame": stop - 88,
                               "end_frame": stop + 37,
                               "tags": {"Type": "Kick Off (Start)" if not index else "Throw-in",
                                        "Team": "N/A"}})
        elif previous not in (None, "D", label):       # una pèrdua: pilota solta abans
            home[start - 15:start] = away[start - 15:start] = 2.5
        previous = label
        start = stop

    x = np.where(visible, x, np.nan)
    y = np.where(visible, y, np.nan)
    home = np.where(visible, home, np.nan)
    away = np.where(visible, away, np.nan)

    features = pd.DataFrame({
        "frame": np.arange(n),
        "ball_visible": visible,
        "ball_x_norm": x,
        "ball_y_norm": y,
        "home_nearest_m": home,
        "away_nearest_m": away,
        "home_mean_speed_mps": speed,
        "away_mean_speed_mps": speed,
    })
    return features, events, blocks, truth
