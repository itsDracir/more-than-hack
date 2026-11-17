"""Features per frame a l'entrada, etiquetes H / A / D / X a la sortida. La capa de decisió del
pipeline.

Contracte amb el constructor de features (el notebook de preprocessament, o qualsevol
possession.py que el substitueixi): un DataFrame amb una fila per frame de vídeo 0..N-1, en
ordre, amb aquestes columnes. Els noms coincideixen amb solution/01_preprocessing.ipynb, així que
el seu CSV s'hi connecta sense canvis.

    ball_x_norm, ball_y_norm      posició de la pilota, coordenades 0-1 del camp, NaN si no es veu
    home_nearest_m                distància pilota -> jugador local més proper (m), NaN si es desconeix
    away_nearest_m                distància pilota -> jugador visitant més proper (m), NaN si es desconeix
    home_mean_speed_mps           velocitat mitjana dels jugadors locals seguits (m/s)
    away_mean_speed_mps           velocitat mitjana dels jugadors visitants seguits (m/s)

Ordre de les decisions:
  1. fora de les parts puntuades              -> X (mai es puntua)
  2. pilota fora de joc (deadball.py)         -> D
  3. en joc: quin equip (teams.py)            -> H / A
  4. en joc però la pilota no es veu mai      -> X (el scorer hi posa l'etiqueta anterior)
"""

import os
import sys

import numpy as np

from solution.deadball import dead_mask
from solution.params import Params
from solution.teams import AWAY, HOME, segment_teams, team_evidence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import games  # noqa: E402

REQUIRED_COLUMNS = ("ball_x_norm", "ball_y_norm", "home_nearest_m", "away_nearest_m",
                    "home_mean_speed_mps", "away_mean_speed_mps")


def check_columns(features):
    missing = [c for c in REQUIRED_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError("features are missing columns {} (contract in solution/labeling.py)"
                         .format(missing))


def collective_speed(features):
    """Mitjana de les dues mitjanes d'equip, ignorant un equip sense cap jugador seguit."""
    speeds = features[["home_mean_speed_mps", "away_mean_speed_mps"]].to_numpy(dtype=float)
    counts = np.isfinite(speeds).sum(axis=1)
    total = np.nansum(speeds, axis=1)
    return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


def graded_mask(n, blocks):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return mask


def label_frames(features, events=None, blocks=None, pitch=None, fps=None, params=None):
    """Array de 'H' / 'A' / 'D' / 'X', una per cada fila de `features`."""
    check_columns(features)
    p = params or Params()
    blocks = blocks if blocks is not None else games.half_video_blocks(games.GAME)
    pitch = pitch or games.GAMES[games.GAME]["pitch"]
    fps = fps or games.GAMES[games.GAME]["fps"]
    n = len(features)

    x = features["ball_x_norm"].to_numpy(dtype=float)
    y = features["ball_y_norm"].to_numpy(dtype=float)
    dead = dead_mask(collective_speed(features), x, y, blocks, events, pitch, fps, p)

    graded = graded_mask(n, blocks)
    evidence = team_evidence(features["home_nearest_m"].to_numpy(dtype=float),
                             features["away_nearest_m"].to_numpy(dtype=float), p)
    teams = segment_teams(evidence, graded & ~dead, p)

    labels = np.full(n, "X", dtype="<U1")
    labels[graded & dead] = "D"
    labels[teams == HOME] = "H"
    labels[teams == AWAY] = "A"
    return labels
