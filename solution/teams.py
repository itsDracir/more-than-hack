"""Quin equip té la pilota, dins dels trams que ja se sap que són en joc.

L'entrada és la distància per frame de la pilota al jugador local més proper i al visitant més
proper -- el que produeix el preprocessament (home_nearest_m / away_nearest_m, NaN quan la pilota
no es veu).

Dos passos:

1. **Viterbi** sobre {local, visitant} amb un cost per cada canvi de possessió. Un desviament o
   un duel perdut dona a l'altre equip uns quants frames d'indici, no prou per pagar dos canvis,
   així que no parteix la possessió. Una intercepció real dona un indici sostingut i sí que ho fa.
2. **Moure cada canvi al primer toc del nou equip.** Entre l'últim toc de l'equip antic i el
   primer del nou, la pilota està solta o per l'aire i l'indici és indiferent, així que Viterbi
   pot posar el canvi a qualsevol punt d'aquest buit. El ground truth dona la pilota al nou equip
   quan n'agafa el control -- les pilotes soltes es queden amb l'últim equip (CHALLENGE.md) -- així
   que el canvi va al primer toc. Un segon tard costa un terç de l'spellF1 (EVALUATION.md).

Els frames sense pilota no aporten cap indici i simplement continuen la possessió actual, que és
el que ha de fer una oclusió dins d'una possessió.
"""

import numpy as np

from solution.params import Params
from solution.viterbi import viterbi

HOME, AWAY, NONE = 0, 1, -1


def team_evidence(home_nearest_m, away_nearest_m, params=None):
    """Indici per frame que l'equip local (positiu) o el visitant (negatiu) té la pilota.

    Només és fort quan algú està realment a prop de la pilota: ponderat per exp(-més_proper / r).
    Un equip sense cap jugador seguit a prop de la pilota es tracta com si fos infinitament lluny.
    """
    p = params or Params()
    home = np.where(np.isfinite(home_nearest_m), home_nearest_m, np.inf)
    away = np.where(np.isfinite(away_nearest_m), away_nearest_m, np.inf)
    with np.errstate(invalid="ignore"):
        closest = np.minimum(home, away)
        margin = np.clip(away - home, -10.0, 10.0)          # inf - inf -> NaN -> sense indici
        evidence = np.exp(-closest / p.control_radius_m) * np.tanh(margin / p.margin_scale_m)
    evidence[~np.isfinite(evidence)] = 0.0
    return evidence + 0.0


def runs_of(mask):
    """[(start, stop)] trams semioberts on `mask` és True."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _touches(evidence, team, threshold):
    return evidence > threshold if team == HOME else evidence < -threshold


def move_changes_to_first_touch(path, evidence, threshold):
    """Desplaça cada canvi d'equip al primer toc del nou equip després de l'últim de l'antic."""
    out = path.copy()
    changes = (np.flatnonzero(path[1:] != path[:-1]) + 1).tolist()
    bounds = [0] + changes + [len(path)]
    for i, change in enumerate(changes):
        old, new = int(path[change - 1]), int(path[change])
        previous_change, next_change = bounds[i], bounds[i + 2]
        old_touches = np.flatnonzero(_touches(evidence[previous_change:change], old, threshold))
        last_old = previous_change + int(old_touches[-1]) if len(old_touches) else previous_change
        new_touches = np.flatnonzero(_touches(evidence[last_old + 1:next_change], new, threshold))
        if not len(new_touches):
            continue
        first_new = last_old + 1 + int(new_touches[0])
        if first_new > change:
            out[change:first_new] = old
        else:
            out[first_new:change] = new
    return out


def segment_teams(evidence, alive, params=None):
    """Codi d'equip per frame: HOME / AWAY dins dels trams en joc, NONE als frames morts i als
    trams en joc on la pilota no es veu mai (el scorer els omple amb l'etiqueta anterior; endevinar
    només hi afegiria soroll)."""
    p = params or Params()
    evidence = np.asarray(evidence, dtype=float)
    codes = np.full(len(evidence), NONE, dtype=np.int8)
    for start, stop in runs_of(alive):
        run = evidence[start:stop]
        if not np.any(run != 0):
            continue
        path = viterbi(np.column_stack([run / 2.0, -run / 2.0]), p.team_switch_cost)
        codes[start:stop] = move_changes_to_first_touch(path, run, p.touch_evidence)
    return codes
