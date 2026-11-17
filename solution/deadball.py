"""La pilota és en joc? Es decideix amb indicis d'una aturada -- mai perquè "el tracker ha perdut
la pilota".

El baseline diu D sempre que la pilota no està seguida. Això barreja dues coses que no tenen res a
veure: la pilota falta en el 41% dels frames puntuats, sobretot *durant* el joc (duels, àrees
plenes, pilotes per l'aire). Cada D espuri parteix una possessió real en dues, i l'spellF1 perd
les dues meitats.

Indicis que es fan servir, tots automàtics:

- **pilota movent-se dins del camp**: recorre més de 2 m en un segon, dins de les línies. És el
  senyal de joc més fort, i el que separa una sortida pausada d'una aturada.
- **velocitat col·lectiva**: quan l'àrbitre atura el joc, els jugadors deixen de córrer -- però el
  joc de possessió lent és igual de lent, així que és un indici feble, relatiu al ritme mitjà de
  la part.
- **pilota més enllà d'una línia**: si el tracker la veu fora, és fora.
- **Smart Tagging SET PIECES**: dins de cada finestra hi ha una represa; el xut es localitza a la
  trajectòria de la pilota, els frames d'abans tendeixen a mort i el segon de després a joc.
- **Kick Off (Start)**: res del que passa abans del xut inicial d'una part és joc.

Un Viterbi de dos estats amb un cost per entrar/sortir d'una aturada ho converteix en intervals
nets.
"""

import numpy as np
import pandas as pd

from solution.params import Params
from solution.viterbi import viterbi

FORCED = 1e3  # indici prou gran perquè cap cost de canvi el pugui contradir


def speed_threshold(values):
    """Velocitat de referència per a "més lent del normal": la mediana de la part, sense NaN.

    No és una separació entre dues poblacions -- al partit real la velocitat col·lectiva suavitzada
    és una única gepa asimètrica (diagnòstic agregat), de manera que qualsevol tall en dues classes com
    Otsu cau enmig del joc i declara mort la major part del partit. La mediana només diu "més lent
    que el ritme típic d'aquesta part", i per això la velocitat és un indici feble i qui decideix és
    el moviment de la pilota.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def ball_speed_mps(x, y, pitch, fps):
    """Velocitat aparent de la pilota entre frames consecutius, en m/s. NaN si en algun dels dos
    frames no hi ha pilota."""
    length, width = pitch
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    speed = np.full(len(x), np.nan)
    if len(x) > 1:
        speed[1:] = np.hypot(np.diff(x) * length, np.diff(y) * width) * fps
    return speed


# Llindars físics de ball_moving (de Marc), no ajustats amb el ground truth.
MIN_STEP_M = 0.04       # 4 cm entre frames (1 m/s): per sota és el tremolor d'una pilota aturada
MAX_STEP_MPS = 45.0     # cap xut no hi arriba: un salt més ràpid és un error del tracker
MIN_VALID_SHARE = 0.70  # part dels passos de la finestra que s'han de veure per poder dir "es mou"


def ball_moving(x, y, pitch, fps, window_s=1.0, min_m=2.0):
    """True on la pilota, vista dins del camp, ha *recorregut* més de `min_m` en `window_s`.

    Es compta la distància recorreguda (la suma dels passos entre frames), no el desplaçament net:
    una paret o una conducció que torna enrere es mou encara que acabi on havia començat. Per no
    sumar soroll, els passos de menys de 4 cm no compten (el tremolor del tracker amb la pilota
    aturada: al partit real, la mediana és de 0,7 cm) i els de més de 45 m/s es descarten com a
    errors. Cal veure com a mínim el 70% dels passos de la finestra i la pilota dins del camp en el
    frame actual. Que el tracker perdi la pilota mai no és, per si sol, indici d'aturada.
    """
    length, width = pitch
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    lag = max(1, int(round(window_s * fps)))
    moving = np.zeros(n, dtype=bool)
    if n <= lag:
        return moving
    with np.errstate(invalid="ignore"):
        inside = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        steps = np.hypot(np.diff(x) * length, np.diff(y) * width)
        valid = inside[:-1] & inside[1:] & np.isfinite(steps) & (steps * fps <= MAX_STEP_MPS)
    travelled = np.r_[0.0, np.cumsum(np.where(valid & (steps >= MIN_STEP_M), steps, 0.0))]
    seen = np.r_[0, np.cumsum(valid.astype(np.int64))]
    window_m = travelled[lag:] - travelled[:-lag]
    window_seen = (seen[lag:] - seen[:-lag]) / lag
    moving[lag:] = inside[lag:] & (window_seen >= MIN_VALID_SHARE) & (window_m > min_m)
    return moving


def detect_restart(ball_speed, start, end, kick_speed=3.0, n_frames=3):
    """Primer frame dins de [start, end) on la pilota va més ràpid que `kick_speed` durant
    `n_frames` frames seguits -- el xut que reprèn el joc. None si la finestra no en té cap."""
    start, end = max(0, int(start)), min(len(ball_speed), int(end))
    if end - start < n_frames:
        return None
    with np.errstate(invalid="ignore"):
        fast = (np.nan_to_num(ball_speed[start:end], nan=0.0) > kick_speed).astype(int)
    running = np.convolve(fast, np.ones(n_frames, dtype=int), mode="valid")
    hits = np.flatnonzero(running == n_frames)
    return start + int(hits[0]) if len(hits) else None


def dead_evidence(speed_smooth, threshold, x, y, pitch, params=None, moving=None):
    """Log-odds per frame que la pilota està morta, abans de la informació de les jugades a pilota
    aturada.

    `moving` (booleà per frame, de ball_moving) marca la pilota viatjant dins del camp, el senyal
    de joc més fort que hi ha: pesa més que uns jugadors lents, de manera que una sortida pausada
    no es confon amb una aturada.
    """
    p = params or Params()
    length, width = pitch
    scale = max(p.speed_scale_frac * threshold, 1e-6) if np.isfinite(threshold) else np.inf
    with np.errstate(invalid="ignore"):
        evidence = p.w_speed * np.tanh(np.nan_to_num((threshold - speed_smooth) / scale, nan=0.0))
        mx, my = p.oob_margin_m / length, p.oob_margin_m / width
        out = (x < -mx) | (x > 1 + mx) | (y < -my) | (y > 1 + my)
    evidence = evidence + p.w_oob * out
    if moving is not None:
        evidence = evidence - p.w_move * np.asarray(moving, dtype=bool)
    return evidence + 0.0  # sense zeros negatius


def smooth_speed(collective_speed, fps, params=None):
    """Mediana mòbil centrada de la velocitat col·lectiva sobre params.speed_smooth_s segons."""
    p = params or Params()
    window = max(1, int(round(p.speed_smooth_s * fps)) | 1)
    return (pd.Series(np.asarray(collective_speed, dtype=float))
            .rolling(window, center=True, min_periods=1).median().to_numpy())


def _is_set_piece(event):
    return (event.get("code") or "").strip().upper() == "SET PIECES"


def _is_kickoff_start(event):
    return "kick off (start)" in (event.get("tags", {}).get("Type") or "").strip().lower()


def dead_mask(collective_speed, x, y, blocks, events, pitch, fps, params=None):
    """Array booleà: True on es considera que la pilota és fora de joc. Només es decideixen els
    blocs puntuats.

    `blocks` és [(half, first_frame, last_frame)] amb l'últim inclòs, tal com el retorna
    games.half_video_blocks(). `events` és la llista de Smart Tagging de
    starter/load_smart_tagging.load(), o None per funcionar sense.
    """
    p = params or Params()
    n = len(collective_speed)
    smooth = smooth_speed(collective_speed, fps, p)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ball_speed = ball_speed_mps(x, y, pitch, fps)
    moving = ball_moving(x, y, pitch, fps, p.move_window_s, p.move_min_m)

    dead = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        stop = last + 1
        threshold = speed_threshold(smooth[first:stop])
        evidence = dead_evidence(smooth[first:stop], threshold, x[first:stop], y[first:stop],
                                 pitch, p, moving[first:stop]) + p.dead_prior

        for event in events or []:
            if not _is_set_piece(event):
                continue
            begin, end = int(event["start_frame"]), int(event["end_frame"])
            if end <= first or begin > last:
                continue
            restart = detect_restart(ball_speed, max(begin, first), min(end, stop),
                                     p.kick_speed_mps, p.kick_frames)
            if _is_kickoff_start(event):
                # Abans del xut (o, si no s'ha trobat, abans de la finestra): encara no hi ha joc.
                # Limitat a 0: la primera part es puntua des del mateix xut inicial, així que la
                # seva finestra comença abans del bloc i un final de llesca negatiu cobriria tota
                # la part.
                evidence[:max(0, (restart if restart is not None else begin) - first)] = FORCED
            if restart is None:
                continue
            evidence[max(begin, first) - first:restart - first] += p.w_setpiece
            evidence[restart - first:min(restart + fps, stop) - first] -= p.w_setpiece

        emissions = np.column_stack([np.zeros(stop - first), evidence])
        dead[first:stop] = viterbi(emissions, p.dead_switch_cost) == 1
    return dead
