"""Tots els números ajustables del pipeline, en un sol lloc, amb el motiu del valor que tenen.

Cap d'ells s'ha ajustat amb els cinc minuts públics -- RULES.md ho prohibeix, i EVALUATION.md
mostra que, a més, tria la resposta equivocada. Es fixen a partir de com és un partit de futbol i
de les unitats en què estan. Si en canvies algun, canvia'l per un motiu que puguis escriure al
document tècnic, no perquè la finestra pública hagi pujat.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # -- pilota morta ----------------------------------------------------------------------
    # La velocitat col·lectiva dels jugadors se suavitza amb una mediana centrada sobre aquests
    # segons: prou llarg per ignorar un replegament trotant, prou curt per mantenir nets els
    # límits d'una aturada.
    speed_smooth_s: float = 2.0
    # La velocitat es compara amb el ritme mitjà (mediana) de la part ("més lent del normal") i
    # se satura dins d'aquesta fracció. És un indici FEBLE: al partit real la distribució de
    # velocitats és una sola gepa asimètrica, no dues poblacions, i el joc de possessió lent és
    # tan lent com una aturada.
    speed_scale_frac: float = 0.15
    w_speed: float = 0.5
    # La pilota que recorre més de move_min_m en move_window_s, dins del camp, és joc. Es compta la
    # distància recorreguda, no el desplaçament net (vegeu deadball.ball_moving). 2 m en 1 s és més
    # lent que qualsevol passada i molt més que el tremolor del tracker sobre una pilota aturada.
    # Pesa més que uns jugadors lents (w_move > w_speed), que és la gràcia.
    move_window_s: float = 1.0
    move_min_m: float = 2.0
    w_move: float = 2.0
    # La pilota vista més enllà d'una línia per més d'aquest marge és fora de joc, facin el que
    # facin els jugadors.
    oob_margin_m: float = 0.5
    w_oob: float = 3.0
    # Dins d'una finestra SET PIECES del Smart Tagging, abans del xut detectat: morta. Just
    # després: en joc.
    w_setpiece: float = 1.5
    # Un xut és la pilota anant més ràpid que això durant aquests frames consecutius.
    kick_speed_mps: float = 3.0
    kick_frames: int = 3
    # Cost d'entrar o sortir d'una aturada: ~1,6 s d'indici de velocitat ple (40 frames x 0,5).
    # Les aturades reals duren uns quants segons; això evita que una pausa breu es declari morta.
    dead_switch_cost: float = 20.0
    # Inclinació per frame cap a "en joc" quan res no diu el contrari. La pilota és en joc la
    # major part del partit i una aturada necessita una causa; sense això, els trams neutres
    # al costat d'una aturada s'hi enganxen perquè quedar-se mort no costa res. 20 / 0,1 = 8 s
    # de frames neutres compensen tornar al joc.
    dead_prior: float = -0.1

    # -- quin equip ------------------------------------------------------------------------
    # Un jugador a menys d'aquesta distància de la pilota probablement la juga; l'indici decau
    # exponencialment més enllà, de manera que una pilota per l'aire lluny de tothom no diu
    # gairebé res.
    control_radius_m: float = 1.5
    # Quant més a prop ha d'estar un equip perquè l'indici se saturi.
    margin_scale_m: float = 1.0
    # Cost d'un canvi de possessió, en unitats d'indici per frame. Un jugador clarament amb la
    # pilota dona ~0,7 per frame, així que això demana ~0,5 s de control clar de l'altre equip:
    # prou llarg per ignorar desviaments i duels d'un toc (CHALLENGE.md: "a simple touch of
    # the ball without control" no és un canvi), prou curt per detectar una intercepció real.
    team_switch_cost: float = 10.0
    # En moure un canvi de possessió al primer toc del nou equip, un frame compta com a toc quan
    # el seu indici és com a mínim així de fort.
    touch_evidence: float = 0.2
