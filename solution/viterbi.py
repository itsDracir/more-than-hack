"""La millor seqüència d'estats donats uns indicis per frame i un cost per cada canvi d'estat.

Substitueix el vot per majoria del baseline. Un vot sobre +/-2 s desplaça cada límit fins a la
mida de la finestra i deixa que un desviament de tres frames canviï l'etiqueta si cau al moment
adequat. Viterbi, en canvi, troba l'única seqüència d'etiquetes que maximitza

    sum_t emissions[t, state_t]  -  sum_t switch_cost[state_{t-1}, state_t]

de manera que el cost de canvi és un control directe i interpretable: "quant d'indici cal per creure
que la possessió ha canviat", en unitats de frames d'indici clar. Això és el que controla el nombre
de possessions, que és exactament el que puntua l'spellF1.
"""

import numpy as np


def viterbi(emissions, switch_cost):
    """El camí més probable. `emissions` és un log-indici (T, K): més alt = més probable.

    `switch_cost` és un escalar (el mateix cost per a qualsevol canvi) o una matriu (K, K) amb
    cost[des_de, cap_a]; la diagonal s'ignora. Els empats van a l'estat d'índex més baix, de forma
    determinista.
    """
    emissions = np.asarray(emissions, dtype=float)
    n_frames, n_states = emissions.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.int8)

    if np.isscalar(switch_cost):
        cost = np.full((n_states, n_states), float(switch_cost))
    else:
        cost = np.array(switch_cost, dtype=float)
    np.fill_diagonal(cost, 0.0)

    back = np.empty((n_frames, n_states), dtype=np.int8)
    score = emissions[0].copy()
    back[0] = np.arange(n_states)
    for t in range(1, n_frames):
        candidates = score[:, None] - cost          # [des_de, cap_a]
        best_from = np.argmax(candidates, axis=0)
        back[t] = best_from
        score = candidates[best_from, np.arange(n_states)] + emissions[t]

    path = np.empty(n_frames, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
