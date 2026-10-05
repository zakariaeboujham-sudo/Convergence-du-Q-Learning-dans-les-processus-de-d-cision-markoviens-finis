"""
Politiques de comportement utilisées par Q-Learning.

* greedy :          a = argmax_a Q(s, a)
* epsilon-greedy :  avec probabilité 1 - eps, a = argmax_a Q(s, a) ;
                    avec probabilité eps, a ~ Uniforme(A).

Les égalités dans l'argmax sont départagées uniformément au hasard
(sinon, avec Q_0 = 0, la politique gloutonne choisirait toujours
l'action d'indice 0 au début, ce qui introduirait un biais arbitraire).
On peut aussi demander un départage déterministe (plus petit indice).
"""

from __future__ import annotations

import numpy as np


def argmax_random_tie(q_row, rng):
    m = q_row.max()
    candidates = np.flatnonzero(q_row == m)
    if len(candidates) == 1:
        return int(candidates[0])
    return int(rng.choice(candidates))


def greedy(Q, s, rng, random_tie=True):
    if random_tie:
        return argmax_random_tie(Q[s], rng)
    return int(np.argmax(Q[s]))


def epsilon_greedy(Q, s, epsilon, rng, random_tie=True):
    if epsilon > 0 and rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return greedy(Q, s, rng, random_tie)


def epsilon_greedy_probabilities(Q, s, epsilon):
    """Loi pi(.|s) de la politique epsilon-greedy (égalités partagées uniformément)."""
    nA = Q.shape[1]
    q = Q[s]
    best = q == q.max()
    probs = np.full(nA, epsilon / nA)
    probs[best] += (1.0 - epsilon) / best.sum()
    return probs
