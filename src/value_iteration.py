"""
Opérateur de Bellman, Value Iteration et évaluation exacte d'une politique.

Notations (MDP fini) :
    P[s, a, s'] = P(s' | s, a),  R[s, a, s'] = R(s, a, s').

Opérateur d'optimalité de Bellman sur les fonctions Q : R^{S x A} -> R^{S x A}
    (TQ)(s, a) = sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * max_{a'} Q(s', a') ].

Opérateur de Bellman d'une politique pi (matrice pi[s, a] = pi(a|s)) :
    (T^pi Q)(s, a) = sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * sum_{a'} pi(a'|s') Q(s', a') ].
"""

from __future__ import annotations

import numpy as np


def expected_reward(P, R):
    """r(s, a) = E[R | s, a] = sum_{s'} P(s'|s,a) R(s,a,s')."""
    return np.einsum("ijk,ijk->ij", P, R)


def bellman_optimality_operator(Q, P, R, gamma):
    """(TQ)(s,a) = r(s,a) + gamma * sum_{s'} P(s'|s,a) max_{a'} Q(s',a')."""
    return expected_reward(P, R) + gamma * P @ Q.max(axis=1)


def bellman_policy_operator(Q, P, R, gamma, pi):
    """(T^pi Q)(s,a) = r(s,a) + gamma * sum_{s'} P(s'|s,a) sum_{a'} pi(a'|s') Q(s',a')."""
    v = (pi * Q).sum(axis=1)
    return expected_reward(P, R) + gamma * P @ v


def value_iteration(P, R, gamma, tol=1e-8, max_iter=100_000, Q0=None):
    """
    Itère Q_{k+1} = T Q_k jusqu'à ce que ||Q_{k+1} - Q_k||_inf < tol.

    D'après l'estimation a posteriori du théorème de Banach,
        ||Q_{k+1} - Q*||_inf <= gamma / (1 - gamma) * ||Q_{k+1} - Q_k||_inf,
    donc à l'arrêt l'erreur est au plus gamma * tol / (1 - gamma).

    Renvoie (Q, historique des incréments ||Q_{k+1} - Q_k||_inf, liste des Q_k).
    """
    nS, nA, _ = P.shape
    Q = np.zeros((nS, nA)) if Q0 is None else np.array(Q0, dtype=float)
    increments, iterates = [], [Q.copy()]
    for _ in range(max_iter):
        Q_new = bellman_optimality_operator(Q, P, R, gamma)
        inc = float(np.max(np.abs(Q_new - Q)))
        increments.append(inc)
        Q = Q_new
        iterates.append(Q.copy())
        if inc < tol:
            break
    else:
        raise RuntimeError("Value Iteration n'a pas convergé en max_iter itérations")
    return Q, np.array(increments), iterates


def greedy_policy(Q):
    """Politique déterministe pi(s) = argmax_a Q(s, a) (plus petit indice en cas d'égalité)."""
    return np.argmax(Q, axis=1)


def optimal_action_sets(Q_star, atol=1e-6):
    """Pour chaque état, l'ensemble des actions a telles que Q*(s,a) = max_a' Q*(s,a') (à atol près)."""
    m = Q_star.max(axis=1, keepdims=True)
    return np.abs(Q_star - m) <= atol  # tableau booléen (nS, nA)


def deterministic_to_matrix(policy, n_actions):
    """Politique déterministe (tableau d'actions) -> matrice pi[s, a]."""
    pi = np.zeros((len(policy), n_actions))
    pi[np.arange(len(policy)), policy] = 1.0
    return pi


def evaluate_policy_exact(P, R, gamma, pi):
    """
    Évaluation exacte d'une politique stationnaire pi.

    V^pi est l'unique solution du système linéaire (I - gamma P_pi) V = r_pi, où
        P_pi[s, s'] = sum_a pi(a|s) P(s'|s,a),   r_pi[s] = sum_a pi(a|s) r(s,a).
    La matrice I - gamma P_pi est inversible car ||gamma P_pi||_inf = gamma < 1
    (série de Neumann). Puis Q^pi(s,a) = r(s,a) + gamma sum_{s'} P(s'|s,a) V^pi(s').
    """
    nS = P.shape[0]
    r = expected_reward(P, R)
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * r).sum(axis=1)
    V = np.linalg.solve(np.eye(nS) - gamma * P_pi, r_pi)
    Q = r + gamma * P @ V
    return V, Q
