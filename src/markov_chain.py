"""Chaînes de Markov finies : matrice de transition, puissances, simulation."""

from __future__ import annotations

import numpy as np


def check_stochastic(P, atol=1e-12):
    """Vérifie p_ij >= 0 et sum_j p_ij = 1 (matrice stochastique)."""
    P = np.asarray(P, dtype=float)
    return bool(np.all(P >= 0) and np.allclose(P.sum(axis=1), 1.0, atol=atol))


def simulate_chain(P, x0, n_steps, rng):
    """Simule X_0 = x0, X_1, ..., X_n avec P(X_{t+1}=j | X_t=i) = P[i, j]."""
    cum = np.cumsum(P, axis=1)
    x = np.empty(n_steps + 1, dtype=int)
    x[0] = x0
    for t in range(n_steps):
        x[t + 1] = min(int(np.searchsorted(cum[x[t]], rng.random(), side="right")), len(P) - 1)
    return x


def empirical_n_step(P, i, n, n_runs, rng):
    """Estimation Monte-Carlo de la loi de X_n sachant X_0 = i (à comparer à la ligne i de P^n)."""
    counts = np.zeros(len(P))
    for _ in range(n_runs):
        counts[simulate_chain(P, i, n, rng)[-1]] += 1
    return counts / n_runs


def stationary_distribution(P):
    """Loi mu telle que mu P = mu (vecteur propre à gauche pour la valeur propre 1)."""
    w, v = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    mu = np.real(v[:, k])
    return mu / mu.sum()
