"""
Partie 2 : exemple de chaîne de Markov à 4 états.
Calcul de P^2, P^3, vérification Monte-Carlo de (P^n)_{ij} = P(X_n = j | X_0 = i),
et loi stationnaire.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.markov_chain import check_stochastic, empirical_n_step, stationary_distribution  # noqa: E402
from src.utils import save_json, savefig  # noqa: E402

SEED = 2026
P = np.array([
    [0.5, 0.3, 0.2, 0.0],
    [0.2, 0.5, 0.2, 0.1],
    [0.1, 0.3, 0.4, 0.2],
    [0.0, 0.2, 0.3, 0.5],
])


def main():
    assert check_stochastic(P)
    rng = np.random.default_rng(SEED)
    P2, P3 = P @ P, P @ P @ P
    P50 = np.linalg.matrix_power(P, 50)
    mu = stationary_distribution(P)
    n_runs = 100_000
    emp3 = empirical_n_step(P, 0, 3, n_runs, rng)
    out = dict(P=P, P2=P2, P3=P3, P50=P50, stationary=mu,
               P2_row_sums=P2.sum(axis=1), P3_row_sums=P3.sum(axis=1),
               mc_n_runs=n_runs, mc_row0_n3=emp3, exact_row0_n3=P3[0],
               mc_max_abs_dev=float(np.max(np.abs(emp3 - P3[0]))),
               mc_std_bound=float(np.sqrt(0.25 / n_runs)), seed=SEED)
    save_json(out, "markov_chain.json")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, M, t in zip(axes, [P, P2, P3, P50], ["$P$", "$P^2$", "$P^3$", "$P^{50}$"]):
        ax.imshow(M, cmap="Blues", vmin=0, vmax=0.6)
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(4), [f"{k+1}" for k in range(4)])
        ax.set_yticks(range(4), [f"{k+1}" for k in range(4)])
        ax.set_xlabel("état d'arrivée $j$")
        ax.set_title(t)
    axes[0].set_ylabel("état de départ $i$")
    savefig(fig, "fig_markov_powers.pdf")
    np.set_printoptions(precision=4, suppress=True)
    print("P^2 =\n", P2, "\nP^3 =\n", P3, "\nP^50 =\n", P50, "\nmu =", mu)
    print("MC ligne 1 de P^3 :", emp3, " exact :", P3[0])


if __name__ == "__main__":
    main()
