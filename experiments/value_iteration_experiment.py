"""
Expérience 1 : Value Iteration et vérifications numériques de la théorie.

* calcul de Q*, V*, politique optimale (gamma = 0.9, critère ||Q_{k+1}-Q_k||_inf < 1e-8) ;
* vitesse de convergence de Value Iteration comparée à la borne gamma^k ||Q_0 - Q*||_inf ;
* vérification empirique de la contraction ||TQ1 - TQ2|| <= gamma ||Q1 - Q2|| ;
* vérification que Q* domine Q^pi pour des politiques tirées au hasard ;
* borne |Q*| <= R_max / (1 - gamma) ;
* borne de perte d'une politique gloutonne 2 eps / (1 - gamma).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment import GridWorld, ACTIONS  # noqa: E402
from src.value_iteration import (  # noqa: E402
    value_iteration, bellman_optimality_operator, greedy_policy, evaluate_policy_exact,
    deterministic_to_matrix, optimal_action_sets,
)
from src.utils import draw_grid, policy_to_text, save_json, savefig, sup_norm, PALETTE  # noqa: E402

SEED = 7
GAMMA = 0.9


def main():
    rng = np.random.default_rng(SEED)
    env = GridWorld()
    P, R = env.P, env.R
    nS, nA = env.n_states, env.n_actions
    out = {"seed": SEED, "gamma": GAMMA, "n_states": nS, "n_actions": nA,
           "cells": env.cells, "slip": env.slip}

    # ---------- Value Iteration (critère du cahier des charges) ----------
    Q_vi, inc, iterates = value_iteration(P, R, GAMMA, tol=1e-8)
    # référence quasi exacte (tolérance 1e-13) pour mesurer ||Q_k - Q*||
    Q_ref, _, _ = value_iteration(P, R, GAMMA, tol=1e-13)
    err = np.array([sup_norm(Qk - Q_ref) for Qk in iterates])
    k = np.arange(len(err))
    bound = GAMMA ** k * err[0]
    out.update(
        vi_iterations=len(inc), vi_last_increment=float(inc[-1]),
        vi_aposteriori_bound=float(GAMMA / (1 - GAMMA) * inc[-1]),
        vi_true_error=float(sup_norm(Q_vi - Q_ref)),
        vi_error0=float(err[0]),
        vi_bound_respected=bool(np.all(err <= bound + 1e-12)),
        vi_ratio_last=float(err[-2] / err[-3]) if len(err) > 3 else None,
        vi_ratios=[float(err[i + 1] / err[i]) for i in range(len(err) - 1) if err[i] > 1e-11],
        vi_iter_needed_by_bound=int(np.ceil(np.log(1e-8 / err[0]) / np.log(GAMMA))),  # plus petit k avec gamma^k e0 <= 1e-8
    )
    Q_star = Q_ref
    V_star = Q_star.max(axis=1)
    pol = greedy_policy(Q_star)
    opt = optimal_action_sets(Q_star)
    out.update(Q_star=Q_star, V_star=V_star, policy=pol,
               policy_text=policy_to_text(env, pol),
               optimal_action_sets={str(env.cells[s]): [ACTIONS[a] for a in range(nA) if opt[s, a]]
                                    for s in range(nS) if s != env.goal},
               V_star_start=float(V_star[env.start]))

    # ---------- Bornitude ----------
    out["r_max"] = env.r_max
    out["return_bound"] = env.r_max / (1 - GAMMA)
    out["Q_star_sup"] = float(sup_norm(Q_star))

    # ---------- Contraction ----------
    ratios = {}
    for g in [0.5, 0.8, 0.9, 0.95, 0.99]:
        rs = []
        for _ in range(20_000):
            scale = 10 ** rng.uniform(-3, 2)
            Q1 = rng.normal(0, scale, (nS, nA))
            Q2 = rng.normal(0, scale, (nS, nA))
            num = sup_norm(bellman_optimality_operator(Q1, P, R, g) - bellman_optimality_operator(Q2, P, R, g))
            rs.append(num / sup_norm(Q1 - Q2))
        # cas d'égalité : Q2 = Q1 + c
        Q1 = rng.normal(0, 1, (nS, nA))
        eq = sup_norm(bellman_optimality_operator(Q1 + 3.0, P, R, g) - bellman_optimality_operator(Q1, P, R, g)) / 3.0
        ratios[g] = dict(max=float(np.max(rs)), mean=float(np.mean(rs)), equality_case=float(eq),
                         samples=np.array(rs))
    out["contraction"] = {str(g): {k2: v for k2, v in d.items() if k2 != "samples"} for g, d in ratios.items()}

    # ---------- Optimalité : Q* domine Q^pi ----------
    V_pistar, Q_pistar = evaluate_policy_exact(P, R, GAMMA, deterministic_to_matrix(pol, nA))
    # Explication du taux asymptotique : rayon spectral de P_{pi*} restreinte aux états non terminaux
    P_pi = np.einsum("sa,sat->st", deterministic_to_matrix(pol, nA), P)
    nt = env.non_terminal_states()
    rho = float(max(abs(np.linalg.eigvals(P_pi[np.ix_(nt, nt)]))))
    out.update(spectral_radius_transient=rho, asymptotic_rate_predicted=GAMMA * rho)
    out["greedy_policy_value_matches_Q_star"] = float(sup_norm(Q_pistar - Q_star))
    worst_gap, n_pol = -np.inf, 20_000
    best_random = -np.inf
    for i in range(n_pol):
        if i % 2 == 0:
            pi = deterministic_to_matrix(rng.integers(0, nA, nS), nA)
        else:
            pi = rng.dirichlet(np.ones(nA), nS)
        V_pi, Q_pi = evaluate_policy_exact(P, R, GAMMA, pi)
        worst_gap = max(worst_gap, float(np.max(Q_pi - Q_star)))
        best_random = max(best_random, float(V_pi[env.start]))
    out.update(n_random_policies=n_pol, max_Qpi_minus_Qstar=worst_gap,
               best_random_policy_value_start=best_random)

    # ---------- Perte d'une politique gloutonne approchée (Singh & Yee) ----------
    losses = []
    for eps in [0.05, 0.2, 0.5, 1.0, 2.0]:
        worst = 0.0
        for _ in range(2000):
            Qa = Q_star + rng.uniform(-eps, eps, (nS, nA))
            Qa[env.goal] = 0
            Vg, _ = evaluate_policy_exact(P, R, GAMMA, deterministic_to_matrix(greedy_policy(Qa), nA))
            worst = max(worst, float(np.max(V_star - Vg)))
        losses.append(dict(eps=eps, worst_loss=worst, bound=2 * eps / (1 - GAMMA)))
    out["greedy_loss"] = losses

    save_json(out, "value_iteration.json")
    np.save(os.path.join(os.path.dirname(__file__), "..", "results", "data", "Q_star_gamma0.9.npy"), Q_star)

    # ---------- Figures ----------
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    draw_grid(env, ax, title="GridWorld $4\\times4$ (14 états)")
    savefig(fig, "fig1_gridworld.pdf")

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    draw_grid(env, ax, values=V_star, policy=pol, title=r"$V^*$ et politique optimale ($\gamma=0.9$)")
    savefig(fig, "fig2_optimal_policy.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.semilogy(k, err, "o-", ms=3, color=PALETTE[0], label=r"$\|Q_k-Q^*\|_\infty$ (mesuré)")
    ax.semilogy(k, bound, "--", color=PALETTE[1], label=r"borne $\gamma^k\|Q_0-Q^*\|_\infty$")
    ax.semilogy(np.arange(1, len(inc) + 1), inc, ":", color=PALETTE[2], label=r"$\|Q_{k}-Q_{k-1}\|_\infty$")
    ax.axhline(1e-8, color="grey", lw=0.8)
    ax.set_xlabel("itération $k$")
    ax.set_title("Convergence de Value Iteration ($\\gamma=0.9$)")
    ax.legend(fontsize=8)
    ax = axes[1]
    gs = list(ratios.keys())
    ax.boxplot([ratios[g]["samples"] for g in gs], tick_labels=[str(g) for g in gs], showfliers=False)
    ax.scatter(range(1, len(gs) + 1), gs, marker="_", s=600, color=PALETTE[1], zorder=3, label=r"$\gamma$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\|TQ_1-TQ_2\|_\infty / \|Q_1-Q_2\|_\infty$")
    ax.set_title("Contraction : 20 000 couples aléatoires par $\\gamma$")
    ax.legend()
    savefig(fig, "fig_vi_convergence_contraction.pdf")

    print(policy_to_text(env, pol))
    print({k2: v for k2, v in out.items() if k2 in (
        "vi_iterations", "vi_last_increment", "vi_aposteriori_bound", "vi_true_error", "vi_error0",
        "vi_bound_respected", "vi_iter_needed_by_bound", "Q_star_sup", "return_bound",
        "greedy_policy_value_matches_Q_star", "max_Qpi_minus_Qstar", "best_random_policy_value_start",
        "V_star_start", "spectral_radius_transient", "asymptotic_rate_predicted")})
    print(out["contraction"])
    print(out["greedy_loss"])
    print(out["optimal_action_sets"])
    print("ratios VI:", out["vi_ratios"][:5], out["vi_ratios"][-5:])


if __name__ == "__main__":
    main()
