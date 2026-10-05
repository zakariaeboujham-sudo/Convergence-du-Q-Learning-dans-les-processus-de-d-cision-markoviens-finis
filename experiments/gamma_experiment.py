"""
Expérience sur gamma in {0.5, 0.8, 0.9, 0.95, 0.99}.
* Value Iteration : nombre d'itérations, borne théorique, taux asymptotique, Q*_gamma, politique.
* Q-Learning (départs aléatoires, eps = 0.1, alpha = 1/N^0.8, 10 graines, 50 000 épisodes) :
  erreur absolue et relative ||Q_t - Q*_gamma||_inf / ||Q*_gamma||_inf.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import run_many, stack, summarize  # noqa: E402
from src.environment import GridWorld  # noqa: E402
from src.value_iteration import value_iteration, greedy_policy, optimal_action_sets, deterministic_to_matrix  # noqa: E402
from src.utils import save_json, savefig, mean_and_band, draw_grid, policy_to_text, sup_norm, PALETTE  # noqa: E402

GAMMAS = [0.5, 0.8, 0.9, 0.95, 0.99]
SEEDS = list(range(10))
N_EP = 50_000


def main():
    env = GridWorld()
    out, curves, vi_err = {}, {}, {}
    for g in GAMMAS:
        Q8, inc, it = value_iteration(env.P, env.R, g, tol=1e-8)
        Qs, _, _ = value_iteration(env.P, env.R, g, tol=1e-13)
        e = np.array([sup_norm(Q - Qs) for Q in it])
        vi_err[g] = e
        pol = greedy_policy(Qs)
        P_pi = np.einsum("sa,sat->st", deterministic_to_matrix(pol, 4), env.P)
        nt = env.non_terminal_states()
        rho = float(max(abs(np.linalg.eigvals(P_pi[np.ix_(nt, nt)]))))
        opt = optimal_action_sets(Qs)
        cfg, runs = run_many(dict(gamma=g, random_start=True, n_episodes=N_EP, eval_every=50), SEEDS)
        s = summarize(cfg, runs)
        E = stack(runs, "error_inf")
        ee = runs[0]["eval_episodes"]
        nQ = sup_norm(Qs)
        curves[g] = (ee, E / nQ, E)
        idx = {c: int(np.searchsorted(ee, c)) for c in [1000, 10_000, 50_000]}
        out[str(g)] = dict(
            vi_iterations=len(inc),
            vi_iter_bound=int(np.ceil(np.log(1e-8 / e[0]) / np.log(g))),  # plus petit k avec g^k e0 <= 1e-8
            asymptotic_rate=g * rho, spectral_radius=rho,
            Q_star_sup=nQ, V_star_start=float(Qs.max(1)[env.start]), return_bound=env.r_max / (1 - g),
            policy=policy_to_text(env, pol),
            n_optimal_actions_per_state=[int(x) for x in opt[:-1].sum(1)],
            ql_err_inf={c: float(E[:, i].mean()) for c, i in idx.items()},
            ql_rel_err={c: float((E[:, i] / nQ).mean()) for c, i in idx.items()},
            ql_frac_optimal_final=s["final_frac_optimal_mean"],
            ql_stabilized=s["n_seeds_stabilized"], ql_stab_median=s["stabilization_median"],
            ql_avg_length_last=s["avg_length_last"],
        )
        print(g, {k: v for k, v in out[str(g)].items() if k != "policy"})
        print(out[str(g)]["policy"])
    save_json(out, "gamma_experiment.json")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c, g in zip(PALETTE, GAMMAS):
        ee, rel, _ = curves[g]
        m, lo, hi = mean_and_band(rel)
        axes[0].loglog(ee[1:], m[1:], color=c, label=rf"$\gamma={g}$")
        axes[0].fill_between(ee[1:], lo[1:], hi[1:], color=c, alpha=0.15)
        e = vi_err[g]
        axes[1].semilogy(np.arange(len(e)), np.maximum(e, 1e-16), "o-", ms=2, color=c, label=rf"$\gamma={g}$")
        axes[1].semilogy(np.arange(len(e)), g ** np.arange(len(e)) * e[0], ":", color=c, lw=1)
    axes[0].set_title(r"Q-Learning : $\|Q_t-Q^*_\gamma\|_\infty / \|Q^*_\gamma\|_\infty$")
    axes[0].set_xlabel("épisode")
    axes[0].legend(fontsize=8)
    axes[1].set_title(r"Value Iteration : $\|Q_k-Q^*_\gamma\|_\infty$ (pointillés : $\gamma^k\|Q_0-Q^*\|$)")
    axes[1].set_xlabel("itération $k$")
    axes[1].set_ylim(1e-9, 200)
    axes[1].set_xlim(0, 45)
    axes[1].legend(fontsize=8)
    savefig(fig, "fig5_gamma.pdf")

    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(3.1 * len(GAMMAS), 3.3))
    for ax, g in zip(axes, GAMMAS):
        Qs, _, _ = value_iteration(env.P, env.R, g, tol=1e-13)
        draw_grid(env, ax, values=Qs.max(1), policy=greedy_policy(Qs), title=rf"$\gamma={g}$", show_labels=False)
    savefig(fig, "fig5b_gamma_policies.pdf")


if __name__ == "__main__":
    main()
