"""
Expérience sur epsilon (départ FIXE en S : l'exploration ne vient que de la politique).
1) epsilon in {0, 0.05, 0.1, 0.3}, Q_0 = 0, gamma = 0.9, alpha = 1/N^0.8, 10 graines, 50 000 épisodes.
2) Exploration contre exploitation : initialisation pessimiste Q_0 = -R_max/(1-gamma) = -100,
   epsilon = 0 contre epsilon = 0.1.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import run_many, stack, summarize, solve_q_star  # noqa: E402
from src.environment import GridWorld  # noqa: E402
from src.value_iteration import greedy_policy  # noqa: E402
from src.utils import save_json, savefig, mean_and_band, moving_average, policy_to_text, PALETTE  # noqa: E402

EPS = [0.0, 0.05, 0.1, 0.3]
SEEDS = list(range(10))
N_EP = 50_000
N_EP_PESS = 5_000


def describe(runs, cfg, env, V_star_S):
    s = summarize(cfg, runs)
    ee = runs[0]["eval_episodes"]
    E, Es, G = stack(runs, "error_inf"), stack(runs, "error_start"), stack(runs, "greedy_value")
    idx = {c: int(np.searchsorted(ee, c)) for c in [100, 1000, 5_000, 10_000, 50_000] if c <= ee[-1]}
    never = [int((r["visits"][:-1] == 0).sum()) for r in runs]
    return dict(
        err_inf={c: float(E[:, i].mean()) for c, i in idx.items()},
        err_start={c: float(Es[:, i].mean()) for c, i in idx.items()},
        greedy_value={c: float(G[:, i].mean()) for c, i in idx.items()},
        greedy_value_final_min=float(G[:, -1].min()), greedy_value_final_max=float(G[:, -1].max()),
        n_seeds_greedy_optimal_from_S=int((np.abs(G[:, -1] - V_star_S) < 1e-6).sum()),
        online_return_last1000=s["avg_return_last"], online_length_last1000=s["avg_length_last"],
        online_return_all=float(np.mean([r["episode_return"].mean() for r in runs])),
        pairs_never_visited_mean=float(np.mean(never)), pairs_never_visited=never,
        frac_optimal_final=s["final_frac_optimal_mean"], stabilized=s["n_seeds_stabilized"],
        min_visits_mean=s["min_visits_mean"],
        n_episodes_truncated=int(sum((r["episode_length"] >= cfg["max_steps"]).sum() for r in runs)),
        policy_seed0=policy_to_text(env, greedy_policy(runs[0]["Q"])),
    ), (ee, E, Es, G, np.stack([moving_average(r["episode_return"], 200) for r in runs]))


def main():
    env = GridWorld()
    Q_star = solve_q_star(0.9)
    V_star_S = float(Q_star.max(1)[env.start])
    out, curves = {"V_star_start": V_star_S}, {}
    for e in EPS:
        cfg, runs = run_many(dict(epsilon=e, n_episodes=N_EP, eval_every=50), SEEDS)
        out[f"eps={e}"], curves[e] = describe(runs, cfg, env, V_star_S)
        print(e, {k: v for k, v in out[f"eps={e}"].items() if k != "policy_seed0"})
        print(out[f"eps={e}"]["policy_seed0"])
    q0 = -env.r_max / (1 - 0.9)
    pess = {}
    for e in [0.0, 0.1]:
        cfg, runs = run_many(dict(epsilon=e, Q0=np.full((env.n_states, env.n_actions), q0),
                                  n_episodes=N_EP_PESS, eval_every=20), SEEDS)
        out[f"pessimistic_eps={e}"], pess[e] = describe(runs, cfg, env, V_star_S)
        print("pess", e, {k: v for k, v in out[f"pessimistic_eps={e}"].items() if k != "policy_seed0"})
        print(out[f"pessimistic_eps={e}"]["policy_seed0"])
    out["pessimistic_Q0"] = q0
    save_json(out, "epsilon_experiment.json")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c, e in zip(PALETTE, EPS):
        ee, E, Es, G, R = curves[e]
        for ax, arr in [(axes[0], E), (axes[1], G)]:
            m, lo, hi = mean_and_band(arr)
            ax.semilogx(ee[1:], m[1:], color=c, label=rf"$\varepsilon={e}$")
            ax.fill_between(ee[1:], lo[1:], hi[1:], color=c, alpha=0.15)
        m, lo, hi = mean_and_band(R)
        x = np.arange(1, len(m) + 1)
        axes[2].semilogx(x, m, color=c, label=rf"$\varepsilon={e}$")
    axes[0].set_yscale("log")
    axes[0].set_title(r"$\|Q_t-Q^*\|_\infty$ (départ fixe)")
    axes[1].axhline(V_star_S, ls="--", color="grey", label=r"$V^*(S)$")
    axes[1].set_title(r"$V^{\pi_t}(S)$ de la politique gloutonne")
    axes[2].set_title("Récompense par épisode pendant l'apprentissage (moy. mobile 200)")
    for ax in axes:
        ax.set_xlabel("épisode")
        ax.legend(fontsize=8)
    savefig(fig, "fig7_epsilon.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for c, e in zip([PALETTE[1], PALETTE[0]], [0.0, 0.1]):
        ee, E, Es, G, R = pess[e]
        for ax, arr in [(axes[0], G), (axes[1], E)]:
            m, lo, hi = mean_and_band(arr)
            ax.semilogx(ee[1:], m[1:], color=c, label=rf"$\varepsilon={e}$")
            ax.fill_between(ee[1:], lo[1:], hi[1:], color=c, alpha=0.2)
    axes[0].axhline(V_star_S, ls="--", color="grey", label=r"$V^*(S)$")
    axes[0].set_title(rf"$V^{{\pi_t}}(S)$, initialisation pessimiste $Q_0={q0:.0f}$")
    axes[1].set_yscale("log")
    axes[1].set_title(r"$\|Q_t-Q^*\|_\infty$, initialisation pessimiste")
    for ax in axes:
        ax.set_xlabel("épisode")
        ax.legend(fontsize=8)
    savefig(fig, "fig7b_exploration_vs_exploitation.pdf")


if __name__ == "__main__":
    main()
