"""
Expérience 2 : Q-Learning de référence.
gamma = 0.9, epsilon = 0.1, alpha_t(s,a) = 1/N_t(s,a)^0.8 (Robbins-Monro), Q_0 = 0,
20 graines, 100 000 épisodes. Mesure principale : E_t = ||Q_t - Q*||_inf.

Deux réglages :
  (A) départ fixe en S (réglage du cahier des charges) ;
  (B) départs aléatoires uniformes (« exploring starts »), pour lequel l'hypothèse
      « chaque paire (s,a) est visitée infiniment souvent » est satisfaite de façon
      beaucoup plus homogène.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import run_many, stack, summarize, solve_q_star  # noqa: E402
from src.environment import GridWorld  # noqa: E402
from src.value_iteration import greedy_policy  # noqa: E402
from src.utils import (save_json, savefig, mean_and_band, moving_average, draw_grid,  # noqa: E402
                       policy_to_text, PALETTE, DATA_DIR)

SEEDS = list(range(20))
N_EP = 100_000
SNAP = (10, 100, 1000, 10_000, N_EP)
CHECKPOINTS = [100, 1000, 10_000, 50_000, 100_000]


def analyse(cfg, runs, env, Q_star):
    ee = runs[0]["eval_episodes"]
    summ = summarize(cfg, runs)
    E, Em, Es = stack(runs, "error_inf"), stack(runs, "error_mean"), stack(runs, "error_start")
    idx = [int(np.searchsorted(ee, c)) for c in CHECKPOINTS]
    summ["checkpoints"] = {c: dict(err_inf=float(E[:, i].mean()), err_inf_median=float(np.median(E[:, i])),
                                   err_mean=float(Em[:, i].mean()), err_start=float(Es[:, i].mean()),
                                   frac_optimal=float(stack(runs, "frac_optimal")[:, i].mean()))
                           for c, i in zip(CHECKPOINTS, idx)}
    late = ee >= 10_000
    summ["loglog_slope_err_inf"] = float(np.polyfit(np.log(ee[late]), np.log(E.mean(0)[late]), 1)[0])
    summ["loglog_slope_err_mean"] = float(np.polyfit(np.log(ee[late]), np.log(Em.mean(0)[late]), 1)[0])
    Vis = np.stack([r["visits"] for r in runs]).mean(0)
    summ["visits_mean"] = Vis
    s_min, a_min = np.unravel_index(np.argmin(Vis[:-1]), Vis[:-1].shape)
    summ["visits_min_pair"] = dict(cell=str(env.cells[s_min]), action=int(a_min), value=float(Vis[s_min, a_min]))
    summ["visits_max"] = float(Vis[:-1].max())
    summ["visits_ratio_max_min"] = float(Vis[:-1].max() / Vis[:-1].min())
    cells = []
    for r in runs:
        d = np.abs(r["Q"] - Q_star)[:-1]
        cells.append(str(env.cells[np.unravel_index(np.argmax(d), d.shape)[0]]))
    summ["argmax_error_cells"] = {c: cells.count(c) for c in sorted(set(cells))}
    d_all = np.concatenate([np.abs(r["Q"] - Q_star)[:-1].ravel() for r in runs])
    v_all = np.concatenate([r["visits"][:-1].ravel() for r in runs])
    summ["corr_logvisits_logerror"] = float(np.corrcoef(np.log(v_all + 1), np.log(d_all + 1e-12))[0, 1])
    summ["policy_seed0_final"] = policy_to_text(env, greedy_policy(runs[0]["Q"]))
    summ["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    return summ


def main():
    env = GridWorld()
    Q_star = solve_q_star(0.9)
    results, all_runs = {}, {}
    for tag, extra in [("fixed_start", dict(random_start=False)), ("random_start", dict(random_start=True))]:
        cfg, runs = run_many(dict(n_episodes=N_EP, eval_every=50, snapshot_episodes=SNAP, **extra), SEEDS)
        results[tag] = analyse(cfg, runs, env, Q_star)
        all_runs[tag] = runs
        np.savez_compressed(os.path.join(DATA_DIR, f"main_qlearning_{tag}_curves.npz"),
                            eval_episodes=runs[0]["eval_episodes"], error_inf=stack(runs, "error_inf"),
                            error_mean=stack(runs, "error_mean"), error_start=stack(runs, "error_start"),
                            frac_optimal=stack(runs, "frac_optimal"), greedy_value=stack(runs, "greedy_value"),
                            episode_return=stack(runs, "episode_return"),
                            episode_length=stack(runs, "episode_length"))
    save_json(results, "main_qlearning.json")

    # ---------------- Figure 3 : ||Q_t - Q*||_inf ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, tag, title in [(axes[0], "fixed_start", "(A) départ fixe en S"),
                           (axes[1], "random_start", "(B) départs aléatoires")]:
        runs = all_runs[tag]
        ee = runs[0]["eval_episodes"]
        for key, lab, col in [("error_inf", r"$\|Q_t-Q^*\|_\infty$", PALETTE[0]),
                              ("error_start", r"$\max_a|Q_t(S,a)-Q^*(S,a)|$", PALETTE[2]),
                              ("error_mean", r"moyenne de $|Q_t-Q^*|$", PALETTE[1])]:
            m, lo, hi = mean_and_band(stack(runs, key))
            ax.loglog(ee[1:], m[1:], color=col, label=lab)
            ax.fill_between(ee[1:], lo[1:], hi[1:], color=col, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("épisode")
    axes[0].set_ylabel("erreur (moyenne sur 20 graines, bande 10 %-90 %)")
    axes[0].legend(fontsize=8)
    fig.suptitle(r"Q-Learning, $\gamma=0.9$, $\varepsilon=0.1$, $\alpha_t=1/N_t(s,a)^{0.8}$", y=1.02)
    savefig(fig, "fig3_error_convergence.pdf")

    # ---------------- Figure 4 : apprentissage (réglage A) ----------------
    runs = all_runs["fixed_start"]
    ee = runs[0]["eval_episodes"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    R_ = np.stack([moving_average(r["episode_return"], 100) for r in runs])
    L_ = np.stack([moving_average(r["episode_length"], 100) for r in runs])
    x = np.arange(1, R_.shape[1] + 1)
    for ax, arr, t in [(axes[0, 0], R_, "Récompense totale par épisode (moy. mobile 100)"),
                       (axes[0, 1], L_, "Longueur des épisodes (moy. mobile 100)")]:
        m, lo, hi = mean_and_band(arr)
        ax.semilogx(x, m, color=PALETTE[0])
        ax.fill_between(x, lo, hi, color=PALETTE[0], alpha=0.2)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("épisode")
    m, lo, hi = mean_and_band(100 * stack(runs, "frac_optimal"))
    axes[1, 0].semilogx(ee[1:], m[1:], color=PALETTE[2])
    axes[1, 0].fill_between(ee[1:], lo[1:], hi[1:], color=PALETTE[2], alpha=0.2)
    axes[1, 0].set_title("% d'états où l'action gloutonne est optimale", fontsize=10)
    axes[1, 0].set_xlabel("épisode")
    m, lo, hi = mean_and_band(stack(runs, "greedy_value"))
    axes[1, 1].semilogx(ee[1:], m[1:], color=PALETTE[4])
    axes[1, 1].fill_between(ee[1:], lo[1:], hi[1:], color=PALETTE[4], alpha=0.2)
    axes[1, 1].axhline(Q_star.max(1)[env.start], ls="--", color="grey", label=r"$V^*(S)$")
    axes[1, 1].set_title(r"$V^{\pi_t}(S)$, $\pi_t$ gloutonne pour $Q_t$", fontsize=10)
    axes[1, 1].set_xlabel("épisode")
    axes[1, 1].legend()
    fig.tight_layout()
    savefig(fig, "fig4_reward_learning.pdf")

    # ---------------- Instantanés (graine 0, réglage A) + visites ----------------
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.6))
    axes = axes.ravel()
    for ax, ep in zip(axes, SNAP):
        Qs = runs[0]["snapshots"][ep]
        draw_grid(env, ax, values=Qs.max(1), policy=greedy_policy(Qs), title=f"épisode {ep}", show_labels=False)
    Vis = results["fixed_start"]["visits_mean"]
    draw_grid(env, axes[-1], values=np.log10(np.asarray(Vis).sum(1) + 1),
              title=r"$\log_{10}\sum_a N(s,a)$ (moy. 20 graines)", value_fmt="{:.1f}", show_labels=False,
              cmap="Oranges")
    fig.suptitle(r"Graine 0, départ fixe : $\max_a Q_t(s,a)$ (couleur) et action gloutonne", y=1.0)
    savefig(fig, "fig_snapshots_visits.pdf")

    for tag in results:
        print("=====", tag)
        print({k: v for k, v in results[tag].items() if k not in ("visits_mean", "stabilization_all", "config")})
        print("stabilisation :", results[tag]["stabilization_all"])


if __name__ == "__main__":
    main()
