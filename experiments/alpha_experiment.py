"""
Expérience sur le taux d'apprentissage alpha.
1) Illustration scalaire de Robbins-Monro : estimer m = E[Y] par x_{n+1} = x_n + alpha_n (Y_n - x_n).
2) Q-Learning (gamma = 0.9, eps = 0.1, départs aléatoires, 10 graines, 50 000 épisodes) avec
   alpha in {0.1, 0.5, 0.9} constants, alpha_t = 1/N_t(s,a), alpha_t = 1/N_t(s,a)^0.8.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import run_many, stack, summarize  # noqa: E402
from src.q_learning import learning_rate_label  # noqa: E402
from src.utils import save_json, savefig, mean_and_band, PALETTE  # noqa: E402

RATES = [0.1, 0.5, 0.9, ("inv",), ("poly", 0.8)]
SEEDS = list(range(10))
N_EP = 50_000
SEED_SCALAR = 123


def scalar_robbins_monro():
    """Y ~ loi de la récompense obtenue en (2,3) avec l'action 'bas' : +10 w.p. 0.8, -1 w.p. 0.1, -5 w.p. 0.1."""
    rng = np.random.default_rng(SEED_SCALAR)
    vals, probs = np.array([10.0, -1.0, -5.0]), np.array([0.8, 0.1, 0.1])
    m = float(vals @ probs)
    var = float(((vals - m) ** 2) @ probs)
    n, runs = 20_000, 200
    schedules = {
        r"$\alpha_n=0.5$": lambda k: 0.5 * np.ones_like(k, dtype=float),
        r"$\alpha_n=0.1$": lambda k: 0.1 * np.ones_like(k, dtype=float),
        r"$\alpha_n=1/n$": lambda k: 1.0 / k,
        r"$\alpha_n=1/n^{0.8}$": lambda k: 1.0 / k**0.8,
        r"$\alpha_n=1/n^2$ ($\sum\alpha_n<\infty$)": lambda k: 1.0 / k**2,
    }
    k = np.arange(1, n + 1)
    res, curves = {}, {}
    for name, f in schedules.items():
        a = f(k)
        Y = rng.choice(vals, size=(runs, n), p=probs)
        x = np.zeros(runs) + (-20.0)  # point de départ volontairement loin : x_0 = -20
        traj = np.empty((runs, n))
        for t in range(n):
            x = x + a[t] * (Y[:, t] - x)
            traj[:, t] = x
        mse = ((traj - m) ** 2).mean(0)
        curves[name] = mse
        res[name] = dict(final_mse=float(mse[-1]), final_mean=float(traj[:, -1].mean()),
                         mse_tail_avg=float(mse[n // 2:].mean()),
                         final_var_across_runs=float(traj[:, -1].var()),
                         sum_alpha=float(a.sum()), sum_alpha2=float((a ** 2).sum()))
    # variances stationnaires théoriques pour alpha constant : alpha * sigma^2 / (2 - alpha)
    theory = {c: c * var / (2 - c) for c in (0.1, 0.5)}
    # pour 1/n^2 : biais résiduel prédit = |x0 - m| * prod(1 - alpha_k) (à partir de k = 2, alpha_1 = 1)
    return dict(mean=m, variance=var, n=n, runs=runs, results=res, const_alpha_stationary_var=theory), curves


def main():
    sc, sc_curves = scalar_robbins_monro()
    out = {"scalar": sc, "qlearning": {}}
    curves = {}
    for lr in RATES:
        cfg, runs = run_many(dict(learning_rate=lr, random_start=True, n_episodes=N_EP, eval_every=50), SEEDS)
        s = summarize(cfg, runs)
        E, Em = stack(runs, "error_inf"), stack(runs, "error_mean")
        ee = runs[0]["eval_episodes"]
        curves[str(lr)] = (ee, E, Em, learning_rate_label(lr))
        idx = {c: int(np.searchsorted(ee, c)) for c in [1000, 10_000, 25_000, 50_000]}
        tail = ee >= 25_000
        out["qlearning"][str(lr)] = dict(
            err_inf={c: float(E[:, i].mean()) for c, i in idx.items()},
            err_mean={c: float(Em[:, i].mean()) for c, i in idx.items()},
            # fluctuation en fin d'apprentissage : écart-type temporel de E_t sur la 2e moitié
            tail_temporal_std=float(np.mean([E[r, tail].std() for r in range(len(runs))])),
            tail_min=float(E[:, tail].min(axis=1).mean()), tail_max=float(E[:, tail].max(axis=1).mean()),
            frac_optimal_final=s["final_frac_optimal_mean"], stabilized=s["n_seeds_stabilized"],
            stab_median=s["stabilization_median"],
        )
        print(lr, out["qlearning"][str(lr)])
    save_json(out, "alpha_experiment.json")
    print(sc)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c, (key, (ee, E, Em, lab)) in zip(PALETTE, curves.items()):
        for ax, arr in [(axes[0], E), (axes[1], Em)]:
            m, lo, hi = mean_and_band(arr)
            ax.loglog(ee[1:], m[1:], color=c, label=lab)
            ax.fill_between(ee[1:], lo[1:], hi[1:], color=c, alpha=0.15)
    axes[0].set_title(r"Q-Learning : $\|Q_t-Q^*\|_\infty$")
    axes[1].set_title(r"Q-Learning : moyenne de $|Q_t-Q^*|$")
    for ax in axes[:2]:
        ax.set_xlabel("épisode")
    axes[0].legend(fontsize=8)
    for c, (name, mse) in zip(PALETTE, sc_curves.items()):
        axes[2].loglog(np.arange(1, len(mse) + 1), mse, color=c, label=name)
    for cst in (0.1, 0.5):
        axes[2].axhline(sc["const_alpha_stationary_var"][cst], ls=":", color="grey", lw=1)
    axes[2].set_title(r"Robbins-Monro scalaire : $\mathbb{E}[(x_n-m)^2]$ (200 répétitions)")
    axes[2].set_xlabel("$n$")
    axes[2].legend(fontsize=7)
    savefig(fig, "fig6_alpha.pdf")


if __name__ == "__main__":
    main()
