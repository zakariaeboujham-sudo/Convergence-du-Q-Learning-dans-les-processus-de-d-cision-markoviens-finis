"""Fonctions communes aux expériences : exécution multi-graines en parallèle."""

from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld  # noqa: E402
from src.q_learning import q_learning  # noqa: E402
from src.value_iteration import value_iteration  # noqa: E402

DEFAULT = dict(gamma=0.9, epsilon=0.1, learning_rate=("poly", 0.8),
               n_episodes=20_000, eval_every=20, max_steps=200, slip=0.2)


def solve_q_star(gamma, slip=DEFAULT["slip"]):
    env = GridWorld(slip=slip)
    Q_star, inc, _ = value_iteration(env.P, env.R, gamma, tol=1e-12)
    return Q_star


def _run_one(args):
    cfg, seed = args
    env = GridWorld(slip=cfg.get("slip", DEFAULT["slip"]), random_start=cfg.get("random_start", False))
    Q_star = solve_q_star(cfg["gamma"], cfg.get("slip", DEFAULT["slip"]))
    Q, h = q_learning(
        env,
        gamma=cfg["gamma"],
        n_episodes=cfg["n_episodes"],
        learning_rate=cfg["learning_rate"],
        epsilon=cfg["epsilon"],
        Q_star=Q_star,
        Q0=cfg.get("Q0"),
        max_steps=cfg["max_steps"],
        eval_every=cfg["eval_every"],
        snapshot_episodes=cfg.get("snapshot_episodes", ()),
        seed=seed,
        random_tie=cfg.get("random_tie", True),
    )
    return dict(
        seed=seed,
        eval_episodes=np.array(h.eval_episodes),
        error_inf=np.array(h.error_inf),
        error_mean=np.array(h.error_mean),
        error_start=np.array(h.error_start),
        frac_optimal=np.array(h.frac_optimal),
        greedy_value=np.array(h.greedy_value),
        episode_return=np.array(h.episode_return),
        episode_length=np.array(h.episode_length),
        visits=h.visits,
        Q=Q,
        snapshots=h.snapshots,
    )


def run_many(cfg, seeds, processes=None):
    full = {**DEFAULT, **cfg}
    with Pool(processes or os.cpu_count()) as pool:
        runs = pool.map(_run_one, [(full, s) for s in seeds])
    return full, runs


def stack(runs, key):
    return np.stack([r[key] for r in runs])


def stabilization_episode(eval_episodes, frac_optimal):
    """Premier épisode à partir duquel la politique gloutonne reste optimale (100 %) jusqu'à la fin."""
    fo = np.asarray(frac_optimal)
    ok = fo >= 1.0 - 1e-12
    if not ok[-1]:
        return None
    k = len(ok) - 1
    while k > 0 and ok[k - 1]:
        k -= 1
    return int(eval_episodes[k])


def summarize(cfg, runs, last_window=1000):
    ee = runs[0]["eval_episodes"]
    err = stack(runs, "error_inf")
    stab = [stabilization_episode(ee, r["frac_optimal"]) for r in runs]
    stab_ok = [s for s in stab if s is not None]
    return dict(
        final_error_inf_mean=float(err[:, -1].mean()),
        final_error_inf_min=float(err[:, -1].min()),
        final_error_inf_max=float(err[:, -1].max()),
        final_error_mean_mean=float(stack(runs, "error_mean")[:, -1].mean()),
        final_error_start_mean=float(stack(runs, "error_start")[:, -1].mean()),
        final_frac_optimal_mean=float(stack(runs, "frac_optimal")[:, -1].mean()),
        final_greedy_value_mean=float(stack(runs, "greedy_value")[:, -1].mean()),
        avg_return_last=float(np.mean([r["episode_return"][-last_window:].mean() for r in runs])),
        avg_length_last=float(np.mean([r["episode_length"][-last_window:].mean() for r in runs])),
        n_seeds=len(runs),
        n_seeds_stabilized=len(stab_ok),
        stabilization_median=(float(np.median(stab_ok)) if stab_ok else None),
        stabilization_all=stab,
        min_visits_mean=float(np.mean([r["visits"][:-1].min() for r in runs])),
    )
