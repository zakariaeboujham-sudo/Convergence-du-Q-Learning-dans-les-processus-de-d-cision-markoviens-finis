"""
Q-Learning tabulaire (Watkins, 1989).

À chaque transition observée (s_t, a_t, r_t, s_{t+1}) :

    delta_t = r_t + gamma * max_a Q_t(s_{t+1}, a) - Q_t(s_t, a_t)     (erreur TD)
    Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + alpha_t(s_t, a_t) * delta_t

et Q_{t+1} = Q_t sur toutes les autres paires. Si s_{t+1} est terminal,
max_a Q_t(s_{t+1}, a) = 0 (la valeur de l'état absorbant G est nulle).

L'agent n'utilise JAMAIS env.P ni env.R : il n'a accès qu'à env.reset()
et env.step(a). Q* n'est utilisé que pour MESURER l'erreur, pas pour apprendre.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .policies import epsilon_greedy
from .value_iteration import optimal_action_sets, evaluate_policy_exact, deterministic_to_matrix


# ----------------------------------------------------------------------
# Taux d'apprentissage
# ----------------------------------------------------------------------
def make_learning_rate(spec):
    """
    spec peut être :
      * un nombre c          -> alpha_t = c (constant : ne vérifie PAS Robbins-Monro) ;
      * ("inv", )            -> alpha_t(s,a) = 1 / N_t(s,a) ;
      * ("poly", omega)      -> alpha_t(s,a) = 1 / N_t(s,a)^omega, 1/2 < omega <= 1.
    N_t(s,a) est le nombre de mises à jour de la paire (s,a), mise à jour courante comprise.
    Pour 1/2 < omega <= 1 :  sum alpha = +inf  et  sum alpha^2 < +inf  (Robbins-Monro).
    """
    if isinstance(spec, (int, float)):
        c = float(spec)
        return lambda n: c
    kind = spec[0]
    if kind == "inv":
        return lambda n: 1.0 / n
    if kind == "poly":
        omega = float(spec[1])
        return lambda n: 1.0 / n**omega
    raise ValueError(f"taux d'apprentissage inconnu : {spec}")


def learning_rate_label(spec):
    if isinstance(spec, (int, float)):
        return rf"$\alpha={spec}$"
    if spec[0] == "inv":
        return r"$\alpha_t=1/N_t(s,a)$"
    return rf"$\alpha_t=1/N_t(s,a)^{{{spec[1]}}}$"


# ----------------------------------------------------------------------
# Résultats
# ----------------------------------------------------------------------
@dataclass
class QLearningHistory:
    eval_episodes: list = field(default_factory=list)   # épisodes où l'on mesure
    error_inf: list = field(default_factory=list)       # ||Q_t - Q*||_inf
    error_mean: list = field(default_factory=list)      # moyenne |Q_t - Q*|
    error_start: list = field(default_factory=list)     # max_a |Q_t(S,a) - Q*(S,a)|
    frac_optimal: list = field(default_factory=list)    # % d'états où greedy(Q_t) est optimale
    greedy_value: list = field(default_factory=list)    # V^{pi_t}(S), pi_t = greedy(Q_t) (évaluation exacte)
    episode_return: list = field(default_factory=list)  # somme des récompenses de l'épisode
    episode_length: list = field(default_factory=list)  # nombre de pas de l'épisode
    snapshots: dict = field(default_factory=dict)       # quelques Q_t sauvegardés
    visits: np.ndarray | None = None                    # N(s, a) final
    Q: np.ndarray | None = None                         # Q final


def q_learning(
    env,
    gamma,
    n_episodes,
    learning_rate=("poly", 0.8),
    epsilon=0.1,
    Q_star=None,
    Q0=None,
    max_steps=200,
    eval_every=10,
    snapshot_episodes=(),
    seed=0,
    random_tie=True,
):
    rng = np.random.default_rng(seed)
    env.rng = np.random.default_rng(seed + 10_000)  # flux aléatoire séparé pour l'environnement
    nS, nA = env.n_states, env.n_actions
    Q = np.zeros((nS, nA)) if Q0 is None else np.array(Q0, dtype=float)
    Q[env.goal] = 0.0  # valeur de l'état absorbant
    N = np.zeros((nS, nA), dtype=np.int64)
    alpha_fn = make_learning_rate(learning_rate)
    hist = QLearningHistory()
    nt = np.array(env.non_terminal_states())
    opt_sets = optimal_action_sets(Q_star) if Q_star is not None else None

    def record(ep):
        hist.eval_episodes.append(ep)
        if Q_star is not None:
            diff = np.abs(Q[nt] - Q_star[nt])
            hist.error_inf.append(float(diff.max()))
            hist.error_mean.append(float(diff.mean()))
            hist.error_start.append(float(np.abs(Q[env.start] - Q_star[env.start]).max()))
            greedy_a = np.argmax(Q[nt], axis=1)
            hist.frac_optimal.append(float(opt_sets[nt, greedy_a].mean()))
            # Mesure de performance (utilise le modèle UNIQUEMENT pour évaluer, jamais pour apprendre)
            pi = deterministic_to_matrix(np.argmax(Q, axis=1), nA)
            V_pi, _ = evaluate_policy_exact(env.P, env.R, gamma, pi)
            hist.greedy_value.append(float(V_pi[env.start]))

    record(0)
    for ep in range(1, n_episodes + 1):
        s = env.reset()
        total, steps = 0.0, 0
        for steps in range(1, max_steps + 1):
            a = epsilon_greedy(Q, s, epsilon, rng, random_tie)
            s_next, r, done = env.step(a)
            N[s, a] += 1
            alpha = alpha_fn(N[s, a])
            target = r + (0.0 if done else gamma * Q[s_next].max())
            delta = target - Q[s, a]          # erreur TD
            Q[s, a] += alpha * delta
            total += r
            s = s_next
            if done:
                break
        hist.episode_return.append(total)
        hist.episode_length.append(steps)
        if ep % eval_every == 0 or ep == n_episodes:
            record(ep)
        if ep in snapshot_episodes:
            hist.snapshots[ep] = Q.copy()

    hist.visits, hist.Q = N, Q
    return Q, hist
