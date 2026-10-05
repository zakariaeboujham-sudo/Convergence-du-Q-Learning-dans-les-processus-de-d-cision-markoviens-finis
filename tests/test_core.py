"""Tests de cohérence entre le code et les résultats théoriques du rapport."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.environment import GridWorld  # noqa: E402
from src.markov_chain import check_stochastic  # noqa: E402
from src.q_learning import q_learning  # noqa: E402
from src.value_iteration import (  # noqa: E402
    bellman_optimality_operator, bellman_policy_operator, value_iteration, greedy_policy,
    evaluate_policy_exact, deterministic_to_matrix,
)

RNG = np.random.default_rng(0)


def test_transition_kernel_is_stochastic():
    for slip in (0.0, 0.2, 0.5):
        env = GridWorld(slip=slip)
        for a in range(env.n_actions):
            assert check_stochastic(env.P[:, a, :])


def test_max_lemma():
    # |max f - max g| <= max |f - g|
    for _ in range(10_000):
        f, g = RNG.normal(size=5), RNG.normal(size=5)
        assert abs(f.max() - g.max()) <= np.abs(f - g).max() + 1e-12


def test_contraction():
    env = GridWorld()
    for gamma in (0.5, 0.9, 0.99):
        for _ in range(2000):
            Q1, Q2 = RNG.normal(0, 10, (2, env.n_states, env.n_actions))
            lhs = np.abs(bellman_optimality_operator(Q1, env.P, env.R, gamma)
                         - bellman_optimality_operator(Q2, env.P, env.R, gamma)).max()
            assert lhs <= gamma * np.abs(Q1 - Q2).max() + 1e-10


def test_fixed_point_and_optimality():
    env = GridWorld()
    gamma = 0.9
    Q, _, _ = value_iteration(env.P, env.R, gamma, tol=1e-12)
    assert np.abs(bellman_optimality_operator(Q, env.P, env.R, gamma) - Q).max() < 1e-10
    # Q^{pi*} = Q* pour la politique gloutonne
    _, Qg = evaluate_policy_exact(env.P, env.R, gamma, deterministic_to_matrix(greedy_policy(Q), 4))
    assert np.abs(Qg - Q).max() < 1e-9
    # Q^pi <= Q* pour des politiques aléatoires, et Q^pi = T^pi Q^pi
    for _ in range(300):
        pi = RNG.dirichlet(np.ones(4), env.n_states)
        _, Qpi = evaluate_policy_exact(env.P, env.R, gamma, pi)
        assert np.all(Qpi <= Q + 1e-9)
        assert np.abs(bellman_policy_operator(Qpi, env.P, env.R, gamma, pi) - Qpi).max() < 1e-9


def test_return_bound():
    env = GridWorld()
    for gamma in (0.5, 0.9, 0.99):
        Q, _, _ = value_iteration(env.P, env.R, gamma, tol=1e-10)
        assert np.abs(Q).max() <= env.r_max / (1 - gamma)


def test_q_learning_deterministic_converges():
    # environnement déterministe, départs aléatoires : Q-Learning doit retrouver Q* très précisément
    env = GridWorld(slip=0.0, random_start=True)
    gamma = 0.9
    Q_star, _, _ = value_iteration(env.P, env.R, gamma, tol=1e-12)
    Q, h = q_learning(env, gamma, 3000, learning_rate=1.0, epsilon=0.3, Q_star=Q_star, seed=1)
    assert h.error_inf[-1] < 1e-6


def test_q_learning_does_not_use_model(monkeypatch=None):
    # Q-Learning doit fonctionner si l'on rend le modèle illisible APRÈS construction du simulateur
    env = GridWorld(slip=0.0, random_start=True)
    cum = env._cumP.copy()
    env.P = None
    env._cumP = cum
    q_learning(env, 0.9, 50, learning_rate=0.5, epsilon=0.2, Q_star=None, seed=0)
