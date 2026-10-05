"""
GridWorld fini : un processus de décision markovien M = (S, A, P, R, gamma).

Grille 4x4 (ligne, colonne), origine en haut à gauche :

    +---+---+---+---+
    | S |   |   |   |
    +---+---+---+---+
    |   | X |   |   |
    +---+---+---+---+
    |   |   | X |   |
    +---+---+---+---+
    |   |   |   | G |
    +---+---+---+---+

* S = (0,0) départ, G = (3,3) objectif (état absorbant, terminal),
  X = obstacles (cases infranchissables, elles ne sont PAS des états).
* Actions : 0 = haut, 1 = bas, 2 = gauche, 3 = droite.
* Dynamique « glissante » : l'action demandée est exécutée avec
  probabilité 1 - slip ; avec probabilité slip/2 chacune, l'agent part
  dans l'une des deux directions perpendiculaires. Si la direction
  réalisée mène hors de la grille ou dans un obstacle, l'agent reste
  sur place (collision).
* Récompense R(s, a, s') (ne dépend en fait que de (s, s')) :
      +10 si s' = G (arrivée sur l'objectif),
       -5 si s' = s  (collision avec un bord ou un obstacle),
       -1 sinon      (déplacement ordinaire).
  Depuis G : transition G -> G avec récompense 0 (état absorbant).

Tout est fini : on peut donc écrire P sous forme d'un tableau
P[s, a, s'] et R sous forme d'un tableau R[s, a, s'].
"""

from __future__ import annotations

import numpy as np

ACTIONS = ("haut", "bas", "gauche", "droite")
ACTION_ARROWS = ("↑", "↓", "←", "→")
# déplacements (dligne, dcolonne)
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
# directions perpendiculaires à chaque action
PERPENDICULAR = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}


class GridWorld:
    def __init__(
        self,
        n_rows: int = 4,
        n_cols: int = 4,
        start=(0, 0),
        goal=(3, 3),
        obstacles=((1, 1), (2, 2)),
        slip: float = 0.2,
        r_goal: float = 10.0,
        r_step: float = -1.0,
        r_collision: float = -5.0,
        seed: int | None = None,
        random_start: bool = False,
    ):
        if not 0.0 <= slip <= 1.0:
            raise ValueError("slip doit être dans [0, 1]")
        self.n_rows, self.n_cols = n_rows, n_cols
        self.start_cell, self.goal_cell = tuple(start), tuple(goal)
        self.obstacles = {tuple(o) for o in obstacles}
        self.slip = slip
        # random_start=True : chaque épisode commence dans un état non terminal
        # tiré uniformément (« exploring starts »), au lieu de toujours partir de S.
        self.random_start = random_start
        self.r_goal, self.r_step, self.r_collision = r_goal, r_step, r_collision

        # Les états sont les cases qui ne sont pas des obstacles.
        self.cells = [
            (i, j)
            for i in range(n_rows)
            for j in range(n_cols)
            if (i, j) not in self.obstacles
        ]
        self.cell_to_state = {c: k for k, c in enumerate(self.cells)}
        self.n_states = len(self.cells)
        self.n_actions = len(ACTIONS)
        self.start = self.cell_to_state[self.start_cell]
        self.goal = self.cell_to_state[self.goal_cell]

        self.P, self.R = self._build_model()
        # fonctions de répartition cumulées, pour simuler rapidement
        self._cumP = np.cumsum(self.P, axis=2)
        self.rng = np.random.default_rng(seed)
        self.state = self.start

    # ------------------------------------------------------------------
    # Modèle (utilisé par Value Iteration, INCONNU de l'agent Q-Learning)
    # ------------------------------------------------------------------
    def _move(self, cell, direction):
        di, dj = MOVES[direction]
        nxt = (cell[0] + di, cell[1] + dj)
        inside = 0 <= nxt[0] < self.n_rows and 0 <= nxt[1] < self.n_cols
        if not inside or nxt in self.obstacles:
            return cell  # collision : on reste sur place
        return nxt

    def _reward(self, s, s_next):
        if s == self.goal:
            return 0.0
        if s_next == self.goal:
            return self.r_goal
        if s_next == s:
            return self.r_collision
        return self.r_step

    def _direction_distribution(self, a):
        """Loi de la direction réellement suivie quand on choisit a."""
        p1, p2 = PERPENDICULAR[a]
        dist = {a: 1.0 - self.slip}
        dist[p1] = dist.get(p1, 0.0) + self.slip / 2
        dist[p2] = dist.get(p2, 0.0) + self.slip / 2
        return dist

    def _build_model(self):
        nS, nA = self.n_states, self.n_actions
        P = np.zeros((nS, nA, nS))
        R = np.zeros((nS, nA, nS))
        for s, cell in enumerate(self.cells):
            for a in range(nA):
                if s == self.goal:
                    P[s, a, s] = 1.0
                    continue
                for d, p in self._direction_distribution(a).items():
                    s_next = self.cell_to_state[self._move(cell, d)]
                    P[s, a, s_next] += p
            for s_next in range(nS):
                R[s, :, s_next] = self._reward(s, s_next)
        # Vérification : chaque P[s, a, .] est une loi de probabilité.
        assert np.all(P >= 0) and np.allclose(P.sum(axis=2), 1.0)
        return P, R

    def expected_reward(self):
        """r(s, a) = sum_{s'} P(s'|s,a) R(s,a,s')."""
        return np.einsum("ijk,ijk->ij", self.P, self.R)

    @property
    def r_max(self):
        return float(np.max(np.abs(self.R)))

    # ------------------------------------------------------------------
    # Interface d'interaction (seule chose que voit Q-Learning)
    # ------------------------------------------------------------------
    def reset(self):
        if self.random_start:
            nt = self.non_terminal_states()
            self.state = nt[int(self.rng.integers(len(nt)))]
        else:
            self.state = self.start
        return self.state

    def step(self, action: int):
        """Simule une transition. Renvoie (s', r, terminé)."""
        s = self.state
        # inversion de la fonction de répartition : s' ~ P(.|s, a)
        u = self.rng.random()
        s_next = int(np.searchsorted(self._cumP[s, action], u, side="right"))
        s_next = min(s_next, self.n_states - 1)  # garde-fou numérique
        r = float(self.R[s, action, s_next])
        self.state = s_next
        return s_next, r, s_next == self.goal

    def non_terminal_states(self):
        return [s for s in range(self.n_states) if s != self.goal]
