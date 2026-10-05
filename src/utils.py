"""Outils : normes, sauvegarde, affichage de l'environnement et des politiques."""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .environment import ACTION_ARROWS, MOVES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figures")
DATA_DIR = os.path.join(ROOT, "results", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
PALETTE = ["#1f5fa8", "#d1495b", "#2e933c", "#e8a33d", "#6c4f9e", "#3a3a3a"]


def sup_norm(x):
    """||x||_inf = max |x_i|."""
    return float(np.max(np.abs(x)))


def save_json(obj, name):
    path = os.path.join(DATA_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_to_builtin)
    return path


def _to_builtin(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def policy_to_text(env, policy):
    rows = []
    for i in range(env.n_rows):
        row = []
        for j in range(env.n_cols):
            c = (i, j)
            if c in env.obstacles:
                row.append("X")
            elif c == env.goal_cell:
                row.append("G")
            else:
                row.append(ACTION_ARROWS[policy[env.cell_to_state[c]]])
        rows.append(" ".join(row))
    return "\n".join(rows)


def draw_grid(env, ax, values=None, policy=None, title=None, value_fmt="{:.2f}",
              show_labels=True, cmap="Blues"):
    """Dessine la grille ; optionnellement les valeurs V(s) (couleur) et une politique (flèches)."""
    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(env.n_rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
    if values is not None:
        vals = np.array([values[env.cell_to_state[c]] for c in env.cells])
        vmin, vmax = vals.min(), vals.max()
        cm = plt.get_cmap(cmap)
    for i in range(env.n_rows):
        for j in range(env.n_cols):
            c = (i, j)
            face = "white"
            if c in env.obstacles:
                face = "#3a3a3a"
            elif values is not None:
                v = values[env.cell_to_state[c]]
                face = cm(0.15 + 0.7 * (v - vmin) / (vmax - vmin + 1e-12))
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="#888", lw=1))
            cx, cy = j + 0.5, i + 0.5
            if c in env.obstacles:
                ax.text(cx, cy, "X", ha="center", va="center", color="white", fontsize=14, weight="bold")
                continue
            s = env.cell_to_state[c]
            if show_labels and c == env.start_cell:
                ax.text(j + 0.08, i + 0.2, "S", fontsize=10, weight="bold", color="#1f5fa8")
            if c == env.goal_cell:
                ax.text(cx, cy, "G", ha="center", va="center", fontsize=16, weight="bold", color="#2e933c")
                if values is not None:
                    ax.text(cx, i + 0.85, value_fmt.format(values[s]), ha="center", fontsize=8)
                continue
            if policy is not None:
                di, dj = MOVES[int(policy[s])]
                ax.annotate("", xy=(cx + 0.28 * dj, cy + 0.28 * di), xytext=(cx - 0.28 * dj, cy - 0.28 * di),
                            arrowprops=dict(arrowstyle="-|>", lw=2, color="#d1495b"))
            if values is not None:
                ax.text(cx, i + 0.88, value_fmt.format(values[s]), ha="center", fontsize=8)
            if show_labels and values is None and policy is None:
                ax.text(j + 0.92, i + 0.92, f"s{s}", ha="right", va="bottom", fontsize=7, color="#777")
    if title:
        ax.set_title(title)


def mean_and_band(curves):
    """Moyenne et quantiles 10 %/90 % d'un ensemble de courbes (une par graine)."""
    a = np.asarray(curves, dtype=float)
    return a.mean(axis=0), np.quantile(a, 0.1, axis=0), np.quantile(a, 0.9, axis=0)


def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    # début : moyenne sur la fenêtre disponible
    head = np.cumsum(x[: w - 1]) / np.arange(1, w)
    return np.concatenate([head, out])
