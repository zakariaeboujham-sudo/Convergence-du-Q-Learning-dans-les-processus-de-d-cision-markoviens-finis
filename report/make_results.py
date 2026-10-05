"""
Génère report/generated/results.tex à partir des fichiers results/data/*.json.

Tous les nombres cités dans la partie expérimentale du rapport passent par ces
macros : aucun résultat n'est recopié à la main.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "results", "data")
OUT = os.path.join(HERE, "generated", "results.tex")
ARROWS_TEX = {"↑": r"$\uparrow$", "↓": r"$\downarrow$", "←": r"$\leftarrow$", "→": r"$\rightarrow$"}


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def f(x, d=2):
    """Nombre décimal ordinaire."""
    return f"{x:.{d}f}"


def sci(x, d=1):
    """Notation scientifique LaTeX."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return rf"{m:.{d}f}\times10^{{{e}}}"


def pct(x, d=0):
    return rf"{100 * x:.{d}f}\,\%"


def pmatrix(M, d=3):
    rows = [" & ".join(f"{v:.{d}f}" for v in r) for r in M]
    return r"\begin{pmatrix}" + r"\\".join(rows) + r"\end{pmatrix}"


def vec(v, d=3):
    return "(" + ",\\ ".join(f"{x:.{d}f}" for x in v) + ")"


macros, tables = {}, {}


def m(name, value, math=True):
    macros[name] = rf"\ensuremath{{{value}}}" if math else value


# ---------------------------------------------------------------- Markov
mk = load("markov_chain.json")
m("MarkovPtwo", pmatrix(mk["P2"], 2))
m("MarkovPthree", pmatrix(mk["P3"], 3))
m("MarkovPtwoOneFour", f(mk["P2"][0][3], 2))
m("MarkovRuns", f"{mk['mc_n_runs']:,}".replace(",", r"\,"))
m("MarkovMC", vec(mk["mc_row0_n3"], 4))
m("MarkovExactRow", vec(mk["exact_row0_n3"], 3))
m("MarkovDev", f(mk["mc_max_abs_dev"], 4))
m("MarkovStd", f(mk["mc_std_bound"], 4))
m("MarkovMu", vec(mk["stationary"], 4))

# ---------------------------------------------------------------- Value Iteration
vi = load("value_iteration.json")
c9 = vi["contraction"]["0.9"]
m("ContrMaxNine", f(c9["max"], 4))
m("ContrEqNine", f(c9["equality_case"], 4))
m("ContrMeanNine", f(c9["mean"], 3))
m("NumGreedyMatch", sci(vi["greedy_policy_value_matches_Q_star"]))
m("NumRandomPolicies", f"{vi['n_random_policies']:,}".replace(",", r"\,"))
m("NumMaxQpiMinusQstar", sci(vi["max_Qpi_minus_Qstar"]))
m("NumBestRandom", f(vi["best_random_policy_value_start"], 3))
m("VstarS", f(vi["V_star_start"], 3))
Q = np.array(vi["Q_star"])
opt = np.abs(Q - Q.max(1, keepdims=True)) <= 1e-6
gaps = []
for s in range(Q.shape[0] - 1):  # dernier état = G
    if (~opt[s]).any():
        gaps.append(Q[s].max() - Q[s][~opt[s]].max())
gap = min(gaps)
m("ActionGap", f(gap, 3))
m("ActionGapHalf", f(gap / 2, 3))
m("VIRatioLast", f(vi["vi_ratios"][-1], 4))
m("SpectralRho", f(vi["spectral_radius_transient"], 4))
m("AsymptoticRate", f(vi["asymptotic_rate_predicted"], 4))
m("VIIters", str(vi["vi_iterations"]))
m("VIBoundIters", str(int(np.ceil(np.log(1e-8 / vi["vi_error0"]) / np.log(vi["gamma"])))))
m("VIAPost", sci(vi["vi_aposteriori_bound"]))
m("VITrueErr", sci(vi["vi_true_error"]))
m("VILastInc", sci(vi["vi_last_increment"]))
m("VIErrZero", f(vi["vi_error0"], 3))
m("QstarSup", f(vi["Q_star_sup"], 3))
m("ReturnBound", f(vi["return_bound"], 0))
loss = {d["eps"]: d for d in vi["greedy_loss"]}
m("LossEpsSmall", f(loss[0.05]["worst_loss"], 3))
m("LossEpsOne", f(loss[1.0]["worst_loss"], 3))
m("LossBoundOne", f(loss[1.0]["bound"], 0))
m("LossEpsTwo", f(loss[2.0]["worst_loss"], 2))
m("LossBoundTwo", f(loss[2.0]["bound"], 0))

# Tableau de Q*
cells = [tuple(c) for c in vi["cells"]]
lines = []
for s, c in enumerate(cells[:-1]):
    row = [f"$({c[0]},{c[1]})$"]
    for a in range(4):
        v = Q[s, a]
        txt = f"{v:.3f}"
        row.append(rf"\textbf{{{txt}}}" if opt[s, a] else txt)
    row.append(f"{Q[s].max():.3f}")
    lines.append(" & ".join(row) + r" \\")
tables["TableQstar"] = "\n".join(lines)

# ---------------------------------------------------------------- Q-Learning principal
mq = load("main_qlearning.json")
for tag, T in [("fixed_start", "A"), ("random_start", "B")]:
    d = mq[tag]
    cp = d["checkpoints"]
    m(f"Main{T}ErrFinal", f(cp["100000"]["err_inf"], 2))
    m(f"Main{T}ErrFinalMedian", f(cp["100000"]["err_inf_median"], 2))
    m(f"Main{T}ErrThousand", f(cp["1000"]["err_inf"], 2))
    m(f"Main{T}ErrTenK", f(cp["10000"]["err_inf"], 2))
    m(f"Main{T}ErrFiftyK", f(cp["50000"]["err_inf"], 2))
    m(f"Main{T}ErrHundred", f(cp["100"]["err_inf"], 2))
    m(f"Main{T}MeanErrFinal", f(cp["100000"]["err_mean"], 3))
    m(f"Main{T}StartErrFinal", f(cp["100000"]["err_start"], 3))
    m(f"Main{T}FracOpt", pct(d["final_frac_optimal_mean"], 1))
    m(f"Main{T}GreedyValue", f(d["final_greedy_value_mean"], 3))
    m(f"Main{T}Stab", str(d["n_seeds_stabilized"]))
    m(f"Main{T}StabMedian", f"{d['stabilization_median']:,.0f}".replace(",", r"\,"))
    m(f"Main{T}Slope", f(d["loglog_slope_err_inf"], 2))
    m(f"Main{T}SlopeMean", f(d["loglog_slope_err_mean"], 2))
    m(f"Main{T}VisitsMin", f(d["visits_min_pair"]["value"], 0))
    m(f"Main{T}VisitsMax", f"{d['visits_max']:,.0f}".replace(",", r"\,"))
    m(f"Main{T}VisitsRatio", f(d["visits_ratio_max_min"], 0))
    m(f"Main{T}Corr", f(d["corr_logvisits_logerror"], 2))
    m(f"Main{T}ErrMax", f(d["final_error_inf_max"], 2))
    m(f"Main{T}ErrMin", f(d["final_error_inf_min"], 2))
    m(f"Main{T}AvgReturn", f(d["avg_return_last"], 2))
    m(f"Main{T}AvgLength", f(d["avg_length_last"], 2))
    m(f"Main{T}NSeeds", str(d["n_seeds"]))
rows = []
for c in ["100", "1000", "10000", "50000", "100000"]:
    a, b = mq["fixed_start"]["checkpoints"][c], mq["random_start"]["checkpoints"][c]
    rows.append(f"{int(c):,}".replace(",", r"\,") + " & " + " & ".join(
        [f(a["err_inf"]), f(a["err_inf_median"]), f(a["err_mean"], 3), pct(a["frac_optimal"], 1),
         f(b["err_inf"]), f(b["err_inf_median"]), f(b["err_mean"], 3), pct(b["frac_optimal"], 1)]) + r" \\")
tables["TableMain"] = "\n".join(rows)
argA = mq["fixed_start"]["argmax_error_cells"]
far = sum(v for k, v in argA.items() if k in ("(0, 3)", "(3, 0)", "(0, 2)", "(2, 0)"))
m("MainAArgmaxFar", str(far))

# ---------------------------------------------------------------- gamma
gm = load("gamma_experiment.json")
rows = []
for g in ["0.5", "0.8", "0.9", "0.95", "0.99"]:
    d = gm[g]
    rows.append(" & ".join([
        g, f(1 / (1 - float(g)), 0), str(d["vi_iterations"]), str(int(np.ceil(np.log(1e-8 / d["Q_star_sup"]) / np.log(float(g))))), f(d["asymptotic_rate"], 3),
        f(d["Q_star_sup"], 2), f(d["V_star_start"], 2), f(d["ql_err_inf"]["1000"], 2), f(d["ql_err_inf"]["50000"], 2),
        pct(d["ql_rel_err"]["50000"], 1), f"{d['ql_stab_median']:,.0f}".replace(",", r"\,"),
    ]) + r" \\")
tables["TableGamma"] = "\n".join(rows)
pols = {g: gm[g]["policy"] for g in gm}
same_sets = len({tuple(gm[g]["n_optimal_actions_per_state"]) for g in gm}) == 1
m("GammaVIItersLow", str(gm["0.5"]["vi_iterations"]))
m("GammaVIItersHigh", str(gm["0.99"]["vi_iterations"]))
m("GammaBoundHigh", f"{int(np.ceil(np.log(1e-8 / gm['0.99']['Q_star_sup']) / np.log(0.99))):,}".replace(",", r"\,"))
rels = [gm[g]["ql_rel_err"]["50000"] for g in gm]
m("GammaRelMin", pct(min(rels), 1))
m("GammaRelMax", pct(max(rels), 1))
m("GammaBoundLow", str(int(np.ceil(np.log(1e-8 / gm["0.5"]["Q_star_sup"]) / np.log(0.5)))))
m("GammaStabHighest", f"{gm['0.99']['ql_stab_median']:,.0f}".replace(",", r"\,"))
m("GammaRelThousandLow", pct(gm["0.5"]["ql_rel_err"]["1000"], 0))
m("GammaRelThousandHigh", pct(gm["0.99"]["ql_rel_err"]["1000"], 0))
m("GammaStabLow", f"{gm['0.5']['ql_stab_median']:,.0f}".replace(",", r"\,"))
m("GammaStabHigh", f"{gm['0.95']['ql_stab_median']:,.0f}".replace(",", r"\,"))

# ---------------------------------------------------------------- alpha
al = load("alpha_experiment.json")
labels = {"0.1": r"$\alpha=0.1$", "0.5": r"$\alpha=0.5$", "0.9": r"$\alpha=0.9$",
          "('inv',)": r"$1/N_t(s,a)$", "('poly', 0.8)": r"$1/N_t(s,a)^{0.8}$"}
rm_ok = {"0.1": "non", "0.5": "non", "0.9": "non", "('inv',)": "oui", "('poly', 0.8)": "oui"}
rows = []
for k, lab in labels.items():
    d = al["qlearning"][k]
    rows.append(" & ".join([lab, rm_ok[k], f(d["err_inf"]["1000"]), f(d["err_inf"]["10000"]),
                            f(d["err_inf"]["50000"]), f(d["err_mean"]["50000"], 3), f(d["tail_temporal_std"], 2),
                            pct(d["frac_optimal_final"], 1), f"{d['stabilized']}/10"]) + r" \\")
tables["TableAlpha"] = "\n".join(rows)
for k, nm in [("0.1", "AlphaOne"), ("0.5", "AlphaFive"), ("0.9", "AlphaNine"), ("('inv',)", "AlphaInv"),
              ("('poly', 0.8)", "AlphaPoly")]:
    d = al["qlearning"][k]
    m(f"{nm}Final", f(d["err_inf"]["50000"]))
    m(f"{nm}TenK", f(d["err_inf"]["10000"]))
    m(f"{nm}Thousand", f(d["err_inf"]["1000"]))
    m(f"{nm}Std", f(d["tail_temporal_std"], 2))
    m(f"{nm}MeanFinal", f(d["err_mean"]["50000"], 3))
    m(f"{nm}TailMin", f(d["tail_min"], 2))
    m(f"{nm}TailMax", f(d["tail_max"], 2))
sc = al["scalar"]
m("ScMean", f(sc["mean"], 2))
m("ScVar", f(sc["variance"], 2))
m("ScN", f"{sc['n']:,}".replace(",", r"\,"))
m("ScRuns", str(sc["runs"]))
m("ScTheoOne", f(sc["const_alpha_stationary_var"]["0.1"], 3))
m("ScTheoFive", f(sc["const_alpha_stationary_var"]["0.5"], 2))
m("ScTheoInv", sci(sc["variance"] / sc["n"], 2))
res = sc["results"]
keys = list(res.keys())
names = ["ScFive", "ScOne", "ScInv", "ScPoly", "ScSq"]
for k, nm in zip(keys, names):
    m(f"{nm}MSE", sci(res[k]["final_mse"], 2) if res[k]["final_mse"] < 0.1 else f(res[k]["final_mse"], 3))
    m(f"{nm}Tail", (sci(res[k]["mse_tail_avg"], 2) if res[k]["mse_tail_avg"] < 0.1 else f(res[k]["mse_tail_avg"], 3))
      if "mse_tail_avg" in res[k] else "--")
    m(f"{nm}Mean", f(res[k]["final_mean"], 3))
rows = []
theo = {"ScFive": f(sc["const_alpha_stationary_var"]["0.5"], 2), "ScOne": f(sc["const_alpha_stationary_var"]["0.1"], 3),
        "ScInv": sci(sc["variance"] / sc["n"], 2), "ScPoly": r"$\to0$", "ScSq": r"$\not\to0$"}
for k, nm in zip(keys, names):
    r_ = res[k]
    tail = r_.get("mse_tail_avg")
    rows.append(" & ".join([
        k, (f"{r_['sum_alpha']:,.0f}".replace(",", r"\,") if r_["sum_alpha"] > 100 else f(r_["sum_alpha"], 2)),
        (f"{r_['sum_alpha2']:,.0f}".replace(",", r"\,") if r_["sum_alpha2"] > 100 else f(r_["sum_alpha2"], 2)),
        "$" + (sci(r_["final_mse"], 2) if r_["final_mse"] < 0.1 else f(r_["final_mse"], 3)) + "$",
        "$" + ((sci(tail, 2) if tail < 0.1 else f(tail, 3)) if tail is not None else "-") + "$",
        theo[nm] if theo[nm].startswith("$") else "$" + theo[nm] + "$",
    ]) + r" \\")
tables["TableScalar"] = "\n".join(rows)

# ---------------------------------------------------------------- epsilon
ep = load("epsilon_experiment.json")
rows = []
for e in ["0.0", "0.05", "0.1", "0.3"]:
    d = ep[f"eps={e}"]
    rows.append(" & ".join([
        e, f(d["err_inf"]["50000"]), f(d["err_start"]["50000"], 3), f(d["greedy_value"]["50000"], 3),
        f"{d['n_seeds_greedy_optimal_from_S']}/10", pct(d["frac_optimal_final"], 1),
        f(d["online_return_all"], 2), f(d["pairs_never_visited_mean"], 1), f(d["min_visits_mean"], 1),
    ]) + r" \\")
tables["TableEps"] = "\n".join(rows)
for e, nm in [("0.0", "EpsZero"), ("0.05", "EpsFive"), ("0.1", "EpsTen"), ("0.3", "EpsThirty")]:
    d = ep[f"eps={e}"]
    m(f"{nm}Err", f(d["err_inf"]["50000"]))
    m(f"{nm}ErrHundred", f(d["err_inf"]["100"]))
    m(f"{nm}ErrThousand", f(d["err_inf"]["1000"]))
    m(f"{nm}Value", f(d["greedy_value"]["50000"], 3))
    m(f"{nm}OptS", str(d["n_seeds_greedy_optimal_from_S"]))
    m(f"{nm}Return", f(d["online_return_all"], 2))
    m(f"{nm}Never", f(d["pairs_never_visited_mean"], 1))
    m(f"{nm}NeverMax", str(max(d["pairs_never_visited"])))
    m(f"{nm}FracOpt", pct(d["frac_optimal_final"], 1))
    m(f"{nm}MinVisits", f(d["min_visits_mean"], 1))
    m(f"{nm}ValueMin", f(d["greedy_value_final_min"], 3))
m("EpsVstar", f(ep["V_star_start"], 3))
m("PessQZero", f(ep["pessimistic_Q0"], 0))
rows = []
for e, nm in [("0.0", "PessZero"), ("0.1", "PessTen")]:
    d = ep[f"pessimistic_eps={e}"]
    last = str(max(int(k) for k in d["err_inf"]))
    m(f"{nm}Err", f(d["err_inf"][last]))
    m(f"{nm}Value", f(d["greedy_value"][last], 3))
    m(f"{nm}OptS", str(d["n_seeds_greedy_optimal_from_S"]))
    m(f"{nm}Never", f(d["pairs_never_visited_mean"], 1))
    m(f"{nm}Trunc", f"{d['n_episodes_truncated']:,}".replace(",", r"\,"))
    m(f"{nm}Return", f(d["online_return_all"], 2))
    m(f"{nm}FracOpt", pct(d["frac_optimal_final"], 1))
    m(f"{nm}ValueMin", f(d["greedy_value_final_min"], 3))
    m(f"{nm}ValueMax", f(d["greedy_value_final_max"], 3))
    rows.append(" & ".join([e, f(d["err_inf"][last]), f(d["greedy_value"][last], 3),
                            f"{d['n_seeds_greedy_optimal_from_S']}/10", pct(d["frac_optimal_final"], 1),
                            f(d["pairs_never_visited_mean"], 1), f(d["online_return_all"], 2),
                            f"{d['n_episodes_truncated']:,}".replace(",", r"\,")]) + r" \\")
tables["TablePess"] = "\n".join(rows)
m("PessEpisodes", f"{max(int(k) for k in ep['pessimistic_eps=0.0']['err_inf']):,}".replace(",", r"\,"))

# ---------------------------------------------------------------- écriture
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("% Fichier généré automatiquement par report/make_results.py. Ne pas éditer.\n")
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in tables.items():
        fh.write(f"\\newcommand{{\\{k}}}{{%\n{v}\n}}\n")
print(f"{len(macros)} macros, {len(tables)} tableaux -> {OUT}")
