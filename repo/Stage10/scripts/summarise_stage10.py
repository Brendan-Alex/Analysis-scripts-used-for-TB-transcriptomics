"""Stage10 summary: per-fold held-out AUC with bootstrap CI vs Stage7 20-gene random panels; side by side with Stage7 original Native
and Stage9 selectors (no forward selection) and mRMR_scaled (Stage9)."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
S10 = Path(__file__).resolve().parents[1]; MT = S10.parent; SEED = 42
LAB = json.load(open(MT / "Stage7/data/labels.json")); Y = {t: {c: np.array(v[1]) for c, v in LAB[t].items()} for t in LAB}
def boot_ci(y, s):
    rng = np.random.default_rng(SEED); I = np.stack([rng.integers(0, len(y), len(y)) for _ in range(1000)]); Yb, Sb = y[I], s[I]
    Rk = rankdata(Sb, axis=1); n1 = Yb.sum(1); n0 = Yb.shape[1] - n1; ok = (n1 > 0) & (n0 > 0)
    b = ((Rk * Yb).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1); return np.percentile(b[ok], [2.5, 97.5])
RA = pd.read_csv(MT / "Stage8/data/random_auc_long.csv"); RA = RA[RA.Panel == "Native"]
rows = []
for r in map(json.loads, open(S10 / "results/records.jsonl")):
    t, h = r["task"], r["heldout"]; T = t.replace("_vs_", " vs "); y = Y[t][h]
    for m, s in r["heldout_scores"].items():
        a = r["heldout_auc"][m]; lo, hi = boot_ci(y, np.asarray(s)); ra = RA[(RA.Task == T) & (RA["Held-out cohort"] == h) & (RA.method == m)].AUC.to_numpy()
        rows.append({"Task": T, "Held-out cohort": h, "Selector": r["arm"], "Method": m, "Candidates": r["n_candidates"], "Training cross-cohort AUC": round(r["train_cross_auc"], 3),
                     "AUC": round(a, 3), "AUC 95% CI lower": round(lo, 3), "AUC 95% CI upper": round(hi, 3), "Random 20-gene median": round(float(np.median(ra)), 3),
                     "Proportion random >= selector": round(float((ra >= a).mean()), 3), "Panel": ";".join(r["panel"])})
S = pd.DataFrame(rows); S.to_csv(S10 / "results/selector_results_stage10.csv", index=False)
S9 = pd.read_csv(MT / "Stage9/results/selector_results.csv")
M7 = pd.read_csv(MT / "Stage7/results/MASTER_results_table.csv"); M7 = M7[M7.Panel == "Native"].copy()
MN = {"Logistic regression": "LR", "Random forest": "RF", "MLP (alpha fixed at 10)": "MLP", "singscore (rank-based, untrained)": "singscore", "Directional average (untrained)": "DirAvg"}
M7 = M7.assign(Selector="original Native (mRMR + FS)", Method=M7.Method.map(MN), **{"Proportion random >= selector": M7["Proportion of random panels with AUC >= panel"]})
cols = ["Task", "Held-out cohort", "Selector", "Method", "AUC", "Proportion random >= selector"]
ALL = pd.concat([M7[cols], S9[cols], S[cols]]); ALL["beats"] = ALL["Proportion random >= selector"] <= 0.05
order = ["original Native (mRMR + FS)", "mRMR_scaled", "MI20_FS", "consistent60_FS", "MI_meanrank", "consistent"]
G = ALL.groupby(["Task", "Selector"]).agg(AUC_min=("AUC", "min"), AUC_median=("AUC", "median"), AUC_max=("AUC", "max"), beats=("beats", "sum"), cells=("beats", "size")).reset_index()
G["Selector"] = pd.Categorical(G.Selector, order); G = G.sort_values(["Task", "Selector"]); G.to_csv(S10 / "results/all_selectors_comparison.csv", index=False)
tf = pd.concat([S9.drop_duplicates(["Task", "Held-out cohort", "Selector"])[["Task", "Selector", "Training cross-cohort AUC"]], S.drop_duplicates(["Task", "Held-out cohort", "Selector"])[["Task", "Selector", "Training cross-cohort AUC"]]])
print(G.round(3).to_string(index=False)); print(tf.groupby(["Task", "Selector"])["Training cross-cohort AUC"].median().round(3).to_string())
print(ALL[ALL.Task == "ATB vs OD"].pivot_table(index=["Held-out cohort", "Method"], columns="Selector", values="AUC")[order].round(2).to_string())
