"""Stage9 summaries. Selectors: held-out AUC with bootstrap CI vs Stage7's 20-gene random panels (same fold, universe, size, seeds, method).
Ablation: curves (held-out and training AUC by panel size) and comparison of training-informed removal orders with 50 random orders."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
S9 = Path(__file__).resolve().parents[1]; MT = S9.parent; R = S9 / "results"; SEED = 42
LAB = json.load(open(MT / "Stage7/data/labels.json")); Y = {t: {c: np.array(v[1]) for c, v in LAB[t].items()} for t in LAB}
def boot_ci(y, s):
    rng = np.random.default_rng(SEED); I = np.stack([rng.integers(0, len(y), len(y)) for _ in range(1000)]); Yb, Sb = y[I], s[I]
    Rk = rankdata(Sb, axis=1); n1 = Yb.sum(1); n0 = Yb.shape[1] - n1; ok = (n1 > 0) & (n0 > 0)
    b = ((Rk * Yb).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1); return np.percentile(b[ok], [2.5, 97.5])
RA = pd.read_csv(MT / "Stage8/data/random_auc_long.csv"); RA = RA[RA.Panel == "Native"]       # Stage7 20-gene random panels
M7 = pd.read_csv(MT / "Stage7/results/MASTER_results_table.csv"); M7 = M7[M7.Panel == "Native"]
MN = {"LR": "Logistic regression", "RF": "Random forest", "MLP": "MLP (alpha fixed at 10)", "singscore": "singscore (rank-based, untrained)", "DirAvg": "Directional average (untrained)"}
rows = []
for r in map(json.loads, open(R / "selectors/selector_records.jsonl")):
    t, h = r["task"], r["heldout"]; T = t.replace("_vs_", " vs "); y = Y[t][h]
    for m, s in r["heldout_scores"].items():
        a = r["heldout_auc"][m]; lo, hi = boot_ci(y, np.asarray(s)); ra = RA[(RA.Task == T) & (RA["Held-out cohort"] == h) & (RA.method == m)].AUC.to_numpy()
        n7 = M7[(M7.Task == T) & (M7["Held-out cohort"] == h) & (M7.Method == MN[m])].AUC.iloc[0]
        rows.append({"Task": T, "Held-out cohort": h, "Selector": r["selector"], "Method": m, "Training cross-cohort AUC": round(r["train_cross_auc"], 3), "AUC": round(a, 3),
                     "AUC 95% CI lower": round(lo, 3), "AUC 95% CI upper": round(hi, 3), "Random 20-gene median": round(float(np.median(ra)), 3),
                     "Proportion random >= selector": round(float((ra >= a).mean()), 3), "Original Native AUC (Stage7)": n7, "Selector minus original Native": round(a - n7, 3),
                     "Sibling flag (GSE19444 in selection)": t == "LTB_vs_CON" and h == "GSE19439", "Panel": ";".join(r["panel"])})
S = pd.DataFrame(rows); S.to_csv(R / "selector_results.csv", index=False)
S["beats"] = S["Proportion random >= selector"] <= 0.05
print(S.groupby(["Task", "Selector"]).agg(AUC_min=("AUC", "min"), AUC_max=("AUC", "max"), beats=("beats", "sum"), cells=("beats", "size"),
      d_med=("Selector minus original Native", "median"), trainfit=("Training cross-cohort AUC", "median")).round(3).to_string())
nat = M7.assign(beats=M7["Proportion of random panels with AUC >= panel"] <= 0.05).groupby("Task").agg(AUC_min=("AUC", "min"), AUC_max=("AUC", "max"), beats=("beats", "sum"), cells=("beats", "size"))
print("original Native:\n", nat.to_string())
# ---------------- ablation
A = []
for line in open(R / "ablation/ablation_records.jsonl"):
    r = json.loads(line); base = dict(task=r["task"], heldout=r["heldout"], panel=r["panel"], order=r["order"], rand=r["rand"], k=r["k"], train=r["train_cross_auc"])
    for m, v in r["heldout_auc"].items(): A.append({**base, "method": m, "auc": v})
A = pd.DataFrame(A); A.to_csv(R / "ablation_long.csv.gz", index=False)
n = A.groupby(["task", "heldout", "panel"]).k.max().rename("n").reset_index(); A = A.merge(n)
# (1) informed vs random removal: mean held-out AUC over all sizes, percentile within the 50 random orders
cmp = []
for (t, h, p, m), g in A[A.method.isin(["LR", "singscore", "DirAvg"])].groupby(["task", "heldout", "panel", "method"]):
    rnd = g[g.order == "random"].groupby("rand").auc.mean()
    for o in ("backward", "univariate"):
        x = g[g.order == o].auc.mean(); cmp.append(dict(task=t, heldout=h, panel=p, method=m, order=o, mean_heldout_over_sizes=round(x, 4),
            random_median=round(float(rnd.median()), 4), percentile_vs_random=round(float((rnd < x).mean() + 0.5 * (rnd == x).mean()), 3)))
C = pd.DataFrame(cmp); C.to_csv(R / "ablation_informed_vs_random.csv", index=False)
print(C.groupby(["task", "panel", "order"]).percentile_vs_random.agg(["median", "min", "max"]).round(2).to_string())
# (2) shape: held-out and training at full, half, quarter, 3, 1 genes (ordered removals)
def at(g, k): x = g[g.k == k]; return x.auc.iloc[0] if len(x) else np.nan
sh = []
for (t, h, p, o, m), g in A[A.order.isin(["backward", "univariate"])].groupby(["task", "heldout", "panel", "order", "method"]):
    nn = g.n.iloc[0]; ks = [nn, max(1, round(nn / 2)), max(1, round(nn / 4)), 3, 1]; tr = g.drop_duplicates("k").set_index("k").train
    sh.append(dict(task=t, heldout=h, panel=p, order=o, method=m, n=nn, **{f"heldout_k{k}": round(at(g, k), 3) for k in ["full", "half", "quarter", "3", "1"] for _ in [0]} if False else {},
                   heldout_full=round(at(g, nn), 3), heldout_half=round(at(g, ks[1]), 3), heldout_quarter=round(at(g, ks[2]), 3), heldout_3=round(at(g, 3), 3), heldout_1=round(at(g, 1), 3),
                   train_full=round(tr.get(nn), 3), train_half=round(tr.get(ks[1]), 3), train_quarter=round(tr.get(ks[2]), 3), train_3=round(tr.get(3), 3), train_1=round(tr.get(1), 3),
                   heldout_best_k=int(g.loc[g.auc.idxmax(), "k"])))
SH = pd.DataFrame(sh); SH["d_half"] = SH.heldout_half - SH.heldout_full; SH["d_train_half"] = SH.train_half - SH.train_full
SH.to_csv(R / "ablation_shape.csv", index=False)
print(SH[SH.method.isin(["LR", "DirAvg"])].groupby(["task", "panel", "order", "method"]).agg(d_half_med=("d_half", "median"), d_half_min=("d_half", "min"), d_half_max=("d_half", "max"),
      dtrain_half=("d_train_half", "median"), up=("d_half", lambda v: int((v > 0).sum())), n=("d_half", "size")).round(3).to_string())
