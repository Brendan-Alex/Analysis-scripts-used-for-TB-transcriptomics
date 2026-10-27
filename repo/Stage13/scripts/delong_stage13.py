"""Stage13b: paired DeLong tests on the external validation scores (same samples, two panels) + master table with external rows.
Reads only Stage13/results/external_scores_long.csv.gz, external_auc.csv, random_panels.jsonl and Stage7 master table. Run: python scripts/delong_stage13.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
S13 = Path(__file__).resolve().parents[1]; MT = S13.parent; R = S13 / "results"
def midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2
def delong(y, s1, s2):
    """fast DeLong (Sun & Xu 2014): returns auc1, auc2, diff, se, z, p (two-sided), 95% CI of diff"""
    y = np.asarray(y).astype(bool); P = np.vstack([s1, s2]); m, n = y.sum(), (~y).sum(); X, Yn = P[:, y], P[:, ~y]
    tx = np.array([midrank(r) for r in X]); ty = np.array([midrank(r) for r in Yn]); tz = np.array([midrank(r) for r in P])
    auc = (tz[:, y].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, y] - tx) / n; v10 = 1 - (tz[:, ~y] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n; d = auc[0] - auc[1]; se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    z = d / se if se > 0 else 0.0; return auc[0], auc[1], d, se, z, 2 * norm.sf(abs(z)), (d - 1.96 * se, d + 1.96 * se)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj
S = pd.read_csv(R / "external_scores_long.csv.gz"); A = pd.read_csv(R / "external_auc.csv"); rows = []
for G, g in S.groupby("cohort"):
    for m, gm in g.groupby("method"):
        W = gm.pivot(index="GSM", columns="panel", values="score"); y = gm.drop_duplicates("GSM").set_index("GSM").y.reindex(W.index).to_numpy()
        for q in [c for c in W.columns if c != "S11_consistent"]:
            a1, a2, d, se, z, p, (lo, hi) = delong(y, W["S11_consistent"].to_numpy(), W[q].to_numpy())
            rows.append(dict(cohort=G, method=m, signature="S11_consistent", comparator=q, AUC_signature=round(a1, 4), AUC_comparator=round(a2, 4),
                             AUC_diff=round(d, 4), diff_CI_lo=round(lo, 4), diff_CI_hi=round(hi, 4), z=round(z, 3), p=p))
D = pd.DataFrame(rows)
# family for multiplicity: per cohort x method, all comparators (Holm)
D["p_Holm_within_cohort_method"] = D.groupby(["cohort", "method"]).p.transform(lambda p: holm(p.to_numpy()))
D["primary_family (DirAvg)"] = D.method.eq("DirAvg"); D["p"] = D.p.round(5); D["p_Holm_within_cohort_method"] = D.p_Holm_within_cohort_method.round(5)
D.to_csv(R / "delong_paired.csv", index=False)
# ---- master table with external rows
M = pd.read_csv(MT / "Stage7/results/MASTER_results_table.csv"); M.insert(0, "Analysis", "Rotating LOCO (Stage7)")
MM = {"LR": "Logistic regression", "RF": "Random forest", "MLP": "MLP (alpha fixed at 10)", "singscore": "singscore (rank-based, untrained)", "DirAvg": "Directional average (untrained)"}
PN = {"S11_consistent": "Stage11 consistent signature (final)", "S11_MI_meanrank": "Stage11 MI-meanrank signature (final)", "Combined8": "Combined 8-gene (Stage11 consistent ATB vs LTB + ATB vs OD)"}
tr = {t: M[M.Task == t].drop_duplicates("Held-out cohort") for t in M.Task.unique()}
rnd = [json.loads(l) for l in open(R / "random_panels.jsonl")]; ext = []
for r in A.itertuples():
    t = r.task.replace("_", " "); T = tr[t]; neg = r.task.split("_vs_")[1]
    row = {c: np.nan for c in M.columns}
    row.update({"Analysis": "External validation (Stage13; locked models trained on all task cohorts)", "Task": t, "Positive class": "ATB", "Negative class": neg,
                "Held-out cohort": r.cohort, "Training cohorts": ";".join(T["Held-out cohort"]), "Training n": int(T["Test n"].sum()),
                "Training positive n": int(T["Test positive n"].sum()), "Training negative n": int(T["Test negative n"].sum()),
                "Test n": r.n_pos + r.n_neg, "Test positive n": r.n_pos, "Test negative n": r.n_neg, "Panel": PN.get(r.panel, r.panel),
                "Panel genes used": r.features_in_all_training, "Panel genes zero-filled in held-out cohort": r.features_zero_filled_in_validation,
                "Method": MM[r.method], "AUC": r.AUC, "AUC 95% CI lower": r.CI_lo, "AUC 95% CI upper": r.CI_hi,
                "Derivation overlap with held-out cohort (original list)": False, "Derivation overlap via sub-series containment (added, accepted)": False,
                "Derivation relationship": "independent validation cohort", "Optimistic (any derivation overlap)": False})
    dist = [x["auc"][r.method] for x in rnd if x["cohort"] == r.cohort and x["k"] == r.panel_genes] if r.panel in PN else []
    if dist:
        row.update({"Random panels (n)": len(dist), "Random panel AUC median": round(float(np.median(dist)), 3), "Random panel AUC 2.5th pct": round(float(np.percentile(dist, 2.5)), 3),
                    "Random panel AUC 97.5th pct": round(float(np.percentile(dist, 97.5)), 3), "Proportion of random panels with AUC >= panel": round(float(np.mean(np.array(dist) >= r.AUC)), 3)})
    ext.append(row)
X = pd.concat([M, pd.DataFrame(ext)], ignore_index=True); X.to_csv(R / "MASTER_results_table_with_external.csv", index=False)
print(len(M), "+", len(ext), "=", len(X), "rows")
pd.set_option("display.width", 250); print(D[D.method.isin(["DirAvg", "singscore", "LR"])].drop(columns=["signature", "primary_family (DirAvg)"]).to_string(index=False))
