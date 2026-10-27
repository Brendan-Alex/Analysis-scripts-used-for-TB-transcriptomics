"""Stage3: AUCs with percentile bootstrap 95% CIs (1000 resamples, seed 42) for curated panels (internal LOCO and external),
size-matched Random distributions (50 draws), MLP convergence and over-fitting diagnostics."""
import os, sys, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__)); from common import *
RES = ST / "results/bench"
from scipy.stats import rankdata
def boot(y, s, n=1000):
    """same resamples as the sequential loop (rng.integers(0, n, n) per replicate, seed 42); AUC by Mann-Whitney, vectorised"""
    rng = np.random.default_rng(SEED); I = np.stack([rng.integers(0, len(y), len(y)) for _ in range(n)])
    Y = y[I]; S = s[I]; R = rankdata(S, axis=1); n1 = Y.sum(1); n0 = Y.shape[1] - n1; ok = (n1 > 0) & (n0 > 0)
    auc = ((R * Y).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1)
    return np.percentile(auc[ok], [2.5, 97.5])
cur, rnd, dg = [], [], []
for task in ["ATB_vs_LTB", "LTB_vs_CON"]:
    D = pickle.load(open(ST / f"data/cache/{task}.pkl", "rb"))
    for line in open(RES / f"{task}__records.jsonl"):
        r = json.loads(line)
        for m, v in r["scores"].items():
            for c, s in v.items():
                y = D[c]["y"].to_numpy(); s = np.array(s); auc = roc_auc_score(y, s)
                if r["kind"] == "random":
                    rnd.append(dict(task=task, size=int(r["panel"][6:]), draw=int(r["heldout_or_draw"]), method=m, cohort=c, AUC=auc))
                else:
                    lo, hi = boot(y, s)
                    cur.append(dict(task=task, evaluation=r["kind"], panel=r["panel"], cohort=c, method=m, n=len(y), n_pos=int(y.sum()),
                                    AUC=round(auc, 3), CI_low=round(lo, 3), CI_high=round(hi, 3), n_features=r["diag"]["n_features"],
                                    zero_filled=r["diag"]["zero_filled_in_test"].get(c)))
        dg.append(dict(task=task, key=r["key"], **{k: v for k, v in r["diag"].items() if k != "zero_filled_in_test"}))
C = pd.DataFrame(cur); R = pd.DataFrame(rnd); G = pd.DataFrame(dg)
# random comparison: size-matched
used = C[C.evaluation == "external"].groupby(["task", "panel"]).n_features.first()
rows = []
for (task, p, c, m), g in C[C.evaluation == "external"].groupby(["task", "panel", "cohort", "method"]):
    rm = "MLP" if m == "MLP_alpha" else m
    dist = R[(R.task == task) & (R["size"] == used[(task, p)]) & (R.method == rm) & (R.cohort == c)].AUC
    a = g.AUC.iloc[0]
    rows.append(dict(task=task, panel=p, cohort=c, method=m, AUC=a, CI=f"{g.CI_low.iloc[0]:.3f}-{g.CI_high.iloc[0]:.3f}", random_size=used[(task, p)],
                     random_median=round(dist.median(), 3), random_p2_5=round(dist.quantile(.025), 3), random_p97_5=round(dist.quantile(.975), 3),
                     random_min=round(dist.min(), 3), random_max=round(dist.max(), 3), prop_random_ge_curated=round((dist >= a).mean(), 2)))
X = pd.DataFrame(rows)
C.to_csv(ST / "results/curated_auc_with_ci.csv", index=False); R.to_csv(ST / "results/random_draw_auc.csv", index=False)
X.to_csv(ST / "results/external_vs_random.csv", index=False); G.to_csv(ST / "results/model_diagnostics.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for task in ["ATB_vs_LTB", "LTB_vs_CON"]:
    print(f"\n===== {task}: EXTERNAL =====")
    x = X[X.task == task].copy(); x["AUC_CI"] = x.AUC.map("{:.3f}".format) + " (" + x.CI + ")"
    x["rand"] = x.random_median.map("{:.3f}".format) + " [" + x.random_p2_5.map("{:.2f}".format) + "-" + x.random_p97_5.map("{:.2f}".format) + "] p=" + x.prop_random_ge_curated.map(str)
    for c in x.cohort.unique():
        print("--", c); print(x[x.cohort == c].pivot(index="panel", columns="method", values="AUC_CI")[["LR", "RF", "MLP", "MLP_alpha", "singscore", "DirAvg"]].to_string())
    print("-- random (median [2.5-97.5%], share of random >= curated)")
    for c in x.cohort.unique(): print(c); print(x[(x.cohort == c)].pivot(index="panel", columns="method", values="rand")[["LR", "RF", "MLP", "singscore", "DirAvg"]].to_string())
    print(f"\n===== {task}: INTERNAL LOCO =====")
    i = C[(C.task == task) & (C.evaluation == "internal")].copy(); i["v"] = i.AUC.map("{:.3f}".format)
    print(i.pivot_table(index=["panel", "cohort"], columns="method", values="v", aggfunc="first")[["LR", "RF", "MLP", "MLP_alpha", "singscore", "DirAvg"]].to_string())
print("\nMLP convergence (curated + random fits):", G.MLP_converged.value_counts().to_dict(), "| MLP_alpha:", G.MLP_alpha_converged.dropna().value_counts().to_dict())
print("alpha chosen:", G.MLP_alpha_chosen.dropna().value_counts().to_dict())
