"""Stage3 diagnostic: is rank-based scoring really immune to the global shift?
For 200 random 20-gene sets (seed 42), score each sample by the plain mean within-sample rank of the set (no direction,
no training) and report the AUC vs class. Under no composition effect, AUCs scatter around 0.5.
Also by expression stratum: random sets drawn from the top / middle / bottom third of genes by cohort mean."""
import os, sys, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__)); from common import *
rows = []
for task in ["ATB_vs_LTB", "LTB_vs_CON"]:
    D = pickle.load(open(ST / f"data/cache/{task}.pkl", "rb"))
    for c, d in D.items():
        R, L, y = d["R"], d["L"], d["y"].to_numpy(); mean = L.mean(1).sort_values(); n = len(mean)
        strata = {"all": mean.index, "low third": mean.index[: n // 3], "mid third": mean.index[n // 3: 2 * n // 3], "high third": mean.index[2 * n // 3:]}
        for sname, pool in strata.items():
            rng = np.random.default_rng(SEED); a = []
            for _ in range(200):
                g = rng.choice(pool, 20, replace=False); a.append(roc_auc_score(y, R.loc[g].mean().to_numpy()))
            a = np.array(a)
            rows.append(dict(task=task, cohort=c, stratum=sname, median_AUC=round(np.median(a), 3), p2_5=round(np.percentile(a, 2.5), 3),
                             p97_5=round(np.percentile(a, 97.5), 3), share_AUC_lt_0_3=round((a < .3).mean(), 2), share_AUC_gt_0_7=round((a > .7).mean(), 2)))
T = pd.DataFrame(rows); T.to_csv(ST / "results/rank_composition_check.csv", index=False); print(T.to_string())
