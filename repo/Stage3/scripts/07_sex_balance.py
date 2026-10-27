"""Stage3 diagnostic: Y-linked genes (RPS4Y1, TXLNGY, KDM5D) entered the Native panels; check sex balance by class.
Sex from GEO 'gender'/'sex' characteristic where present, else inferred from RPS4Y1 (log2, 2-means split)."""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
G = pd.read_csv(CR / "qc/geo_samples_long.csv", low_memory=False)
rows = []
for task in ["ATB_vs_LTB", "LTB_vs_CON"]:
    D = pickle.load(open(ST / f"data/cache/{task}.pkl", "rb")); t = TASKS[task]
    for c, d in D.items():
        g = G[G.cohort == c].set_index("GSM"); col = next((k for k in ["ch:gender", "ch:sex", "ch:Sex"] if k in g.columns and g[k].notna().any()), None)
        if col:
            s = g.loc[d["y"].index, col].astype(str).str.upper().str[0].map({"M": "M", "F": "F"}); src = "GEO " + col[3:]
        else:
            v = d["L"].loc["RPS4Y1"] if "RPS4Y1" in d["L"].index else None
            if v is None: continue
            thr = (v.min() + v.max()) / 2
            for _ in range(20): thr = (v[v > thr].mean() + v[v <= thr].mean()) / 2
            s = (v > thr).map({True: "M", False: "F"}); src = "RPS4Y1 inferred"
        y = d["y"]
        rows.append(dict(task=task, cohort=c, sex_source=src, male_frac_pos=round((s[y == 1] == "M").mean(), 2), male_frac_neg=round((s[y == 0] == "M").mean(), 2),
                         n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum())))
T = pd.DataFrame(rows); T.to_csv(ST / "results/sex_balance.csv", index=False); print(T.to_string())
