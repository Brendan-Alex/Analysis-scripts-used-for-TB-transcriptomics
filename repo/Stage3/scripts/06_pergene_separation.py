"""Stage3 diagnostic: how much of the transcriptome separates the classes? per-gene AUC (log2 values) in each cohort."""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(__file__)); from common import *
rows = []
for task in ["ATB_vs_LTB", "LTB_vs_CON"]:
    D = pickle.load(open(ST / f"data/cache/{task}.pkl", "rb"))
    for c, d in D.items():
        L, y = d["L"].to_numpy(), d["y"].to_numpy(); R = rankdata(L, axis=1); n1 = y.sum(); n0 = len(y) - n1
        auc = (R[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0); dev = np.abs(auc - 0.5)
        rows.append(dict(task=task, cohort=c, genes=len(auc), share_gene_AUC_dev_gt_0_1=round((dev > .1).mean(), 3), share_dev_gt_0_2=round((dev > .2).mean(), 3),
                         share_dev_gt_0_3=round((dev > .3).mean(), 3), share_genes_lower_in_pos=round((auc < .5).mean(), 3)))
T = pd.DataFrame(rows); T.to_csv(ST / "results/pergene_separation.csv", index=False); print(T.to_string())
