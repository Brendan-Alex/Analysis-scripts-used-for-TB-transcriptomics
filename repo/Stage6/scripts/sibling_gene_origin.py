"""Supplementary diagnostic for the GSE19439 fold (LTB_vs_CON): which genes of the Stage5 Native panel carry the GSE19439 result?
Splits the Stage5 GSE19439-fold panel by origin (genes from GSE19444's mRMR pool vs the other training cohorts' pools) and scores
GSE19439 with DirAvg and LR trained on (a) GSE28623+GSE107994 only (sibling excluded) and (b) GSE19444 only (sibling alone).
Runs in the Cowork VM on a cache rebuilt there by the Stage5 prepare code (STAGE5_DIR env; Windows pickles need pandas 3);
cross-check: (a) with the full panel must reproduce sibling_exclusion.csv (Native_stage5)."""
import os, sys, json, importlib.util, numpy as np, pandas as pd
from pathlib import Path
S6 = Path(__file__).resolve().parents[1]; MT = S6.parent
os.environ["STAGE5_OUT"] = os.environ.get("STAGE5_DIR", str(MT / "Stage5"))
sp = importlib.util.spec_from_file_location("s5", str(MT / "Stage5/scripts/run_stage5.py")); s5 = importlib.util.module_from_spec(sp); sp.loader.exec_module(s5)
from sklearn.metrics import roc_auc_score
t, h = "LTB_vs_CON", "GSE19439"; D = s5.load(t)
panel = json.load(open(MT / "Stage5/results/native" / f"{t}__fold_{h}.json"))["panel"]
pool44 = set(json.load(open(MT / "Stage5/results/native" / f"{t}__GSE19444__cohort.json"))["mrmr_top20"])
sets = {"full Stage5 panel (20)": panel, "genes from GSE19444 pool": [g for g in panel if g in pool44], "genes from other pools": [g for g in panel if g not in pool44]}
rows = []
for name, G in sets.items():
    for trn_name, trn in [("GSE28623+GSE107994 (sibling excluded)", ["GSE28623", "GSE107994"]), ("GSE19444 only (sibling alone)", ["GSE19444"])]:
        out, diag, feats = s5.fit_eval(D, trn, [h], G, ["LR", "DirAvg"])
        rows.append(dict(gene_set=name, n_genes=len(G), training=trn_name, **{f"AUC_{m}": round(roc_auc_score(D[h]["y"], out[m][h]), 3) for m in out}))
R = pd.DataFrame(rows); R.to_csv(S6 / "results/sibling_gene_origin.csv", index=False); print(R.to_string())
