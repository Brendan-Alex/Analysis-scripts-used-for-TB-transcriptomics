"""Stage4 step 0.
 - per cohort: label-free z-score per gene over ALL of the cohort's included (Stage3) samples, whatever their class
   (user decision 19 Oct: the held-out cohort is standardised with its own label-free statistics; nothing else is fitted on it)
 - per task: cache {cohort: Z (task samples), L (log2, task samples), R (within-sample ranks over all genes), y}
 - sex-chromosome gene list from GEO platform annotations (GPL6947, GPL10558 'Chromosome'; GPL11532 'seqname'),
   symbols HGNC-mapped: a gene is sex-linked if any probe of it is annotated X, Y or XY
 - fold table and per-fold global-shift diagnostic of the held-out cohort"""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__)); from common4 import *
sys.path.insert(0, str(CR / "scripts")); from hgnc_alias import HGNCMapper
HG = HGNCMapper(); FD = DATA_ROOT / "Full data"
sex, anno = set(), set()
for f, col in [(FD / "Train folder/GSE-19491/GPL6947-13512.txt", "Chromosome"), (FD / "Train folder/GSE-37250/GPL10558-50081 (Master).txt", "Chromosome")]:
    t = pd.read_csv(f, sep="\t", comment="#", dtype=str, low_memory=False)[["Symbol", col]].dropna()
    t["g"] = t.Symbol.str.strip().map(lambda x: HG.map(x)["current"]); anno |= set(t.g)
    sex |= set(t[t[col].str.strip().str.upper().isin(["X", "Y", "XY"])].g)
a = pd.read_csv(FD / "Train folder/GSE-73408/GPL11532-32230.txt", sep="\t", comment="#", dtype=str, low_memory=False)
a["first"] = a.gene_assignment.fillna("---").str.split(" /// ").str[0].str.split(" // ").str[1].fillna("").str.strip()
a = a[a["first"] != ""]; a["g"] = a["first"].map(lambda x: HG.map(x)["current"]); anno |= set(a.g)
sex |= set(a[a.seqname.isin(["chrX", "chrY"])].g)
pd.DataFrame({"Gene": sorted(sex)}).to_csv(S4 / "data/sex_chromosome_genes.csv", index=False)
print("sex-linked genes:", len(sex), "| annotated genes:", len(anno))
M = manifest(); inc = M[M.included_stage3]
folds, shift = [], []
for task, t in TASKS.items():
    d = {}
    for c in t["cohorts"]:
        X = expr(c); m = inc[inc.cohort == c].set_index("GSM"); X = X[[s for s in X.columns if s in m.index]]
        Zall = X.sub(X.mean(1), axis=0).div(X.std(1, ddof=0).replace(0, np.nan), axis=0).fillna(0.0)
        keep = m.index[m.class_label.isin([t["pos"], t["neg"]])]; keep = [s for s in X.columns if s in keep]
        y = (m.loc[keep, "class_label"] == t["pos"]).astype(int)
        L = X[keep]; R = pd.DataFrame(rankdata(L.to_numpy(), axis=0), index=L.index, columns=keep)
        d[c] = dict(Z=Zall[keep], L=L, R=R, y=y, n_cohort_all=X.shape[1])
        yy = y.to_numpy(); s = L.mean().to_numpy(); auc = roc_auc_score(yy, s)
        rng = np.random.default_rng(SEED); I = np.stack([rng.integers(0, len(yy), len(yy)) for _ in range(1000)]); Y = yy[I]; S = s[I]
        Rk = rankdata(S, axis=1); n1 = Y.sum(1); n0 = Y.shape[1] - n1; ok = (n1 > 0) & (n0 > 0)
        b = ((Rk * Y).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1); lo, hi = np.percentile(b[ok], [2.5, 97.5])
        Rg = rankdata(L.to_numpy(), axis=1); ga = (Rg[:, yy == 1].sum(1) - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (len(yy) - yy.sum()))
        shift.append(dict(task=task, heldout=c, allgene_mean_AUC=round(auc, 3), CI=f"{lo:.3f}-{hi:.3f}", share_genes_AUCdev_gt_0_2=round(float((np.abs(ga - .5) > .2).mean()), 3),
                          quantile_normalised=c in ("GSE19491", "GSE42834")))
    pickle.dump(d, open(S4 / f"data/cache/{task}.pkl", "wb"))
    for h in t["cohorts"]:
        tr = [c for c in t["cohorts"] if c != h]
        folds.append(dict(task=task, heldout=h, training_cohorts=";".join(tr), n_train=sum(len(d[c]["y"]) for c in tr),
                          train_pos=sum(int(d[c]["y"].sum()) for c in tr), train_neg=sum(int((1 - d[c]["y"]).sum()) for c in tr),
                          n_test=len(d[h]["y"]), test_pos=int(d[h]["y"].sum()), test_neg=int((1 - d[h]["y"]).sum()),
                          z_reference_samples=d[h]["n_cohort_all"], all_training_have_both_classes=all(0 < d[c]["y"].sum() < len(d[c]["y"]) for c in tr)))
F = pd.DataFrame(folds); F.to_csv(S4 / "qc/folds.csv", index=False); print(F.to_string())
G = pd.DataFrame(shift); G.to_csv(S4 / "results/global_shift_heldout.csv", index=False); print(G.to_string())
