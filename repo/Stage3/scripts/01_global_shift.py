"""Stage3 item 2: per-sample all-gene mean (log2, every gene in the cohort's file) scored against class.
AUC: positive = first-named class (ATB in ATB_vs_LTB, LTB in LTB_vs_CON ...). AUC < 0.5 = positive class has LOWER mean.
Bootstrap 95% CI: 1000 resamples, seed 42 (stratified by nothing, as in the benchmark).
Also the per-sample median and the mean over the genes shared by all 10 cohorts.
Normalisation caveat column: quantile-normalised cohorts (GSE19491 stage3, GSE42834) have near-constant all-gene
distributions by construction; their GEO-processed versions are reported as extra rows."""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__)); from common import *
M = manifest(); inc = M[M.included_stage3]
COH = ["GSE19491", "GSE37250", "GSE73408", "GSE19439", "GSE19444", "GSE28623", "GSE101705", "GSE107994", "GSE42834", "GSE83456"]
E = {c: expr(c) for c in COH}
shared = sorted(set.intersection(*[set(e.index) for e in E.values()]))
E["GSE19491 (GEO-processed values)"] = pd.read_csv(CR / "expression/GSE19491_log2_genelevel.csv.gz", index_col=0)
E["GSE42834 (GEO-normalised values)"] = None
CONTRASTS = [("ATB", "LTB"), ("LTB", "CON"), ("ATB", "CON"), ("ATB", "OD")]
ROLE = {}
for task, t in TASKS.items():
    for r in ["train", "test"]:
        for c in t[r]: ROLE.setdefault((c, t["pos"], t["neg"]), []).append(f"{task}:{r}" + (" (on hold)" if task == "ATB_vs_OD" else ""))
NORM = {"GSE19491": "quantile (joint, stage3)", "GSE42834": "quantile (Clean_rebuild raw rebuild)", "GSE37250": "GEO (GenomeStudio quantile)",
        "GSE73408": "GEO RMA", "GSE19439": "GEO BeadStudio scaled", "GSE19444": "GEO BeadStudio scaled", "GSE28623": "GEO Agilent processed",
        "GSE101705": "author normalised counts", "GSE107994": "edgeR log2 CPM", "GSE83456": "GEO, per-gene median-centred",
        "GSE19491 (GEO-processed values)": "GEO BeadStudio scaled per sub-series", "GSE42834 (GEO-normalised values)": "GEO, per-gene median-centred per sub-series"}
def boot(y, s, n=1000):
    rng = np.random.default_rng(SEED); a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2: a.append(roc_auc_score(y[i], s[i]))
    return np.percentile(a, [2.5, 97.5])
rows = []
for name, X in E.items():
    c = name.split(" ")[0]
    if X is None:   # GSE42834 GEO-normalised series-matrix values, same probes as Clean_rebuild would have used
        G = pd.read_csv(DATA_ROOT / "Full data/Test folder/GSE-42834/GSE42834_expression.csv", index_col=0)
        sel = pd.read_csv(CR / "qc/probe_selection_GSE42834.csv"); sel = sel[sel.selected]
        X = G.loc[sel.probe.astype(str)].copy(); X.index = sel.symbol_hgnc.values
    m = inc[inc.cohort == c].set_index("GSM"); m = m[m.index.isin(X.columns)]
    for p, n in CONTRASTS:
        a, b = m[m.class_label == p].index, m[m.class_label == n].index
        if len(a) == 0 or len(b) == 0: continue
        S = X[list(a) + list(b)]; y = np.r_[np.ones(len(a)), np.zeros(len(b))]
        r = dict(cohort=name, contrast=f"{p}_vs_{n}", role=";".join(ROLE.get((c, p, n), ["not used in a task"])), n_pos=len(a), n_neg=len(b),
                 normalisation=NORM[name], genes=X.shape[0])
        for lab, s in [("mean_all", S.mean().to_numpy()), ("median_all", S.median().to_numpy()), ("mean_shared", S.reindex(shared).mean().to_numpy())]:
            auc = roc_auc_score(y, s); lo, hi = boot(y, s); r[f"AUC_{lab}"] = round(auc, 3); r[f"CI_{lab}"] = f"{lo:.3f}-{hi:.3f}"
        r["pos_minus_neg_mean_log2"] = round(float(S[a].mean().mean() - S[b].mean().mean()), 4)
        r["sd_of_sample_means"] = round(float(S.mean().std()), 4)
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(ST / "results/global_shift_all_cohorts.csv", index=False)
pd.set_option("display.width", 250); print(R[["cohort", "contrast", "role", "n_pos", "n_neg", "AUC_mean_all", "CI_mean_all", "AUC_median_all", "AUC_mean_shared", "pos_minus_neg_mean_log2", "sd_of_sample_means"]].to_string())
print("shared genes:", len(shared))
