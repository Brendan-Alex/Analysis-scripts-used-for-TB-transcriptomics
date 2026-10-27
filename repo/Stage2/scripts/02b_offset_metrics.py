"""Stage2 item 2 metrics: does the per-sample all-gene mean separate sub-series, before vs after joint normalisation?
AUC = Mann-Whitney AUC of the per-sample mean (0.5 = no separation; reported as max(AUC, 1-AUC) too).
Also: all-probe mean after quantile normalisation (identical by construction), and a disease-constant batch check:
GSE22098 healthy controls vs London controls (GSE19439/GSE19444), same class, different sub-series."""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
from pathlib import Path
import itertools, sys
import numpy as np, pandas as pd
from scipy.stats import kruskal, mannwhitneyu, ttest_ind
S = Path(__file__).resolve().parents[1]; CR = S.parent / "Clean_rebuild"
T = pd.read_csv(S / "qc/GSE19491_sample_means.csv")
def auc(a, b):
    u = mannwhitneyu(a, b).statistic; return u / (len(a) * len(b))
inc = T[T.role == "included"]; rows = []
for col in ["mean_before", "mean_after_primary"]:
    subs = sorted(inc.subseries.unique())
    H = kruskal(*[inc[inc.subseries == s][col] for s in subs]); rows.append(dict(metric=col, comparison="Kruskal-Wallis across 5 sub-series", value=round(H.statistic, 2), p=H.pvalue))
    for s in subs:
        a = auc(inc[inc.subseries == s][col], inc[inc.subseries != s][col])
        rows.append(dict(metric=col, comparison=f"{s} vs other sub-series (AUC)", value=round(a, 3), sep=round(max(a, 1 - a), 3)))
    for s in [x for x in subs if x != "GSE22098"]:
        a = auc(inc[inc.subseries == "GSE22098"][col], inc[inc.subseries == s][col])
        rows.append(dict(metric=col, comparison=f"GSE22098 (all OD) vs {s} (AUC)", value=round(a, 3), sep=round(max(a, 1 - a), 3)))
    a = auc(inc[inc.class_label == "ATB"][col], inc[inc.class_label == "OD"][col])
    rows.append(dict(metric=col, comparison="ATB vs OD (class AUC of sample mean)", value=round(a, 3), sep=round(max(a, 1 - a), 3)))
    a = auc(inc[inc.class_label == "ATB"][col], inc[inc.class_label == "LTB"][col])
    rows.append(dict(metric=col, comparison="ATB vs LTB (class AUC of sample mean)", value=round(a, 3), sep=round(max(a, 1 - a), 3)))
# diagnostic: same class, different sub-series
hc = T[T.role != "included"]; lon = T[(T.role == "included") & (T.class_label == "CON")]
for col in ["mean_before", "mean_after_diagnostic"]:
    a = auc(hc[col], lon[col]); rows.append(dict(metric=col, comparison="GSE22098 healthy controls vs London controls (AUC)", value=round(a, 3), sep=round(max(a, 1 - a), 3)))
R = pd.DataFrame(rows); R.to_csv(S / "qc/GSE19491_offset_metrics.csv", index=False); print(R.to_string())
# ---- all-probe mean after QN (constant by construction) and gene-level residual batch (disease held constant)
from scipy.stats import false_discovery_control
import glob, os
E2 = pd.read_csv(S / "expression/GSE19491_log2_genelevel_jointnorm.csv.gz", index_col=0)
print("after primary: per-sample mean over the gene-level file  sd =", round(float(E2.mean().std()), 4))
sm = pd.read_csv(DATA_ROOT / "Full data/Train folder/GSE-19491/GSE19491_series_matrix.txt", sep="\t", comment="!", index_col=0)
old = pd.read_csv(CR / "qc/probe_selection_GSE19491.csv"); osel = old[old.selected]
H, Lc = hc.GSM.tolist(), lon.GSM.tolist()
B = np.log2(np.clip(sm.loc[osel.probe, H + Lc], 1, None)); B.index = osel.symbol_hgnc.values
exec(open(S / "scripts/_diag_matrix.py").read())     # builds D (after, diagnostic normalisation) genes x (H+Lc)
out = []
for name, X in [("before (GEO values)", B), ("after (joint quantile)", D)]:
    common = X.index
    t = ttest_ind(X[H].to_numpy(), X[Lc].to_numpy(), axis=1, equal_var=False)
    q = false_discovery_control(np.nan_to_num(t.pvalue, nan=1), method="bh"); dmean = X[H].mean(1) - X[Lc].mean(1)
    out.append(dict(data=name, genes=len(common), FDR05=int((q < 0.05).sum()), FDR05_absdiff_gt1=int(((q < 0.05) & (dmean.abs() > 1)).sum()),
                    median_abs_diff=round(float(dmean.abs().median()), 3)))
O = pd.DataFrame(out); O.to_csv(S / "qc/GSE19491_batch_check_controls.csv", index=False); print(O.to_string())
