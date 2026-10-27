"""Sex balance by class in the ATB_vs_OD cohorts (Stage3 rule, unchanged): GEO 'gender'/'sex' characteristic where present, else
inferred from RPS4Y1 log2 expression (iterative 2-means split). Reads expression/labels directly (no cache needed)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
S7 = Path(__file__).resolve().parents[1]; MT = S7.parent; CR = MT / "Clean_rebuild"; S3 = MT / "Stage3"
G = pd.read_csv(CR / "qc/geo_samples_long.csv", low_memory=False); M = pd.read_csv(S3 / "data/sample_manifest_stage3.csv"); M = M[M.included_stage3]
def expr(c): return pd.read_csv(S7 / "data/GSE83456_log2_genelevel_rebuilt.csv.gz" if c == "GSE83456" else CR / f"expression/{c}_log2_genelevel.csv.gz", index_col=0)
rows = []
for c in ["GSE37250", "GSE73408", "GSE42834", "GSE83456"]:
    m = M[(M.cohort == c) & M.class_label.isin(["ATB", "OD"])].set_index("GSM"); y = (m.class_label == "ATB").astype(int)
    g = G[G.cohort == c].set_index("GSM"); col = next((k for k in ["ch:gender", "ch:sex", "ch:Sex"] if k in g.columns and g.loc[g.index.intersection(m.index), k].notna().any()), None)
    if col:
        s = g.loc[y.index, col].astype(str).str.upper().str[0].map({"M": "M", "F": "F"}); src = "GEO " + col[3:]
    else:
        v = expr(c).loc["RPS4Y1", y.index]; thr = (v.min() + v.max()) / 2
        for _ in range(20): thr = (v[v > thr].mean() + v[v <= thr].mean()) / 2
        s = (v > thr).map({True: "M", False: "F"}); src = "RPS4Y1 inferred"
    rows.append(dict(task="ATB_vs_OD", cohort=c, sex_source=src, male_frac_ATB=round((s[y == 1] == "M").mean(), 2), male_frac_OD=round((s[y == 0] == "M").mean(), 2),
                     n_ATB=int(y.sum()), n_OD=int((1 - y).sum()), n_sex_unknown=int(s.isna().sum())))
T = pd.DataFrame(rows); T.to_csv(S7 / "results/sex_balance_ATB_vs_OD.csv", index=False); print(T.to_string())
