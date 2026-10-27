"""Step 2: link GEO *non-normalised* supplementary columns (which carry no GSM IDs: 'Sample N' or chip barcodes)
to GSMs, by expression identity against the GSM-labelled GEO series-matrix values. Never by column order.
Method: raw -> log2 -> quantile-normalise; keep probes with mean log2 > 7; per-probe centre both matrices, correlate every raw column with every GSM column on
shared probes; accept the best match if it is mutual-best, r_best >= 0.90 and r_best - r_second >= 0.10.
Also reports raw columns that are identical (or r>0.999) to each other = duplicate arrays.
Output: qc/raw_column_to_GSM.csv, qc/raw_duplicate_arrays.csv
"""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import gzip, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; RAW = DATA_ROOT / "geo_raw"
FD = DATA_ROOT / "Full data/Test folder"
G = pd.read_csv(R / "qc/geo_samples_long.csv", low_memory=False)
JOBS = {"GSE42834": (["GSE42825", "GSE42826", "GSE42827", "GSE42830", "GSE42831", "GSE42832"], FD / "GSE-42834/GSE42834_expression.csv"),
        "GSE83456": (["GSE83456"], FD / "GSE-83456/GSE83456_expression.csv")}
def read_raw(sub):
    f = RAW / (f"{sub}_non-normalized.txt.gz" if sub != "GSE83456" else "GSE83456_matrix_EPTB_unnormalized.txt.gz")
    d = pd.read_csv(f, sep="\t", index_col=0, low_memory=False)
    d = d.loc[:, [i % 2 == 0 for i in range(d.shape[1])]]            # value columns (pval columns follow each)
    d = d.loc[:, ~d.columns.str.contains("pval|p_value", case=False)]
    d = d.dropna(axis=1, how="all").apply(pd.to_numeric, errors="coerce")
    d.columns = [f"{sub}::{c}" for c in d.columns]
    return d
def qnorm_log2(x):
    """log2 (floor 1) then quantile-normalise across columns, so array-wide intensity differences
    do not drive the match; the GEO normalised values are themselves quantile-type normalised."""
    a = np.log2(np.clip(x, 1, None)); rk = a.rank(axis=0).to_numpy()
    ms = np.sort(a.to_numpy(), axis=0).mean(1); grid = np.arange(1, len(ms) + 1)
    return pd.DataFrame(np.column_stack([np.interp(rk[:, k], grid, ms) for k in range(a.shape[1])]),
                        index=a.index, columns=a.columns)
maps, dups = [], []
for cohort, (subs, normf) in JOBS.items():
    N = pd.read_csv(normf, index_col=0); N = N[[c for c in N.columns if c.startswith("GSM")]].apply(pd.to_numeric, errors="coerce")
    for sub in subs:
        raw = read_raw(sub)
        gsms = G[(G.cohort == cohort) & G.series_id.str.contains(sub)].GSM.tolist()
        Nn = N[gsms]
        common = raw.index.intersection(Nn.index)
        ok = raw.loc[common].notna().all(1) & Nn.loc[common].notna().all(1); common = common[ok.to_numpy()]
        A = qnorm_log2(raw.loc[common]); B = Nn.loc[common]
        expr = A.mean(1) > 7                                               # expressed probes only (log2 > 7)
        A = A[expr].sub(A[expr].mean(1), axis=0); B = B[expr].sub(B[expr].mean(1), axis=0)
        a = A.to_numpy(); b = B.to_numpy()
        a = (a - a.mean(0)) / a.std(0); b = (b - b.mean(0)) / b.std(0)
        C = a.T @ b / a.shape[0]                                           # raw x gsm
        for i, rc in enumerate(raw.columns):
            o = np.argsort(C[i])[::-1]; j = o[0]
            mutual = int(np.argmax(C[:, j])) == i
            maps.append(dict(cohort=cohort, subseries=sub, raw_column=rc, GSM=gsms[j], r_best=C[i, j],
                             r_second=C[i, o[1]] if len(o) > 1 else np.nan, mutual_best=mutual,
                             n_raw_cols=raw.shape[1], n_gsm_in_subseries=len(gsms)))
        print(cohort, sub, "raw cols", raw.shape[1], "GSMs", len(gsms), "probes used", a.shape[0], flush=True)
    # duplicate arrays across all raw files of this cohort (identical raw values)
    allraw = pd.concat([read_raw(s) for s in subs], axis=1, join="inner")
    X = allraw.to_numpy(); X = np.where(np.isfinite(X), X, 0)
    Xs = (X - X.mean(0)) / X.std(0); CC = Xs.T @ Xs / X.shape[0]; np.fill_diagonal(CC, 0)
    for i, j in zip(*np.where(np.triu(CC) > 0.999)):
        dups.append(dict(cohort=cohort, raw_a=allraw.columns[i], raw_b=allraw.columns[j], r=CC[i, j],
                         identical=bool(np.allclose(X[:, i], X[:, j]))))
    print(cohort, "raw columns", allraw.shape[1], "duplicate pairs r>0.999:", sum(d["cohort"] == cohort for d in dups), flush=True)
M = pd.DataFrame(maps); M["accepted"] = M.mutual_best & (M.r_best >= 0.90) & ((M.r_best - M.r_second) >= 0.10)
M.to_csv(R / "qc/raw_column_to_GSM.csv", index=False)
pd.DataFrame(dups, columns=["cohort","raw_a","raw_b","r","identical"]).to_csv(R / "qc/raw_duplicate_arrays.csv", index=False)
print(M.groupby(["cohort", "subseries"]).agg(n=("GSM", "size"), accepted=("accepted", "sum"), uniqueGSM=("GSM", "nunique"),
                                             rmin=("r_best", "min"), r2max=("r_second", "max")).round(3).to_string())
