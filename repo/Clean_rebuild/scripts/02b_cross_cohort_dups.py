"""Step 2b: identical arrays across GSE42834 sub-series and GSE83456 (both Illumina HT-12 v4, London).
Raw (non-normalised) values compared on shared probes; r>0.999 or numerically identical is a duplicate."""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; RAW = DATA_ROOT / "geo_raw"
def read_raw(f, tag):
    d = pd.read_csv(f, sep="\t", index_col=0, low_memory=False)
    d = d.loc[:, [i % 2 == 0 for i in range(d.shape[1])]]
    d = d.loc[:, ~d.columns.str.contains("pval|p_value", case=False)].dropna(axis=1, how="all").apply(pd.to_numeric, errors="coerce")
    d.columns = [f"{tag}::{c}" for c in d.columns]; return d
A = pd.concat([read_raw(RAW / f"{s}_non-normalized.txt.gz", s) for s in ["GSE42825","GSE42826","GSE42827","GSE42830","GSE42831","GSE42832"]], axis=1, join="inner")
B = read_raw(RAW / "GSE83456_matrix_EPTB_unnormalized.txt.gz", "GSE83456")
c = A.index.intersection(B.index); a = A.loc[c].to_numpy(); b = B.loc[c].to_numpy()
ok = np.isfinite(a).all(1) & np.isfinite(b).all(1); a, b = a[ok], b[ok]
az = (a - a.mean(0)) / a.std(0); bz = (b - b.mean(0)) / b.std(0); C = az.T @ bz / a.shape[0]
print("shared probes", ok.sum(), "max cross r", C.max().round(5), "pairs r>0.999:", int((C > 0.999).sum()))
i, j = np.unravel_index(np.argsort(C, axis=None)[::-1][:5], C.shape)
for x, y in zip(i, j): print("  top:", A.columns[x], B.columns[y], round(float(C[x, y]), 5))
pd.DataFrame([(A.columns[x], B.columns[y], C[x, y]) for x, y in zip(*np.where(C > 0.999))], columns=["GSE42834_raw", "GSE83456_raw", "r"]).to_csv(R / "qc/cross_cohort_duplicates_42834_83456.csv", index=False)
