"""Step 2c: duplicate arrays inside the GSE19491 SuperSeries, from the GEO series matrix (local GEO original).
Pairs of different GSMs with sample-sample Pearson r > 0.995 on all probes are listed; 'identical' = equal values."""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
f = DATA_ROOT / "Full data/Train folder/GSE-19491/GSE19491_series_matrix.txt"
X = pd.read_csv(f, sep="\t", comment="!", index_col=0).apply(pd.to_numeric, errors="coerce").dropna()
A = X.to_numpy(np.float64); Z = (A - A.mean(0)) / A.std(0); C = Z.T @ Z / A.shape[0]; np.fill_diagonal(C, 0)
rows = [dict(GSM_a=X.columns[i], GSM_b=X.columns[j], r=C[i, j], identical=bool(np.allclose(A[:, i], A[:, j])),
             max_abs_diff=float(np.abs(A[:, i] - A[:, j]).max()), ratio_median=float(np.median(A[:, i] / A[:, j])))
        for i, j in zip(*np.where(np.triu(C) > 0.995))]
pd.DataFrame(rows).to_csv(R / "qc/gse19491_duplicate_pairs.csv", index=False)
print(X.shape); print(pd.DataFrame(rows).round(5).to_string())
