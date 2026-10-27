"""Step 5: how many genes change probe relative to the current (old) selection.
Old rule (repo scripts): within the per-cohort z-scored file, sort rows by mean z across all samples and keep the
first row per gene. Because z-scored rows have mean ~1e-15, that choice is effectively arbitrary.
Both rules are evaluated on the same HGNC-mapped symbols so only the probe-choice rule differs.
Output: qc/probe_selection_vs_old.csv
"""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; TT = DATA_ROOT / "Test+Train Datasets"
sys.path.insert(0, str(Path(__file__).resolve().parent)); from hgnc_alias import HGNCMapper
HG = HGNCMapper()
Z = {"GSE19439": "Train", "GSE19444": "Train", "GSE19491": "Train", "GSE37250": "Train", "GSE28623": "Train", "GSE73408": "Train",
     "GSE42834": "Test", "GSE83456": "Test"}
rows = []
for g in sys.argv[1:]:
    s = Z[g]; zf = TT / f"{s} Datasets/{s} Datasets Normalized/{g}_zscored_final.csv"
    z = pd.read_csv(zf, low_memory=False)
    num = z.select_dtypes("number"); num = num[[c for c in num.columns if c != "ILMN_ID"]]
    if "ILMN_ID" in z.columns:
        probe = z["ILMN_ID"].astype(str)
    else:   # GSE73408: z-file rows are in the GEO canonical-table order (verified by gene names below)
        can = pd.read_csv(DATA_ROOT / "Full data/Train folder/GSE-73408/GSE73408_canonical.csv", usecols=[0, 1])
        assert len(can) == len(z) and (can.iloc[:, 1].astype(str).values == z["Gene"].astype(str).values).mean() > 0.99
        probe = can.iloc[:, 0].astype(str)
    sel = pd.read_csv(R / f"qc/probe_selection_{g}.csv", dtype={"probe": str})
    p2h = sel.set_index("probe").symbol_hgnc
    old = pd.DataFrame({"probe": probe.values, "mean_z": num.mean(1).values})
    old = old[old.probe.isin(p2h.index)]; old["symbol_hgnc"] = old.probe.map(p2h)
    old = old.sort_values("mean_z", ascending=False, kind="stable").drop_duplicates("symbol_hgnc")
    new = sel[sel.selected][["symbol_hgnc", "probe"]]
    m = new.merge(old[["symbol_hgnc", "probe"]], on="symbol_hgnc", suffixes=("_new", "_old"))
    mult = sel.groupby("symbol_hgnc").size(); mg = set(mult[mult > 1].index)
    mm = m[m.symbol_hgnc.isin(mg)]
    rows.append(dict(cohort=g, genes_compared=len(m), multi_probe_genes=len(mm), genes_probe_changed=int((mm.probe_new != mm.probe_old).sum()),
                     pct_of_multi_probe_genes_changed=round(100 * (mm.probe_new != mm.probe_old).mean(), 1),
                     max_abs_mean_z_in_old_file=float(num.mean(1).abs().max())))
    print(rows[-1], flush=True)
p = R / "qc/probe_selection_vs_old.csv"
old_ = pd.read_csv(p) if p.exists() else pd.DataFrame(); new_ = pd.DataFrame(rows)
if len(old_): old_ = old_[~old_.cohort.isin(new_.cohort)]
pd.concat([old_, new_]).to_csv(p, index=False)
