"""Stage3 step 0: apply the decisions and build task datasets.
 - GSE19491: drop the 59 OD (GSE22098) and the 7 GSE19435 ATB; renormalise the remaining samples together from the
   GEO raw files (same method as Stage2: log2(max(x,1)) + quantile normalisation; probe = highest mean raw log2)
 - no per-sample age rule (paediatric exclusion stays at sub-series level)
Outputs: data/sample_manifest_stage3.csv, data/GSE19491_log2_genelevel_stage3.csv.gz, qc/task_cohorts.csv"""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common import *
sys.path.insert(0, str(CR / "scripts")); from hgnc_alias import HGNCMapper
M = pd.read_csv(CR / "sample_manifest.csv"); M["included_stage3"] = M.included; M["stage3_exclusion"] = ""
od = (M.cohort == "GSE19491") & M.included & (M.class_label == "OD")
s35 = (M.cohort == "GSE19491") & M.included & (M.subseries == "GSE19435")
M.loc[od, ["included_stage3", "stage3_exclusion"]] = [False, "GSE19491 OD dropped (batch = OD in GSE22098; decision 19 Oct)"]
M.loc[s35, ["included_stage3", "stage3_exclusion"]] = [False, "GSE19435 ATB dropped (no LTB in sub-series; decision 19 Oct)"]
print("dropped OD", int(od.sum()), "| dropped GSE19435 ATB", int(s35.sum()))
M.to_csv(ST / "data/sample_manifest_stage3.csv", index=False)
keep = M[(M.cohort == "GSE19491") & M.included_stage3].GSM.tolist()
files = {os.path.basename(f).split("_")[0]: f for f in glob.glob(str(DATA_ROOT / "geo_raw/GSE19491_RAW/GSM*.txt.gz"))}
def rd(g):
    d = pd.read_csv(files[g], sep="\t"); d = d[d.iloc[:, 0].notna()]
    v = pd.to_numeric(d.iloc[:, 1], errors="coerce"); v.index = d.iloc[:, 0].astype(str).values
    return v[~v.index.duplicated(keep=False)].rename(g)
R = pd.concat([rd(g) for g in keep], axis=1, join="inner"); R = R[R.notna().all(1)]; R = R.drop(index="ILMN_2038777", errors="ignore")
L = np.log2(np.clip(R, 1, None)); rk = L.rank(axis=0).to_numpy(); ms = np.sort(L.to_numpy(), axis=0).mean(1); grid = np.arange(1, len(ms) + 1)
Q = pd.DataFrame(np.column_stack([np.interp(rk[:, k], grid, ms) for k in range(L.shape[1])]), index=L.index, columns=L.columns)
HG = HGNCMapper()
gpl = pd.read_csv(DATA_ROOT / "Full data/Train folder/GSE-19491/GPL6947-13512.txt", sep="\t", comment="#", dtype=str).set_index("ID").Symbol
sym = gpl.reindex(L.index).fillna("").str.strip(); ok = (sym != "") & ~sym.str.contains(r"[;,/]")
hg = sym[ok].map(lambda x: HG.map(x)["current"])
df = pd.DataFrame({"probe": hg.index, "symbol_hgnc": hg.values, "m": L.loc[hg.index].mean(1).values}).sort_values(["symbol_hgnc", "m", "probe"], ascending=[True, False, True])
ch = df.drop_duplicates("symbol_hgnc")
out = Q.loc[ch.probe]; out.index = ch.symbol_hgnc.values; out.index.name = "Gene"
assert out.index.is_unique and set(out.columns) == set(keep)
out.astype("float32").to_csv(ST / "data/GSE19491_log2_genelevel_stage3.csv.gz", float_format="%.5f")
print("GSE19491 stage3:", out.shape)
rows = []
for task, t in TASKS.items():
    for role in ["train", "test"]:
        for c in t[role]:
            m = M[(M.cohort == c) & M.included_stage3]
            n_pos, n_neg = int((m.class_label == t["pos"]).sum()), int((m.class_label == t["neg"]).sum())
            rows.append(dict(task=task, role=role, cohort=c, **{t["pos"]: n_pos, t["neg"]: n_neg}, both_classes=(n_pos > 0 and n_neg > 0)))
T = pd.DataFrame(rows); T.to_csv(ST / "qc/task_cohorts.csv", index=False); print(T.to_string())
