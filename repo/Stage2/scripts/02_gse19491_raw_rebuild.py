"""Stage2 item 2: GSE19491 rebuilt from the GEO per-sample raw files (GSE19491_RAW.tar), all sub-series normalised together.
Raw files: one per GSM (file names start with the GSM accession), background-subtracted, unnormalised intensities
(GEO's processed series-matrix values are these x a per-sample scale factor, r = 1.000).
Method (same as the Clean_rebuild GSE42834 rebuild): log2(max(x,1)) -> quantile normalisation across samples.
  primary   : the 213 Clean_rebuild included samples
  diagnostic: the 213 + the 39 adult GSE22098 healthy controls (ASLE/Still controls), used ONLY to measure residual
              sub-series batch with disease held constant (GSE22098 controls vs London controls)
Probe per gene: highest mean raw log2 (before quantile normalisation) over the 213 included samples.
Outputs: expression/GSE19491_log2_genelevel_jointnorm.csv.gz, qc/GSE19491_probe_selection_jointnorm.csv,
         qc/GSE19491_sample_means.csv
"""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd
S = Path(__file__).resolve().parents[1]; CR = S.parent / "Clean_rebuild"
sys.path.insert(0, str(CR / "scripts")); from hgnc_alias import HGNCMapper
HG = HGNCMapper()
M = pd.read_csv(CR / "sample_manifest.csv"); m = M[M.cohort == "GSE19491"].set_index("GSM")
inc = m[m.included].index.tolist()
hc = m[m.geo_basis.str.contains(r"healthy control=(ASLE|Still)", regex=True)].index.tolist()
files = {os.path.basename(f).split("_")[0]: f for f in glob.glob(str(DATA_ROOT / "geo_raw/GSE19491_RAW/GSM*.txt.gz"))}
assert len(files) == 498
DUP = set()
def rd(g):
    d = pd.read_csv(files[g], sep="\t")
    d = d[d.iloc[:, 0].notna()]                       # GSM484599: 16,732 blank padding rows (65,535-row export) dropped
    v = pd.to_numeric(d.iloc[:, 1], errors="coerce"); v.index = d.iloc[:, 0].astype(str).values
    dup = v.index[v.index.duplicated(keep=False)]; DUP.update(dup)
    return v[~v.index.isin(dup)].rename(g)                # ILMN_2038777 appears twice with different values in 13 GSE22098 files: dropped
need = inc + hc
R = pd.concat([rd(g) for g in need], axis=1, join="inner")
R = R[R.notna().all(1)]; R = R[~R.index.isin(DUP)]
print("raw matrix", R.shape, "| probes dropped as duplicated within a file:", sorted(DUP), flush=True)
def qn(a):
    rk = a.rank(axis=0).to_numpy(); ms = np.sort(a.to_numpy(), axis=0).mean(1); grid = np.arange(1, len(ms) + 1)
    return pd.DataFrame(np.column_stack([np.interp(rk[:, k], grid, ms) for k in range(a.shape[1])]), index=a.index, columns=a.columns)
L = np.log2(np.clip(R, 1, None))
Q = qn(L[inc]); Qd = qn(L[need])
gpl = pd.read_csv(DATA_ROOT / "Full data/Train folder/GSE-19491/GPL6947-13512.txt", sep="\t", comment="#", dtype=str).set_index("ID").Symbol
sym = gpl.reindex(L.index).fillna("").str.strip(); ok = (sym != "") & ~sym.str.contains(r"[;,/]")
hg = sym[ok].map(lambda x: HG.map(x)["current"])
df = pd.DataFrame({"probe": hg.index, "symbol_hgnc": hg.values, "rank_mean_raw_log2": L.loc[hg.index, inc].mean(1).values})
df = df.sort_values(["symbol_hgnc", "rank_mean_raw_log2", "probe"], ascending=[True, False, True])
ch = df.drop_duplicates("symbol_hgnc"); df["selected"] = df.probe.isin(ch.probe); df.to_csv(S / "qc/GSE19491_probe_selection_jointnorm.csv", index=False)
out = Q.loc[ch.probe]; out.index = ch.symbol_hgnc.values; out.index.name = "Gene"
assert out.index.is_unique and set(out.columns) == set(inc)
out.astype("float32").to_csv(S / "expression/GSE19491_log2_genelevel_jointnorm.csv.gz", float_format="%.5f")
old = pd.read_csv(CR / "qc/probe_selection_GSE19491.csv"); oc = old[old.selected].set_index("symbol_hgnc").probe
print("genes", out.shape[0], "| probe choice differs from Clean_rebuild for", int((ch.set_index("symbol_hgnc").probe.reindex(oc.index) != oc).sum()), "of", len(oc), flush=True)
# per-sample all-gene means: before (Clean_rebuild gene-level file) and after (this file); diagnostic set too
before = pd.read_csv(CR / "expression/GSE19491_log2_genelevel.csv.gz", index_col=0)
dg = Qd.loc[ch.probe]; dg.index = ch.symbol_hgnc.values
# "before" for the 39 HCs: GEO series-matrix values, same transform and probes as Clean_rebuild
sm = pd.read_csv(DATA_ROOT / "Full data/Train folder/GSE-19491/GSE19491_series_matrix.txt", sep="\t", comment="!", index_col=0)
bsel = old[old.selected]; bh = np.log2(np.clip(sm.loc[bsel.probe, hc], 1, None)); bh.index = bsel.symbol_hgnc.values
T = pd.DataFrame({"GSM": inc + hc,
                  "role": ["included"] * len(inc) + ["GSE22098 healthy control (diagnostic only)"] * len(hc)})
T["subseries"] = T.GSM.map(m.subseries); T["class_label"] = T.GSM.map(m.class_label).where(T.role == "included", "CON(GSE22098)")
T["mean_before"] = list(before[inc].mean().values) + list(bh[hc].mean().values)
T["mean_after_primary"] = list(out[inc].mean().values) + [np.nan] * len(hc)
T["mean_after_diagnostic"] = dg[inc + hc].mean().values
T.to_csv(S / "qc/GSE19491_sample_means.csv", index=False)
print(T.groupby(["subseries", "class_label"])[["mean_before", "mean_after_primary", "mean_after_diagnostic"]].agg(["median", "std"]).round(4).to_string())
