"""Stage2 item 1, GSE107994: GEO edgeR xlsx vs (a) the local CSV Clean_rebuild read, (b) the rebuilt gene-level file.
GSM <-> column title link taken from the GEO series matrix (independent of the SOFT file); labels re-derived from it."""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import gzip, re
from pathlib import Path
import numpy as np, pandas as pd
S = Path(__file__).resolve().parents[1]; CR = S.parent / "Clean_rebuild"
FD = DATA_ROOT / "Full data/Test folder/GSE-107994"
X = pd.read_excel(DATA_ROOT / "geo_raw/GSE107994_edgeR_normalized_Leicester_with_progressor_longitudinal.xlsx")
print("xlsx", X.shape, list(X.columns[:4]))
meta = {}
for line in gzip.open(DATA_ROOT / "geo_raw/GSE107994_series_matrix.txt.gz", "rt"):
    if line.startswith("!Sample_"):
        k, *v = line.rstrip("\n").split("\t"); meta.setdefault(k[8:], []).append([x.strip('"') for x in v])
gsm, title = meta["geo_accession"][0], meta["title"][0]
ch = {}
for row in meta["characteristics_ch1"]:
    for i, v in enumerate(row):
        if ":" in v: k, val = v.split(":", 1); ch.setdefault(i, {})[k.strip()] = val.strip()
t2g = dict(zip(title, gsm)); assert len(t2g) == len(gsm) == 175
idcol, symcol = X.columns[0], X.columns[1]
datacols = [c for c in X.columns if c in t2g]; print("xlsx sample columns matched to GEO titles:", len(datacols))
L = pd.read_csv(FD / "GSE107994_edgeR_normalized_final.csv")
print("local csv", L.shape)
# (a) local csv vs xlsx, full
ok_ids = (L.Gene_ID.astype(str).values == X[idcol].astype(str).values).all()
sym_x = X[symcol].astype(str); sym_l = L.Gene.astype(str)
diffsym = (sym_x.values != sym_l.values)
print("row Ensembl IDs identical & same order:", ok_ids, "| symbol differences:", int(diffsym.sum()))
print("  examples (xlsx -> local csv):", list(zip(sym_x[diffsym][:8], sym_l[diffsym][:8])))
xl_types = X[symcol][diffsym].map(type).value_counts().to_dict(); print("  python types of those xlsx cells:", xl_types)
same_cols = [c for c in datacols if c in L.columns]
d = np.abs(X[same_cols].to_numpy(float) - L[same_cols].to_numpy(float))
print("local csv vs xlsx: sample columns", len(same_cols), "values", d.size, "max abs diff", float(d.max()), "n>1e-9", int((d > 1e-9).sum()))
# (b) rebuilt gene-level file vs xlsx
M = pd.read_csv(CR / "sample_manifest.csv"); man = M[(M.cohort == "GSE107994") & M.included].set_index("GSM").class_label
E = pd.read_csv(CR / "expression/GSE107994_log2_genelevel.csv.gz", index_col=0)
sel = pd.read_csv(CR / "qc/probe_selection_GSE107994.csv"); ch_ = sel[sel.selected].set_index("symbol_hgnc").probe
Xi = X.set_index(idcol)[datacols].rename(columns=t2g)
V = Xi.loc[ch_.reindex(E.index).values, E.columns]
dd = np.abs(V.to_numpy(float) - E.to_numpy(float))
print("rebuilt vs GEO xlsx: genes", E.shape[0], "samples", E.shape[1], "values", dd.size, "max abs diff", float(dd.max()), "n>1e-4", int((dd > 1e-4).sum()))
# labels and membership from series matrix
grp = pd.Series({gsm[i]: ch[i].get("group") for i in range(175)}); outl = pd.Series({gsm[i]: ch[i].get("outlier") for i in range(175)})
site = pd.Series({gsm[i]: ch[i].get("tb_disease_type") for i in range(175)})
intended = set(pd.read_csv(FD / "GSE107994_baseline_GSM_patient_class.csv").GSM)
lab = grp.map({"Active_TB": "ATB", "LTBI": "LTB", "Control": "CON"})
keep = lab.notna() & lab.index.isin(intended) & (outl == "No") & ~((lab == "ATB") & (site == "Non-Pulmonary"))
print("membership by series-matrix rules == manifest:", set(lab[keep].index) == set(man.index), "| label mismatches:", int((lab[keep].reindex(man.index) != man).sum()))
il = pd.read_csv(FD / "GSE107994_baseline_GSM_patient_class.csv").set_index("GSM").Class
print("intended-list class vs GEO group (all 152):", int((il != lab.reindex(il.index)).sum()), "mismatches")
