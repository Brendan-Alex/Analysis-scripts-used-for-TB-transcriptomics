"""Stage2 item 1: full (every value, every sample) verification of Clean_rebuild expression files against GEO.
For series-matrix cohorts: GEO series matrix (downloaded, checksummed) -> same transform -> the probe Clean_rebuild
selected for each gene -> compare every gene x sample. Membership and labels are re-derived from the series matrix's
own sample characteristics (an independent copy of the GEO metadata from the SOFT file used by Clean_rebuild).
Also re-derives the probe choice (highest mean log2 over included samples) from GEO values independently.
Usage: python 01_full_verify.py GSE73408 [...]
"""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import gzip, io, sys, re
from pathlib import Path
import numpy as np, pandas as pd
S = Path(__file__).resolve().parents[1]; CR = S.parent / "Clean_rebuild"; RAW = DATA_ROOT / "geo_raw"
M = pd.read_csv(CR / "sample_manifest.csv")
RULES = {   # class from series-matrix characteristics (independent re-implementation)
 "GSE37250": lambda ch: (None if "HIV positive" in ch.get("hiv status", "") else
                         {"active tuberculosis": "ATB", "latent TB infection": "LTB", "other disease": "OD"}.get(ch.get("disease state"))),
 "GSE28623": lambda ch: {"tuberculosis patients": "ATB", "healthy donors latently infected with M. tuberculosis": "LTB",
                         "healthy non-infected donors": "CON"}.get(ch.get("group")),
 "GSE73408": lambda ch: {"TB": "ATB", "LTBI": "LTB", "PNA": "OD"}.get(ch.get("clinical group")),
 "GSE83456": lambda ch: {"PTB": "ATB", "HC": "CON", "Sarcoid": "OD"}.get(ch.get("disease state")),   # EPTB excluded
}
SCALE = {"GSE37250": "linear", "GSE28623": "linear", "GSE73408": "log", "GSE83456": "log"}
def read_series_matrix(f):
    meta, rows = {}, []
    with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    head, _, rest = txt.partition("!series_matrix_table_begin")
    for line in head.split("\n"):
        if line.startswith("!Sample_"):
            k, *v = line.rstrip("\r").split("\t"); meta.setdefault(k[8:], []).append([x.strip('"') for x in v])
    table = rest.partition("!series_matrix_table_end")[0].strip("\n")
    X = pd.read_csv(io.StringIO(table), sep="\t", index_col=0)
    X.index = X.index.astype(str).str.strip('"'); X.columns = X.columns.str.strip('"')
    return meta, X
report = []
for g in sys.argv[1:]:
    meta, X = read_series_matrix(RAW / f"{g}_series_matrix.txt.gz")
    gsm = meta["geo_accession"][0]; n = len(gsm)
    chars = [dict() for _ in range(n)]
    for row in meta.get("characteristics_ch1", []):
        for i, v in enumerate(row):
            if ":" in v: k, val = v.split(":", 1); chars[i][k.strip().lower()] = val.strip()
    cls = pd.Series([RULES[g](c) for c in chars], index=gsm)
    want = cls.dropna()
    man = M[(M.cohort == g) & M.included].set_index("GSM").class_label
    memb_ok = set(want.index) == set(man.index); lab_mis = int((want.reindex(man.index) != man).sum())
    E = pd.read_csv(CR / f"expression/{g}_log2_genelevel.csv.gz", index_col=0)
    sel = pd.read_csv(CR / f"qc/probe_selection_{g}.csv", dtype={"probe": str}); ch = sel[sel.selected].set_index("symbol_hgnc").probe
    assert list(E.columns) == list(man.index) or set(E.columns) == set(man.index)
    V = X.loc[ch.reindex(E.index).values, E.columns].apply(pd.to_numeric, errors="coerce")
    V = np.log2(np.clip(V, 1, None)) if SCALE[g] == "linear" else V
    d = np.abs(V.to_numpy(float) - E.to_numpy(float))
    # independent probe choice
    cand = sel.copy(); Xi = X.loc[cand.probe.astype(str), E.columns].apply(pd.to_numeric, errors="coerce")
    Li = np.log2(np.clip(Xi, 1, None)) if SCALE[g] == "linear" else Xi
    cand["m"] = Li.mean(1).values
    pick = cand.sort_values(["symbol_hgnc", "m", "probe"], ascending=[True, False, True]).drop_duplicates("symbol_hgnc").set_index("symbol_hgnc").probe
    probe_mis = "n/a (ranked on GEO non-normalised data)" if g == "GSE83456" else int((pick.reindex(ch.index) != ch).sum())
    r = dict(cohort=g, GEO_samples_in_matrix=n, GEO_probes=X.shape[0], included_by_series_matrix_rules=len(want), manifest_included=len(man),
             membership_identical=memb_ok, label_mismatches=lab_mis, genes=E.shape[0], values_compared=int(d.size),
             values_differing_gt_1e4=int((d > 1e-4).sum()), max_abs_diff=float(np.nanmax(d)), NaN_in_GEO_selected=int(np.isnan(V.to_numpy()).sum()),
             probe_choice_mismatches=probe_mis)
    report.append(r); print(r, flush=True)
f = S / "qc/full_verification.csv"; old = pd.read_csv(f) if f.exists() else pd.DataFrame()
new = pd.DataFrame(report); old = old[~old.cohort.isin(new.cohort)] if len(old) else old
pd.concat([old, new]).to_csv(f, index=False)
