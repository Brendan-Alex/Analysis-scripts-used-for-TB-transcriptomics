"""Step 4: per-cohort gene-level log2 expression for the samples the manifest includes.
  * columns are GSM IDs; every column is selected by GSM from sample_manifest.csv (included == True)
  * the matrix column -> GSM link is the GSM header itself (GEO series-matrix-derived tables), or, where
    GEO only supplies titles (GSE107994 edgeR table, GSE101705 counts), the GEO title -> GSM map from the
    downloaded GEO metadata, asserted 1:1
  * probe -> symbol from the GEO platform (GPL) table where it is on disk; probes with no symbol or with
    several symbols are dropped; symbols harmonised with HGNCMapper (same resources as Rerun_corrected)
  * one probe per gene: highest mean pre-normalisation log2 expression over the cohort's included samples
    (for GSE42834/GSE83456, whose GEO matrices are already per-gene centred, the ranking uses the GEO
    non-normalised supplementary values linked to GSMs in qc/raw_column_to_GSM.csv)
  * log2 rule: Illumina/Agilent linear intensities -> log2(max(x, 1)); RNA-seq counts -> log2(x + 1);
    already-log data unchanged
  * GSE42834 (user decision): rebuilt from the GEO non-normalised supplementary files (columns linked to GSMs by
    expression identity, qc/raw_column_to_GSM.csv): log2(max(x,1)) then quantile normalisation across the included
    samples; probe ranking on the log2 values before quantile normalisation
  * GSE73408 (user decision): multi-gene transcript clusters assigned to the first gene of GPL11532 gene_assignment
Outputs: expression/<GSE>_log2_genelevel.csv.gz, qc/probe_selection_<GSE>.csv, qc/expression_build_summary.csv
Usage: python 04_build_expression.py GSE19439 [GSE...]
"""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; MT = R.parent; FD = DATA_ROOT / "Full data"
sys.path.insert(0, str(Path(__file__).resolve().parent)); from hgnc_alias import HGNCMapper
HG = HGNCMapper()
MAN = pd.read_csv(R / "sample_manifest.csv"); GEO = pd.read_csv(R / "qc/geo_samples_long.csv", low_memory=False)
GPL6947 = FD / "Train folder/GSE-19491/GPL6947-13512.txt"; GPL10558 = FD / "Train folder/GSE-37250/GPL10558-50081 (Master).txt"
CFG = {
 "GSE19491": dict(src=FD / "Train folder/GSE-19491/GSE19491_series_matrix.txt", kind="series_matrix", gpl=GPL6947, scale="linear"),
 "GSE19439": dict(src=FD / "Train folder/GSE-19439/GSE19439_expression.csv", kind="gsm_csv", gpl=GPL6947, scale="linear"),
 "GSE19444": dict(src=FD / "Train folder/GSE-19444/GSE19444_expression.csv", kind="gsm_csv", gpl=GPL6947, scale="linear"),
 "GSE37250": dict(src=FD / "Train folder/GSE-37250/GSE37250_expression.csv", kind="gsm_csv", gpl=GPL10558, scale="linear"),
 "GSE42834": dict(src=None, kind="raw_illumina", gpl=GPL10558, scale="linear_quantile"),
 "GSE83456": dict(src=FD / "Test folder/GSE-83456/GSE83456_expression.csv", kind="gsm_csv", gpl=GPL10558, scale="log_centred", rank="raw"),
 "GSE28623": dict(src=FD / "Test folder/GSE 28623/GSE28623_expression.csv", kind="gsm_csv", gpl=FD / "Test folder/GSE 28623/GPL4133_probe_to_gene.csv", scale="linear"),
 "GSE73408": dict(src=FD / "Train folder/GSE-73408/GSE73408_canonical.csv", kind="gsm_csv", gpl=FD / "Train folder/GSE-73408/GPL11532-32230.txt", scale="log"),
 "GSE107994": dict(src=FD / "Test folder/GSE-107994/GSE107994_edgeR_normalized_final.csv", kind="title_csv", scale="log", title_prefix=""),
 "GSE101705": dict(src=DATA_ROOT / "geo_raw/GSE101705_normalizedCounts.xlsx", kind="title_xlsx", scale="counts", drop_symbols=["MARCH1"]),
}
def gpl_map(path):
    if path.suffix == ".csv":
        m = pd.read_csv(path, dtype=str); return m.set_index(m.columns[0]).iloc[:, 0]
    t = pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)
    if "gene_assignment" in t.columns:   # Affymetrix gene ST: primary (first-listed) gene of gene_assignment (user decision)
        first = t.gene_assignment.fillna("---").str.split(" /// ").str[0].str.split(" // ").str[1].fillna("").str.strip()
        return pd.Series(first.values, index=t.ID.values)
    return t.set_index("ID")["Symbol"]
def quantile_normalise(a):
    rk = a.rank(axis=0, method="average").to_numpy(); ms = np.sort(a.to_numpy(), axis=0).mean(1); grid = np.arange(1, len(ms) + 1)
    return pd.DataFrame(np.column_stack([np.interp(rk[:, k], grid, ms) for k in range(a.shape[1])]), index=a.index, columns=a.columns)
def read_raw(sb):
    f = DATA_ROOT / "geo_raw" / (f"{sb}_non-normalized.txt.gz" if sb != "GSE83456" else "GSE83456_matrix_EPTB_unnormalized.txt.gz")
    d = pd.read_csv(f, sep="\t", index_col=0, low_memory=False)
    d = d.loc[:, [i % 2 == 0 for i in range(d.shape[1])]]
    d = d.loc[:, ~d.columns.str.contains("pval|p_value", case=False)].dropna(axis=1, how="all")
    d.columns = [f"{sb}::{x}" for x in d.columns]; d.index = d.index.astype(str); return d
def to_log2(v, scale, c=None):
    if scale == "linear": return np.log2(np.clip(v, 1, None))
    if scale == "linear_quantile": return quantile_normalise(np.log2(np.clip(v, 1, None)))   # one method across all included samples
    if scale == "counts": return np.log2(v + 1)
    return v
_h = pd.read_csv(R / "resources/hgnc_entries.tsv", sep="\t", dtype=str)
_h = _h[(_h.Status == "Approved") & _h["Ensembl gene ID"].notna()]
HGNC_ENS = _h.set_index("Ensembl gene ID")["Approved symbol"]; HGNC_ENS = HGNC_ENS[~HGNC_ENS.index.duplicated(keep=False)]
DATEFIX = []
summ = []
for g in sys.argv[1:]:
    c = CFG[g]; inc = MAN[(MAN.cohort == g) & MAN.included]; want = inc.GSM.tolist()
    if c["kind"] == "series_matrix":
        X = pd.read_csv(c["src"], sep="\t", comment="!", index_col=0)
    elif c["kind"] in ("gsm_csv", "gsm_csv_with_gene"):
        X = pd.read_csv(c["src"], index_col=0, low_memory=False)
    elif c["kind"] == "raw_illumina":   # GEO non-normalised supplementary, columns linked to GSMs in qc/raw_column_to_GSM.csv
        M2 = pd.read_csv(R / "qc/raw_column_to_GSM.csv"); M2 = M2[(M2.cohort == g) & M2.accepted]
        r2g = M2.set_index("raw_column").GSM; assert r2g.is_unique
        X = pd.concat([read_raw(sb) for sb in M2.subseries.unique()], axis=1, join="inner")
        X = X[[x for x in X.columns if x in r2g.index]].rename(columns=r2g.to_dict())
    elif c["kind"] == "title_csv":
        X = pd.read_csv(c["src"])
    else:
        X = pd.read_excel(c["src"]); X = X.rename(columns={X.columns[0]: "Gene"})
        X["Gene"] = X["Gene"].apply(lambda v: v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else v)
    X.index = X.index.astype(str)
    # ---- link columns to GSM
    if c["kind"].startswith("title"):
        t2g = GEO[GEO.cohort == g].set_index("title").GSM
        assert t2g.index.is_unique, "GEO titles not unique"
        datacols = [x for x in X.columns if x in t2g.index]
        assert len(datacols) == len(t2g), f"{g}: {len(datacols)} of {len(t2g)} GEO titles found as columns"
        gene = X["Gene"].astype(str)
        probe = (X["Gene_ID"].astype(str) if "Gene_ID" in X.columns else pd.Series([f"row{i}" for i in range(len(X))]))   # unique row keys
        assert probe.is_unique
        V = X[datacols].rename(columns=t2g.to_dict()); V.index = probe.values; sym_raw = pd.Series(gene.values, index=probe.values)
        # Excel date corruption of gene symbols (SEPT*/MARCH* -> dates)
        is_date = sym_raw.str.match(r"^\d{1,2}/\d{1,2}/\d{4}") | sym_raw.str.match(r"^\d{4}-\d{2}-\d{2}")
        if g == "GSE107994":   # recover from the Ensembl gene ID via the HGNC export
            ens = HGNC_ENS.reindex(sym_raw.index[is_date]); assert ens.notna().all(), "unrecoverable date-corrupted symbols"
            DATEFIX.extend(dict(cohort=g, row=k, corrupted=sym_raw[k], recovered=ens[k], method="Ensembl gene ID -> HGNC approved symbol") for k in ens.index)
            sym_raw[is_date] = ens.values
        else:                  # no stable ID to recover from: drop, and drop symbols that collide with a corrupted entry
            for k in sym_raw.index[is_date]: DATEFIX.append(dict(cohort=g, row=k, corrupted=sym_raw[k], recovered="", method="dropped: no gene ID in file"))
            amb = [x for x in c.get("drop_symbols", [])]
            for k in sym_raw.index[sym_raw.isin(amb)]: DATEFIX.append(dict(cohort=g, row=k, corrupted=sym_raw[k], recovered="", method="dropped: duplicated symbol, one copy is a date-mangled restoration (cannot tell which)"))
            sym_raw[is_date | sym_raw.isin(amb)] = ""
    else:
        gsmcols = [x for x in X.columns if str(x).startswith("GSM")]
        V = X[gsmcols]
        if c["kind"] == "gsm_csv_with_gene":
            sym_raw = X["Gene"].astype(str)
        else:
            mp = gpl_map(c["gpl"]); mp.index = mp.index.astype(str); sym_raw = mp.reindex(V.index)
    missing = sorted(set(want) - set(V.columns)); assert not missing, f"{g}: included GSMs missing from matrix: {missing}"
    V = V[want].apply(pd.to_numeric, errors="coerce")
    n_probe0 = len(V)
    # ---- probe -> symbol -> HGNC
    s = sym_raw.reindex(V.index).fillna("").astype(str).str.strip()
    multi = s.str.contains(r"[;,/]| /// ", regex=True); empty = s.isin(["", "nan", "NA", "---"])
    keep = ~multi & ~empty & V.notna().all(1)
    V, s = V[keep], s[keep]
    hg = s.map(lambda x: HG.map(x)["current"])
    # ---- ranking values (pre-normalisation, log2)
    L = to_log2(V, c["scale"])
    if c.get("rank") == "raw":
        M2 = pd.read_csv(R / "qc/raw_column_to_GSM.csv"); M2 = M2[(M2.cohort == g) & M2.accepted]
        subs = M2.subseries.unique(); raws = []
        for sb in subs:
            f = DATA_ROOT / "geo_raw" / (f"{sb}_non-normalized.txt.gz" if sb != "GSE83456" else "GSE83456_matrix_EPTB_unnormalized.txt.gz")
            d = pd.read_csv(f, sep="\t", index_col=0, low_memory=False)
            d = d.loc[:, [i % 2 == 0 for i in range(d.shape[1])]]
            d = d.loc[:, ~d.columns.str.contains("pval|p_value", case=False)].dropna(axis=1, how="all")
            d.columns = [f"{sb}::{x}" for x in d.columns]; raws.append(d)
        raw = pd.concat(raws, axis=1); raw.index = raw.index.astype(str)
        r2g = M2.set_index("raw_column").GSM; raw = raw[[x for x in raw.columns if x in r2g.index]].rename(columns=r2g.to_dict())
        use = [x for x in want if x in raw.columns]
        rank_src = f"GEO non-normalised supplementary values for {len(use)}/{len(want)} included GSMs" + (
            " (GSE42832 columns could not be linked to GSMs; ranking uses the other sub-series)" if len(use) < len(want) else "")
        rk = np.log2(np.clip(raw.reindex(V.index)[use].apply(pd.to_numeric, errors="coerce"), 1, None)).mean(1)
    elif c["scale"] == "linear_quantile":
        rank_src = "GEO non-normalised values, log2, before quantile normalisation"; rk = np.log2(np.clip(V, 1, None)).mean(1)
    else:
        rank_src = "matrix values on log2 scale"; rk = L.mean(1)
    df = pd.DataFrame({"probe": V.index, "symbol_raw": s.values, "symbol_hgnc": hg.values, "rank_mean_log2": rk.values})
    df = df.dropna(subset=["rank_mean_log2"]).sort_values(["symbol_hgnc", "rank_mean_log2", "probe"], ascending=[True, False, True])
    chosen = df.drop_duplicates("symbol_hgnc", keep="first")
    df["selected"] = df.probe.isin(chosen.probe)
    df.to_csv(R / f"qc/probe_selection_{g}.csv", index=False)
    out = L.loc[chosen.probe]; out.index = chosen.symbol_hgnc.values; out.index.name = "Gene"
    assert out.index.is_unique and out.columns.is_unique and list(out.columns) == want
    out.astype("float32").to_csv(R / f"expression/{g}_log2_genelevel.csv.gz", float_format="%.5f")
    summ.append(dict(cohort=g, samples=out.shape[1], genes=out.shape[0], probes_in_matrix=n_probe0, probes_no_symbol=int(empty.sum()),
                     probes_multi_symbol=int((multi & ~empty).sum()), multi_probe_genes=int((df.groupby("symbol_hgnc").size() > 1).sum()),
                     symbols_changed_by_HGNC=int((chosen.symbol_raw.str.upper() != chosen.symbol_hgnc).sum()),
                     value_scale=c["scale"], probe_rank_source=rank_src))
    print(summ[-1], flush=True)
if DATEFIX:
    fp = R / "qc/date_corrupted_symbols.csv"; old = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
    nf = pd.DataFrame(DATEFIX); old = old[~old.cohort.isin(nf.cohort)] if len(old) else old
    pd.concat([old, nf]).to_csv(fp, index=False)
p = R / "qc/expression_build_summary.csv"
old = pd.read_csv(p) if p.exists() else pd.DataFrame()
new = pd.DataFrame(summ); old = old[~old.cohort.isin(new.cohort)] if len(old) else old
pd.concat([old, new]).to_csv(p, index=False)
