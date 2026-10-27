"""Stage7 step 1: rebuild GSE83456 at absolute expression level from the GEO non-normalised file.
Same rules as the Clean_rebuild GSE42834 rebuild (scale 'linear_quantile'):
  - source: Clean_rebuild/geo_raw/GSE83456_matrix_EPTB_unnormalized.txt.gz (raw_value columns; detection p-value columns dropped)
  - raw column -> GSM: Clean_rebuild/qc/raw_column_to_GSM.csv (expression-identity match, accepted rows only)
  - samples: the 155 GSMs the Clean_rebuild manifest includes (untreated whole blood, EPTB excluded), in the same order
  - probes: exactly the probe set of Clean_rebuild/qc/probe_selection_GSE83456.csv (mapped, single-symbol probes),
    log2(max(x, 1)) then quantile normalisation across the 155 samples over that probe set, then the probe
    Clean_rebuild selected for each gene (its selection already used these raw values)
Output: data/GSE83456_log2_genelevel_rebuilt.csv.gz and qc/gse83456_rebuild_checks.csv. Nothing in Clean_rebuild is modified."""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
S7 = Path(__file__).resolve().parents[1]; MT = S7.parent; CR = MT / "Clean_rebuild"; S3 = MT / "Stage3"
def quantile_normalise(a):      # identical to Clean_rebuild/scripts/04_build_expression.py
    rk = a.rank(axis=0, method="average").to_numpy(); ms = np.sort(a.to_numpy(), axis=0).mean(1); grid = np.arange(1, len(ms) + 1)
    return pd.DataFrame(np.column_stack([np.interp(rk[:, k], grid, ms) for k in range(a.shape[1])]), index=a.index, columns=a.columns)
d = pd.read_csv(DATA_ROOT / "geo_raw/GSE83456_matrix_EPTB_unnormalized.txt.gz", sep="\t", index_col=0, low_memory=False)
d = d.loc[:, [i % 2 == 0 for i in range(d.shape[1])]]
d = d.loc[:, ~d.columns.str.contains("pval|p_value", case=False)].dropna(axis=1, how="all")
d.columns = [f"GSE83456::{x}" for x in d.columns]; d.index = d.index.astype(str)
M2 = pd.read_csv(CR / "qc/raw_column_to_GSM.csv"); M2 = M2[(M2.cohort == "GSE83456") & M2.accepted]; r2g = M2.set_index("raw_column").GSM; assert r2g.is_unique
d = d[[x for x in d.columns if x in r2g.index]].rename(columns=r2g.to_dict())
MAN = pd.read_csv(CR / "sample_manifest.csv"); inc = MAN[(MAN.cohort == "GSE83456") & MAN.included]; want = inc.GSM.tolist()
old = pd.read_csv(CR / "expression/GSE83456_log2_genelevel.csv.gz", index_col=0)
assert list(old.columns) == want, "Clean_rebuild column order differs from manifest"
ps = pd.read_csv(CR / "qc/probe_selection_GSE83456.csv"); ps["probe"] = ps.probe.astype(str)
V = d.reindex(ps.probe)[want].apply(pd.to_numeric, errors="coerce")
n_missing = int(V.isna().any(axis=1).sum())
L = quantile_normalise(np.log2(np.clip(V.dropna(), 1, None)))
sel = ps[ps.selected]; sel = sel[sel.probe.isin(L.index)]
out = L.loc[sel.probe]; out.index = sel.symbol_hgnc.values; out.index.name = "Gene"; out = out.loc[old.index.intersection(out.index)]
out.astype("float32").to_csv(S7 / "data/GSE83456_log2_genelevel_rebuilt.csv.gz", float_format="%.5f")
# ---- checks
M3 = pd.read_csv(S3 / "data/sample_manifest_stage3.csv"); m3 = M3[(M3.cohort == "GSE83456") & M3.included_stage3]
g = out.index.intersection(old.index); zo = old.loc[g].sub(old.loc[g].mean(1), axis=0).div(old.loc[g].std(1, ddof=0), axis=0)
zn = out.loc[g].sub(out.loc[g].mean(1), axis=0).div(out.loc[g].std(1, ddof=0), axis=0)
gene_r = np.array([np.corrcoef(zo.iloc[i], zn.iloc[i])[0, 1] for i in range(len(g))])
chk = dict(samples_new=out.shape[1], samples_clean_rebuild=old.shape[1], same_GSMs_same_order=list(out.columns) == list(old.columns),
           stage3_included=len(m3), stage3_GSMs_equal=set(m3.GSM) == set(out.columns),
           labels_stage3=str(m3.class_label.value_counts().to_dict()), labels_clean_rebuild=str(inc.class_label.value_counts().to_dict()),
           labels_identical=bool((m3.set_index("GSM").class_label.reindex(inc.GSM).values == inc.class_label.values).all()),
           genes_new=out.shape[0], genes_clean_rebuild=old.shape[0], genes_shared=len(g), selected_probes_missing_in_raw=int((~ps[ps.selected].probe.isin(L.index)).sum()),
           probes_with_NA_dropped=n_missing, value_range_new=f"{float(out.min().min()):.2f} to {float(out.max().max()):.2f}",
           value_range_old=f"{float(old.min().min()):.2f} to {float(old.max().max()):.2f}",
           median_gene_mean_new=round(float(out.mean(1).median()), 3), median_gene_mean_old=round(float(old.mean(1).median()), 3),
           median_within_sample_r_new=round(float(np.median(np.corrcoef(out.rank().T.to_numpy())[np.triu_indices(out.shape[1], 1)])), 3),
           median_within_sample_r_old=round(float(np.median(np.corrcoef(old.rank().T.to_numpy())[np.triu_indices(old.shape[1], 1)])), 3),
           per_gene_z_corr_new_vs_old_median=round(float(np.nanmedian(gene_r)), 4), per_gene_z_corr_5th_pct=round(float(np.nanpercentile(gene_r, 5)), 4),
           per_gene_z_corr_min=round(float(np.nanmin(gene_r)), 4))
pd.DataFrame([chk]).T.rename(columns={0: "value"}).to_csv(S7 / "qc/gse83456_rebuild_checks.csv"); print(pd.DataFrame([chk]).T.to_string())
