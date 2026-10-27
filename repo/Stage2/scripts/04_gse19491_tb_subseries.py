"""Stage2 item 6: GSE19491 ATB/LTB by sub-series, and batch between sub-series with class held constant."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ttest_ind, false_discovery_control, mannwhitneyu
S = Path(__file__).resolve().parents[1]; CR = S.parent / "Clean_rebuild"
M = pd.read_csv(CR / "sample_manifest.csv"); m = M[(M.cohort == "GSE19491") & M.included]
G = pd.read_csv(CR / "qc/geo_samples_long.csv", low_memory=False); g = G[G.cohort == "GSE19491"].set_index("GSM")
m = m.assign(region=m.GSM.map(g["ch:geographical region"]), source=m.GSM.map(g.source_name_ch1))
tab = pd.crosstab(m.subseries, m.class_label); tab["region"] = m.groupby("subseries").region.agg(lambda s: ";".join(sorted(s.dropna().unique())))
print(tab.to_string()); tab.to_csv(S / "qc/GSE19491_class_by_subseries.csv")
E = pd.read_csv(S / "expression/GSE19491_log2_genelevel_jointnorm.csv.gz", index_col=0)
rows = []
def de(a, b, label):
    t = ttest_ind(E[a].to_numpy(), E[b].to_numpy(), axis=1, equal_var=False); q = false_discovery_control(np.nan_to_num(t.pvalue, nan=1))
    d = E[a].mean(1) - E[b].mean(1)
    rows.append(dict(comparison=label, n_a=len(a), n_b=len(b), genes_FDR05=int((q < .05).sum()), genes_FDR05_absdiff_gt1=int(((q < .05) & (d.abs() > 1)).sum())))
sel = lambda s, c: m[(m.subseries == s) & (m.class_label == c)].GSM.tolist()
de(sel("GSE19439", "CON"), sel("GSE19444", "CON"), "CON: GSE19439 vs GSE19444 (batch, class constant)")
de(sel("GSE19439", "LTB"), sel("GSE19444", "LTB"), "LTB: GSE19439 vs GSE19444 (batch, class constant)")
de(sel("GSE19439", "ATB"), sel("GSE19444", "ATB"), "ATB: GSE19439 vs GSE19444 (batch, class constant)")
de(sel("GSE19442", "LTB"), sel("GSE19439", "LTB") + sel("GSE19444", "LTB"), "LTB: GSE19442 (S Africa) vs London")
de(sel("GSE19435", "ATB"), sel("GSE19439", "ATB") + sel("GSE19444", "ATB"), "ATB: GSE19435 vs other London ATB")
for s in ["GSE19439", "GSE19442", "GSE19444"]:
    de(sel(s, "ATB"), sel(s, "LTB"), f"ATB vs LTB within {s} (biology, for scale)")
R = pd.DataFrame(rows); print(R.to_string()); R.to_csv(S / "qc/GSE19491_tb_batch_checks.csv", index=False)
