"""Stage2 item 3/4: per-sample age from GEO metadata vs what the files include."""
import re
from pathlib import Path
import numpy as np, pandas as pd
S = Path(__file__).resolve().parents[1]; CR = S.parent / "Clean_rebuild"; MT = S.parent
G = pd.read_csv(CR / "qc/geo_samples_long.csv", low_memory=False)
M = pd.read_csv(CR / "sample_manifest.csv")
OLD = pd.read_csv(MT / "Sample_ID_audit/all_cohorts_sample_map.csv")      # old files' GSM membership (8 cohorts)
def age_of(row):
    for k in ("ch:age", "ch:age_at_baseline_visit"):
        v = row.get(k)
        if isinstance(v, str) or (isinstance(v, float) and not np.isnan(v)):
            s = str(v); m = re.search(r"(\d+(?:\.\d+)?)", s)
            if m:
                unit = "months" if re.search(r"month", s, re.I) else "years"
                a = float(m.group(1)); return (a / 12 if unit == "months" else a), s
    return np.nan, ""
A = G.apply(lambda r: pd.Series(age_of(r), index=["age_years", "age_raw"]), axis=1)
G = pd.concat([G[["cohort", "GSM", "series_id"]], A], axis=1)
D = M.merge(G, on=["cohort", "GSM"], how="left")
D["in_old_file"] = D.apply(lambda r: r.GSM in set(OLD[OLD.Cohort == r.cohort].GSM) if r.cohort in set(OLD.Cohort) else np.nan, axis=1)
D.to_csv(S / "qc/age_per_sample.csv", index=False)
rows = []
for c, d in D.groupby("cohort"):
    has = d.age_years.notna()
    u16 = d.age_years < 16
    rows.append(dict(cohort=c, GEO_samples=len(d), samples_with_age=int(has.sum()), GEO_under16=int(u16.sum()),
                     under16_included_CleanRebuild=int((u16 & d.included).sum()),
                     under16_excluded_CleanRebuild_for_age=0,
                     under16_in_old_file=(int((u16 & (d.in_old_file == True)).sum()) if d.in_old_file.notna().any() else "old file has no GSMs"),
                     min_age=d.age_years.min(), max_age=d.age_years.max(),
                     unit_note="; ".join(sorted(set(d.age_raw[d.age_raw.str.contains("month", case=False, na=False)].head(3))))))
R = pd.DataFrame(rows); R.to_csv(S / "qc/age_audit_by_cohort.csv", index=False); print(R.to_string())
od = D[(D.cohort == "GSE19491") & D.included & (D.class_label == "OD")].copy()
od["disease"] = od.geo_basis.str.extract(r"illness=(\w+)")[0]
for dis, g in od.groupby("disease"):
    a = g.age_years
    print(f"\n{dis}: n={len(g)} age min {a.min()} q1 {a.quantile(.25)} median {a.median()} q3 {a.quantile(.75)} max {a.max()}; >=16: {(a>=16).sum()}  <16: {(a<16).sum()}  missing: {a.isna().sum()}")
    print("  ages:", sorted(a.tolist()))
od[["GSM", "disease", "age_years", "age_raw", "title"]].sort_values(["disease", "age_years"]).to_csv(S / "qc/GSE19491_OD_ages.csv", index=False)
