"""Step 3: sample_manifest.csv — one row per (cohort, GSM) for every GEO sample of the 10 series.
Every inclusion decision and class label is derived from GEO metadata fields keyed by GSM
(qc/geo_samples_long.csv). Titles are never used to decide class or inclusion.
User decisions (2026-10-19) are encoded below and cited in exclusion_reason:
  - GSE107994: intended 152-GSM list; progressors excluded; GEO-flagged outliers excluded; EPTB excluded
  - EPTB excluded wherever GEO annotates TB site (GSE107994, GSE83456)
  - longitudinal sub-series: keep pre-treatment samples, drop on-treatment samples
  - GSE42834: rebuilt from GEO non-normalised data (GSE42832 whole-blood samples therefore excluded); whole blood only; non-active sarcoidosis excluded; GSE42830 'Baseline' = pneumonia -> OD
  - GSE19491: GSE22098 healthy controls removed from OD
  - duplicate arrays (same array under two GSMs): one copy kept; the copy in the longitudinal sub-series is dropped
"""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; MT = R.parent
G = pd.read_csv(R / "qc/geo_samples_long.csv", low_memory=False)
def ch(df, k): return df.get("ch:" + k, pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.strip()
out = []
def add(df, cls, inc, reason, sub=None, note=""):
    for i in df.index:
        out.append(dict(cohort=df.at[i, "cohort"], GSM=df.at[i, "GSM"], subseries=sub.at[i] if sub is not None else df.at[i, "cohort"],
                        title=df.at[i, "title"], source=df.at[i, "source_name_ch1"], class_label=cls.at[i] if hasattr(cls, "at") else cls,
                        included=bool(inc.at[i]) if hasattr(inc, "at") else inc, exclusion_reason=reason.at[i] if hasattr(reason, "at") else reason,
                        geo_basis=note.at[i] if hasattr(note, "at") else note))
def sub_of(df, superid):
    return df.series_id.str.split(";").apply(lambda s: ";".join(x for x in s if x != superid) or superid)

# ---- GSE107994 (RNA-seq, Leicester) -------------------------------------------------------------
d = G[G.cohort == "GSE107994"].copy()
grp, outl, site, tp = ch(d, "group"), ch(d, "outlier"), ch(d, "tb_disease_type"), ch(d, "timepoint_months")
intended = set(pd.read_csv(DATA_ROOT / "Full data/Test folder/GSE-107994/GSE107994_baseline_GSM_patient_class.csv").GSM)
cls = grp.map({"Active_TB": "ATB", "LTBI": "LTB", "Control": "CON"}).fillna("")
reason = pd.Series("", index=d.index, dtype=object)
reason[~d.GSM.isin(intended)] = "not in intended baseline GSM list (LTBI progressor sample)"
reason[(grp == "LTBI_Progressor") & (tp == "Baseline")] = "LTBI progressor baseline (was labelled LTB in old z-file) — not in intended list"
reason[(reason == "") & (outl == "Yes")] = "GEO submitter QC flag outlier: Yes (user decision: exclude)"
reason[(reason == "") & (cls == "ATB") & (site == "Non-Pulmonary")] = "extrapulmonary TB (tb_disease_type: Non-Pulmonary; user decision: exclude EPTB everywhere)"
add(d, cls, reason == "", reason, note="group=" + grp + "; outlier=" + outl + "; tb_disease_type=" + site + "; timepoint=" + tp)

# ---- GSE19491 (SuperSeries, Berry 2010) -----------------------------------------------------------
d = G[G.cohort == "GSE19491"].copy(); sub = sub_of(d, "GSE19491")
ill, hc, src, tpt = ch(d, "illness"), ch(d, "healthy control"), d.source_name_ch1.fillna(""), ch(d, "time post initiation of treatment")
wb = src.str.contains("whole blood", case=False)
cls = pd.Series("", index=d.index, dtype=object)
cls[ill.eq("PTB")] = "ATB"; cls[ill.str.upper().isin(["LATENT", "LATENT TB"])] = "LTB"
cls[ill.str.startswith("Control")] = "CON"; cls[ill.isin(["Still", "ASLE"])] = "OD"
reason = pd.Series("", index=d.index, dtype=object)
reason[(sub == "GSE19435") & (tpt != "0_months")] = "on treatment (GSE19435, time post initiation of treatment: " + tpt + ")"
reason[(reason == "") & ~wb] = "not whole blood (sorted cells, GSE19443)"
reason[(reason == "") & (hc != "")] = "healthy control of GSE22098 (characteristic 'healthy control: " + hc + "') — not OD"
reason[(reason == "") & ill.isin(["PSLE", "Staph", "Strep"])] = "GSE22098 disease not in OD definition (old analysis used ASLE + Still only)"
reason[(reason == "") & (cls == "")] = "no class"
add(d, cls.where(reason == "", cls), reason == "", reason, sub, "illness=" + ill + "; healthy control=" + hc + "; time=" + tpt + "; source=" + src)

# ---- GSE42834 (SuperSeries, Bloom 2013) -----------------------------------------------------------
d = G[G.cohort == "GSE42834"].copy(); sub = sub_of(d, "GSE42834")
ds, cell, src = ch(d, "disease state").str.lower(), ch(d, "cell type"), d.source_name_ch1.fillna("")
cls = pd.Series("", index=d.index, dtype=object)
cls[ds == "tb"] = "ATB"; cls[ds == "control"] = "CON"
cls[ds.isin(["active sarcoid", "active sarcoidosis", "sarcoid", "lung cancer", "pneumonia"])] = "OD"
cls[(ds == "baseline") & (sub == "GSE42830")] = "OD"     # = pneumonia pre-treatment (user decision; 5/8 byte-identical to GSE42827 pre-treatment arrays)
reason = pd.Series("", index=d.index, dtype=object)
tx = (ch(d, "treament") + ch(d, "treatment")).str.strip()
long_ = sub.isin(["GSE42827", "GSE42831"])
reason[long_ & (tx != "Pre-treatment")] = "on treatment (" + sub + ", treatment: " + tx + ")"
cls[long_ & (tx == "Pre-treatment")] = "OD"   # GSE42827 = pneumonia; GSE42831 = sarcoidosis (titles/summary; its design line says 'pneumonia'), OD either way
reason[(reason == "") & ((cell != "") | ~src.str.contains("whole blood", case=False))] = "not whole blood (" + src + ")"
reason[(reason == "") & (ds == "non-active sarcoidosis")] = "non-active sarcoidosis (user decision: exclude)"
reason[(reason == "") & (sub == "GSE42832")] = "GSE42832 whole blood: its non-normalised columns cannot be linked to GSMs reliably (user decision: rebuild GSE42834 from non-normalised data)"
reason[(reason == "") & (cls == "")] = "no class"
note = "disease state=" + ch(d, "disease state") + "; cell type=" + cell + "; treatment=" + ch(d, "treament") + ch(d, "treatment") + "; source=" + src
note[(ds == "baseline")] += "; GEO disease state 'Baseline' in GSE42830 = pneumonia patient (series design: 8 pneumonia)"
add(d, cls, reason == "", reason, sub, note)

# ---- GSE37250 -----------------------------------------------------------------------------------
d = G[G.cohort == "GSE37250"].copy(); ds, hiv = ch(d, "disease state"), ch(d, "hiv status")
cls = ds.map({"active tuberculosis": "ATB", "latent TB infection": "LTB", "other disease": "OD"}).fillna("")
reason = np.where(hiv != "HIV negative", "HIV positive (" + hiv + ")", np.where(cls == "", "no class", ""))
add(d, cls, pd.Series(reason == "", index=d.index), pd.Series(reason, index=d.index), note="disease state=" + ds + "; hiv=" + hiv)
# ---- GSE73408 -----------------------------------------------------------------------------------
d = G[G.cohort == "GSE73408"].copy(); cg = ch(d, "clinical group")
cls = cg.map({"TB": "ATB", "LTBI": "LTB", "PNA": "OD"}).fillna("")
add(d, cls, cls != "", cls.map(lambda c: "" if c else "no class"), note="clinical group=" + cg)
# ---- GSE19439 / GSE19444 ------------------------------------------------------------------------
for c in ["GSE19439", "GSE19444"]:
    d = G[G.cohort == c].copy(); ill = ch(d, "illness")
    cls = pd.Series(np.select([ill == "PTB", ill == "Latent", ill.str.startswith("Control")], ["ATB", "LTB", "CON"], ""), index=d.index)
    add(d, cls, cls != "", cls.map(lambda x: "" if x else "no class"), note="illness=" + ill + "; source=" + d.source_name_ch1)
# ---- GSE28623 -----------------------------------------------------------------------------------
d = G[G.cohort == "GSE28623"].copy(); g = ch(d, "group")
cls = g.map({"tuberculosis patients": "ATB", "healthy donors latently infected with M. tuberculosis": "LTB", "healthy non-infected donors": "CON"}).fillna("")
add(d, cls, cls != "", cls.map(lambda x: "" if x else "no class"), note="group=" + g + "; source=" + d.source_name_ch1)
# ---- GSE101705 ----------------------------------------------------------------------------------
d = G[G.cohort == "GSE101705"].copy(); g = ch(d, "condition")
cls = g.map({"TB": "ATB", "latent TB infection": "LTB"}).fillna("")
add(d, cls, cls != "", cls.map(lambda x: "" if x else "no class"), note="condition=" + g)
# ---- GSE83456 -----------------------------------------------------------------------------------
d = G[G.cohort == "GSE83456"].copy(); ds = ch(d, "disease state")
cls = ds.map({"PTB": "ATB", "HC": "CON", "Sarcoid": "OD", "EPTB": "EPTB"}).fillna("")
reason = cls.map(lambda x: "extrapulmonary TB (user decision: exclude EPTB everywhere)" if x == "EPTB" else ("" if x else "no class"))
add(d, cls.replace("EPTB", "ATB"), reason == "", reason, note="disease state=" + ds)

M = pd.DataFrame(out)
M.loc[~M.included & (M.exclusion_reason == "no class"), "class_label"] = ""
# cross-cohort GSM reuse (SuperSeries containing other series' samples)
multi = M.groupby("GSM").cohort.agg(lambda s: ";".join(sorted(s)))
M["GSM_also_in_cohort_file"] = M.apply(lambda r: ";".join(x for x in multi[r.GSM].split(";") if x != r.cohort), axis=1)
# known duplicate arrays (different GSMs, same array)
dup = {}
D = pd.read_csv(R / "qc/raw_duplicate_arrays.csv"); mp = pd.read_csv(R / "qc/raw_column_to_GSM.csv").set_index("raw_column").GSM
for _, r in D.iterrows(): dup[mp[r.raw_a]] = mp[r.raw_b]; dup[mp[r.raw_b]] = mp[r.raw_a]
for a, b in pd.read_csv(R / "qc/gse19491_duplicate_pairs.csv")[["GSM_a", "GSM_b"]].itertuples(index=False): dup[a] = b; dup[b] = a
M["duplicate_array_GSM"] = M.GSM.map(dup).fillna("")
LONG = {"GSE19435", "GSE42827", "GSE42831"}
for i in M.index[M.included & (M.duplicate_array_GSM != "")]:
    part = M[(M.cohort == M.at[i, "cohort"]) & (M.GSM == M.at[i, "duplicate_array_GSM"]) & M.included]
    if len(part) and M.at[i, "subseries"] in LONG and part.subseries.iloc[0] not in LONG:
        M.at[i, "included"] = False
        M.at[i, "exclusion_reason"] = f"duplicate array of {part.GSM.iloc[0]} ({part.subseries.iloc[0]} copy kept)"
assert not M.duplicated(["cohort", "GSM"]).any()
assert set(M[M.included].class_label) <= {"ATB", "LTB", "CON", "OD"}
inc = M[M.included]; both = inc[inc.duplicate_array_GSM.isin(set(inc.GSM))]
assert len(both) == 0, f"both copies of a duplicate array included: {both.GSM.tolist()}"
M.to_csv(R / "sample_manifest.csv", index=False)
print(M.groupby(["cohort", "included", "class_label"]).size().unstack(fill_value=0).to_string())
print("\nexclusion reasons:"); print(M[~M.included].groupby(["cohort", "exclusion_reason"]).size().to_string())
