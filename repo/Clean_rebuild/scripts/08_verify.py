"""Step 8: independent checks of the rebuild. Fails loudly; writes qc/verification_report.txt.
 V1 labels: manifest class vs an independent GEO coding (Sample_ID_audit/geo/geo_codes.txt, checksum-verified on 18 Oct;
    GSE107994 vs its separately transcribed GEO table)
 V2 expression files: columns == manifest included GSMs (order-free), unique genes, no NaN, finite
 V3 GSE107994 specifics; V4 duplicate arrays: no included pair; V5 patient-level uniqueness where GEO gives IDs
 V6 GSMs shared between cohort files; V7 class counts vs the old files
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
M = pd.read_csv(R / "sample_manifest.csv"); out = []; fail = []
def say(*a): out.append(" ".join(str(x) for x in a)); print(*a)
# ---- V1
codes = {}
for line in open(MT / "Sample_ID_audit/geo/geo_codes.txt").read().split("\n"):
    g, runs = line.split("\t")
    for r in runs.split():
        st, s = r.split(":"); st = int(st)
        for i, ch in enumerate(s): codes[(g, f"GSM{st+i}")] = ch
OK = {"ATB": set("AT"), "LTB": set("L"), "CON": set("CK"), "OD": set("OBSN")}   # T/S/N = longitudinal pre/post; checked below
inc = M[M.included]; bad = []
for _, r in inc.iterrows():
    if r.cohort == "GSE107994": continue
    c = codes.get((r.cohort, r.GSM))
    if c is None or c not in OK[r.class_label]: bad.append((r.cohort, r.GSM, r.class_label, c))
say("V1 included non-GSE107994 samples checked:", len(inc[inc.cohort != "GSE107994"]), "label mismatches vs independent GEO coding:", len(bad))
if bad: fail.append(f"V1 {bad[:10]}")
# T (ATB on treatment) must never be included; check the GSE19435 included ATB are 0_months via geo_basis
t = inc[(inc.cohort == "GSE19491") & (inc.subseries == "GSE19435")]
assert t.geo_basis.str.contains("time=0_months").all(); say("V1 GSE19435 included samples all time=0_months:", len(t))
x = inc[inc.cohort == "GSE42834"]
assert not x.geo_basis.str.contains("treatment=immunosuppressive|treatment=Treated", regex=True).any()
assert not x.geo_basis.str.contains("cell type=CD|cell type=PBMC", regex=True).any()
say("V1 GSE42834 included: no on-treatment, no sorted-cell samples")
g7 = pd.read_csv(MT / "Sample_ID_audit/geo/GSE107994_geo_samples.csv")
e = inc[inc.cohort == "GSE107994"].merge(g7, on="GSM")
mp = {"Active_TB": "ATB", "LTBI": "LTB", "Control": "CON"}
nbad = int((e.group.map(mp) != e.class_label).sum()); say("V1 GSE107994 labels vs separately transcribed GEO table: mismatches", nbad, "of", len(e))
if nbad: fail.append("V1 GSE107994")
# ---- V2
for c in sorted(M.cohort.unique()):
    X = pd.read_csv(R / f"expression/{c}_log2_genelevel.csv.gz", index_col=0)
    want = set(M[(M.cohort == c) & M.included].GSM)
    ok = set(X.columns) == want and X.columns.is_unique and X.index.is_unique and np.isfinite(X.to_numpy()).all()
    say(f"V2 {c}: {X.shape[0]} genes x {X.shape[1]} samples; columns==manifest {set(X.columns)==want}; unique {X.columns.is_unique and X.index.is_unique}; finite {np.isfinite(X.to_numpy()).all()}")
    if not ok: fail.append(f"V2 {c}")
# ---- V3
s = M[M.cohort == "GSE107994"]
five = {"S14": "GSM2886286", "S43": "GSM2886310", "S123": "GSM2886389", "S131": "GSM2886397", "S176": "GSM2886440"}
for k, gsm in five.items():
    r = s[s.GSM == gsm].iloc[0]; say(f"V3 {k} {gsm}: in GEO yes, included {r.included}, class {r.class_label}, {r.geo_basis}")
prog = s[s.exclusion_reason.str.startswith("LTBI progressor baseline", na=False)].GSM.tolist(); say("V3 8 progressor baselines:", ";".join(prog)); assert len(prog) == 8
intended = set(pd.read_csv(DATA_ROOT / "Full data/Test folder/GSE-107994/GSE107994_baseline_GSM_patient_class.csv").GSM)
assert set(s[s.included].GSM) <= intended; say("V3 all included GSE107994 GSMs are in the intended list")
# ---- V4
both = inc[inc.duplicate_array_GSM.isin(set(inc.GSM))]; say("V4 included samples whose duplicate array is also included:", len(both))
if len(both): fail.append("V4")
# ---- V5
pid = e.merge(g7[["GSM"]], on="GSM").patient_id; say("V5 GSE107994 included patients unique:", pid.is_unique)
G = pd.read_csv(R / "qc/geo_samples_long.csv", low_memory=False)
sub = G[(G.cohort == "GSE42834") & G["ch:subject"].notna()][["GSM", "ch:subject"]].merge(inc[["GSM"]], on="GSM")
say("V5 GSE42834 included samples carrying a GEO subject ID:", len(sub), "unique subjects:", sub["ch:subject"].nunique())
# ---- V6
shared = inc.groupby("GSM").cohort.nunique(); sh = inc[inc.GSM.isin(shared[shared > 1].index)]
say("V6 included GSMs present in more than one cohort file:", sh.GSM.nunique(), dict(sh.groupby("cohort").size()))
# ---- V7
old = pd.read_csv(MT / "Sample_ID_audit/all_cohorts_sample_map.csv")
for c in sorted(M.cohort.unique()):
    new = inc[inc.cohort == c].class_label.value_counts().to_dict()
    o = old[old.Cohort == c].file_label.value_counts().to_dict() if c in set(old.Cohort) else ({"GSE101705": "28 ATB / 16 LTB (author file, no GSMs)", "GSE107994": "138-sample z-file: 42 ATB / 50 LTB / 46 CON, 8 of the LTB = progressors"}[c])
    say(f"V7 {c}: old {o} -> new {new}")
say("RESULT:", "PASS" if not fail else f"FAIL {fail}")
(R / "qc/verification_report.txt").write_text("\n".join(out) + "\n")
