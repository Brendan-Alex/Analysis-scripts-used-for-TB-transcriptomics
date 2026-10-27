"""Stage4 shared definitions: rotating leave-one-cohort-out over every cohort of each task."""
from pathlib import Path
import numpy as np, pandas as pd
S4 = Path(__file__).resolve().parents[1]; MT = S4.parent; CR = MT / "Clean_rebuild"; S3 = MT / "Stage3"
SEED = 42
TASKS = {
 "ATB_vs_LTB": dict(pos="ATB", neg="LTB", cohorts=["GSE19491", "GSE37250", "GSE73408", "GSE101705", "GSE107994"],
                    panels=["Native", "Zak16", "Kaforou_ATB_v_LTB", "Berry86", "Sweeney3"]),
 "ATB_vs_OD":  dict(pos="ATB", neg="OD", cohorts=["GSE37250", "GSE73408", "GSE42834", "GSE83456"],
                    panels=["Native", "Zak16", "Kaforou_ATB_v_OD", "Berry86", "Sweeney3"]),
 "LTB_vs_CON": dict(pos="LTB", neg="CON", cohorts=["GSE19439", "GSE19444", "GSE28623", "GSE107994"],
                    panels=["Native", "Zak16", "Berry86", "Sweeney3"]),          # Kaforou omitted (user decision)
}
# cohorts where each curated panel was derived / tuned / validated (Stage3 table; GSE19491 contains GSE19439/19442/19444)
OVERLAP = {
 "Berry86": {"GSE19444": "derivation", "GSE19439": "validation", "GSE19491": "contains derivation/validation sub-series GSE19444, GSE19439, GSE19442"},
 "Kaforou_ATB_v_LTB": {"GSE37250": "derivation", "GSE19491": "validation"},
 "Kaforou_ATB_v_OD": {"GSE37250": "derivation", "GSE19491": "validation"},
 "Zak16": {"GSE19439": "parameterisation (Illumina version)", "GSE19444": "testing", "GSE37250": "testing", "GSE42834": "testing",
           "GSE19491": "contains tested sub-series GSE19439, GSE19444, GSE19442"},
 "Sweeney3": {"GSE19491": "discovery", "GSE37250": "discovery", "GSE42834": "discovery", "GSE28623": "validation",
              "GSE19439": "sub-series of discovery dataset GSE19491", "GSE19444": "sub-series of discovery dataset GSE19491"},
}
def manifest(): return pd.read_csv(S3 / "data/sample_manifest_stage3.csv")
def expr(c):
    f = S3 / "data/GSE19491_log2_genelevel_stage3.csv.gz" if c == "GSE19491" else CR / f"expression/{c}_log2_genelevel.csv.gz"
    return pd.read_csv(f, index_col=0)
def panel(p): return pd.read_csv(CR / f"panels/{p}_hgnc.csv").Gene.astype(str).tolist()
