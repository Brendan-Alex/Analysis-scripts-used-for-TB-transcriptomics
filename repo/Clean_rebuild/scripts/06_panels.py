"""Step 6: HGNC-map the six panels (same mapper as cohorts) and report per-cohort coverage.
Outputs: panels/<panel>_hgnc.csv, qc/panel_gene_mapping.csv, qc/panel_coverage.csv"""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[1]; PD = DATA_ROOT / "Supplementary/Gene Panels"
sys.path.insert(0, str(Path(__file__).resolve().parent)); from hgnc_alias import HGNCMapper
HG = HGNCMapper()
PANELS = {"Native20": "Native 20 genes.csv", "Zak16": "Zak16.csv", "Berry86": "Berry86_clean.csv", "Sweeney3": "Sweeney3.csv",
          "Kaforou_ATB_v_LTB": "Kaforou ATB v LTB.csv", "Kaforou_ATB_v_OD": "Kaforou ATB v OD.csv"}
COH = ["GSE19491", "GSE37250", "GSE73408", "GSE19439", "GSE19444", "GSE28623", "GSE101705", "GSE107994", "GSE42834", "GSE83456"]
genes = {c: set(pd.read_csv(R / f"expression/{c}_log2_genelevel.csv.gz", usecols=[0]).iloc[:, 0]) for c in COH}
changed = {}
for c in COH:
    f = R / f"qc/probe_selection_{c}.csv"
    changed[c] = f
maprows, cov = [], []
for p, f in PANELS.items():
    raw = pd.read_csv(PD / f)["Gene"].astype(str).tolist()
    m = [dict(panel=p, gene_in_panel_file=g, **HG.map(g)) for g in raw]
    maprows += m
    cur = sorted({x["current"] for x in m})
    pd.DataFrame({"Gene": cur}).to_csv(R / f"panels/{p}_hgnc.csv", index=False)
    for c in COH:
        miss = [g for g in cur if g not in genes[c]]
        cov.append(dict(panel=p, cohort=c, panel_genes=len(cur), present=len(cur) - len(miss), missing=";".join(miss)))
pd.DataFrame(maprows).to_csv(R / "qc/panel_gene_mapping.csv", index=False)
C = pd.DataFrame(cov); C.to_csv(R / "qc/panel_coverage.csv", index=False)
print(C.pivot(index="panel", columns="cohort", values="present").to_string())
print(pd.DataFrame(maprows).query("status != 'approved'")[["panel", "gene_in_panel_file", "current", "status"]].to_string())
