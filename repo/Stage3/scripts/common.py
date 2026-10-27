"""Stage3 shared definitions."""
from pathlib import Path
import numpy as np, pandas as pd
ST = Path(__file__).resolve().parents[1]; MT = ST.parent; CR = MT / "Clean_rebuild"; S2 = MT / "Stage2"
SEED = 42
TASKS = {
 "ATB_vs_LTB": dict(pos="ATB", neg="LTB", train=["GSE19491", "GSE37250", "GSE73408"], test=["GSE101705", "GSE107994"],
                    panels=["Native_task", "Zak16", "Sweeney3", "Kaforou_ATB_v_LTB"]),
 "LTB_vs_CON": dict(pos="LTB", neg="CON", train=["GSE19439", "GSE19444", "GSE28623"], test=["GSE107994"],
                    panels=["Native_task", "Zak16", "Berry86", "Sweeney3"]),
 "ATB_vs_OD":  dict(pos="ATB", neg="OD", train=["GSE19491", "GSE37250", "GSE73408"], test=["GSE42834", "GSE83456"],
                    panels=["Native_task", "Zak16", "Berry86", "Kaforou_ATB_v_OD"]),          # on hold
}
PANELS = ["Zak16", "Sweeney3", "Berry86", "Kaforou_ATB_v_LTB", "Kaforou_ATB_v_OD"]
def manifest():
    return pd.read_csv(ST / "data/sample_manifest_stage3.csv")
def expr(c):
    f = ST / "data/GSE19491_log2_genelevel_stage3.csv.gz" if c == "GSE19491" else CR / f"expression/{c}_log2_genelevel.csv.gz"
    return pd.read_csv(f, index_col=0)
def panel(p):
    return pd.read_csv(CR / f"panels/{p}_hgnc.csv").Gene.astype(str).tolist()
def task_matrix(task, c, z=True):
    """log2 (z=False) or per-cohort z-scored over the task's two classes (z=True); genes x samples, labels 1=pos."""
    t = TASKS[task]; m = manifest(); m = m[(m.cohort == c) & m.included_stage3 & m.class_label.isin([t["pos"], t["neg"]])]
    X = expr(c)[m.GSM.tolist()]
    y = (m.set_index("GSM").class_label.loc[X.columns] == t["pos"]).astype(int)
    if z:
        sd = X.std(axis=1, ddof=0); X = X.sub(X.mean(axis=1), axis=0).div(sd.replace(0, np.nan), axis=0).fillna(0.0)
    return X, y
