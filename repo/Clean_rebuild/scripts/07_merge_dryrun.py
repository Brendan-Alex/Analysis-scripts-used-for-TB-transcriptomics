"""Step 7: exercise load_and_merge on each task's training cohorts (data check only; no models)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import tb_data
TASKS = {"ATB_vs_LTB / ATB_vs_OD training": ["GSE19491", "GSE37250", "GSE73408"],
         "LTB_vs_CON training": ["GSE19439", "GSE19444", "GSE28623"]}
for t, cs in TASKS.items():
    print(t); X, y, coh = tb_data.load_and_merge(cs)
    print("  final", X.shape, "class counts", y.value_counts().to_dict())
print("deliberate failure check: GSE19491 + GSE19439 (shared GSMs)")
try:
    tb_data.load_and_merge(["GSE19491", "GSE19439"], verbose=False); print("  NOT CAUGHT")
except AssertionError as e:
    print("  caught:", str(e)[:120])
