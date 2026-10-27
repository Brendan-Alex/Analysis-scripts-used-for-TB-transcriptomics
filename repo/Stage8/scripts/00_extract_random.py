"""Extracts every random-panel held-out AUC (50 draws per task x fold x panel x method) from the Stage7 benchmark records into
data/random_auc_long.csv. Read-only on Stage7; no refitting."""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
S8 = Path(__file__).resolve().parents[1]; S7 = S8.parent / "Stage7"
LAB = json.load(open(S7 / "data/labels.json")); rows = []
for t in ["ATB_vs_LTB", "ATB_vs_OD", "LTB_vs_CON"]:
    for line in open(S7 / f"results/bench/{t}__records.jsonl"):
        r = json.loads(line)
        if r["kind"] != "rnd": continue
        h = r["heldout"]; y = np.array(LAB[t][h][1])
        for m, v in r["scores"].items(): rows.append((t.replace("_vs_", " vs "), h, r["panel"], m, r["draw"], r["size"], roc_auc_score(y, np.nan_to_num(v[h]))))
D = pd.DataFrame(rows, columns=["Task", "Held-out cohort", "Panel", "method", "draw", "size", "AUC"]); D.to_csv(S8 / "data/random_auc_long.csv", index=False); print(D.shape, D.groupby("Task").size().to_dict())
