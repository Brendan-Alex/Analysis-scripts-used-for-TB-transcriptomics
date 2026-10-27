"""Recomputes the Stage6 per-cohort gene rankings (ATB_vs_OD, all 4 cohorts) and the ATB_vs_OD cohort relatedness with the
rebuilt GSE83456, using the Stage6 functions unchanged (Stage6/scripts/run_stage6.py, read-only).
Run in the Cowork VM on an ATB_vs_OD cache built there by the Stage7 prepare step (STAGE7_CACHE_DIR; Windows pickles need pandas 3);
cross-check: the first 20 of each mRMR order must equal the Stage7 (Windows) per-cohort pools."""
import os, sys, json, importlib.util
from pathlib import Path
S7 = Path(__file__).resolve().parents[1]; MT = S7.parent
os.environ["STAGE5_DIR"] = os.environ["STAGE7_CACHE_DIR"]; os.environ["STAGE6_OUT"] = str(S7)
sp = importlib.util.spec_from_file_location("s6", str(MT / "Stage6/scripts/run_stage6.py")); s6 = importlib.util.module_from_spec(sp); sp.loader.exec_module(s6)
for sub in ("results/rank", "results/related", "logs"): (S7 / sub).mkdir(parents=True, exist_ok=True)
import pandas as pd
which = sys.argv[1]
if which == "rank":
    for c in sys.argv[2:]:
        r = s6.rank_job("ATB_vs_OD", c); print(r, flush=True)
        with open(S7 / "results/rank/rank_check.jsonl", "a") as fh: fh.write(json.dumps(r) + "\n")
elif which == "seeds":
    (S7 / "results/seeds").mkdir(parents=True, exist_ok=True)
    rows = [s6.seeds_job("ATB_vs_OD", h) for h in s6.TASKS["ATB_vs_OD"]["cohorts"]]
    with open(S7 / "results/seeds/seeds_ATB_vs_OD.jsonl", "w") as fh: fh.write("".join(json.dumps(r) + "\n" for r in rows))
    print(rows)
else:
    rows = s6.related_job("ATB_vs_OD"); pd.DataFrame(rows).to_csv(S7 / "results/related/pairs_ATB_vs_OD.csv", index=False); print(pd.DataFrame(rows).iloc[:, :12].to_string())
