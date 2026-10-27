"""Stage7 supplementary: Native panels (any task via --task; default ATB_vs_OD) with X/Y-linked genes ALLOWED (comparison only; the primary panels exclude them),
recomputed with the rebuilt GSE83456. Same Stage5 algorithm and universe definition except that X/Y genes are not removed:
universe = genes measured in every ATB_vs_OD cohort minus blank/LOC/C*ORF/PHAGE names. Uses the Stage7 ATB_vs_OD cache (read-only);
outputs to Stage7/withsex/. Usage (from the Stage7 folder): python scripts/run_stage7_withsex.py --workers 10 [--task ATB_vs_LTB|ATB_vs_OD|LTB_vs_CON]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import sys, json, pickle, argparse, platform, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
S7 = Path(__file__).resolve().parents[1]; MT = S7.parent
CACHE = Path(os.environ.get("STAGE7_CACHE_DIR", str(S7)))
WS = Path(os.environ.get("STAGE7_WS_OUT", str(S7 / "withsex"))); os.environ["STAGE5_OUT"] = str(WS)
_sp = importlib.util.spec_from_file_location("s5", str(MT / "Stage5/scripts/run_stage5.py")); s5 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(s5)
T = next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--task"), "ATB_vs_OD")
def load7(task): return pickle.load(open(CACHE / "data/cache" / f"{task}.pkl", "rb"))
s5.load = load7
UF = WS / f"data/universe_withsex_{T}.csv"
def universe_ws(task): return pd.read_csv(UF).Gene.astype(str).tolist()
s5.universe = universe_ws
def w_cohort(c): return s5.native_cohort(T, c)
def w_fold(h): return s5.native_fold(T, h, False)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--only", default=""); ap.add_argument("--task", default="ATB_vs_OD"); a = ap.parse_args()
    for sub in ("data", "results/native", "logs"): (WS / sub).mkdir(parents=True, exist_ok=True)
    s5.log("Stage7 withsex start | python", platform.python_version(), "| workers", a.workers, "| cache", CACHE)
    if not UF.exists():
        d = load7(T); allg = set.intersection(*[set(d[c]["Z"].index.astype(str)) for c in s5.TASKS[T]["cohorts"]])
        pd.DataFrame({"Gene": sorted(g for g in allg if not s5.badname(g))}).to_csv(UF, index=False)
    coh = s5.TASKS[T]["cohorts"] if not a.only else a.only.split(",")
    J = [(c, (c,)) for c in coh if not s5.P("results", "native", f"{T}__{c}__cohort.json").exists()]
    s5.run_pool("withsex native_cohort", J, w_cohort, a.workers, lambda lab, r: s5.log("  done", r))
    if a.only: return
    J = [(h, (h,)) for h in coh if not s5.P("results", "native", f"{T}__fold_{h}.json").exists()]
    s5.run_pool("withsex native_fold", J, w_fold, a.workers, lambda lab, r: s5.log("  done", r))
    SEX = set(pd.read_csv(MT / "Stage5/data/sex_chromosome_genes.csv").Gene.astype(str)); rows = []
    for h in coh:
        w = json.load(open(s5.P("results", "native", f"{T}__fold_{h}.json")))["panel"]; n = json.load(open(S7 / "results/native" / f"{T}__fold_{h}.json"))["panel"]
        rows.append(dict(task=T, heldout=h, sex_genes_in_withsex_panel=";".join(g for g in w if g in SEX), n_sex_genes=sum(g in SEX for g in w),
                         genes_shared_with_primary_panel=len(set(w) & set(n)), only_in_withsex=";".join(g for g in w if g not in n), only_in_primary=";".join(g for g in n if g not in w)))
    pd.DataFrame(rows).to_csv(WS / f"sex_gene_exclusion_{T}.csv", index=False); s5.log("withsex finished"); print(pd.DataFrame(rows).to_string())
if __name__ == "__main__":
    main()
