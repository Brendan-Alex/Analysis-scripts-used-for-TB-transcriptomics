"""Stage7 run (native, parallel). Reuses the Stage5 pipeline code read-only (Stage5/scripts/run_stage5.py), with:
  - GSE83456 read from Stage7/data/GSE83456_log2_genelevel_rebuilt.csv.gz (absolute levels; rebuilt by 01_rebuild_gse83456.py)
  - MLP at FIXED alpha = 10 as the primary 'MLP' (curated and random panels)
  - 'MLP_alpha' = tuned alpha on the Stage6 extended grid {1e-4..1000} by inner leave-one-cohort-out (comparison column)
  - everything else exactly as Stage5 (label-free per-cohort z, full-coverage universe minus X/Y, models, seeds, random draws)
ATB_vs_OD is rerun from scratch (cache, GSE83456 mRMR pool, Native panels + canaries, benchmark).
ATB_vs_LTB and LTB_vs_CON contain no GSE83456: their Native panels are copied unchanged from Stage5 (provenance in
results/native/COPIED_FROM_STAGE5.txt); their benchmark is rerun so MLP is at alpha 10 and LR/RF/singscore/DirAvg can be diffed against Stage5.
Usage (from the Stage7 folder): python scripts/run_stage7.py --workers 10"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import sys, json, time, shutil, argparse, warnings, platform, importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
S7 = Path(__file__).resolve().parents[1]; MT = S7.parent; S5 = MT / "Stage5"
os.environ["STAGE5_OUT"] = os.environ.get("STAGE7_OUT", str(S7))          # Stage5 code writes into Stage7
_sp = importlib.util.spec_from_file_location("s5", str(S5 / "scripts/run_stage5.py")); s5 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(s5)
OUT = s5.OUT; REBUILT = S7 / "data/GSE83456_log2_genelevel_rebuilt.csv.gz"
_expr5 = s5.expr
def expr7(c): return pd.read_csv(REBUILT, index_col=0) if c == "GSE83456" else _expr5(c)
s5.expr = expr7
s5.ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
ALPHA_FIXED = 10.0; TASKS = s5.TASKS; P = s5.P; log = s5.log
def w_native_cohort(t, c): return s5.native_cohort(t, c)
def w_native_fold(t, h, can): return s5.native_fold(t, h, can)
def w_canary(t, h): return s5.canary_bench(t, h)
def bench7(task, key):
    warnings.filterwarnings("ignore")
    D = s5._getD(task); t = TASKS[task]; parts = key.split("|"); kind, h, p = parts[0], parts[1], parts[2]
    train = s5.fold_train(task, D, h); genes = s5.panel_genes(task, h, p)
    if kind == "cur":
        out, diag, feats = s5.fit_eval(D, train, [h] + train, genes, s5.CUR, alpha_fixed=ALPHA_FIXED, held=h)
        rec = dict(key=key, kind=kind, heldout=h, panel=p, training=train, features=feats, diag=diag)
    else:
        k = int(parts[3]); size = len([g for g in genes if all(g in D[c]["Z"].index for c in train)])
        rng = np.random.default_rng([s5.SEED, s5.TASKLIST.index(task), t["cohorts"].index(h), size, k]); g2 = list(rng.choice(s5.universe(task), size=size, replace=False))
        out, diag, feats = s5.fit_eval(D, train, [h], g2, s5.RND, alpha_fixed=ALPHA_FIXED)
        diag.pop("directions", None); rec = dict(key=key, kind=kind, heldout=h, panel=p, draw=k, size=size, features=feats, diag=diag)
    rec["scores"] = {m: {c: [float(x) for x in s] for c, s in v.items()} for m, v in out.items()}
    return rec
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--step", default="all")
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); W = a.workers; tasks = a.tasks.split(",")
    for sub in ("data/cache", "results/native", "results/bench", "qc", "logs"): P(*sub.split("/")).mkdir(parents=True, exist_ok=True)
    import sklearn, scipy
    log("Stage7 start | python", platform.python_version(), "| numpy", np.__version__, "| pandas", pd.__version__, "| scipy", scipy.__version__, "| sklearn", sklearn.__version__,
        "| workers", W, "| cpu", os.cpu_count(), "|", platform.platform(), "| GSE83456 from", REBUILT.name, "| MLP alpha fixed", ALPHA_FIXED)
    steps = ["prepare", "native", "bench_curated", "bench_random", "canary_bench"]; todo = steps if a.step == "all" else a.step.split(",")
    if "prepare" in todo: s5.step_prepare(tasks)
    if "native" in todo:
        for t in tasks:
            if t == "ATB_vs_OD": continue
            for f in (S5 / "results/native").glob(f"{t}__*.json"):
                if "checkpoint" not in f.name and not P("results", "native", f.name).exists(): shutil.copy(f, P("results", "native", f.name))
            with open(P("results", "native", "COPIED_FROM_STAGE5.txt"), "a") as fh: fh.write(f"{t}: per-cohort pools, fold panels and canary panels copied unchanged from Stage5/results/native (task contains no GSE83456)\n")
        if "ATB_vs_OD" in tasks:
            t = "ATB_vs_OD"
            J = [(f"{t}|{c}", (t, c)) for c in TASKS[t]["cohorts"] if not P("results", "native", f"{t}__{c}__cohort.json").exists()]
            s5.run_pool("native_cohort", J, w_native_cohort, W, lambda lab, r: log("  done", r))
            J = [(f"{t}|{tag}|{h}", (t, h, tag == "canary")) for tag in ("fold", "canary") for h in TASKS[t]["cohorts"]
                 if not P("results", "native", f"{t}__{tag}_{h}.json").exists() and not (a.smoke and tag == "canary")]
            s5.run_pool("native_fold", J, w_native_fold, W, lambda lab, r: log("  done", r))
            rows = []
            for h in TASKS[t]["cohorts"]:
                f1, f2 = P("results", "native", f"{t}__fold_{h}.json"), P("results", "native", f"{t}__canary_{h}.json")
                if f1.exists() and f2.exists():
                    x, y = json.load(open(f1)), json.load(open(f2))
                    rows.append(dict(task=t, heldout=h, panel_identical=x["panel"] == y["panel"], cross_auc_identical=x["mean_cross_auc"] == y["mean_cross_auc"], allowed_identical=x["allowed"] == y["allowed"]))
            if rows: pd.DataFrame(rows).to_csv(P("qc", "canary_native_ATB_vs_OD.csv"), index=False); log("native canary ATB_vs_OD all identical =", all(all(r[k] for k in r if k not in ("task", "heldout")) for r in rows))
    for kind in ("bench_curated", "bench_random"):
        if kind not in todo: continue
        for t in tasks:
            rec_f, done_f = P("results", "bench", f"{t}__records.jsonl"), P("results", "bench", f"{t}__done.txt")
            done = set(open(done_f).read().split("\n")) if done_f.exists() else set()
            nd = 2 if a.smoke else 50
            keys = [f"cur|{h}|{p}" for h in TASKS[t]["cohorts"] for p in TASKS[t]["panels"]] if kind == "bench_curated" else \
                   [f"rnd|{h}|{p}|{k}" for h in TASKS[t]["cohorts"] for p in TASKS[t]["panels"] for k in range(nd)]
            J = [(k, (t, k)) for k in keys if k not in done]
            def write(lab, rec, rec_f=rec_f, done_f=done_f):
                with open(rec_f, "a") as fh: fh.write(json.dumps(rec) + "\n")
                with open(done_f, "a") as fh: fh.write(lab + "\n")
            s5.run_pool(f"{kind} {t}", J, bench7, W, write)
    if "canary_bench" in todo:
        f = P("qc", "canary_bench.jsonl"); have = {(r["task"], r["heldout"]) for r in map(json.loads, open(f))} if f.exists() else set()
        J = [(f"{t}|{h}", (t, h)) for t in tasks if t == "ATB_vs_OD" for h in TASKS[t]["cohorts"] if (t, h) not in have]
        def w(lab, r):
            with open(f, "a") as fh: fh.write(json.dumps(r) + "\n")
            log("  canary", r["task"], r["heldout"], "PASS" if all(v for k, v in r.items() if k not in ("task", "heldout")) else f"FAIL {r}")
        s5.run_pool("canary_bench", J, w_canary, W, w)
    log("Stage7 finished requested steps:", todo)
if __name__ == "__main__":
    main()
