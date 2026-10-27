"""Stage10: the original Native pipeline with mRMR removed. Isolates the two suspected causes of the Native failure:
the mRMR shortlist step and the forward-selection step. Nothing upstream is modified; Stage5 code imported read-only;
data, folds, universe, models and seeds are Stage7's (MLP alpha fixed at 10). Gene choice uses training cohorts only.

Arms (20-gene panel per fold):
  MI20_FS          candidates = union of each training cohort's top-20 genes by mutual information (Stage6/7 rankings, random_state 42)
                   - i.e. the original per-cohort shortlist WITHOUT the mRMR redundancy step -> original forward selection + swaps
  consistent60_FS  candidates = the fold's 60 most consistent genes (same direction in every training cohort, ranked by the weakest
                   |AUC - 0.5|; 60 ~ the original candidate-set size) -> original forward selection + swaps
Forward selection + swaps = Stage5 Native step unchanged: seed = 5 genes with the best summed per-cohort mRMR rank within the
candidates; add genes maximising mean cross-cohort AUC (liblinear LR trained on one training cohort, scored on each other) to 20;
then single-gene swaps while the training cross-cohort AUC improves.
Compare with Stage9 (no forward selection): MI_meanrank, consistent; and Stage7 original Native.
Benchmark as Stage7/9: LR, RF, MLP (alpha 10), singscore, DirAvg on the held-out cohort; random reference in the summary = Stage7's
20-gene random panels. Canary: held-out cohort replaced by noise + random labels -> panels must be identical.
Usage (from the Stage10 folder): python scripts/run_stage10.py --workers 10 [--tasks ...]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import sys, json, time, pickle, argparse, warnings, platform, importlib.util, shutil
from pathlib import Path
import numpy as np, pandas as pd
S10 = Path(__file__).resolve().parents[1]; MT = S10.parent
S7 = Path(os.environ.get("STAGE7_DIR", str(MT / "Stage7"))); OUT = Path(os.environ.get("STAGE10_OUT", str(S10)))
os.environ["STAGE5_OUT"] = str(OUT)
_sp = importlib.util.spec_from_file_location("s5", str(MT / "Stage5/scripts/run_stage5.py")); s5 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(s5)
s5.load = lambda task: pickle.load(open(S7 / "data/cache" / f"{task}.pkl", "rb"))
s5.universe = lambda task: pd.read_csv(S7 / "data" / f"universe_{task}.csv").Gene.astype(str).tolist()
TASKS = s5.TASKS; P = s5.P; log = s5.log; A10 = 10.0; ARMS = ["MI20_FS", "consistent60_FS"]
RANKDIR = lambda t: (MT / "Stage7/results/rank") if t == "ATB_vs_OD" else (MT / "Stage6/results/rank")
_D = {}
def getD(t):
    if t not in _D: _D.clear(); _D[t] = s5.load(t)
    return _D[t]
def auc_rows(X, y):
    from scipy.stats import rankdata
    R = rankdata(X, axis=1); n1 = y.sum(); n0 = len(y) - n1; return (R[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
def candidates(t, tr, arm, D):
    U = s5.universe(t)
    if arm == "MI20_FS":
        out = set()
        for c in tr:
            r = pd.read_csv(RANKDIR(t) / f"{t}__{c}.csv.gz").set_index("Gene").loc[U]; out |= set(r.sort_values("MI_rank").index[:20])
        return sorted(out)
    a = np.column_stack([auc_rows(D[c]["Z"].loc[U].to_numpy(), D[c]["y"].to_numpy()) for c in tr]) - 0.5
    same = (np.sign(a) == np.sign(a[:, [0]])).all(1) & (a != 0).all(1); score = np.where(same, np.abs(a).min(1), -1)
    o = np.lexsort((np.array(U), -score)); return sorted(U[i] for i in o[:60])
def forward_select(t, tr, allowed, D, ck):
    """Stage5 native_fold forward selection + swaps, unchanged, on a given candidate list (training cohorts only)"""
    from sklearn.feature_selection import mutual_info_classif
    U = s5.universe(t); Dl = [(D[c]["Z"].loc[U].T, D[c]["y"].to_numpy()) for c in tr]
    def mrmr_rank(Xd, yd):
        mi = mutual_info_classif(Xd.to_numpy(), yd, random_state=0); return [Xd.columns[i] for i in s5.mrmr_topk(Xd.to_numpy(), mi, Xd.shape[1])]
    ranks = [mrmr_rank(Xd[allowed], yd) for Xd, yd in Dl]; score = {g: sum(r.index(g) for r in ranks) for g in allowed}
    if ck is not None and ck.exists(): st = json.load(open(ck)); panel, cur, swaps = st["panel"], st["cur"], st["swaps"]
    else:
        panel = sorted(allowed, key=lambda g: (score[g], g))[:5]
        while len(panel) < min(20, len(allowed)):
            bg, ba = None, -np.inf
            for g in sorted(set(allowed) - set(panel)):
                a = s5.cross_auc(Dl, panel + [g])
                if a > ba: bg, ba = g, a
            panel.append(bg)
        cur = s5.cross_auc(Dl, panel); swaps = 0
        if ck is not None: json.dump(dict(panel=panel, cur=cur, swaps=swaps), open(ck, "w"))
    improved = True
    while improved:
        improved = False
        for i in range(len(panel)):
            for gin in sorted(set(allowed) - set(panel)):
                trial = panel.copy(); trial[i] = gin; a = s5.cross_auc(Dl, trial)
                if a > cur + 1e-12:
                    panel, cur, improved = trial, a, True; swaps += 1
                    if ck is not None: json.dump(dict(panel=panel, cur=cur, swaps=swaps), open(ck, "w"))
                    break
            if improved: break
    return panel, cur, swaps
def job(t, h, arm, canary=False):
    warnings.filterwarnings("ignore")
    D = getD(t)
    if canary:
        D = {c: dict(v) for c, v in D.items()}; rng = np.random.default_rng(7); x = D[h]
        x["Z"] = pd.DataFrame(rng.normal(size=x["Z"].shape), index=x["Z"].index, columns=x["Z"].columns); x["y"] = pd.Series(rng.integers(0, 2, len(x["y"])), index=x["y"].index)
    tr = [c for c in TASKS[t]["cohorts"] if c != h]; assert h not in tr
    allowed = candidates(t, tr, arm, D)
    ck = None if canary else P("results", "checkpoints", f"{t}__{arm}__{h}.json")
    panel, cur, swaps = forward_select(t, tr, allowed, D, ck)
    if canary: return dict(task=t, heldout=h, arm=arm, panel=panel)
    out, diag, feats = s5.fit_eval(D, tr, [h], panel, s5.RND, alpha_fixed=A10); y = D[h]["y"].to_numpy()
    return dict(task=t, heldout=h, arm=arm, training=tr, n_candidates=len(allowed), candidates=allowed, panel=panel,
                train_cross_auc=cur, swaps=swaps, heldout_auc={m: float(s5.auc(y, v[h])) for m, v in out.items()}, heldout_scores={m: [float(x) for x in v[h]] for m, v in out.items()})
def canary_job(t, h, arm): return job(t, h, arm, canary=True)
def appender(f):
    def w(lab, r):
        with open(f, "a") as fh: fh.write(json.dumps(r) + "\n")
        with open(str(f) + ".done", "a") as fh: fh.write(lab + "\n")
    return w
def done(f): return set(open(str(f) + ".done").read().split("\n")) if Path(str(f) + ".done").exists() else set()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--step", default="all"); a = ap.parse_args(); W = a.workers; tasks = a.tasks.split(",")
    for sub in ("results/checkpoints", "qc", "logs", "data"): P(*sub.split("/")).mkdir(parents=True, exist_ok=True)
    shutil.copy(MT / "Stage7/data/sex_chromosome_genes.csv", P("data", "sex_chromosome_genes.csv"))   # read by the Stage5 fit_eval
    import sklearn; log("Stage10 start | python", platform.python_version(), "| sklearn", sklearn.__version__, "| workers", W, "| Stage7 data", S7)
    todo = ["select", "canary"] if a.step == "all" else a.step.split(",")
    if "select" in todo:
        f = P("results", "records.jsonl"); dn = done(f)
        J = [(f"{t}|{h}|{arm}", (t, h, arm)) for t in tasks for h in TASKS[t]["cohorts"] for arm in ARMS if f"{t}|{h}|{arm}" not in dn]
        J.sort(key=lambda j: j[1][0] == "LTB_vs_CON"); s5.run_pool("select+benchmark", J, job, W, appender(f))
    if "canary" in todo:
        f = P("qc", "canary.jsonl"); dn = done(f)
        J = [(f"{t}|{h}|{arm}", (t, h, arm)) for t in tasks for h in TASKS[t]["cohorts"] for arm in ARMS if f"{t}|{h}|{arm}" not in dn]
        s5.run_pool("canary", J, canary_job, W, appender(f))
        real = {(r["task"], r["heldout"], r["arm"]): r["panel"] for r in map(json.loads, open(P("results", "records.jsonl")))}
        rows = [dict(task=r["task"], heldout=r["heldout"], arm=r["arm"], panel_identical=real.get((r["task"], r["heldout"], r["arm"])) == r["panel"]) for r in map(json.loads, open(f))]
        pd.DataFrame(rows).to_csv(P("qc", "canary.csv"), index=False); log("canary: all identical =", all(x["panel_identical"] for x in rows))
    log("Stage10 finished requested steps:", todo)
if __name__ == "__main__":
    main()
