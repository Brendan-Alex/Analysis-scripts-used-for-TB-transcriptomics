"""Stage11: how many genes does a transferable task signature need? Ablation of the cross-cohort selectors (Stage9/10) from 20 genes
down to 1, with the panel size chosen WITHOUT the held-out cohort, size-matched random panels at every size, and final per-task
signatures. Nothing upstream is modified; Stage5 code imported read-only; data/folds/universe/models/seeds are Stage7's (MLP alpha 10).

Rankings (training cohorts only):
  consistent   genes with the same direction of effect (AUC vs 0.5) in every ranking cohort, ordered by their minimum |AUC - 0.5|
  MI_meanrank  mean of per-cohort mutual-information ranks (Stage6/7 rankings, random_state 42)
Panel of size k = top-k of the ranking ("drop the least useful gene one at a time" with a training-only criterion).

Per task x outer fold x ranking x method (LR, RF, MLP alpha 10, singscore, DirAvg):
  outer curve   held-out AUC at k = 1..20 (ranking on all training cohorts)                         [description only]
  inner curve   leave-one-training-cohort-out: rank on the other training cohorts, score the left-out one; mean over them
  chosen k      smallest k whose mean inner AUC is within 0.01 of the best inner AUC (k = 1..20)  -> scored on the held-out cohort
Random panels: 50 draws per fold and size k = 1..20 from the Stage7 task universe, seed default_rng([42, task, fold, k, draw]) - the
Stage7 formula, so k = 20 reproduces Stage7's 20-gene random panels. All 5 methods.
Final signatures (deliverable): per task x ranking, rank on ALL cohorts of the task; k_final by the same inner rule run as
leave-one-cohort-out over all cohorts (primary method DirAvg; every method reported). Its expected performance is the outer LOCO estimate.
Canary: held-out cohort replaced by noise + random labels -> outer ranking and chosen k must be identical.
Usage (from the Stage11 folder): python scripts/run_stage11.py --workers 10"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import sys, json, pickle, argparse, warnings, platform, importlib.util, shutil
from pathlib import Path
import numpy as np, pandas as pd
S11 = Path(__file__).resolve().parents[1]; MT = S11.parent
S7 = Path(os.environ.get("STAGE7_DIR", str(MT / "Stage7"))); OUT = Path(os.environ.get("STAGE11_OUT", str(S11)))
os.environ["STAGE5_OUT"] = str(OUT)
_sp = importlib.util.spec_from_file_location("s5", str(MT / "Stage5/scripts/run_stage5.py")); s5 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(s5)
s5.load = lambda task: pickle.load(open(S7 / "data/cache" / f"{task}.pkl", "rb"))
s5.universe = lambda task: pd.read_csv(S7 / "data" / f"universe_{task}.csv").Gene.astype(str).tolist()
TASKS = s5.TASKS; TL = list(TASKS); P = s5.P; log = s5.log; A10 = 10.0; KMAX = 20; TOL = 0.01; RANKINGS = ["consistent", "MI_meanrank"]
METHODS = s5.RND; NRAND = int(os.environ.get("STAGE11_NRAND", "50"))
RANKDIR = lambda t: (MT / "Stage7/results/rank") if t == "ATB_vs_OD" else (MT / "Stage6/results/rank")
_D, _MI = {}, {}
def getD(t):
    if t not in _D: _D.clear(); _D[t] = s5.load(t)
    return _D[t]
def mi_rank(t, c):
    if (t, c) not in _MI: _MI[(t, c)] = pd.read_csv(RANKDIR(t) / f"{t}__{c}.csv.gz").set_index("Gene")["MI_rank"]
    return _MI[(t, c)]
def auc_rows(X, y):
    from scipy.stats import rankdata
    R = rankdata(X, axis=1); n1 = y.sum(); n0 = len(y) - n1; return (R[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
def ranking(t, coh, how, D):
    U = s5.universe(t)
    if how == "MI_meanrank":
        rk = pd.concat([mi_rank(t, c).reindex(U).rename(c) for c in coh], axis=1); return rk.mean(1).sort_values(kind="stable").index[:KMAX].tolist()
    a = np.column_stack([auc_rows(D[c]["Z"].loc[U].to_numpy(), D[c]["y"].to_numpy()) for c in coh]) - 0.5
    same = (np.sign(a) == np.sign(a[:, [0]])).all(1) & (a != 0).all(1); score = np.where(same, np.abs(a).min(1), -1)
    o = np.lexsort((np.array(U), -score)); return [U[i] for i in o[:KMAX]]
def scores(D, train, test, genes):
    out, diag, feats = s5.fit_eval(D, train, [test], genes, METHODS, alpha_fixed=A10); y = D[test]["y"].to_numpy()
    return {m: float(s5.auc(y, v[test])) for m, v in out.items()}, {m: [float(x) for x in v[test]] for m, v in out.items()}
def inner_choice(t, coh, how, D):
    """leave-one-cohort-out over 'coh' only; returns mean inner AUC per method per k and the chosen k per method"""
    cur = {m: np.zeros(KMAX) for m in METHODS}
    for i in coh:
        rest = [c for c in coh if c != i]; rk = ranking(t, rest, how, D)
        for k in range(1, KMAX + 1):
            a, _ = scores(D, rest, i, rk[:k])
            for m in METHODS: cur[m][k - 1] += a[m] / len(coh)
    choice = {m: int(1 + np.flatnonzero(cur[m] >= cur[m].max() - TOL)[0]) for m in METHODS}
    return {m: [round(float(x), 5) for x in v] for m, v in cur.items()}, choice
def noised(D, h):
    D = {c: dict(v) for c, v in D.items()}; rng = np.random.default_rng(7); x = D[h]
    x["Z"] = pd.DataFrame(rng.normal(size=x["Z"].shape), index=x["Z"].index, columns=x["Z"].columns); x["y"] = pd.Series(rng.integers(0, 2, len(x["y"])), index=x["y"].index); return D
def fold_job(t, h, how, canary=False):
    warnings.filterwarnings("ignore")
    D = getD(t); D = noised(D, h) if canary else D
    tr = [c for c in TASKS[t]["cohorts"] if c != h]; assert h not in tr
    assert not any(set(D[c]["Z"].columns) & set(D[h]["Z"].columns) for c in tr)
    rk = ranking(t, tr, how, D); inner, choice = inner_choice(t, tr, how, D)
    if canary: return dict(task=t, heldout=h, ranking=how, genes=rk, choice=choice, inner=inner)
    curve = []
    for k in range(1, KMAX + 1):
        a, s = scores(D, tr, h, rk[:k]); curve.append(dict(k=k, auc=a, scores=s))
    return dict(task=t, heldout=h, ranking=how, training=tr, genes=rk, inner=inner, choice=choice, curve=curve)
def random_job(t, h, k):
    warnings.filterwarnings("ignore")
    D = getD(t); tr = [c for c in TASKS[t]["cohorts"] if c != h]; U = s5.universe(t); res = []
    for d in range(NRAND):
        rng = np.random.default_rng([s5.SEED, TL.index(t), TASKS[t]["cohorts"].index(h), k, d]); g = list(rng.choice(U, size=k, replace=False))
        a, _ = scores(D, tr, h, g); res.append(a)
    return dict(task=t, heldout=h, k=k, auc={m: [r[m] for r in res] for m in METHODS})
def final_job(t, how):
    warnings.filterwarnings("ignore")
    D = getD(t); coh = TASKS[t]["cohorts"]; rk = ranking(t, coh, how, D); inner, choice = inner_choice(t, coh, how, D)
    return dict(task=t, ranking=how, cohorts=coh, genes=rk, inner=inner, choice=choice, primary_method="DirAvg", final_signature=rk[:choice["DirAvg"]])
def appender(f):
    def w(lab, r):
        with open(f, "a") as fh: fh.write(json.dumps(r) + "\n")
        with open(str(f) + ".done", "a") as fh: fh.write(lab + "\n")
    return w
def done(f): return set(open(str(f) + ".done").read().split("\n")) if Path(str(f) + ".done").exists() else set()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--step", default="all"); a = ap.parse_args(); W = a.workers; tasks = a.tasks.split(",")
    for sub in ("results", "qc", "logs", "data"): P(sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(MT / "Stage7/data/sex_chromosome_genes.csv", P("data", "sex_chromosome_genes.csv"))   # read by the Stage5 fit_eval
    import sklearn; log("Stage11 start | python", platform.python_version(), "| sklearn", sklearn.__version__, "| workers", W, "| Stage7 data", S7, "| random draws", NRAND)
    todo = ["folds", "final", "random", "canary"] if a.step == "all" else a.step.split(",")
    if "folds" in todo:
        f = P("results", "fold_curves.jsonl"); dn = done(f)
        s5.run_pool("fold ablation", [(f"{t}|{h}|{r}", (t, h, r)) for t in tasks for h in TASKS[t]["cohorts"] for r in RANKINGS if f"{t}|{h}|{r}" not in dn], fold_job, W, appender(f))
    if "final" in todo:
        f = P("results", "final_signatures.jsonl"); dn = done(f)
        s5.run_pool("final signatures", [(f"{t}|{r}", (t, r)) for t in tasks for r in RANKINGS if f"{t}|{r}" not in dn], final_job, W, appender(f))
    if "random" in todo:
        f = P("results", "random_by_size.jsonl"); dn = done(f)
        s5.run_pool("random panels", [(f"{t}|{h}|{k}", (t, h, k)) for t in tasks for h in TASKS[t]["cohorts"] for k in range(1, KMAX + 1) if f"{t}|{h}|{k}" not in dn], random_job, W, appender(f))
    if "canary" in todo:
        f = P("qc", "canary.jsonl"); dn = done(f)
        s5.run_pool("canary", [(f"{t}|{h}|{r}", (t, h, r, True)) for t in tasks for h in TASKS[t]["cohorts"] for r in RANKINGS if f"{t}|{h}|{r}" not in dn], fold_job, W, appender(f))
        real = {(r["task"], r["heldout"], r["ranking"]): (r["genes"], r["choice"], r["inner"]) for r in map(json.loads, open(P("results", "fold_curves.jsonl")))}
        rows = [dict(task=r["task"], heldout=r["heldout"], ranking=r["ranking"], genes_identical=real[(r["task"], r["heldout"], r["ranking"])][0] == r["genes"],
                     chosen_k_identical=real[(r["task"], r["heldout"], r["ranking"])][1] == r["choice"], inner_curve_identical=real[(r["task"], r["heldout"], r["ranking"])][2] == r["inner"])
                for r in map(json.loads, open(f))]
        pd.DataFrame(rows).to_csv(P("qc", "canary.csv"), index=False); log("canary: all identical =", all(all(v for k, v in x.items() if k.endswith("identical")) for x in rows))
    log("Stage11 finished requested steps:", todo)
if __name__ == "__main__":
    main()
