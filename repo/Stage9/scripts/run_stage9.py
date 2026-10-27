"""Stage9: is the Native panel noise, and does a corrected selector transfer? Runs natively, in parallel. Nothing upstream is modified.
Data, folds, universe, normalisation, models and seeds are exactly Stage7's (Stage7 caches, rebuilt GSE83456, MLP alpha fixed at 10);
the Stage5 pipeline code is imported read-only. Everything is chosen on the fold's TRAINING cohorts only; the held-out cohort is
only ever scored.

PART A - gene ablation (panels: Native fold panel, Zak16, Kaforou [ATB tasks]; every fold)
  Removal orders, all decided on training cohorts only:
    backward    remove, one at a time, the gene whose removal hurts the training cross-cohort AUC least
                (cross-cohort AUC = Stage5 Native objective: liblinear LR trained on one training cohort, scored on each other)
    univariate  remove first the gene with the weakest separation in the training cohorts (mean |AUC - 0.5| over training cohorts)
    random      50 random removal orders (seed [42, 9, task, fold, panel, order]) = null for "any gene removed"
  At every size k = n..1: training cross-cohort AUC and held-out AUC. Ordered removals: LR, RF, MLP (alpha 10), singscore, DirAvg
  (held-out scores kept for CIs). Random orders: LR, singscore, DirAvg (AUC only).
  Reading: if held-out AUC stays flat or rises while training AUC falls as genes go, the removed genes were fitting noise.

PART B - corrected selectors (20 genes, per fold, training cohorts only; universe = Stage7 task universe)
    MI_meanrank      mean of per-training-cohort mutual-information ranks (Stage6/7 rankings, MI random_state 42) -> top 20
    mRMR_scaled      Stage5 Native algorithm unchanged EXCEPT relevance is MI / max(MI) in that cohort, putting it on the same
                     0-1 scale as the |r| redundancy term; per-cohort top-20 pools -> Stage5 forward selection + swaps
    consistent       genes with the same direction of effect in every training cohort, ranked by their minimum |AUC - 0.5|
                     across training cohorts -> top 20
  Each panel is benchmarked like Stage7 (LR, RF, MLP alpha 10, singscore, DirAvg; held-out scores for CIs). The random-panel
  reference is Stage7's 20-gene random panels for the same fold (same universe, size and seeds) - read in the summary step.
  Leakage canary for the selectors: held-out cohort replaced by noise + random labels -> selected panels must be identical.

Usage (from the Stage9 folder):  python scripts/run_stage9.py --workers 10        [--step ablation|selectors|canary] [--tasks ...]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import sys, json, time, pickle, argparse, warnings, platform, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
S9 = Path(__file__).resolve().parents[1]; MT = S9.parent
S7 = Path(os.environ.get("STAGE7_DIR", str(MT / "Stage7")))                 # caches + universes (read-only)
OUT = Path(os.environ.get("STAGE9_OUT", str(S9)))
os.environ["STAGE5_OUT"] = str(OUT)
_sp = importlib.util.spec_from_file_location("s5", str(MT / "Stage5/scripts/run_stage5.py")); s5 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(s5)
def _load(task): return pickle.load(open(S7 / "data/cache" / f"{task}.pkl", "rb"))
def _universe(task): return pd.read_csv(S7 / "data" / f"universe_{task}.csv").Gene.astype(str).tolist()
s5.load = _load; s5.universe = _universe
TASKS = s5.TASKS; TL = list(TASKS); SEED = 42; P = s5.P; log = s5.log; A10 = 10.0
NATIVE = lambda t: (MT / "Stage7/results/native")                           # Stage7 native panels (ATB_vs_OD rebuilt; others = Stage5 copies)
RANKDIR = lambda t: (MT / "Stage7/results/rank") if t == "ATB_vs_OD" else (MT / "Stage6/results/rank")
ABL_PANELS = {"ATB_vs_LTB": ["Native", "Zak16", "Kaforou_ATB_v_LTB"], "ATB_vs_OD": ["Native", "Zak16", "Kaforou_ATB_v_OD"], "LTB_vs_CON": ["Native", "Zak16"]}
NRAND = int(os.environ.get("STAGE9_NRAND", "50"))
_D = {}
def getD(t):
    if t not in _D: _D.clear(); _D[t] = s5.load(t)
    return _D[t]
def train_of(t, h, D):
    tr = [c for c in TASKS[t]["cohorts"] if c != h]; assert h not in tr
    assert not any(set(D[c]["Z"].columns) & set(D[h]["Z"].columns) for c in tr); return tr
def auc_rows(X, y):
    """per-gene AUC for a genes x samples matrix (rank based, ties averaged)"""
    from scipy.stats import rankdata
    R = rankdata(X, axis=1); n1 = y.sum(); n0 = len(y) - n1; return (R[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
def panel_genes(t, h, p, D, tr):
    g = json.load(open(NATIVE(t) / f"{t}__fold_{h}.json"))["panel"] if p == "Native" else s5.panel(p)
    return [x for x in g if all(x in D[c]["Z"].index for c in tr)]
def heldout_eval(D, tr, h, genes, methods):
    out, diag, feats = s5.fit_eval(D, tr, [h], genes, methods, alpha_fixed=A10)
    y = D[h]["y"].to_numpy(); return {m: float(s5.auc(y, v[h])) for m, v in out.items()}, {m: [float(x) for x in v[h]] for m, v in out.items()}
def train_cross(D, tr, genes):
    Dl = [(D[c]["Z"].reindex(genes).T.fillna(0.0), D[c]["y"].to_numpy()) for c in tr]; return s5.cross_auc(Dl, genes)
# ------------------------------------------------------------------ PART A
def ablation_job(t, h, p, order, k_rand):
    warnings.filterwarnings("ignore")
    D = getD(t); tr = train_of(t, h, D); genes = panel_genes(t, h, p, D, tr); cur = list(genes); seq = []
    if order == "univariate":
        sep = {g: np.mean([abs(auc_rows(D[c]["Z"].loc[[g]].to_numpy(), D[c]["y"].to_numpy())[0] - 0.5) for c in tr]) for g in genes}
        drop_order = sorted(genes, key=lambda g: (sep[g], g))
    elif order == "random":
        rng = np.random.default_rng([SEED, 9, TL.index(t), TASKS[t]["cohorts"].index(h), ABL_PANELS[t].index(p), k_rand]); drop_order = list(rng.permutation(genes))
    full = order != "random"; methods = s5.RND if full else ["LR", "singscore", "DirAvg"]
    while cur:
        a, sc = heldout_eval(D, tr, h, cur, methods)
        rec = dict(task=t, heldout=h, panel=p, order=order, rand=k_rand, k=len(cur), genes=list(cur), train_cross_auc=train_cross(D, tr, cur) if len(cur) >= 1 else None, heldout_auc=a)
        if full: rec["heldout_scores"] = sc
        seq.append(rec)
        if len(cur) == 1: break
        if order == "backward":
            best = None
            for g in sorted(cur):
                rest = [x for x in cur if x != g]; v = train_cross(D, tr, rest)
                if best is None or v > best[1] + 1e-12: best = (g, v)
            cur.remove(best[0]); rec["removed_next"] = best[0]
        else:
            nxt = [x for x in drop_order if x in cur][0]; cur.remove(nxt); rec["removed_next"] = nxt
    return seq
# ------------------------------------------------------------------ PART B
def mi_table(t, c): return pd.read_csv(RANKDIR(t) / f"{t}__{c}.csv.gz").set_index("Gene")
def scaled_pool(t, c):
    """Stage5 per-cohort step with relevance rescaled to 0-1; MI values identical to Stage5/6/7 (random_state 42)"""
    D = getD(t); U = s5.universe(t); X = D[c]["Z"].loc[U]; mi = mi_table(t, c).loc[U, "MI"].to_numpy()
    top = np.argsort(-mi, kind="stable")[:2000]; A = X.T.to_numpy()[:, top]; rel = mi[top] / mi[top].max()
    sel = s5.mrmr_topk(A, rel, 20); r = dict(task=t, cohort=c, relevance="MI/max(MI)", mrmr_top20=[U[top[i]] for i in sel])
    json.dump(r, open(P("results", "native", f"{t}__{c}__cohort.json"), "w"), indent=1); return r
def selector_job(t, h, sel, canary=False):
    warnings.filterwarnings("ignore")
    D = getD(t)
    if canary:
        D = {c: dict(v) for c, v in D.items()}; rng = np.random.default_rng(7); x = D[h]
        x["Z"] = pd.DataFrame(rng.normal(size=x["Z"].shape), index=x["Z"].index, columns=x["Z"].columns); x["y"] = pd.Series(rng.integers(0, 2, len(x["y"])), index=x["y"].index)
    tr = train_of(t, h, D); U = s5.universe(t)
    if sel == "MI_meanrank":
        rk = pd.concat([mi_table(t, c).loc[U, "MI_rank"].rename(c) for c in tr], axis=1); panel = rk.mean(1).sort_values(kind="stable").index[:20].tolist()
    elif sel == "consistent":
        a = np.column_stack([auc_rows(D[c]["Z"].loc[U].to_numpy(), D[c]["y"].to_numpy()) for c in tr]) - 0.5
        same = (np.sign(a) == np.sign(a[:, [0]])).all(1) & (a != 0).all(1); score = np.where(same, np.abs(a).min(1), -1)
        o = np.lexsort((np.array(U), -score)); panel = [U[i] for i in o[:20]]
    else:   # mRMR_scaled: Stage5 forward selection + swaps on the scaled pools (training cohorts only); canary = Stage5's own canary mode
        tag = "canary" if canary else "fold"; f = P("results", "native", f"{t}__{tag}_{h}.json")
        if not f.exists(): s5.native_fold(t, h, canary)
        panel = json.load(open(f))["panel"]
    if canary: return dict(task=t, heldout=h, selector=sel, panel=panel)
    a, sc = heldout_eval(D, tr, h, panel, s5.RND)
    return dict(task=t, heldout=h, selector=sel, training=tr, panel=panel, train_cross_auc=train_cross(D, tr, panel), heldout_auc=a, heldout_scores=sc)
def canary_selector(t, h, sel): return selector_job(t, h, sel, canary=True)
# ------------------------------------------------------------------ driver
def run_pool(name, jobs, fn, W, on_result): s5.run_pool(name, jobs, fn, W, on_result)
def appender(f):
    def w(lab, r):
        rows = r if isinstance(r, list) else [r]
        with open(f, "a") as fh: fh.write("".join(json.dumps(x) + "\n" for x in rows))
        with open(str(f) + ".done", "a") as fh: fh.write(lab + "\n")
    return w
def done(f): return set(open(str(f) + ".done").read().split("\n")) if Path(str(f) + ".done").exists() else set()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--step", default="all")
    ap.add_argument("--tasks", default=",".join(TASKS)); a = ap.parse_args(); W = a.workers; tasks = a.tasks.split(",")
    for sub in ("results/ablation", "results/selectors", "results/native", "qc", "logs", "data"): P(*sub.split("/")).mkdir(parents=True, exist_ok=True)
    import shutil; shutil.copy(MT / "Stage7/data/sex_chromosome_genes.csv", P("data", "sex_chromosome_genes.csv"))   # read by the Stage5 fit_eval
    import sklearn, scipy
    log("Stage9 start | python", platform.python_version(), "| numpy", np.__version__, "| pandas", pd.__version__, "| sklearn", sklearn.__version__, "| workers", W, "| Stage7 data", S7)
    todo = ["selectors", "canary", "ablation"] if a.step == "all" else a.step.split(",")
    if "selectors" in todo:
        J = [(f"{t}|{c}", (t, c)) for t in tasks for c in TASKS[t]["cohorts"] if not P("results", "native", f"{t}__{c}__cohort.json").exists()]
        run_pool("mRMR_scaled pools", J, scaled_pool, W, lambda lab, r: None)
        f = P("results", "selectors", "selector_records.jsonl"); dn = done(f)
        J = [(f"{t}|{h}|{s}", (t, h, s)) for s in ["mRMR_scaled", "MI_meanrank", "consistent"] for t in tasks for h in TASKS[t]["cohorts"] if f"{t}|{h}|{s}" not in dn]
        run_pool("selectors", J, selector_job, W, appender(f))
    if "canary" in todo:
        f = P("qc", "canary_selectors.jsonl"); dn = done(f)
        J = [(f"{t}|{h}|{s}", (t, h, s)) for s in ["mRMR_scaled", "MI_meanrank", "consistent"] for t in tasks for h in TASKS[t]["cohorts"] if f"{t}|{h}|{s}" not in dn]
        run_pool("selector canary", J, canary_selector, W, appender(f))
        real = {(r["task"], r["heldout"], r["selector"]): r["panel"] for r in map(json.loads, open(P("results", "selectors", "selector_records.jsonl")))}
        rows = [dict(task=r["task"], heldout=r["heldout"], selector=r["selector"], panel_identical=real.get((r["task"], r["heldout"], r["selector"])) == r["panel"])
                for r in map(json.loads, open(f))]
        pd.DataFrame(rows).to_csv(P("qc", "canary_selectors.csv"), index=False); log("selector canary: all identical =", all(x["panel_identical"] for x in rows))
    if "ablation" in todo:
        f = P("results", "ablation", "ablation_records.jsonl"); dn = done(f)
        J = [(f"{t}|{h}|{p}|{o}|0", (t, h, p, o, 0)) for t in tasks for h in TASKS[t]["cohorts"] for p in ABL_PANELS[t] for o in ("backward", "univariate")]
        J += [(f"{t}|{h}|{p}|random|{k}", (t, h, p, "random", k)) for t in tasks for h in TASKS[t]["cohorts"] for p in ABL_PANELS[t] for k in range(NRAND)]
        J = [j for j in J if j[0] not in dn]; J.sort(key=lambda j: j[1][3] == "random")
        run_pool("ablation", J, ablation_job, W, appender(f))
    log("Stage9 finished requested steps:", todo)
if __name__ == "__main__":
    main()
