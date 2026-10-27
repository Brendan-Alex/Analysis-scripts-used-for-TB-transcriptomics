"""Stage6 (continues Stage5; nothing in Stage5 is modified - its caches, panels and code are imported read-only).
Steps (each resumable; finished jobs are skipped on re-run):
  rank        per task x cohort: mutual information over the full task universe (random_state 42, as Stage5) and the full mRMR
              order within the MI top-2000 (Stage5's top-20 pool = first 20 of this order; checked).
  related     per task x cohort pair: platform-robust relatedness (see related_job docstring).
  alpha       per task x fold x panel (Stage5 panels): MLP alpha grid extended to {1e-4,1e-3,1e-2,1e-1,1,10,100,1000};
              inner leave-one-cohort-out AUC (per inner cohort) and held-out AUC at every alpha; chosen = best inner mean AUC.
  seeds       per fold: the 5 seed genes of the Native forward selection (recomputed exactly as Stage5) - provenance only.
  sib_native  sibling-exclusion sensitivity: LTB_vs_CON GSE19439 held out with GSE19444 removed from training (and the mirror,
              GSE19444 held out without GSE19439); Native re-selected on the remaining training cohorts (Stage5 algorithm/universe).
  rand_mlp    random-panel MLP refitted at the newly chosen alpha (same Stage5 gene draws) where the chosen alpha changed.
  sib_bench   sibling folds: all panels + Stage5's own Native panel, all six methods (MLP_alpha on the extended grid),
              50 size-matched random draws per panel.
Usage (from the Stage6 folder): python scripts/run_stage6.py --workers 10   [--step NAME] [--tasks ...]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import sys, json, time, argparse, warnings, platform, importlib.util, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
S6 = Path(__file__).resolve().parents[1]; MT = S6.parent
S5 = Path(os.environ.get("STAGE5_DIR", str(MT / "Stage5")))          # read-only
OUT = Path(os.environ.get("STAGE6_OUT", str(S6)))
os.environ["STAGE5_OUT"] = str(S5)
_spec = importlib.util.spec_from_file_location("s5", str(MT / "Stage5/scripts/run_stage5.py")); s5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s5)
ALPHAS8 = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]; s5.ALPHAS = ALPHAS8      # MLP_alpha inside s5.fit_eval uses the extended grid (sib_bench)
SEED = 42; TASKS = s5.TASKS; TASKLIST = list(TASKS)
SIB = {"GSE19439": "GSE19444", "GSE19444": "GSE19439"}      # held-out -> sibling removed from training (LTB_vs_CON)
def P(*a): return OUT.joinpath(*a)
def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(msg, flush=True)
    with open(P("logs", "run.log"), "a", encoding="utf-8") as fh: fh.write(msg + "\n")
_D = {}
def getD(task):
    if task not in _D: _D.clear(); _D[task] = s5.load(task)
    return _D[task]
def s5panel(task, held, p): return json.load(open(S5 / "results/native" / f"{task}__fold_{held}.json"))["panel"] if p == "Native" else s5.panel(p)
# ------------------------------------------------------------------ rank
def rank_job(task, c):
    warnings.filterwarnings("ignore")
    from sklearn.feature_selection import mutual_info_classif
    d = getD(task)[c]; U = s5.universe(task); X = d["Z"].loc[U]; A = X.T.to_numpy(); y = d["y"].to_numpy()
    mi = mutual_info_classif(A, y, random_state=SEED); order = np.argsort(-mi, kind="stable"); top = order[:2000]
    B = A[:, top]; C = np.nan_to_num(np.abs(np.corrcoef(B.T))); np.fill_diagonal(C, 0); m = mi[top]
    sel, rem = [], list(range(len(top)))           # exact Stage5 mRMR (mean |r| to selected), full length
    acc = np.zeros(len(top))
    for _ in range(len(top)):
        r = np.array(rem); s = m[r] if not sel else m[r] - acc[r] / len(sel)
        b = int(r[int(np.argmax(s))]); sel.append(b); rem.remove(b); acc += C[:, b]
    mi_rank = np.empty(len(U), int); mi_rank[order] = np.arange(1, len(U) + 1)
    mr = {U[top[i]]: k + 1 for k, i in enumerate(sel)}
    pool5 = json.load(open(S5 / "results/native" / f"{task}__{c}__cohort.json"))["mrmr_top20"]
    df = pd.DataFrame({"Gene": U, "MI": mi, "MI_rank": mi_rank, "mRMR_rank_in_top2000": [mr.get(g, np.nan) for g in U]})
    df.to_csv(P("results", "rank", f"{task}__{c}.csv.gz"), index=False)
    return dict(task=task, cohort=c, top20_matches_stage5=[U[top[i]] for i in sel[:20]] == pool5)
# ------------------------------------------------------------------ relatedness
def related_job(task):
    """Per cohort pair on the task universe (genes measured in every cohort; X/Y and LOC-type names excluded):
       mean_profile_spearman  Spearman r between the two cohorts' per-gene mean log2 expression (all task samples)
       sd_profile_spearman    same for per-gene SD
       signal_spearman        Spearman r between per-gene class effects (pos-minus-neg mean of label-free z, divided by pooled SD)
       signal_top200_jaccard  overlap of the 200 genes with largest |effect| in each cohort
       cohort_origin_AUC      5-fold CV logistic regression telling the two cohorts' samples apart, on within-sample ranks of the
                              2000 most variable universe genes (1.0 = trivially separable; 0.5 = indistinguishable)
       max_cross_sample_r     largest Pearson r between any sample of A and any sample of B on within-sample ranks of all universe genes (duplicate check)
       same_platform          from GEO platform of the cohort files (filled in the summary)"""
    warnings.filterwarnings("ignore")
    from scipy.stats import spearmanr, rankdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    D = getD(task); U = s5.universe(task); coh = TASKS[task]["cohorts"]; st = {}
    for c in coh:
        L = D[c]["L"].loc[U]; Z = D[c]["Z"].loc[U]; y = D[c]["y"].to_numpy()
        e = (Z.loc[:, y == 1].mean(1) - Z.loc[:, y == 0].mean(1)) / np.sqrt((Z.loc[:, y == 1].var(1) + Z.loc[:, y == 0].var(1)) / 2).replace(0, np.nan)
        Rk = pd.DataFrame(rankdata(L.to_numpy(), axis=0) / L.shape[0], index=U, columns=L.columns)
        st[c] = dict(mean=L.mean(1), sd=L.std(1), eff=e.fillna(0), R=Rk)
    out = []
    for a, b in itertools.combinations(coh, 2):
        A_, B_ = st[a], st[b]; top = lambda e: set(e.abs().sort_values(ascending=False).index[:200])
        RR = pd.concat([A_["R"], B_["R"]], axis=1); v = RR.var(1).sort_values(ascending=False).index[:2000]
        X = RR.loc[v].T.to_numpy(); yy = np.r_[np.zeros(A_["R"].shape[1]), np.ones(B_["R"].shape[1])]
        pr = cross_val_predict(LogisticRegression(max_iter=2000), X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        Xa, Xb = A_["R"].to_numpy(), B_["R"].to_numpy()                 # all universe genes for the duplicate check
        Xa = (Xa - Xa.mean(0)) / Xa.std(0); Xb = (Xb - Xb.mean(0)) / Xb.std(0); cr = Xa.T @ Xb / Xa.shape[0]
        i, j = np.unravel_index(np.argmax(cr), cr.shape)
        wa = Xa.T @ Xa / Xa.shape[0]; wb = Xb.T @ Xb / Xb.shape[0]; iu = lambda w: w[np.triu_indices(w.shape[0], 1)]
        out.append(dict(task=task, cohort_a=a, cohort_b=b, n_a=A_["R"].shape[1], n_b=B_["R"].shape[1], genes=len(U),
                        mean_profile_spearman=round(float(spearmanr(A_["mean"], B_["mean"])[0]), 4), sd_profile_spearman=round(float(spearmanr(A_["sd"], B_["sd"])[0]), 4),
                        signal_spearman=round(float(spearmanr(A_["eff"], B_["eff"])[0]), 4), signal_top200_jaccard=round(len(top(A_["eff"]) & top(B_["eff"])) / len(top(A_["eff"]) | top(B_["eff"])), 4),
                        cohort_origin_AUC=round(float(roc_auc_score(yy, pr)), 4), max_cross_sample_r=round(float(cr.max()), 4),
                        max_pair=f"{A_['R'].columns[i]}~{B_['R'].columns[j]}", median_cross_sample_r=round(float(np.median(cr)), 4),
                        median_within_a_r=round(float(np.median(iu(wa))), 4), median_within_b_r=round(float(np.median(iu(wb))), 4), max_within_a_r=round(float(iu(wa).max()), 4), max_within_b_r=round(float(iu(wb).max()), 4), n_cross_pairs_r_gt_0_95=int((cr > 0.95).sum())))
    return out
# ------------------------------------------------------------------ alpha grid
def alpha_job(task, held, p):
    warnings.filterwarnings("ignore")
    D = getD(task); train = s5.fold_train(task, D, held); genes = s5panel(task, held, p)
    feats = [g for g in genes if all(g in D[c]["Z"].index for c in train)]; X, y = s5.stack(D, train, feats)
    inner, per, held_auc, held_scores, sd, it, iit = {}, {}, {}, {}, {}, {}, {}
    for a in ALPHAS8:
        aucs = {}; its = []
        for h in train:
            tr = [c for c in train if c != h]; Xi, yi = s5.stack(D, tr, feats); m = s5.make("MLP", a).fit(Xi, yi); its.append(int(m.n_iter_))
            aucs[h] = s5.auc(D[h]["y"], s5.predict(m, D, h, feats))
        per[str(a)] = aucs; inner[str(a)] = float(np.mean(list(aucs.values()))); iit[str(a)] = its
        m = s5.make("MLP", a).fit(X, y); s = s5.predict(m, D, held, feats); it[str(a)] = int(m.n_iter_)
        held_auc[str(a)] = s5.auc(D[held]["y"], s); held_scores[str(a)] = [float(x) for x in s]; sd[str(a)] = float(np.std(s))
    best = None
    for a in ALPHAS8:                                   # same rule as Stage5: first alpha with strictly greater mean inner AUC
        if best is None or inner[str(a)] > best[1] + 1e-12: best = (a, inner[str(a)])
    return dict(task=task, heldout=held, panel=p, training=train, n_features=len(feats), chosen_alpha=best[0], inner_AUC=inner, inner_AUC_per_cohort=per,
                heldout_AUC=held_auc, heldout_scores=held_scores, heldout_pred_sd=sd, n_iter_full=it, n_iter_inner=iit)
# ------------------------------------------------------------------ Native helpers (Stage5 algorithm, training list given explicitly)
def native_select(task, train, held, ck):
    warnings.filterwarnings("ignore")
    from sklearn.feature_selection import mutual_info_classif
    data = getD(task); U = s5.universe(task); assert held not in train
    Dl = [(data[c]["Z"].loc[U].T, data[c]["y"].to_numpy()) for c in train]
    pools = {c: json.load(open(S5 / "results/native" / f"{task}__{c}__cohort.json"))["mrmr_top20"] for c in train}
    allowed = sorted(set().union(*pools.values()))
    def mrmr_rank(Xd, yd):
        mi = mutual_info_classif(Xd.to_numpy(), yd, random_state=0); return [Xd.columns[i] for i in s5.mrmr_topk(Xd.to_numpy(), mi, Xd.shape[1])]
    ranks = [mrmr_rank(Xd[allowed], yd) for Xd, yd in Dl]; score = {g: sum(r.index(g) for r in ranks) for g in allowed}
    seed = sorted(allowed, key=lambda g: (score[g], g))[:5]
    if ck is None: return dict(seed=seed, allowed=allowed, pools=pools)
    if ck.exists(): st = json.load(open(ck)); panel_, cur, swaps = st["panel"], st["cur"], st["swaps"]
    else:
        panel_ = list(seed)
        while len(panel_) < min(20, len(allowed)):
            bg, ba = None, -np.inf
            for g in sorted(set(allowed) - set(panel_)):
                a = s5.cross_auc(Dl, panel_ + [g])
                if a > ba: bg, ba = g, a
            panel_.append(bg)
        cur = s5.cross_auc(Dl, panel_); swaps = 0; json.dump(dict(panel=panel_, cur=cur, swaps=swaps), open(ck, "w"))
    improved = True
    while improved:
        improved = False
        for i in range(len(panel_)):
            for gin in sorted(set(allowed) - set(panel_)):
                trial = panel_.copy(); trial[i] = gin; a = s5.cross_auc(Dl, trial)
                if a > cur + 1e-12:
                    panel_, cur, improved = trial, a, True; swaps += 1; json.dump(dict(panel=panel_, cur=cur, swaps=swaps), open(ck, "w")); break
            if improved: break
    return dict(task=task, heldout=held, training_cohorts=train, allowed=len(allowed), seed=seed, panel=panel_, mean_cross_auc=cur, swaps=swaps)
def seeds_job(task, held):
    train = [c for c in TASKS[task]["cohorts"] if c != held]; r = native_select(task, train, held, None)
    return dict(task=task, heldout=held, seed=r["seed"], allowed=len(r["allowed"]))
def sib_native_job(held):
    task = "LTB_vs_CON"; train = [c for c in TASKS[task]["cohorts"] if c not in (held, SIB[held])]
    r = native_select(task, train, held, P("results", "sibling", f"native_{held}_without_{SIB[held]}.checkpoint.json"))
    r["excluded_sibling"] = SIB[held]; return r
# ------------------------------------------------------------------ random MLP at new alpha (Stage5 draws reproduced)
def rand_mlp_job(task, held, p, alpha):
    warnings.filterwarnings("ignore")
    D = getD(task); t = TASKS[task]; train = s5.fold_train(task, D, held); genes = s5panel(task, held, p)
    size = len([g for g in genes if all(g in D[c]["Z"].index for c in train)]); U = s5.universe(task); res = []
    for k in range(50):
        rng = np.random.default_rng([SEED, TASKLIST.index(task), t["cohorts"].index(held), size, k]); g2 = list(rng.choice(U, size=size, replace=False))
        X, y = s5.stack(D, train, g2); m = s5.make("MLP", alpha).fit(X, y); res.append(s5.auc(D[held]["y"], s5.predict(m, D, held, g2)))
    return dict(task=task, heldout=held, panel=p, alpha=alpha, size=size, random_MLP_AUC=res)
# ------------------------------------------------------------------ sibling-fold benchmark
def sib_bench_job(held, p, kind, k):
    warnings.filterwarnings("ignore")
    task = "LTB_vs_CON"; D = getD(task); train = [c for c in TASKS[task]["cohorts"] if c not in (held, SIB[held])]
    assert held not in train and SIB[held] not in train
    if p == "Native_reselected": genes = json.load(open(P("results", "sibling", f"native_{held}_without_{SIB[held]}.json")))["panel"]
    elif p == "Native_stage5": genes = json.load(open(S5 / "results/native" / f"{task}__fold_{held}.json"))["panel"]
    else: genes = s5.panel(p)
    if kind == "cur":
        out, diag, feats = s5.fit_eval(D, train, [held] + train, genes, s5.CUR, held=held)
        rec = dict(key=f"cur|{held}|{p}", kind=kind, heldout=held, excluded_sibling=SIB[held], panel=p, training=train, features=feats, diag=diag)
    else:
        alpha = float(kind.split("@")[1]); size = len([g for g in genes if all(g in D[c]["Z"].index for c in train)])
        rng = np.random.default_rng([SEED, 6, TASKS[task]["cohorts"].index(held), size, k]); g2 = list(rng.choice(s5.universe(task), size=size, replace=False))
        out, diag, feats = s5.fit_eval(D, train, [held], g2, s5.RND, alpha_fixed=alpha); diag.pop("directions", None)
        rec = dict(key=f"rnd|{held}|{p}|{k}", kind="rnd", heldout=held, excluded_sibling=SIB[held], panel=p, draw=k, size=size, features=feats, diag=diag)
    rec["scores"] = {m: {c: [float(x) for x in s] for c, s in v.items()} for m, v in out.items()}; return rec
SIB_PANELS = ["Native_reselected", "Native_stage5", "Zak16", "Berry86", "Berry86_noTLR7", "Sweeney3"]
# ------------------------------------------------------------------ driver
def run_pool(name, jobs, fn, W, on_result):
    if not jobs: log(name, ": nothing to do"); return
    log(name, ":", len(jobs), "jobs on", W, "workers"); t0 = time.time(); n = 0
    with ProcessPoolExecutor(max_workers=W) as ex:
        fut = {ex.submit(fn, *args): lab for lab, args in jobs}
        for f in as_completed(fut):
            on_result(fut[f], f.result()); n += 1
            if n % max(1, len(jobs) // 10) == 0 or n == len(jobs):
                el = time.time() - t0; log(f"{name}: {n}/{len(jobs)} done, {el/60:.1f} min elapsed, ~{el/n*(len(jobs)-n)/60:.1f} min left")
def appender(f):
    def w(lab, r):
        with open(f, "a") as fh: fh.write(json.dumps(r) + "\n")
        with open(str(f) + ".done", "a") as fh: fh.write(lab + "\n")
    return w
def done(f): return set(open(str(f) + ".done").read().split("\n")) if Path(str(f) + ".done").exists() else set()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); ap.add_argument("--step", default="all")
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); W = a.workers; tasks = a.tasks.split(",")
    for sub in ("results/rank", "results/alpha", "results/sibling", "results/related", "results/seeds", "logs"): P(*sub.split("/")).mkdir(parents=True, exist_ok=True)
    import sklearn, scipy
    log("Stage6 start | python", platform.python_version(), "| numpy", np.__version__, "| pandas", pd.__version__, "| scipy", scipy.__version__, "| sklearn", sklearn.__version__,
        "| workers", W, "| cpu", os.cpu_count(), "|", platform.platform(), "| Stage5 read from", S5)
    steps = ["rank", "related", "alpha", "seeds", "sib_native", "rand_mlp", "sib_bench"]; todo = steps if a.step == "all" else a.step.split(",")
    if "rank" in todo:
        f = P("results", "rank", "rank_check.jsonl"); dn = done(f)
        run_pool("rank", [(f"{t}|{c}", (t, c)) for t in tasks for c in TASKS[t]["cohorts"] if f"{t}|{c}" not in dn], rank_job, W, appender(f))
    if "related" in todo:
        f = P("results", "related", "pairs.jsonl"); dn = done(f)
        def w(lab, rows):
            with open(f, "a") as fh: fh.write("".join(json.dumps(r) + "\n" for r in rows))
            with open(str(f) + ".done", "a") as fh: fh.write(lab + "\n")
        run_pool("related", [(t, (t,)) for t in tasks if t not in dn], related_job, W, w)
    if "alpha" in todo:
        f = P("results", "alpha", "alpha_grid.jsonl"); dn = done(f)
        J = [(f"{t}|{h}|{p}", (t, h, p)) for t in tasks for h in TASKS[t]["cohorts"] for p in TASKS[t]["panels"] if f"{t}|{h}|{p}" not in dn]
        J.sort(key=lambda j: (j[1][0] != "ATB_vs_OD", j[1][0] != "ATB_vs_LTB")); run_pool("alpha", J, alpha_job, W, appender(f))
    if "seeds" in todo:
        f = P("results", "seeds", "seeds.jsonl"); dn = done(f)
        run_pool("seeds", [(f"{t}|{h}", (t, h)) for t in tasks for h in TASKS[t]["cohorts"] if f"{t}|{h}" not in dn], seeds_job, W, appender(f))
    if "LTB_vs_CON" in tasks and "sib_native" in todo:
        J = [(h, (h,)) for h in SIB if not P("results", "sibling", f"native_{h}_without_{SIB[h]}.json").exists()]
        def w(lab, r): json.dump(r, open(P("results", "sibling", f"native_{lab}_without_{SIB[lab]}.json"), "w"), indent=1); log("  sibling native", lab, r["panel"], round(r["mean_cross_auc"], 4))
        run_pool("sib_native", J, sib_native_job, W, w)
    if "rand_mlp" in todo:
        f = P("results", "alpha", "random_mlp_new_alpha.jsonl"); dn = done(f); new = {}
        for line in open(P("results", "alpha", "alpha_grid.jsonl")): r = json.loads(line); new[(r["task"], r["heldout"], r["panel"])] = r["chosen_alpha"]
        old = {}
        for t in tasks:
            for line in open(S5 / "results/bench" / f"{t}__records.jsonl"):
                r = json.loads(line)
                if r["kind"] == "cur": old[(t, r["heldout"], r["panel"])] = r["diag"]["MLP_alpha_chosen"]
        J = [(f"{t}|{h}|{p}", (t, h, p, new[(t, h, p)])) for (t, h, p) in new if t in tasks and new[(t, h, p)] != old[(t, h, p)] and f"{t}|{h}|{p}" not in dn]
        log("rand_mlp: chosen alpha changed in", len([1 for k in new if k[0] in tasks and new[k] != old[k]]), "of", len([1 for k in new if k[0] in tasks]), "fold x panel cells")
        run_pool("rand_mlp", J, rand_mlp_job, W, appender(f))
    if "LTB_vs_CON" in tasks and "sib_bench" in todo:
        f = P("results", "sibling", "sib_records.jsonl"); dn = done(f)
        J = [(f"cur|{h}|{p}", (h, p, "cur", 0)) for h in SIB for p in SIB_PANELS if f"cur|{h}|{p}" not in dn]
        run_pool("sib_bench curated", J, sib_bench_job, W, appender(f))
        al = {}
        for line in open(f):
            r = json.loads(line)
            if r["kind"] == "cur": al[(r["heldout"], r["panel"])] = r["diag"]["MLP_alpha_chosen"]
        nd = 2 if a.smoke else 50
        J = [(f"rnd|{h}|{p}|{k}", (h, p, f"rnd@{al[(h, p)]}", k)) for h in SIB for p in SIB_PANELS for k in range(nd) if f"rnd|{h}|{p}|{k}" not in dn]
        run_pool("sib_bench random", J, sib_bench_job, W, appender(f))
    log("Stage6 finished requested steps:", todo)
if __name__ == "__main__":
    main()
