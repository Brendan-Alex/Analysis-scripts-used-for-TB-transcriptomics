"""Stage5: Stage4 rotating leave-one-cohort-out benchmark with the Stage4 decisions applied. Runs natively, in parallel.

Changes from Stage4 (user decisions)
  - selection universe per task = genes measured in EVERY cohort of the task (held-out cohort's gene LIST used; never its values/labels),
    minus X/Y genes (Stage4 list) and minus names Native always excluded (blank/NA, PHAGE, LOC*, C*ORF*).
    Native per-cohort pools, fold selection and random panels all draw from this universe -> no Native gene can be zero-filled.
  - Berry86 reported twice: 'Berry86' (as published, includes X-linked TLR7) and 'Berry86_noTLR7'.
  - curated panels otherwise fixed; any of their genes absent from a cohort is still zero-filled and reported.
Unchanged: label-free per-cohort z over all included samples; models, seeds, singscore/DirAvg, bootstrap (done in the summary step).
Added diagnostics: per-alpha inner-LOCO AUC per inner cohort, and held-out AUC at every alpha (reporting only, never used to choose).

Steps (run in order, each resumable; finished work is skipped on re-run):
  prepare -> native_cohort -> native_fold (+ native canary) -> bench_curated -> bench_random -> canary_bench
Usage (from the Stage5 folder):  python scripts/run_stage5.py              (everything)
                                 python scripts/run_stage5.py --workers 6   (default: CPU count - 1)
                                 python scripts/run_stage5.py --step bench_random
RF: n_jobs=1 inside each worker (results do not depend on n_jobs for a fixed random_state); parallelism is across jobs."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import sys, json, time, pickle, argparse, warnings, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

S5 = Path(__file__).resolve().parents[1]; MT = S5.parent; CR = MT / "Clean_rebuild"; S3 = MT / "Stage3"; S4 = MT / "Stage4"
OUT = Path(os.environ.get("STAGE5_OUT", str(S5)))          # override only for smoke tests
SEED = 42; ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
TASKS = {
 "ATB_vs_LTB": dict(pos="ATB", neg="LTB", cohorts=["GSE19491", "GSE37250", "GSE73408", "GSE101705", "GSE107994"],
                    panels=["Native", "Zak16", "Kaforou_ATB_v_LTB", "Berry86", "Berry86_noTLR7", "Sweeney3"]),
 "ATB_vs_OD":  dict(pos="ATB", neg="OD", cohorts=["GSE37250", "GSE73408", "GSE42834", "GSE83456"],
                    panels=["Native", "Zak16", "Kaforou_ATB_v_OD", "Berry86", "Berry86_noTLR7", "Sweeney3"]),
 "LTB_vs_CON": dict(pos="LTB", neg="CON", cohorts=["GSE19439", "GSE19444", "GSE28623", "GSE107994"],
                    panels=["Native", "Zak16", "Berry86", "Berry86_noTLR7", "Sweeney3"]),
}
TASKLIST = list(TASKS)
def P(*a): return OUT.joinpath(*a)
def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(msg, flush=True)
    with open(P("logs", "run.log"), "a", encoding="utf-8") as fh: fh.write(msg + "\n")
def manifest(): return pd.read_csv(S3 / "data/sample_manifest_stage3.csv")
def expr(c):
    f = S3 / "data/GSE19491_log2_genelevel_stage3.csv.gz" if c == "GSE19491" else CR / f"expression/{c}_log2_genelevel.csv.gz"
    return pd.read_csv(f, index_col=0)
def panel(p):
    if p == "Berry86_noTLR7": return [g for g in panel("Berry86") if g != "TLR7"]
    return pd.read_csv(CR / f"panels/{p}_hgnc.csv").Gene.astype(str).tolist()
def badname(g):
    import re
    return g.upper() in ("", "NA", "NAN", "---") or "PHAGE" in g.upper() or re.match(r"^(LOC\d+|C\d+ORF\d+)$", g, re.I) is not None
def load(task): return pickle.load(open(P("data", "cache", f"{task}.pkl"), "rb"))
def universe(task): return pd.read_csv(P("data", f"universe_{task}.csv")).Gene.astype(str).tolist()
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))

# ---------------------------------------------------------------- prepare
def step_prepare(tasks):
    from scipy.stats import rankdata
    SEX = set(pd.read_csv(S4 / "data/sex_chromosome_genes.csv").Gene.astype(str))
    pd.DataFrame({"Gene": sorted(SEX)}).to_csv(P("data", "sex_chromosome_genes.csv"), index=False)
    M = manifest(); inc = M[M.included_stage3]; folds, shift, sep, labels, uni_rows = [], [], [], {}, []
    for task in tasks:
        t = TASKS[task]
        if P("data", "cache", f"{task}.pkl").exists() and P("data", f"universe_{task}.csv").exists():
            log("prepare: cache exists", task); continue
        d = {}
        for c in t["cohorts"]:
            X = expr(c); X.index = X.index.astype(str); m = inc[inc.cohort == c].set_index("GSM"); X = X[[s for s in X.columns if s in m.index]]
            Zall = X.sub(X.mean(1), axis=0).div(X.std(1, ddof=0).replace(0, np.nan), axis=0).fillna(0.0)
            keep = [s for s in X.columns if m.loc[s, "class_label"] in (t["pos"], t["neg"])]
            y = (m.loc[keep, "class_label"] == t["pos"]).astype(int); L = X[keep]
            R = pd.DataFrame(rankdata(L.to_numpy(), axis=0), index=L.index, columns=keep)
            d[c] = dict(Z=Zall[keep], L=L, R=R, y=y, n_cohort_all=X.shape[1])
        allg = set.intersection(*[set(d[c]["Z"].index) for c in t["cohorts"]])
        n_bad = sum(badname(g) for g in allg); n_sex = sum((g in SEX) and not badname(g) for g in allg)
        U = sorted(g for g in allg if g not in SEX and not badname(g))
        pd.DataFrame({"Gene": U}).to_csv(P("data", f"universe_{task}.csv"), index=False)
        uni_rows.append(dict(task=task, genes_in_all_cohorts=len(allg), removed_bad_names=n_bad, removed_sex_genes=n_sex, universe=len(U),
                             **{f"genes_measured_{c}": d[c]["Z"].shape[0] for c in t["cohorts"]}))
        pickle.dump(d, open(P("data", "cache", f"{task}.pkl"), "wb"), protocol=4)
        labels[task] = {c: [d[c]["y"].index.tolist(), d[c]["y"].astype(int).tolist()] for c in t["cohorts"]}
        for c in t["cohorts"]:
            L, yy = d[c]["L"], d[c]["y"].to_numpy(); n1 = yy.sum(); n0 = len(yy) - n1
            s = L.mean().to_numpy(); a = auc(yy, s)
            rng = np.random.default_rng(SEED); I = np.stack([rng.integers(0, len(yy), len(yy)) for _ in range(1000)]); Yb, Sb = yy[I], s[I]
            Rk = rankdata(Sb, axis=1); b1 = Yb.sum(1); b0 = Yb.shape[1] - b1; ok = (b1 > 0) & (b0 > 0)
            b = ((Rk * Yb).sum(1) - b1 * (b1 + 1) / 2) / np.where(ok, b1 * b0, 1); lo, hi = np.percentile(b[ok], [2.5, 97.5])
            Rg = rankdata(L.to_numpy(), axis=1); ga = (Rg[:, yy == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
            shift.append(dict(task=task, heldout=c, allgene_mean_AUC=round(a, 3), CI=f"{lo:.3f}-{hi:.3f}",
                              share_genes_AUCdev_gt_0_2=round(float((np.abs(ga - .5) > .2).mean()), 3), quantile_normalised=c in ("GSE19491", "GSE42834")))
            inU = L.index.isin(U)
            for scope, msk in (("all_measured", np.ones(len(ga), bool)), ("task_universe", inU)):
                g = ga[msk]
                sep.append(dict(task=task, cohort=c, scope=scope, n_genes=int(msk.sum()), n_pos=int(n1), n_neg=int(n0),
                                prop_AUC_gt_0_8=round(float((g > 0.8).mean()), 4), prop_AUC_lt_0_2=round(float((g < 0.2).mean()), 4),
                                prop_either_dev_gt_0_3=round(float((np.abs(g - .5) > .3).mean()), 4),
                                prop_either_dev_gt_0_2=round(float((np.abs(g - .5) > .2).mean()), 4), median_abs_dev=round(float(np.median(np.abs(g - .5))), 4)))
        for h in t["cohorts"]:
            tr = [c for c in t["cohorts"] if c != h]
            folds.append(dict(task=task, heldout=h, training_cohorts=";".join(tr), n_train=sum(len(d[c]["y"]) for c in tr),
                              train_pos=sum(int(d[c]["y"].sum()) for c in tr), train_neg=sum(int((1 - d[c]["y"]).sum()) for c in tr),
                              n_test=len(d[h]["y"]), test_pos=int(d[h]["y"].sum()), test_neg=int((1 - d[h]["y"]).sum()), z_reference_samples=d[h]["n_cohort_all"]))
        log("prepare:", task, "universe", len(U), "of", len(allg), "genes in all cohorts")
    def app(df, f):
        f = P(*f); df = pd.DataFrame(df)
        if len(df): df.to_csv(f, mode="a", header=not f.exists(), index=False)
    app(folds, ("qc", "folds.csv")); app(shift, ("results", "global_shift_heldout.csv")); app(sep, ("results", "gene_level_separability.csv"))
    app(uni_rows, ("qc", "selection_universe.csv"))
    if labels:
        lf = P("data", "labels.json"); old = json.load(open(lf)) if lf.exists() else {}; old.update(labels); json.dump(old, open(lf, "w"))

# ---------------------------------------------------------------- Native (Stage4 algorithm, universe restricted)
def mrmr_topk(A, mi, k):
    C = np.nan_to_num(np.abs(np.corrcoef(A.T))); np.fill_diagonal(C, 0); sel, rem = [], list(range(A.shape[1]))
    for _ in range(min(k, A.shape[1])):
        s = mi[rem] if not sel else mi[rem] - C[np.ix_(rem, sel)].mean(1); b = rem[int(np.argmax(s))]; sel.append(b); rem.remove(b)
    return sel
def native_cohort(task, c):
    warnings.filterwarnings("ignore")
    from sklearn.feature_selection import mutual_info_classif
    d = load(task)[c]; U = universe(task); X = d["Z"].loc[U]; A = X.T.to_numpy(); y = d["y"].to_numpy()
    mi = mutual_info_classif(A, y, random_state=SEED); top = np.argsort(-mi, kind="stable")[:2000]; sel = mrmr_topk(A[:, top], mi[top], 20)
    r = dict(task=task, cohort=c, universe=len(U), mrmr_top20=[X.index[top[i]] for i in sel])
    json.dump(r, open(P("results", "native", f"{task}__{c}__cohort.json"), "w"), indent=1); return f"{task} {c}"
def cross_auc(D, genes):
    from sklearn.linear_model import LogisticRegression
    a = []
    for i, (Xi, yi) in enumerate(D):
        m = LogisticRegression(solver="liblinear", max_iter=2000).fit(Xi[genes], yi)
        for j, (Xj, yj) in enumerate(D):
            if i != j: a.append(auc(yj, m.predict_proba(Xj[genes])[:, 1]))
    return float(np.mean(a))
def native_fold(task, held, canary=False):
    warnings.filterwarnings("ignore")
    from sklearn.feature_selection import mutual_info_classif
    t = TASKS[task]; tr = [c for c in t["cohorts"] if c != held]; assert held not in tr
    data = load(task); U = universe(task)
    if canary:                              # held-out cohort replaced by noise and random labels; nothing below may change
        rng = np.random.default_rng(7); h = dict(data[held])
        h["Z"] = pd.DataFrame(rng.normal(size=h["Z"].shape), index=h["Z"].index, columns=h["Z"].columns)
        h["y"] = pd.Series(rng.integers(0, 2, len(h["y"])), index=h["y"].index); data[held] = h
    D = [(data[c]["Z"].loc[U].T, data[c]["y"].to_numpy()) for c in tr]
    pools = {c: json.load(open(P("results", "native", f"{task}__{c}__cohort.json")))["mrmr_top20"] for c in tr}
    allowed = sorted(set().union(*pools.values()))
    def mrmr_rank(X, y):
        mi = mutual_info_classif(X.to_numpy(), y, random_state=0); return [X.columns[i] for i in mrmr_topk(X.to_numpy(), mi, X.shape[1])]
    ranks = [mrmr_rank(Xd[allowed], yd) for Xd, yd in D]; score = {g: sum(r.index(g) for r in ranks) for g in allowed}
    tag = "canary" if canary else "fold"; ck = P("results", "native", f"{task}__{tag}_{held}.checkpoint.json"); t0 = time.time()
    if ck.exists():
        st = json.load(open(ck)); panel_, cur, swaps = st["panel"], st["cur"], st["swaps"]
    else:
        panel_ = sorted(allowed, key=lambda g: (score[g], g))[:5]
        while len(panel_) < min(20, len(allowed)):
            best, ba = None, -np.inf
            for g in sorted(set(allowed) - set(panel_)):
                a = cross_auc(D, panel_ + [g])
                if a > ba: best, ba = g, a
            panel_.append(best)
        cur = cross_auc(D, panel_); swaps = 0; json.dump(dict(panel=panel_, cur=cur, swaps=swaps), open(ck, "w"))
    improved = True
    while improved:
        improved = False
        for i in range(len(panel_)):
            for gin in sorted(set(allowed) - set(panel_)):
                trial = panel_.copy(); trial[i] = gin; a = cross_auc(D, trial)
                if a > cur + 1e-12:
                    panel_, cur, improved = trial, a, True; swaps += 1; json.dump(dict(panel=panel_, cur=cur, swaps=swaps), open(ck, "w")); break
            if improved: break
    r = dict(task=task, heldout=held, training_cohorts=tr, universe=len(U), allowed=len(allowed), panel=panel_, mean_cross_auc=cur, swaps=swaps,
             zero_filled_in_heldout=[g for g in panel_ if g not in data[held]["Z"].index], seconds_swap_phase=round(time.time() - t0))
    json.dump(r, open(P("results", "native", f"{task}__{tag}_{held}.json"), "w"), indent=1)
    return f"{task} {tag} {held}: cross-AUC {cur:.4f}, swaps {swaps}, zero-filled {len(r['zero_filled_in_heldout'])}"

# ---------------------------------------------------------------- benchmark (Stage4 02_bench.py logic)
_D = {}
def _getD(task):
    if task not in _D: _D.clear(); _D[task] = load(task)
    return _D[task]
def panel_genes(task, held, p): return json.load(open(P("results", "native", f"{task}__fold_{held}.json")))["panel"] if p == "Native" else panel(p)
def stack(D, coh, genes):
    X = pd.concat([D[c]["Z"].reindex(genes).T for c in coh]).fillna(0.0); y = np.concatenate([D[c]["y"].to_numpy() for c in coh]); return X, y
def direction(D, coh, genes):
    X, y = stack(D, coh, genes); d = np.sign(X[y == 1].mean() - X[y == 0].mean()); return d.replace(0, 1.0)
def singscore(D, c, dirn):
    R = D[c]["R"]; N = R.shape[0]; up = [g for g in dirn.index if dirn[g] > 0 and g in R.index]; dn = [g for g in dirn.index if dirn[g] < 0 and g in R.index]
    def part(G, rev):
        if not G: return np.zeros(R.shape[1])
        r = R.loc[G]; r = (N + 1 - r) if rev else r; n = len(G); lo, hi = (n + 1) / 2, (2 * N - n + 1) / 2
        return ((r.mean() - lo) / (hi - lo) - 0.5).to_numpy()
    return part(up, False) + part(dn, True)
def diravg(D, c, dirn):
    Z = D[c]["Z"]; g = [x for x in dirn.index if x in Z.index]
    return np.zeros(Z.shape[1]) if not g else (Z.loc[g].mul(dirn[g], axis=0)).mean().to_numpy()   # no gene present -> constant (zero-fill)
def make(m, alpha=1e-4):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    if m == "LR": return LogisticRegression(max_iter=1000)
    if m == "RF": return RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=1)
    return MLPClassifier(hidden_layer_sizes=(50,), max_iter=5000, random_state=SEED, alpha=alpha)
def predict(mdl, D, c, feats): return mdl.predict_proba(D[c]["Z"].reindex(feats).T.fillna(0.0))[:, 1]
SEXG = None
def fit_eval(D, train, tests, genes, methods, alpha_fixed=None, held=None):
    global SEXG
    if SEXG is None: SEXG = set(pd.read_csv(P("data", "sex_chromosome_genes.csv")).Gene.astype(str))
    feats = [g for g in genes if all(g in D[c]["Z"].index for c in train)]
    X, y = stack(D, train, feats); out, diag = {}, {"n_features": len(feats), "n_sex_features": sum(g in SEXG for g in feats)}
    for m in methods:
        if m in ("LR", "RF", "MLP"):
            a = alpha_fixed if (m == "MLP" and alpha_fixed is not None) else 1e-4
            mdl = make(m, a).fit(X, y)
            if m == "MLP": diag.update(MLP_alpha_used=a, MLP_n_iter=int(mdl.n_iter_), MLP_converged=bool(mdl.n_iter_ < 5000))
            out[m] = {c: predict(mdl, D, c, feats) for c in tests}
        elif m == "MLP_alpha":
            best, inner, per = None, {}, {}
            for a in ALPHAS:
                aucs = []
                for h in train:
                    tr = [c for c in train if c != h]; Xi, yi = stack(D, tr, feats)
                    aucs.append(auc(D[h]["y"], predict(make("MLP", a).fit(Xi, yi), D, h, feats)))
                inner[str(a)] = float(np.mean(aucs)); per[str(a)] = dict(zip(train, map(float, aucs)))
                if best is None or np.mean(aucs) > best[1] + 1e-12: best = (a, float(np.mean(aucs)))
            mdl = make("MLP", best[0]).fit(X, y)
            diag.update(MLP_alpha_chosen=best[0], MLP_alpha_inner_AUC=inner, MLP_alpha_inner_AUC_per_cohort=per,
                        MLP_alpha_n_iter=int(mdl.n_iter_), MLP_alpha_converged=bool(mdl.n_iter_ < 5000))
            out[m] = {c: predict(mdl, D, c, feats) for c in tests}
            if held is not None:            # diagnostic only: held-out AUC at every alpha (never used for the choice)
                diag["MLP_alpha_grid_heldout_AUC"] = {str(a): (auc(D[held]["y"], predict(mdl, D, held, feats)) if a == best[0] else
                                                         auc(D[held]["y"], predict(make("MLP", a).fit(X, y), D, held, feats))) for a in ALPHAS}
        else:
            dirn = direction(D, train, feats); diag.update(n_up=int((dirn > 0).sum()), n_down=int((dirn < 0).sum()), directions={g: int(v) for g, v in dirn.items()})
            f = singscore if m == "singscore" else diravg
            out[m] = {c: f(D, c, dirn) for c in tests}
    diag["zero_filled_in_test"] = {c: int(sum(g not in D[c]["Z"].index for g in feats)) for c in tests}
    return out, diag, feats
CUR = ["LR", "RF", "MLP", "MLP_alpha", "singscore", "DirAvg"]; RND = ["LR", "RF", "MLP", "singscore", "DirAvg"]
def fold_train(task, D, h):
    train = [c for c in TASKS[task]["cohorts"] if c != h]; assert h not in train
    assert not any(set(D[c]["Z"].columns) & set(D[h]["Z"].columns) for c in train), "held-out GSM in training"
    return train
def bench_job(task, key, alpha=None):
    warnings.filterwarnings("ignore")
    D = _getD(task); t = TASKS[task]; parts = key.split("|"); kind, h, p = parts[0], parts[1], parts[2]
    train = fold_train(task, D, h); genes = panel_genes(task, h, p)
    if kind == "cur":
        out, diag, feats = fit_eval(D, train, [h] + train, genes, CUR, held=h)
        rec = dict(key=key, kind=kind, heldout=h, panel=p, training=train, features=feats, diag=diag)
    else:
        k = int(parts[3]); size = len([g for g in genes if all(g in D[c]["Z"].index for c in train)])
        rng = np.random.default_rng([SEED, TASKLIST.index(task), t["cohorts"].index(h), size, k]); g2 = list(rng.choice(universe(task), size=size, replace=False))
        out, diag, feats = fit_eval(D, train, [h], g2, RND, alpha_fixed=alpha)
        diag.pop("directions", None); rec = dict(key=key, kind=kind, heldout=h, panel=p, draw=k, size=size, features=feats, diag=diag)
    rec["scores"] = {m: {c: [float(x) for x in s] for c, s in v.items()} for m, v in out.items()}
    return rec
def canary_bench(task, held):
    warnings.filterwarnings("ignore")
    D = _getD(task); train = fold_train(task, D, held); genes = panel_genes(task, held, "Native")
    rng = np.random.default_rng(999); N = {c: dict(v) for c, v in D.items()}; h = N[held]
    h["Z"] = pd.DataFrame(rng.normal(size=h["Z"].shape), index=h["Z"].index, columns=h["Z"].columns)
    h["L"] = pd.DataFrame(rng.normal(8, 2, size=h["L"].shape), index=h["L"].index, columns=h["L"].columns)
    h["R"] = h["L"].rank(axis=0); h["y"] = pd.Series(rng.integers(0, 2, len(h["y"])), index=h["y"].index)
    a, da, fa = fit_eval(D, train, train + [held], genes, CUR); b, db, fb = fit_eval(N, train, train + [held], genes, CUR)
    return dict(task=task, heldout=held, features_identical=fa == fb, directions_identical=da["directions"] == db["directions"],
                alpha_identical=da["MLP_alpha_chosen"] == db["MLP_alpha_chosen"], inner_alpha_AUCs_identical=da["MLP_alpha_inner_AUC"] == db["MLP_alpha_inner_AUC"],
                training_predictions_identical=all(np.array_equal(a[m][c], b[m][c]) for m in CUR for c in train),
                heldout_predictions_changed=any(not np.array_equal(a[m][held], b[m][held]) for m in CUR))

# ---------------------------------------------------------------- driver
def run_pool(name, jobs, fn, workers, on_result):
    """jobs: list of (label, args); fn(*args) in worker processes; on_result(label, result) in the main process (writes files)."""
    if not jobs: log(name, ": nothing to do"); return
    log(name, ":", len(jobs), "jobs on", workers, "workers"); t0 = time.time(); n = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(fn, *args): lab for lab, args in jobs}
        for f in as_completed(fut):
            on_result(fut[f], f.result()); n += 1
            if n % max(1, len(jobs) // 20) == 0 or n == len(jobs):
                el = time.time() - t0; log(f"{name}: {n}/{len(jobs)} done, {el/60:.1f} min elapsed, ~{el/n*(len(jobs)-n)/60:.1f} min left")
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--step", default="all"); ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); W = a.workers; tasks = a.tasks.split(",")
    for sub in ("data/cache", "results/native", "results/bench", "qc", "logs"): P(*sub.split("/")).mkdir(parents=True, exist_ok=True)
    import sklearn, scipy
    log("Stage5 start | python", platform.python_version(), "| numpy", np.__version__, "| pandas", pd.__version__, "| scipy", scipy.__version__,
        "| sklearn", sklearn.__version__, "| workers", W, "| cpu", os.cpu_count(), "|", platform.platform())
    steps = ["prepare", "native_cohort", "native_fold", "bench_curated", "bench_random", "canary_bench"]
    todo = steps if a.step == "all" else [a.step]
    if "prepare" in todo: step_prepare(tasks)
    if "native_cohort" in todo:
        J = [(f"{t}|{c}", (t, c)) for t in tasks for c in TASKS[t]["cohorts"] if not P("results", "native", f"{t}__{c}__cohort.json").exists()]
        run_pool("native_cohort", J, native_cohort, W, lambda lab, r: log("  done", r))
    if "native_fold" in todo:
        J = [(f"{t}|{tag}|{h}", (t, h, tag == "canary")) for tag in ("fold", "canary") for t in tasks for h in TASKS[t]["cohorts"]
             if not P("results", "native", f"{t}__{tag}_{h}.json").exists() and not (a.smoke and tag == "canary")]
        run_pool("native_fold", J, native_fold, W, lambda lab, r: log("  done", r))
        rows = []
        for t in tasks:
            for h in TASKS[t]["cohorts"]:
                f1, f2 = P("results", "native", f"{t}__fold_{h}.json"), P("results", "native", f"{t}__canary_{h}.json")
                if f1.exists() and f2.exists():
                    x, y = json.load(open(f1)), json.load(open(f2))
                    rows.append(dict(task=t, heldout=h, panel_identical=x["panel"] == y["panel"], cross_auc_identical=x["mean_cross_auc"] == y["mean_cross_auc"],
                                     allowed_identical=x["allowed"] == y["allowed"]))
        if rows: pd.DataFrame(rows).to_csv(P("qc", "canary_native.csv"), index=False); log("native canary: all identical =", all(r["panel_identical"] and r["cross_auc_identical"] and r["allowed_identical"] for r in rows))
    for kind in ("bench_curated", "bench_random"):
        if kind not in todo: continue
        for t in tasks:
            rec_f, done_f = P("results", "bench", f"{t}__records.jsonl"), P("results", "bench", f"{t}__done.txt")
            done = set(open(done_f).read().split("\n")) if done_f.exists() else set()
            alpha = {}
            if rec_f.exists():
                for line in open(rec_f):
                    r = json.loads(line)
                    if r["kind"] == "cur": alpha[(r["heldout"], r["panel"])] = r["diag"]["MLP_alpha_chosen"]
            ndraw = 2 if a.smoke else 50
            if kind == "bench_curated": keys = [f"cur|{h}|{p}" for h in TASKS[t]["cohorts"] for p in TASKS[t]["panels"]]
            else: keys = [f"rnd|{h}|{p}|{k}" for h in TASKS[t]["cohorts"] for p in TASKS[t]["panels"] for k in range(ndraw)]
            J = [(k, (t, k, None if kind == "bench_curated" else alpha[(k.split("|")[1], k.split("|")[2])])) for k in keys if k not in done]
            def write(lab, rec, rec_f=rec_f, done_f=done_f):
                with open(rec_f, "a") as fh: fh.write(json.dumps(rec) + "\n")
                with open(done_f, "a") as fh: fh.write(lab + "\n")
            run_pool(f"{kind} {t}", J, bench_job, W, write)
    if "canary_bench" in todo:
        f = P("qc", "canary_bench.jsonl"); have = {(r["task"], r["heldout"]) for r in map(json.loads, open(f))} if f.exists() else set()
        J = [(f"{t}|{h}", (t, h)) for t in tasks for h in TASKS[t]["cohorts"] if (t, h) not in have]
        def w(lab, r):
            with open(f, "a") as fh: fh.write(json.dumps(r) + "\n")
            log("  canary", r["task"], r["heldout"], "PASS" if all(v for k, v in r.items() if k not in ("task", "heldout")) else f"FAIL {r}")
        run_pool("canary_bench", J, canary_bench, W, w)
    log("Stage5 finished requested steps:", todo)
if __name__ == "__main__":
    main()
