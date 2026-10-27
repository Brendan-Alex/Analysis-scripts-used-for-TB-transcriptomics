"""Stage3 item 6-7: benchmark ATB_vs_LTB and LTB_vs_CON (ATB_vs_OD held).
Inputs per cohort: log2 gene-level (for singscore) and per-cohort z-scores over the task's two classes (models, directional average).
Features = panel genes present in every training cohort of the fit; genes absent from a test cohort are zero-filled (z = cohort mean).
Methods
  LR   LogisticRegression(max_iter=1000)
  RF   RandomForestClassifier(500 trees, random_state 42, n_jobs=-1)
  MLP  MLPClassifier((50,), max_iter=5000, random_state 42)                     [convergence recorded]
  MLP_alpha  same, alpha chosen from {1e-4,1e-3,1e-2,1e-1,1,10} by inner leave-one-cohort-out AUC within the training cohorts
  singscore  per-sample rank score (Foroutan 2018 simpleScore, centred), up/down sets from the training direction of each gene
  DirAvg     mean of direction-signed z-scores
  direction of a gene = sign(mean z in positive class - mean z in negative class), pooled training samples of the fit
Evaluations: internal LOCO (Native panel re-derived inside the fold), external (all training cohorts -> each test cohort),
             Random: 50 draws per curated-panel size (size = genes the curated panel actually uses), drawn from genes present in
             every training AND test cohort of the task, seed default_rng([42, size, draw]); external only.
Usage: python 03_benchmark.py cache | run TASK [seconds]
"""
import os, sys, json, time, pickle, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
sys.path.insert(0, os.path.dirname(__file__)); from common import *
warnings.filterwarnings("ignore", category=ConvergenceWarning)
CACHE = ST / "data/cache"; CACHE.mkdir(exist_ok=True); RES = ST / "results/bench"; RES.mkdir(exist_ok=True)
RUN_TASKS = ["ATB_vs_LTB", "LTB_vs_CON"]
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
def build_cache():
    for task in RUN_TASKS:
        t = TASKS[task]; d = {}
        for c in t["train"] + t["test"]:
            Z, y = task_matrix(task, c, z=True); L, _ = task_matrix(task, c, z=False)
            R = pd.DataFrame(rankdata(L.to_numpy(), axis=0), index=L.index, columns=L.columns)   # within-sample ranks, ties averaged
            d[c] = dict(Z=Z, L=L, R=R, y=y)
        pickle.dump(d, open(CACHE / f"{task}.pkl", "wb")); print(task, {c: v["Z"].shape for c, v in d.items()})
def load(task): return pickle.load(open(CACHE / f"{task}.pkl", "rb"))
def panels_for(task, fold="ALL"):
    P = {}
    for p in TASKS[task]["panels"]:
        P[p] = json.load(open(ST / f"results/native/{task}__fold_{fold}.json"))["panel"] if p == "Native_task" else panel(p)
    return P
def stack(D, coh, genes):
    X = pd.concat([D[c]["Z"].reindex(genes).T for c in coh]).fillna(0.0); y = np.concatenate([D[c]["y"].to_numpy() for c in coh]); return X, y
def direction(D, coh, genes):
    X, y = stack(D, coh, genes); d = np.sign(X[y == 1].mean() - X[y == 0].mean()); return d.replace(0, 1.0)
def singscore(D, c, dirn):
    R = D[c]["R"]; N = R.shape[0]; up = [g for g in dirn.index if dirn[g] > 0 and g in R.index]; dn = [g for g in dirn.index if dirn[g] < 0 and g in R.index]
    def part(G, rev):
        if not G: return 0.0
        r = R.loc[G]; r = (N + 1 - r) if rev else r; n = len(G); lo, hi = (n + 1) / 2, (2 * N - n + 1) / 2
        return ((r.mean() - lo) / (hi - lo) - 0.5).to_numpy()
    return part(up, False) + part(dn, True)
def diravg(D, c, dirn):
    Z = D[c]["Z"]; g = [x for x in dirn.index if x in Z.index]; return (Z.loc[g].mul(dirn[g], axis=0)).mean().to_numpy()
def make(m, alpha=1e-4):
    if m == "LR": return LogisticRegression(max_iter=1000)
    if m == "RF": return RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    return MLPClassifier(hidden_layer_sizes=(50,), max_iter=5000, random_state=SEED, alpha=alpha)
def fit_eval(D, train, tests, genes, methods):
    """returns {method: {cohort: scores}} plus diagnostics"""
    feats = [g for g in genes if all(g in D[c]["Z"].index for c in train)]
    X, y = stack(D, train, feats); out, diag = {}, {"n_features": len(feats)}
    for m in methods:
        if m in ("LR", "RF", "MLP"):
            mdl = make(m).fit(X, y)
            if m == "MLP": diag["MLP_n_iter"] = int(mdl.n_iter_); diag["MLP_converged"] = bool(mdl.n_iter_ < 5000)
            diag[f"{m}_train_AUC"] = roc_auc_score(y, mdl.predict_proba(X)[:, 1])
            out[m] = {c: mdl.predict_proba(D[c]["Z"].reindex(feats).T.fillna(0.0))[:, 1] for c in tests}
        elif m == "MLP_alpha":
            best = None
            for a in ALPHAS:
                aucs = []
                for h in train:
                    tr = [c for c in train if c != h]; Xi, yi = stack(D, tr, feats)
                    mi = make("MLP", a).fit(Xi, yi); aucs.append(roc_auc_score(D[h]["y"], mi.predict_proba(D[h]["Z"].reindex(feats).T.fillna(0.0))[:, 1]))
                if best is None or np.mean(aucs) > best[1] + 1e-12: best = (a, np.mean(aucs))
            mdl = make("MLP", best[0]).fit(X, y); diag["MLP_alpha_chosen"] = best[0]; diag["MLP_alpha_inner_AUC"] = best[1]
            diag["MLP_alpha_converged"] = bool(mdl.n_iter_ < 5000); diag["MLP_alpha_train_AUC"] = roc_auc_score(y, mdl.predict_proba(X)[:, 1])
            out[m] = {c: mdl.predict_proba(D[c]["Z"].reindex(feats).T.fillna(0.0))[:, 1] for c in tests}
        else:
            dirn = direction(D, train, feats)
            diag["n_up"] = int((dirn > 0).sum()); diag["n_down"] = int((dirn < 0).sum())
            f = singscore if m == "singscore" else diravg
            out[m] = {c: f(D, c, dirn) for c in tests}
    diag["zero_filled_in_test"] = {c: int(sum(g not in D[c]["Z"].index for g in feats)) for c in tests}
    return out, diag, feats
ALLM = ["LR", "RF", "MLP", "MLP_alpha", "singscore", "DirAvg"]
def jobs(task):
    t = TASKS[task]; J = []
    for p in t["panels"]:
        for h in t["train"]: J.append(("internal", p, h, None))
        J.append(("external", p, "ALL", None))
    P = panels_for(task)
    sizes = sorted({len([g for g in P[p] if 1]) for p in t["panels"]})
    return J
def run(task, budget):
    D = load(task); t = TASKS[task]; t0 = time.time()
    done_f = RES / f"{task}__done.txt"; done = set(open(done_f).read().split("\n")) if done_f.exists() else set()
    P = panels_for(task)
    # sizes = genes each curated panel uses in the external fit
    used = {p: [g for g in P[p] if all(g in D[c]["Z"].index for c in t["train"])] for p in P}
    universe = sorted(set.intersection(*[set(D[c]["Z"].index) for c in t["train"] + t["test"]]))
    J = []
    for p in t["panels"]:
        for h in t["train"]: J.append(f"internal|{p}|{h}")
        J.append(f"external|{p}|ALL")
    for p in t["panels"]:
        for k in range(50): J.append(f"random|{len(used[p])}|{k}")
    J = list(dict.fromkeys(J))
    for key in J:
        if key in done: continue
        if time.time() - t0 > budget: print("budget reached; remaining", sum(1 for j in J if j not in done)); return
        kind, a, b = key.split("|")
        if kind == "internal":
            genes = json.load(open(ST / f"results/native/{task}__fold_{b}.json"))["panel"] if a == "Native_task" else P[a]
            train = [c for c in t["train"] if c != b]; out, diag, feats = fit_eval(D, train, [b], genes, ALLM)
        elif kind == "external":
            train = t["train"]; out, diag, feats = fit_eval(D, train, t["test"], P[a], ALLM)
        else:
            size, k = int(a), int(b); rng = np.random.default_rng([SEED, size, k]); genes = list(rng.choice(universe, size=size, replace=False))
            train = t["train"]; out, diag, feats = fit_eval(D, train, t["test"], genes, ["LR", "RF", "MLP", "singscore", "DirAvg"])
        rec = dict(key=key, kind=kind, panel=a if kind != "random" else f"Random{a}", heldout_or_draw=b, features=feats, diag=diag,
                   scores={m: {c: list(map(float, s)) for c, s in v.items()} for m, v in out.items()})
        with open(RES / f"{task}__records.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
        with open(done_f, "a") as fh: fh.write(key + "\n")
        done.add(key)
    print("all jobs done", len(J))
if __name__ == "__main__":
    if sys.argv[1] == "cache": build_cache()
    else: run(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 150)
