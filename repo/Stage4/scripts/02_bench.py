"""Stage4 benchmark, rotating leave-one-cohort-out (every cohort of a task held out in turn).
Everything fitted uses ONLY the fold's training cohorts:
  - Native panel: results/native/<task>__fold_<held>__nosex.json (selected on training cohorts only, 01_native.py)
  - features = panel genes present in every training cohort; genes absent from the held-out cohort are zero-filled (z = cohort mean)
  - gene directions (singscore up/down sets, DirAvg signs): pooled training samples
  - MLP_alpha: alpha from {1e-4..10} by inner leave-one-cohort-out over the training cohorts
  - normalisation: label-free per-cohort z (user decision); no parameter is estimated across cohorts
Methods: LR (max_iter 1000), RF (500 trees, rs 42, n_jobs -1), MLP ((50,), max_iter 5000, rs 42, alpha 1e-4), MLP_alpha, singscore, DirAvg
Curated jobs score the held-out cohort AND every training cohort (resubstitution, for the generalisation gap).
Random jobs: 50 draws per curated panel, size = genes that panel uses in the fold, drawn from the training-cohort gene intersection
  minus X/Y genes and minus the names Native excludes (LOC*, C*ORF*, PHAGE, blank); seed default_rng([42, task_idx, held_idx, size, draw]);
  methods LR, RF, MLP (alpha = alpha chosen for that curated panel in the fold), singscore, DirAvg; held-out only.
Canary jobs (one per fold, Native panel): held-out Z/L/R replaced by N(0,1) noise / random ranks; all training-side outputs must be identical.
Usage: python 02_bench.py run TASK SECONDS | python 02_bench.py canary_native TASK HELDOUT"""
import os, sys, json, time, pickle, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
sys.path.insert(0, os.path.dirname(__file__)); from common4 import *
warnings.filterwarnings("ignore", category=ConvergenceWarning)
RES = S4 / "results/bench"; QC = S4 / "qc"; SEX = set(pd.read_csv(S4 / "data/sex_chromosome_genes.csv").Gene)
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]; TASKLIST = list(TASKS)
def load(task): return pickle.load(open(S4 / f"data/cache/{task}.pkl", "rb"))
def panel_genes(task, held, p):
    return json.load(open(S4 / f"results/native/{task}__fold_{held}__nosex.json"))["panel"] if p == "Native" else panel(p)
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
    Z = D[c]["Z"]; g = [x for x in dirn.index if x in Z.index]; return (Z.loc[g].mul(dirn[g], axis=0)).mean().to_numpy()
def make(m, alpha=1e-4):
    if m == "LR": return LogisticRegression(max_iter=1000)
    if m == "RF": return RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    return MLPClassifier(hidden_layer_sizes=(50,), max_iter=5000, random_state=SEED, alpha=alpha)
def predict(mdl, D, c, feats): return mdl.predict_proba(D[c]["Z"].reindex(feats).T.fillna(0.0))[:, 1]
def fit_eval(D, train, tests, genes, methods, alpha_fixed=None):
    feats = [g for g in genes if all(g in D[c]["Z"].index for c in train)]
    X, y = stack(D, train, feats); out, diag = {}, {"n_features": len(feats), "n_sex_features": sum(g in SEX for g in feats)}
    for m in methods:
        if m in ("LR", "RF", "MLP"):
            a = alpha_fixed if (m == "MLP" and alpha_fixed is not None) else 1e-4
            mdl = make(m, a).fit(X, y)
            if m == "MLP": diag["MLP_alpha_used"] = a; diag["MLP_n_iter"] = int(mdl.n_iter_); diag["MLP_converged"] = bool(mdl.n_iter_ < 5000)
            out[m] = {c: predict(mdl, D, c, feats) for c in tests}
        elif m == "MLP_alpha":
            best, inner = None, {}
            for a in ALPHAS:
                aucs = []
                for h in train:
                    tr = [c for c in train if c != h]; Xi, yi = stack(D, tr, feats)
                    aucs.append(roc_auc_score(D[h]["y"], predict(make("MLP", a).fit(Xi, yi), D, h, feats)))
                inner[str(a)] = float(np.mean(aucs))
                if best is None or np.mean(aucs) > best[1] + 1e-12: best = (a, float(np.mean(aucs)))
            mdl = make("MLP", best[0]).fit(X, y)
            diag.update(MLP_alpha_chosen=best[0], MLP_alpha_inner_AUC=inner, MLP_alpha_n_iter=int(mdl.n_iter_), MLP_alpha_converged=bool(mdl.n_iter_ < 5000))
            out[m] = {c: predict(mdl, D, c, feats) for c in tests}
        else:
            dirn = direction(D, train, feats); diag["n_up"] = int((dirn > 0).sum()); diag["n_down"] = int((dirn < 0).sum()); diag["directions"] = {g: int(v) for g, v in dirn.items()}
            f = singscore if m == "singscore" else diravg
            out[m] = {c: f(D, c, dirn) for c in tests}
    diag["zero_filled_in_test"] = {c: int(sum(g not in D[c]["Z"].index for g in feats)) for c in tests}
    return out, diag, feats
CUR = ["LR", "RF", "MLP", "MLP_alpha", "singscore", "DirAvg"]; RND = ["LR", "RF", "MLP", "singscore", "DirAvg"]
BADNAME = lambda g: g in ("", "NA", "NAN", "---") or "PHAGE" in g.upper() or pd.Series([g]).str.match(r"^(LOC\d+|C\d+ORF\d+)$", case=False).iloc[0]
def fold_info(task, D, held):
    t = TASKS[task]; train = [c for c in t["cohorts"] if c != held]; assert held not in train
    assert not any(set(D[c]["Z"].columns) & set(D[held]["Z"].columns) for c in train), "held-out GSM in training"
    uni = set.intersection(*[set(D[c]["Z"].index.astype(str)) for c in train]); uni = sorted(g for g in uni if g not in SEX and not BADNAME(g))
    return train, uni
def keys(task, D):
    t = TASKS[task]; J = []
    for h in t["cohorts"]:
        for p in t["panels"]: J.append(f"cur|{h}|{p}")
    for h in t["cohorts"]:
        for p in t["panels"]:
            for k in range(50): J.append(f"rnd|{h}|{p}|{k}")
    return J
def run(task, budget):
    D = load(task); t = TASKS[task]; t0 = time.time(); ti = TASKLIST.index(task)
    done_f = RES / f"{task}__done.txt"; done = set(open(done_f).read().split("\n")) if done_f.exists() else set()
    rec_f = RES / f"{task}__records.jsonl"; alpha = {}
    if rec_f.exists():
        for line in open(rec_f):
            r = json.loads(line)
            if r["kind"] == "cur": alpha[(r["heldout"], r["panel"])] = r["diag"]["MLP_alpha_chosen"]
    FI = {}; J = keys(task, D); n = 0
    for key in J:
        if key in done: continue
        if time.time() - t0 > budget: print(task, "budget reached; remaining", sum(1 for j in J if j not in done)); return
        parts = key.split("|"); kind, h, p = parts[0], parts[1], parts[2]
        if h not in FI: FI[h] = fold_info(task, D, h)
        train, uni = FI[h]; genes = panel_genes(task, h, p)
        if kind == "cur":
            out, diag, feats = fit_eval(D, train, [h] + train, genes, CUR); alpha[(h, p)] = diag["MLP_alpha_chosen"]
            rec = dict(key=key, kind=kind, heldout=h, panel=p, training=train, features=feats, diag=diag)
        else:
            k = int(parts[3]); size = len([g for g in genes if all(g in D[c]["Z"].index for c in train)])
            rng = np.random.default_rng([SEED, ti, t["cohorts"].index(h), size, k]); g2 = list(rng.choice(uni, size=size, replace=False))
            out, diag, feats = fit_eval(D, train, [h], g2, RND, alpha_fixed=alpha[(h, p)])
            rec = dict(key=key, kind=kind, heldout=h, panel=p, draw=k, size=size, features=feats, diag={k2: v for k2, v in diag.items() if k2 != "directions"})
        rec["scores"] = {m: {c: list(map(float, s)) for c, s in v.items()} for m, v in out.items()}
        with open(rec_f, "a") as fh: fh.write(json.dumps(rec) + "\n")
        with open(done_f, "a") as fh: fh.write(key + "\n")
        done.add(key); n += 1
    print(task, "all jobs done", len(J), "| this call", n)
def canary(task, held):
    """identical training-side outputs with the held-out cohort's data replaced by noise"""
    D = load(task); t = TASKS[task]; train, _ = fold_info(task, D, held); genes = panel_genes(task, held, "Native")
    rng = np.random.default_rng(999); N = {c: dict(v) for c, v in D.items()}; h = N[held]
    h["Z"] = pd.DataFrame(rng.normal(size=h["Z"].shape), index=h["Z"].index, columns=h["Z"].columns)
    h["L"] = pd.DataFrame(rng.normal(8, 2, size=h["L"].shape), index=h["L"].index, columns=h["L"].columns)
    h["R"] = h["L"].rank(axis=0); h["y"] = pd.Series(rng.integers(0, 2, len(h["y"])), index=h["y"].index)
    a, da, fa = fit_eval(D, train, train + [held], genes, CUR); b, db, fb = fit_eval(N, train, train + [held], genes, CUR)
    res = dict(task=task, heldout=held, features_identical=fa == fb, directions_identical=da["directions"] == db["directions"],
               alpha_identical=da["MLP_alpha_chosen"] == db["MLP_alpha_chosen"], inner_alpha_AUCs_identical=da["MLP_alpha_inner_AUC"] == db["MLP_alpha_inner_AUC"],
               training_predictions_identical=all(np.array_equal(a[m][c], b[m][c]) for m in CUR for c in train),
               heldout_predictions_changed=any(not np.array_equal(a[m][held], b[m][held]) for m in CUR))
    with open(QC / "canary_bench.jsonl", "a") as fh: fh.write(json.dumps(res) + "\n")
    print(res)
if __name__ == "__main__":
    if sys.argv[1] == "run": run(sys.argv[2], float(sys.argv[3]))
    elif sys.argv[1] == "canary": canary(sys.argv[2], sys.argv[3])
