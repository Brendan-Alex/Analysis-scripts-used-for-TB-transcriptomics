"""Stage4 Native panel per fold (original algorithm, as Stage3 02_native_select.py), using ONLY the fold's training cohorts.
Sex-chromosome genes (data/sex_chromosome_genes.csv) are removed from the selection universe (variant 'nosex');
variant 'withsex' keeps them, run only to report whether the exclusion changes the panels.
Per-cohort step (MI top 2000 -> mRMR top 20) depends only on that cohort and is cached; a fold uses the caches of its
training cohorts only (asserted).
Usage: python 01_native.py cohort TASK COHORT VARIANT | fold TASK HELDOUT VARIANT"""
import os, sys, json, time, pickle
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__)); from common4 import *
OUT = S4 / "results/native"; SEX = set(pd.read_csv(S4 / "data/sex_chromosome_genes.csv").Gene)
def load(task): return pickle.load(open(S4 / f"data/cache/{task}.pkl", "rb"))
def clean(X, variant):
    g = pd.Index(X.index.astype(str))
    bad = g.isin(["", "NA", "NAN", "---"]) | g.str.contains("PHAGE", case=False) | g.str.match(r"^(LOC\d+|C\d+ORF\d+)$", case=False)
    n_sex = int((~bad & g.isin(SEX)).sum())
    if variant == "nosex": bad = bad | g.isin(SEX)
    return X[~bad], n_sex
def mrmr_topk(A, mi, k):
    C = np.nan_to_num(np.abs(np.corrcoef(A.T))); np.fill_diagonal(C, 0); sel, rem = [], list(range(A.shape[1]))
    for _ in range(min(k, A.shape[1])):
        s = mi[rem] if not sel else mi[rem] - C[np.ix_(rem, sel)].mean(1); b = rem[int(np.argmax(s))]; sel.append(b); rem.remove(b)
    return sel
def per_cohort(task, c, variant):
    d = load(task)[c]; X, n_sex = clean(d["Z"], variant); A = X.T.to_numpy(); y = d["y"].to_numpy()
    mi = mutual_info_classif(A, y, random_state=SEED); top = np.argsort(-mi, kind="stable")[:2000]
    sel = mrmr_topk(A[:, top], mi[top], 20)
    json.dump(dict(task=task, cohort=c, variant=variant, genes_after_clean=int(X.shape[0]), sex_genes_in_cohort_universe=n_sex,
                   sex_genes_removed=n_sex if variant == "nosex" else 0, mrmr_top20=[X.index[top[i]] for i in sel]),
              open(OUT / f"{task}__{c}__{variant}__cohort.json", "w"), indent=1)
    print(task, c, variant, "sex genes in universe:", n_sex)
def cross_auc(D, genes):
    a = []
    for i, (Xi, yi) in enumerate(D):
        m = LogisticRegression(solver="liblinear", max_iter=2000).fit(Xi[genes], yi)
        for j, (Xj, yj) in enumerate(D):
            if i != j: a.append(roc_auc_score(yj, m.predict_proba(Xj[genes])[:, 1]))
    return float(np.mean(a))
def mrmr_rank(X, y):
    mi = mutual_info_classif(X.to_numpy(), y, random_state=0); return [X.columns[i] for i in mrmr_topk(X.to_numpy(), mi, X.shape[1])]
def fold(task, held, variant):
    t = TASKS[task]; tr = [c for c in t["cohorts"] if c != held]; assert held not in tr
    data = load(task); D = []
    for c in tr:
        X, _ = clean(data[c]["Z"], variant); D.append((X.T, data[c]["y"].to_numpy()))
    pools = {c: json.load(open(OUT / f"{task}__{c}__{variant}__cohort.json"))["mrmr_top20"] for c in tr}
    union = sorted(set().union(*pools.values())); allowed = [g for g in union if all(g in Xd.columns for Xd, _ in D)]
    ranks = [mrmr_rank(Xd[allowed], yd) for Xd, yd in D]; score = {g: sum(r.index(g) for r in ranks) for g in allowed}
    ck = OUT / f"{task}__fold_{held}__{variant}.checkpoint.json"; t0 = time.time(); BUDGET = float(os.environ.get("BUDGET", "160"))
    if ck.exists():
        st = json.load(open(ck)); panel, cur, swaps = st["panel"], st["cur"], st["swaps"]; print("resuming", len(panel), swaps)
    else:
        seed = sorted(allowed, key=lambda g: (score[g], g))[:5]; panel = list(seed)
        while len(panel) < min(20, len(allowed)):
            best, ba = None, -np.inf
            for g in sorted(set(allowed) - set(panel)):
                a = cross_auc(D, panel + [g])
                if a > ba: best, ba = g, a
            panel.append(best)
        cur = cross_auc(D, panel); swaps = 0
        json.dump(dict(panel=panel, cur=cur, swaps=swaps), open(ck, "w"))
    improved = True
    while improved:
        if time.time() - t0 > BUDGET: print("budget reached; checkpoint saved, rerun to resume", swaps); return
        improved = False
        for i in range(len(panel)):
            for gin in sorted(set(allowed) - set(panel)):
                trial = panel.copy(); trial[i] = gin; a = cross_auc(D, trial)
                if a > cur + 1e-12:
                    panel, cur, improved = trial, a, True; swaps += 1
                    json.dump(dict(panel=panel, cur=cur, swaps=swaps), open(ck, "w")); break
            if improved: break
    json.dump(dict(task=task, heldout=held, variant=variant, training_cohorts=tr, allowed=len(allowed), panel=panel, mean_cross_auc=cur, swaps=swaps,
                   sex_genes_in_panel=[g for g in panel if g in SEX]), open(OUT / f"{task}__fold_{held}__{variant}.json", "w"), indent=1)
    print(task, "held", held, variant, "| allowed", len(allowed), "| cross-AUC", round(cur, 4), "| swaps", swaps, "| sex genes:", [g for g in panel if g in SEX], "|", round(time.time() - t0), "s swap")
if __name__ == "__main__":
    if sys.argv[1] == "cohort": per_cohort(*sys.argv[2:5])
    else: fold(*sys.argv[2:5])
