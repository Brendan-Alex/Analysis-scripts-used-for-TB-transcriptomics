"""Stage3 items 4-5: Native per-task panel, re-derived with the ORIGINAL algorithm (Work/Code/Gene Compression and
processing: 'MI + mrmr.py' then '20 from Gene Universe') on the corrected data, run INSIDE each LOCO fold.
Per training cohort (depends only on that cohort, so cached and reused by every fold that contains it):
  clean names (drop ''/NA, *phage*, LOC#, C#orf#) -> mutual information (random_state 42) top 2000 -> mRMR top 20
  (relevance MI, redundancy = mean |Pearson r| with already-selected genes; the original per-cohort script used signed r,
  the original unification script used |r|; |r| is used here throughout)
Per fold (the training cohorts of that fold):
  union of per-cohort top-20 -> keep genes present in ALL fold cohorts -> mRMR full ranking per cohort (MI random_state 0)
  -> summed rank -> seed = top 5 -> forward selection to 20 maximising mean cross-cohort AUC (LogisticRegression liblinear,
  max_iter 2000, fit on one cohort, score each other cohort) -> swap refinement (first improving swap, repeat until none)
Data: per-cohort z-scores over the task's two classes (Stage3 common.task_matrix).
Usage: python 02_native_select.py dataset TASK COHORT | fold TASK HELDOUT|ALL
"""
import os, sys, json, time, re
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__)); from common import *
OUT = ST / "results/native"; OUT.mkdir(parents=True, exist_ok=True)
def clean(X):
    g = pd.Index(X.index.astype(str))
    bad = g.isin(["", "NA", "NAN", "---"]) | g.str.contains("PHAGE", case=False) | g.str.match(r"^(LOC\d+|C\d+ORF\d+)$", case=False)
    return X[~bad]
def mrmr_topk(X, y, k, mi):
    C = np.abs(np.corrcoef(X.T)); np.fill_diagonal(C, 0); C = np.nan_to_num(C)
    sel = []; rem = list(range(X.shape[1]))
    for _ in range(min(k, X.shape[1])):
        if not sel: s = mi[rem]
        else: s = mi[rem] - C[np.ix_(rem, sel)].mean(1)
        b = rem[int(np.argmax(s))]; sel.append(b); rem.remove(b)
    return sel
def dataset(task, c):
    X, y = task_matrix(task, c); X = clean(X)
    A = X.T.to_numpy(); yy = y.to_numpy()
    mi = mutual_info_classif(A, yy, random_state=SEED)
    top = np.argsort(-mi, kind="stable")[:2000]
    sel = mrmr_topk(A[:, top], yy, 20, mi[top])
    genes = [X.index[top[i]] for i in sel]
    json.dump(dict(task=task, cohort=c, n=int(len(yy)), genes_in_file=int(X.shape[0]), mi_top20=[X.index[i] for i in top[:20]], mrmr_top20=genes),
              open(OUT / f"{task}__{c}__dataset.json", "w"), indent=1)
    print(task, c, genes)
def cross_auc(D, genes):
    aucs = []
    for i, (Xi, yi) in enumerate(D):
        m = LogisticRegression(solver="liblinear", max_iter=2000).fit(Xi[genes], yi)
        for j, (Xj, yj) in enumerate(D):
            if i != j: aucs.append(roc_auc_score(yj, m.predict_proba(Xj[genes])[:, 1]))
    return float(np.mean(aucs))
def mrmr_rank(X, y):
    mi = mutual_info_classif(X.to_numpy(), y, random_state=0)
    return [X.columns[i] for i in mrmr_topk(X.to_numpy(), y, X.shape[1], mi)]
def fold(task, held):
    t = TASKS[task]; coh = [c for c in t["train"] if c != held]
    pools = {c: json.load(open(OUT / f"{task}__{c}__dataset.json"))["mrmr_top20"] for c in coh}
    D = []
    for c in coh:
        X, y = task_matrix(task, c); D.append((X.T, y.to_numpy()))
    union = sorted(set().union(*pools.values()))
    allowed = [g for g in union if all(g in Xd.columns for Xd, _ in D)]
    ranks = [mrmr_rank(Xd[allowed], yd) for Xd, yd in D]
    score = {g: sum(r.index(g) for r in ranks) for g in allowed}
    ranked = sorted(allowed, key=lambda g: (score[g], g)); seed = ranked[:5]
    panel = list(seed); flog = []
    while len(panel) < min(20, len(allowed)):
        best, ba = None, -np.inf
        for g in sorted(set(allowed) - set(panel)):
            a = cross_auc(D, panel + [g])
            if a > ba: best, ba = g, a
        panel.append(best); flog.append(dict(added=best, mean_cross_auc=ba))
    cur = cross_auc(D, panel); slog = []; improved = True; t0 = time.time()
    while improved:
        improved = False
        for i, gout in enumerate(list(panel)):
            for gin in sorted(set(allowed) - set(panel)):
                trial = panel.copy(); trial[i] = gin; a = cross_auc(D, trial)
                if a > cur + 1e-12:
                    slog.append(dict(out=gout, inn=gin, before=cur, after=a)); panel, cur, improved = trial, a, True; break
            if improved: break
    res = dict(task=task, heldout=held, train_cohorts=coh, union_size=len(union), allowed_size=len(allowed), seed=seed,
               panel=panel, final_mean_cross_auc=cur, forward_log=flog, swap_log=slog, swap_seconds=round(time.time() - t0, 1))
    json.dump(res, open(OUT / f"{task}__fold_{held}.json", "w"), indent=1)
    print(task, "held", held, "| allowed", len(allowed), "| panel", len(panel), "| cross-AUC", round(cur, 4), "| swaps", len(slog))
if __name__ == "__main__":
    if sys.argv[1] == "dataset": dataset(sys.argv[2], sys.argv[3])
    else: fold(sys.argv[2], sys.argv[3])
