"""Stage4 summaries, per fold (never averaged across folds).
AUC CI: 1000 bootstrap resamples of the held-out cohort, seed 42 (same draws as sequential rng.integers(0,n,n)), percentile 95%.
Random comparison: the 50 size-matched draws of the same fold and method (MLP_alpha is compared with random MLP run at that
panel's chosen alpha; plain MLP at alpha 1e-4 is compared with the same random MLP and flagged as such)."""
import os, sys, json, pickle, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__)); from common4 import *
R = S4 / "results"; SEX = set(pd.read_csv(S4 / "data/sex_chromosome_genes.csv").Gene)
F = pd.read_csv(S4 / "qc/folds.csv"); G = pd.read_csv(R / "global_shift_heldout.csv")
def boot_ci(y, s, reps=1000):
    rng = np.random.default_rng(SEED); I = np.stack([rng.integers(0, len(y), len(y)) for _ in range(reps)]); Y, S = y[I], s[I]
    Rk = rankdata(S, axis=1); n1 = Y.sum(1); n0 = Y.shape[1] - n1; ok = (n1 > 0) & (n0 > 0)
    b = ((Rk * Y).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1); return np.percentile(b[ok], [2.5, 97.5])
USER_LIST = {"Berry86": {"GSE19444", "GSE19439"}, "Kaforou_ATB_v_LTB": {"GSE37250", "GSE19491"}, "Kaforou_ATB_v_OD": {"GSE37250", "GSE19491"},
             "Zak16": {"GSE19439", "GSE19444", "GSE19442", "GSE37250", "GSE42834"}, "Sweeney3": {"GSE19491", "GSE37250", "GSE42834", "GSE28623"}}
def flag(p, h):
    rel = OVERLAP.get(p, {}).get(h); return rel or ""
rows, gap, sexrows, NANFIX = [], [], [], []
for task, t in TASKS.items():
    D = pickle.load(open(S4 / f"data/cache/{task}.pkl", "rb")); Y = {c: D[c]["y"].to_numpy() for c in D}
    recs = [json.loads(l) for l in open(R / f"bench/{task}__records.jsonl")]
    rnd, rzf, nanfix = {}, {}, []
    for r in recs:
        for m, v in r["scores"].items():
            for c in v:
                if np.isnan(v[c]).any():   # DirAvg over zero genes present (every drawn gene absent from the cohort): zero-fill convention -> constant score
                    nanfix.append(dict(task=task, key=r["key"], method=m, cohort=c)); v[c] = list(np.nan_to_num(np.asarray(v[c]), nan=0.0))
        if r["kind"] == "rnd":
            rzf.setdefault((r["heldout"], r["panel"]), []).append(r["diag"]["zero_filled_in_test"][r["heldout"]] / max(r["diag"]["n_features"], 1))
            for m, v in r["scores"].items(): rnd.setdefault((r["heldout"], r["panel"], m), []).append(roc_auc_score(Y[r["heldout"]], v[r["heldout"]]))
    NANFIX.extend(nanfix)
    for r in [r for r in recs if r["kind"] == "cur"]:
        h, p, d = r["heldout"], r["panel"], r["diag"]; f = F[(F.task == task) & (F.heldout == h)].iloc[0]; gs = G[(G.task == task) & (G.heldout == h)].iloc[0]
        for m, v in r["scores"].items():
            s = np.asarray(v[h]); a = roc_auc_score(Y[h], s); lo, hi = boot_ci(Y[h], s)
            rk = "MLP" if m in ("MLP", "MLP_alpha") else m; ra = np.asarray(rnd.get((h, p, rk), []))
            tr = {c: roc_auc_score(Y[c], v[c]) for c in r["training"]}
            pooled = roc_auc_score(np.concatenate([Y[c] for c in r["training"]]), np.concatenate([v[c] for c in r["training"]]))
            rel = flag(p, h)
            rows.append(dict(task=task, heldout=h, training_cohorts=";".join(r["training"]), n_train=f.n_train, train_pos=f.train_pos, train_neg=f.train_neg,
                n_test=f.n_test, test_pos=f.test_pos, test_neg=f.test_neg, panel=p, method=m, n_features=d["n_features"], n_sex_features=d["n_sex_features"],
                zero_filled_in_heldout=d["zero_filled_in_test"][h], AUC=round(a, 3), CI_low=round(lo, 3), CI_high=round(hi, 3),
                MLP_alpha=d["MLP_alpha_chosen"] if m == "MLP_alpha" else (1e-4 if m == "MLP" else None),
                MLP_converged=d["MLP_alpha_converged"] if m == "MLP_alpha" else (d["MLP_converged"] if m == "MLP" else None),
                rand_n=len(ra), rand_median=round(float(np.median(ra)), 3) if len(ra) else None,
                rand_p2_5=round(float(np.percentile(ra, 2.5)), 3) if len(ra) else None, rand_p97_5=round(float(np.percentile(ra, 97.5)), 3) if len(ra) else None,
                rand_mean_zero_filled_fraction=round(float(np.mean(rzf[(h, p)])), 3) if (h, p) in rzf else None,
                prop_random_ge_curated=round(float((ra >= a).mean()), 3) if len(ra) else None,
                random_reference=("random MLP at this panel's chosen alpha" if m in ("MLP", "MLP_alpha") else m) if len(ra) else "",
                derivation_overlap=rel, overlap_in_user_list=h in USER_LIST.get(p, set()),
                overlap_added_subseries_containment=bool(rel) and h not in USER_LIST.get(p, set()), optimistic=bool(rel),
                heldout_allgene_mean_AUC=gs.allgene_mean_AUC, heldout_allgene_mean_CI=gs.CI, heldout_share_genes_dev_gt_0_2=gs.share_genes_AUCdev_gt_0_2,
                heldout_quantile_normalised=gs.quantile_normalised))
            gap.append(dict(task=task, heldout=h, panel=p, method=m, train_pooled_AUC=round(pooled, 3),
                train_min_cohort_AUC=round(min(tr.values()), 3), train_max_cohort_AUC=round(max(tr.values()), 3),
                train_per_cohort=";".join(f"{c}={x:.3f}" for c, x in tr.items()), heldout_AUC=round(a, 3), gap_pooled_minus_heldout=round(pooled - a, 3)))
pd.DataFrame(NANFIX).to_csv(S4 / "qc/nan_scores_set_to_constant.csv", index=False)
A = pd.DataFrame(rows); A.to_csv(R / "per_fold_auc.csv", index=False)
pd.DataFrame(gap).to_csv(R / "generalisation_gap.csv", index=False)
# Native stability and sex exclusion
st, pw = [], []
for task, t in TASKS.items():
    P = {h: json.load(open(R / f"native/{task}__fold_{h}__nosex.json"))["panel"] for h in t["cohorts"]}
    W = {h: json.load(open(R / f"native/{task}__fold_{h}__withsex.json"))["panel"] for h in t["cohorts"]}
    inter = set.intersection(*map(set, P.values())); cnt = pd.Series([g for v in P.values() for g in v]).value_counts()
    st.append(dict(task=task, n_folds=len(P), genes_per_fold=";".join(f"{h}:{len(v)}" for h, v in P.items()), n_shared_all_folds=len(inter),
                   shared_all_folds=";".join(sorted(inter)), n_distinct_genes=len(cnt), genes_in_ge_half_folds=";".join(sorted(cnt[cnt >= len(P) / 2].index))))
    for a, b in itertools.combinations(P, 2):
        i = len(set(P[a]) & set(P[b])); pw.append(dict(task=task, fold_a=a, fold_b=b, overlap=i, jaccard=round(i / len(set(P[a]) | set(P[b])), 3)))
    for h in t["cohorts"]:
        cj = {c: json.load(open(R / f"native/{task}__{c}__nosex__cohort.json")) for c in t["cohorts"] if c != h}
        sexrows.append(dict(task=task, heldout=h, sex_genes_removed_per_training_cohort=";".join(f"{c}:{v['sex_genes_removed']}" for c, v in cj.items()),
            panel_nosex=";".join(P[h]), panel_withsex=";".join(W[h]), n_shared=len(set(P[h]) & set(W[h])),
            sex_genes_in_withsex_panel=";".join(g for g in W[h] if g in SEX), only_in_nosex=";".join(g for g in P[h] if g not in W[h]),
            only_in_withsex=";".join(g for g in W[h] if g not in P[h])))
    for p in t["panels"]:
        if p != "Native": sexrows.append(dict(task=task, heldout="(curated panel, not altered)", panel_nosex=p, sex_genes_in_withsex_panel=";".join(g for g in panel(p) if g in SEX)))
pd.DataFrame(st).to_csv(R / "native_stability.csv", index=False); pd.DataFrame(pw).to_csv(R / "native_pairwise_overlap.csv", index=False)
pd.DataFrame(sexrows).to_csv(R / "sex_gene_exclusion.csv", index=False)
# ATB vs OD comparator type
M = manifest(); inc = M[M.included_stage3]; ty = {"GSE37250": "indication-defined", "GSE73408": "disease-defined (pneumonia)",
      "GSE42834": "disease-defined (sarcoid, lung cancer, pneumonia)", "GSE83456": "disease-defined (sarcoid, lung cancer, pneumonia)"}
D = pickle.load(open(S4 / "data/cache/ATB_vs_OD.pkl", "rb")); od = {c: int((1 - D[c]["y"]).sum()) for c in D}
c9 = []
for h in TASKS["ATB_vs_OD"]["cohorts"]:
    tr = [c for c in TASKS["ATB_vs_OD"]["cohorts"] if c != h]; fi = sum(od[c] for c in tr if c == "GSE37250") / sum(od[c] for c in tr)
    sub = A[(A.task == "ATB_vs_OD") & (A.heldout == h)]
    c9.append(dict(heldout=h, heldout_OD_type=ty[h], training_OD_indication_defined_fraction=round(fi, 3),
                   **{f"{p}|{m}": a for p, m, a in zip(sub.panel, sub.method, sub.AUC)}))
pd.DataFrame(c9).to_csv(R / "atb_vs_od_comparator_type.csv", index=False)
print("rows", len(A)); print(A.groupby("task").size())
