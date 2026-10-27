"""Stage5 summaries (run after run_stage5.py). Per fold, never averaged across folds.
Bootstrap CI: 1000 resamples of the held-out cohort, seed 42, percentile. Random comparison as Stage4.
Outputs in results/: per_fold_auc.csv, generalisation_gap.csv, change_vs_stage4.csv, berry86_tlr7_effect.csv,
native_stability_*.csv, task_difficulty.csv, alpha_tuning.csv, alpha_grid_long.csv"""
import os, sys, json, re, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
S5 = __import__("pathlib").Path(__file__).resolve().parents[1]; MT = S5.parent; S4 = MT / "Stage4"; CR = MT / "Clean_rebuild"
R = S5 / "results"; SEED = 42; SEX = set(pd.read_csv(S5 / "data/sex_chromosome_genes.csv").Gene.astype(str))
TASKS = {"ATB_vs_LTB": ["GSE19491", "GSE37250", "GSE73408", "GSE101705", "GSE107994"], "ATB_vs_OD": ["GSE37250", "GSE73408", "GSE42834", "GSE83456"],
         "LTB_vs_CON": ["GSE19439", "GSE19444", "GSE28623", "GSE107994"]}
USER = {"Berry86": {"GSE19444", "GSE19439"}, "Kaforou_ATB_v_LTB": {"GSE37250", "GSE19491"}, "Kaforou_ATB_v_OD": {"GSE37250", "GSE19491"},
        "Zak16": {"GSE19439", "GSE19444", "GSE19442", "GSE37250", "GSE42834"}, "Sweeney3": {"GSE19491", "GSE37250", "GSE42834", "GSE28623"}}
ADDED = {"Berry86": {"GSE19491"}, "Zak16": {"GSE19491"}, "Sweeney3": {"GSE19439", "GSE19444"}}   # accepted by user (Stage5)
for d in (USER, ADDED): d["Berry86_noTLR7"] = d.get("Berry86", set())
LAB = json.load(open(S5 / "data/labels.json")); Y = {t: {c: np.array(v[1]) for c, v in LAB[t].items()} for t in LAB}
F = pd.read_csv(S5 / "qc/folds.csv"); G = pd.read_csv(R / "global_shift_heldout.csv")
def boot_ci(y, s):
    rng = np.random.default_rng(SEED); I = np.stack([rng.integers(0, len(y), len(y)) for _ in range(1000)]); Yb, Sb = y[I], s[I]
    Rk = rankdata(Sb, axis=1); n1 = Yb.sum(1); n0 = Yb.shape[1] - n1; ok = (n1 > 0) & (n0 > 0)
    b = ((Rk * Yb).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1); return np.percentile(b[ok], [2.5, 97.5])
rows, gap, alpha_rows, grid_rows = [], [], [], []
for task, coh in TASKS.items():
    yy = Y[task]; rnd = {}; cur = []
    for line in open(R / f"bench/{task}__records.jsonl"):
        r = json.loads(line); h = r["heldout"]
        if r["kind"] == "rnd":
            for m, v in r["scores"].items(): rnd.setdefault((h, r["panel"], m), []).append(roc_auc_score(yy[h], np.nan_to_num(v[h])))
            rnd.setdefault((h, r["panel"], "_zf"), []).append(r["diag"]["zero_filled_in_test"][h])
        else: cur.append(r)
    for r in cur:
        h, p, d = r["heldout"], r["panel"], r["diag"]; f = F[(F.task == task) & (F.heldout == h)].iloc[0]; gs = G[(G.task == task) & (G.heldout == h)].iloc[0]
        for m, v in r["scores"].items():
            s = np.asarray(v[h]); a = roc_auc_score(yy[h], s); lo, hi = boot_ci(yy[h], s)
            ra = np.asarray(rnd.get((h, p, "MLP" if m == "MLP_alpha" else m), []))
            tr = {c: roc_auc_score(yy[c], v[c]) for c in r["training"]}
            pooled = roc_auc_score(np.concatenate([yy[c] for c in r["training"]]), np.concatenate([v[c] for c in r["training"]]))
            u, ad = h in USER.get(p, set()), h in ADDED.get(p, set())
            rows.append(dict(task=task, heldout=h, training_cohorts=";".join(r["training"]), n_train=f.n_train, train_pos=f.train_pos, train_neg=f.train_neg,
                n_test=f.n_test, test_pos=f.test_pos, test_neg=f.test_neg, panel=p, method=m, n_features=d["n_features"], n_sex_features=d["n_sex_features"],
                zero_filled_in_heldout=d["zero_filled_in_test"][h], AUC=round(a, 3), CI_low=round(lo, 3), CI_high=round(hi, 3),
                MLP_alpha=d["MLP_alpha_chosen"] if m == "MLP_alpha" else (1e-4 if m == "MLP" else None),
                MLP_converged=d["MLP_alpha_converged"] if m == "MLP_alpha" else (d["MLP_converged"] if m == "MLP" else None),
                rand_n=len(ra), rand_median=round(float(np.median(ra)), 3), rand_p2_5=round(float(np.percentile(ra, 2.5)), 3),
                rand_p97_5=round(float(np.percentile(ra, 97.5)), 3), prop_random_ge_curated=round(float((ra >= a).mean()), 3),
                rand_max_zero_filled=int(max(rnd[(h, p, "_zf")])), random_reference="random MLP at this panel's chosen alpha" if m.startswith("MLP") else m,
                overlap_user_list=u, overlap_added_accepted=ad, optimistic=u or ad,
                heldout_allgene_mean_AUC=gs.allgene_mean_AUC, heldout_allgene_mean_CI=gs.CI, heldout_share_genes_dev_gt_0_2=gs.share_genes_AUCdev_gt_0_2,
                heldout_quantile_normalised=gs.quantile_normalised))
            gap.append(dict(task=task, heldout=h, panel=p, method=m, train_pooled_AUC=round(pooled, 3), train_min_cohort_AUC=round(min(tr.values()), 3),
                train_max_cohort_AUC=round(max(tr.values()), 3), heldout_AUC=round(a, 3), gap_pooled_minus_heldout=round(pooled - a, 3)))
        inner, grid = d["MLP_alpha_inner_AUC"], d["MLP_alpha_grid_heldout_AUC"]; al = list(inner)
        iv, gv = np.array([inner[k] for k in al]), np.array([grid[k] for k in al]); ch = str(d["MLP_alpha_chosen"])
        alpha_rows.append(dict(task=task, heldout=h, panel=p, n_training_cohorts=len(r["training"]), chosen_alpha=d["MLP_alpha_chosen"],
            chosen_is_extreme=d["MLP_alpha_chosen"] in (1e-4, 10.0), inner_AUC_chosen=round(inner[ch], 3), inner_AUC_range=round(iv.max() - iv.min(), 3),
            inner_margin_best_minus_second=round(float(np.sort(iv)[-1] - np.sort(iv)[-2]), 4),
            heldout_AUC_chosen=round(grid[ch], 3), heldout_AUC_min_grid=round(gv.min(), 3), heldout_AUC_max_grid=round(gv.max(), 3),
            heldout_AUC_range_grid=round(gv.max() - gv.min(), 3), heldout_AUC_alpha_1e4=round(grid["0.0001"], 3),
            heldout_best_alpha=float(al[int(np.argmax(gv))]), spearman_inner_vs_heldout_over_grid=round(float(spearmanr(iv, gv)[0]), 3) if np.ptp(iv) > 0 and np.ptp(gv) > 0 else None))
        for k in al: grid_rows.append(dict(task=task, heldout=h, panel=p, alpha=float(k), inner_LOCO_AUC=round(inner[k], 4), heldout_AUC=round(grid[k], 4),
                                           **{f"inner_{c}": round(x, 4) for c, x in d["MLP_alpha_inner_AUC_per_cohort"][k].items()}))
A = pd.DataFrame(rows); A.to_csv(R / "per_fold_auc.csv", index=False); pd.DataFrame(gap).to_csv(R / "generalisation_gap.csv", index=False)
AL = pd.DataFrame(alpha_rows); AL.to_csv(R / "alpha_tuning.csv", index=False); pd.DataFrame(grid_rows).to_csv(R / "alpha_grid_long.csv", index=False)
# ---- change vs Stage4
B = pd.read_csv(S4 / "results/per_fold_auc.csv"); k = ["task", "heldout", "panel", "method"]
C = A.merge(B[k + ["AUC", "CI_low", "CI_high", "n_features", "zero_filled_in_heldout", "rand_median", "prop_random_ge_curated", "MLP_alpha"]], on=k, how="left", suffixes=("", "_S4"))
C["delta_AUC"] = (C.AUC - C.AUC_S4).round(3); C["delta_rand_median"] = (C.rand_median - C.rand_median_S4).round(3)
C["what_changed"] = np.where(C.panel == "Native", "panel re-selected on full-coverage universe", np.where(C.panel == "Berry86_noTLR7", "new panel (no Stage4 counterpart)",
                    "panel unchanged; random universe changed; any AUC change = platform/library"))
C[k + ["what_changed", "n_features_S4", "n_features", "zero_filled_in_heldout_S4", "zero_filled_in_heldout", "AUC_S4", "AUC", "delta_AUC", "CI_low_S4", "CI_high_S4", "CI_low", "CI_high",
       "MLP_alpha_S4", "MLP_alpha", "rand_median_S4", "rand_median", "delta_rand_median", "prop_random_ge_curated_S4", "prop_random_ge_curated"]].to_csv(R / "change_vs_stage4.csv", index=False)
nat = []
for task, coh in TASKS.items():
    for h in coh:
        p4 = json.load(open(S4 / f"results/native/{task}__fold_{h}__nosex.json")); p5 = json.load(open(R / f"native/{task}__fold_{h}.json"))
        nat.append(dict(task=task, heldout=h, allowed_S4=p4["allowed"], allowed_S5=p5["allowed"], train_cross_AUC_S4=round(p4["mean_cross_auc"], 3), train_cross_AUC_S5=round(p5["mean_cross_auc"], 3),
                        genes_shared_S4_S5=len(set(p4["panel"]) & set(p5["panel"])), zero_filled_S5=len(p5["zero_filled_in_heldout"])))
pd.DataFrame(nat).to_csv(R / "native_change_vs_stage4.csv", index=False)
# ---- TLR7
b1 = A[A.panel == "Berry86"].set_index(k[:2] + ["method"]); b2 = A[A.panel == "Berry86_noTLR7"].set_index(k[:2] + ["method"])
T = pd.DataFrame({"AUC_with_TLR7": b1.AUC, "CI_with": b1.CI_low.astype(str) + "-" + b1.CI_high.astype(str), "AUC_without_TLR7": b2.AUC,
                  "CI_without": b2.CI_low.astype(str) + "-" + b2.CI_high.astype(str), "n_features_with": b1.n_features, "n_features_without": b2.n_features,
                  "prop_random_ge_with": b1.prop_random_ge_curated, "prop_random_ge_without": b2.prop_random_ge_curated})
T["delta_without_minus_with"] = (T.AUC_without_TLR7 - T.AUC_with_TLR7).round(3); T.reset_index().to_csv(R / "berry86_tlr7_effect.csv", index=False)
# ---- Native stability (final)
H = pd.read_csv(CR / "resources/hgnc_entries.tsv", sep="\t", dtype=str, usecols=["Approved symbol", "Approved name", "Locus type", "Status"])
H = H[H.Status == "Approved"].drop_duplicates("Approved symbol").set_index("Approved symbol")
def category(g):
    lt = H["Locus type"].get(g); nm = H["Approved name"].get(g, "")
    if lt is None: return "not in HGNC table", ""
    fl = []
    if re.match(r"^OR\d+[A-Z]+\d+", g) or "olfactory receptor" in str(nm).lower(): fl.append("olfactory receptor")
    if lt == "RNA, micro" or re.match(r"^MIR\d", g): fl.append("microRNA")
    if "pseudogene" in str(lt).lower(): fl.append("pseudogene")
    if lt == "RNA, long non-coding": fl.append("lncRNA")
    return lt, ";".join(fl)
gl, fs, pw = [], [], []
for task, coh in TASKS.items():
    P = {h: json.load(open(R / f"native/{task}__fold_{h}.json"))["panel"] for h in coh}; cnt = pd.Series([g for v in P.values() for g in v]).value_counts()
    for g in sorted(cnt.index, key=lambda g: (-cnt[g], g)):
        lt, fl = category(g)
        gl.append(dict(task=task, gene=g, n_folds_selected=int(cnt[g]), of_folds=len(coh), **{f"held_{h}": int(g in P[h]) for h in coh},
                       hgnc_locus_type=lt, hgnc_name=H["Approved name"].get(g, ""), flag=fl, sex_linked=g in SEX))
    inter = set.intersection(*map(set, P.values())); cats = pd.Series([category(g)[1] or "none" for g in cnt.index])
    fs.append(dict(task=task, n_folds=len(coh), genes_per_fold=20, n_distinct_genes=len(cnt), n_shared_all_folds=len(inter), shared_all_folds=";".join(sorted(inter)),
                   n_in_ge_half_folds=int((cnt >= len(coh) / 2).sum()), n_selected_once=int((cnt == 1).sum()),
                   n_olfactory_receptor=int(sum("olfactory" in category(g)[1] for g in cnt.index)), n_microRNA=int(sum("microRNA" in category(g)[1] for g in cnt.index)),
                   n_pseudogene=int(sum("pseudogene" in category(g)[1] for g in cnt.index)), n_lncRNA=int(sum("lncRNA" in category(g)[1] for g in cnt.index)),
                   n_protein_coding=int(sum(category(g)[0] == "gene with protein product" for g in cnt.index)), n_not_in_HGNC=int(sum(category(g)[0] == "not in HGNC table" for g in cnt.index))))
    for a_, b_ in itertools.combinations(coh, 2):
        i = len(set(P[a_]) & set(P[b_])); pw.append(dict(task=task, fold_a=a_, fold_b=b_, overlap=i, jaccard=round(i / len(set(P[a_]) | set(P[b_])), 3)))
    pools = {c: set(json.load(open(R / f"native/{task}__{c}__cohort.json"))["mrmr_top20"]) for c in coh}
    fs[-1]["per_cohort_mRMR_pool_pairwise_overlap"] = ";".join(f"{a_}/{b_}:{len(pools[a_] & pools[b_])}" for a_, b_ in itertools.combinations(coh, 2))
pd.DataFrame(gl).to_csv(R / "native_stability_genes.csv", index=False); pd.DataFrame(fs).to_csv(R / "native_stability_summary.csv", index=False)
pd.DataFrame(pw).to_csv(R / "native_stability_pairwise.csv", index=False)
# ---- task difficulty
S = pd.read_csv(R / "gene_level_separability.csv"); S = S[S.scope == "all_measured"]; td = []
for _, s in S.iterrows():
    sub = A[(A.task == s.task) & (A.heldout == s.cohort)]
    row = dict(task=s.task, cohort=s.cohort, n_ATB_or_pos=s.n_pos, n_neg=s.n_neg, genes_measured=s.n_genes, prop_genes_AUC_gt_0_8=s.prop_AUC_gt_0_8,
               prop_genes_AUC_lt_0_2=s.prop_AUC_lt_0_2, prop_genes_either_dev_gt_0_3=s.prop_either_dev_gt_0_3)
    for p in ["Sweeney3", "Zak16", "Native", "Berry86_noTLR7"]:
        q = sub[(sub.panel == p) & (sub.method == "DirAvg")]
        if len(q): row[f"random_median_DirAvg_size{int(q.n_features.iloc[0])}"] = q.rand_median.iloc[0]
        q = sub[(sub.panel == p) & (sub.method == "LR")]
        if len(q): row[f"random_median_LR_size{int(q.n_features.iloc[0])}"] = q.rand_median.iloc[0]
    row["best_curated_AUC_any_method"] = sub[sub.panel != "Native"].AUC.max(); row["Native_AUC_range"] = f"{sub[sub.panel=='Native'].AUC.min():.3f}-{sub[sub.panel=='Native'].AUC.max():.3f}"
    td.append(row)
pd.DataFrame(td).to_csv(R / "task_difficulty.csv", index=False)
print("rows", len(A), "| alpha rows", len(AL)); print(A.groupby("task").size())
