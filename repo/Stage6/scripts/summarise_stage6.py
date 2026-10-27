"""Stage6 summaries from the run_stage6.py outputs (plus Stage5 labels/records, read-only). Per fold, never averaged."""
import json, itertools, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
S6 = Path(__file__).resolve().parents[1]; MT = S6.parent; S5 = MT / "Stage5"; R = S6 / "results"; SEED = 42
TASKS = {"ATB_vs_LTB": ["GSE19491", "GSE37250", "GSE73408", "GSE101705", "GSE107994"], "ATB_vs_OD": ["GSE37250", "GSE73408", "GSE42834", "GSE83456"],
         "LTB_vs_CON": ["GSE19439", "GSE19444", "GSE28623", "GSE107994"]}
PLAT = {"GSE19439": "GPL6947", "GSE19444": "GPL6947", "GSE19491": "GPL6947", "GSE37250": "GPL10558", "GSE42834": "GPL10558", "GSE83456": "GPL10558",
        "GSE28623": "GPL4133", "GSE73408": "GPL11532", "GSE101705": "GPL18573", "GSE107994": "GPL20301"}
LAB = json.load(open(S5 / "data/labels.json")); Y = {t: {c: np.array(v[1]) for c, v in LAB[t].items()} for t in LAB}
ALPHAS = ["0.0001", "0.001", "0.01", "0.1", "1.0", "10.0", "100.0", "1000.0"]; COLLAPSE = 1e-3
def boot_ci(y, s):
    rng = np.random.default_rng(SEED); I = np.stack([rng.integers(0, len(y), len(y)) for _ in range(1000)]); Yb, Sb = y[I], s[I]
    Rk = rankdata(Sb, axis=1); n1 = Yb.sum(1); n0 = Yb.shape[1] - n1; ok = (n1 > 0) & (n0 > 0)
    b = ((Rk * Yb).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1); return np.percentile(b[ok], [2.5, 97.5])
# ================================================================= 1. alpha
old, oldrnd = {}, {}
for t in TASKS:
    for line in open(S5 / f"results/bench/{t}__records.jsonl"):
        r = json.loads(line); h = r["heldout"]
        if r["kind"] == "cur": old[(t, h, r["panel"])] = r["diag"]["MLP_alpha_chosen"]
        else: oldrnd.setdefault((t, h, r["panel"]), []).append(roc_auc_score(Y[t][h], r["scores"]["MLP"][h]))
newrnd = {(r["task"], r["heldout"], r["panel"]): r["random_MLP_AUC"] for r in map(json.loads, open(R / "alpha/random_mlp_new_alpha.jsonl"))}
rows, long = [], []
for r in map(json.loads, open(R / "alpha/alpha_grid.jsonl")):
    t, h, p = r["task"], r["heldout"], r["panel"]; y = Y[t][h]
    inn = np.array([r["inner_AUC"][a] for a in ALPHAS]); ho = np.array([r["heldout_AUC"][a] for a in ALPHAS]); sd = np.array([r["heldout_pred_sd"][a] for a in ALPHAS])
    ch = str(r["chosen_alpha"]); ci = ALPHAS.index(ch); s = np.array(r["heldout_scores"][ch]); lo, hi = boot_ci(y, s)
    ok = sd >= COLLAPSE; hb = int(np.argmax(np.where(ok, ho, -1)))          # best held-out among non-collapsed points
    top3 = ho[5:]; plateau = bool(np.ptp(ho[4:]) <= 0.01)                    # alpha 1..1000 within 0.01
    if ci == 7: status = "upper boundary (1000)"
    elif ci == 0: status = "lower boundary (1e-4)"
    else: status = "interior"
    ra = np.asarray(newrnd.get((t, h, p), oldrnd[(t, h, p)])); a_ch = ho[ci]
    rows.append(dict(task=t, heldout=h, panel=p, n_training_cohorts=len(r["training"]), n_features=r["n_features"],
        chosen_alpha_stage5=old[(t, h, p)], chosen_alpha_stage6=r["chosen_alpha"], chosen_changed=old[(t, h, p)] != r["chosen_alpha"], chosen_status=status,
        chosen_prediction_collapsed=bool(sd[ci] < COLLAPSE), heldout_pred_sd_at_chosen=round(float(sd[ci]), 5),
        **{f"heldout_AUC_a{a}": round(float(x), 3) for a, x in zip(ALPHAS, ho)}, **{f"inner_AUC_a{a}": round(float(x), 3) for a, x in zip(ALPHAS, inn)},
        **{f"pred_sd_a{a}": round(float(x), 5) for a, x in zip(ALPHAS, sd)},
        heldout_AUC_chosen=round(float(a_ch), 3), CI_low=round(lo, 3), CI_high=round(hi, 3), heldout_best_alpha=float(ALPHAS[int(np.argmax(ho))]),
        heldout_best_alpha_noncollapsed=float(ALPHAS[hb]), heldout_peak_interior=0 < int(np.argmax(ho)) < 7, heldout_plateau_alpha_ge1=plateau,
        heldout_AUC_range=round(float(np.ptp(ho)), 3), chosen_minus_best=round(float(a_ch - ho.max()), 3), chosen_minus_alpha10=round(float(a_ch - ho[5]), 3),
        chosen_minus_alpha1e4=round(float(a_ch - ho[0]), 3), spearman_inner_vs_heldout=round(float(spearmanr(inn, ho)[0]), 3) if np.ptp(inn) > 0 and np.ptp(ho) > 0 else None,
        spearman_inner_vs_heldout_noncollapsed=round(float(spearmanr(inn[ok], ho[ok])[0]), 3) if ok.sum() > 2 and np.ptp(inn[ok]) > 0 and np.ptp(ho[ok]) > 0 else None,
        inner_margin=round(float(np.sort(inn)[-1] - np.sort(inn)[-2]), 4), rand_MLP_median=round(float(np.median(ra)), 3), prop_random_ge=round(float((ra >= a_ch).mean()), 3),
        random_refit_at_new_alpha=(t, h, p) in newrnd, n_iter_full_at_chosen=r["n_iter_full"][ch]))
    for a, x, z, q in zip(ALPHAS, inn, ho, sd): long.append(dict(task=t, heldout=h, panel=p, alpha=float(a), inner_AUC=round(float(x), 4), heldout_AUC=round(float(z), 4), heldout_pred_sd=q,
                                                              collapsed=q < COLLAPSE, **{f"inner_{c}": round(v, 4) for c, v in r["inner_AUC_per_cohort"][a].items()}))
AL = pd.DataFrame(rows).sort_values(["task", "heldout", "panel"]); AL.to_csv(R / "alpha_extended.csv", index=False); pd.DataFrame(long).to_csv(R / "alpha_extended_long.csv", index=False)
# ================================================================= 2. consensus failure
rk = {}
for t, coh in TASKS.items():
    for c in coh: rk[(t, c)] = pd.read_csv(R / f"rank/{t}__{c}.csv.gz").set_index("Gene")
seeds = {(r["task"], r["heldout"]): r["seed"] for r in map(json.loads, open(R / "seeds/seeds.jsonl"))}
def depth(M):
    """M: genes x cohorts rank matrix (inf = not ranked). depth_2 = smallest k at which some gene is in the top-k of >=2 cohorts;
       depth_all = smallest k at which some gene is in the top-k of every cohort."""
    S = np.sort(M, axis=1); return float(S[:, 1].min()), float(S[:, -1].min())
rng = np.random.default_rng(SEED); NULL = {}
def null_depth(N, m, reps=1000):
    key = (N, m)
    if key not in NULL:
        d2, da = [], []
        for _ in range(reps):
            M = np.stack([rng.permutation(N) + 1 for _ in range(m)], 1); a, b = depth(M); d2.append(a); da.append(b)
        NULL[key] = (np.median(d2), np.percentile(d2, [2.5, 97.5]), np.median(da), np.percentile(da, [2.5, 97.5]), np.array(d2), np.array(da))
    return NULL[key]
def expct(N, m, k):
    from scipy.stats import binom
    q = k / N; return N * binom.sf(1, m, q), N * q ** m
cons, prov, pairs = [], [], []
for t, coh in TASKS.items():
    for h in coh:
        tr = [c for c in coh if c != h]; U = rk[(t, tr[0])].index; N = len(U)
        MI = np.stack([rk[(t, c)].loc[U, "MI_rank"].to_numpy(float) for c in tr], 1)
        MR = np.stack([rk[(t, c)].loc[U, "mRMR_rank_in_top2000"].fillna(np.inf).to_numpy(float) for c in tr], 1)
        pools = {c: set(json.load(open(S5 / f"results/native/{t}__{c}__cohort.json"))["mrmr_top20"]) for c in tr}
        panel = json.load(open(S5 / f"results/native/{t}__fold_{h}.json"))["panel"]
        mi2, miall = depth(MI); mr2, mrall = depth(MR); nd = null_depth(N, len(tr))
        po = [len(pools[a] & pools[b]) for a, b in itertools.combinations(tr, 2)]
        for a, b in itertools.combinations(tr, 2): pairs.append(dict(task=t, heldout=h, cohort_a=a, cohort_b=b, top20_overlap=len(pools[a] & pools[b])))
        supp = {}
        for k in (20, 100, 500, 2000):
            n_any2 = int(((MI <= k).sum(1) >= 2).sum()); n_all = int(((MI <= k).sum(1) == len(tr)).sum())
            supp[k] = (n_any2, n_all)
        gi = {g: i for i, g in enumerate(U)}; pr = []
        for g in panel:
            ranks = {c: int(rk[(t, c)].loc[g, "MI_rank"]) for c in tr}; home = [c for c in tr if g in pools[c]]
            other = [ranks[c] for c in tr if c not in home]; best_other = min(other) if other else None
            pr.append(dict(task=t, heldout=h, gene=g, home_pools=";".join(home), n_home_pools=len(home), in_seed=g in seeds[(t, h)],
                           **{f"MI_rank_{c}": ranks[c] for c in tr}, best_MI_rank_other_training_cohort=best_other,
                           **{f"support_top{k}": bool(len(home) >= 2 or (best_other is not None and best_other <= k)) for k in (20, 100, 500, 2000)}))
        prov += pr; P = pd.DataFrame(pr)
        cons.append({"Task": t, "Held-out cohort": h, "Training cohorts": ";".join(tr), "Universe (genes)": N,
            "Top-20 pool pairwise overlap (min–max)": f"{min(po)}–{max(po)}", "Training-cohort pairs with any top-20 overlap": f"{sum(x > 0 for x in po)}/{len(po)}",
            "MI depth: first gene in ≥2 lists": int(mi2), "MI depth: first gene in all lists": int(miall),
            "Null (random ranking) depth ≥2, median [95%]": f"{nd[0]:.0f} [{nd[1][0]:.0f}–{nd[1][1]:.0f}]", "Null depth all, median [95%]": f"{nd[2]:.0f} [{nd[3][0]:.0f}–{nd[3][1]:.0f}]",
            "P(null depth ≥2 ≤ observed)": round(float((nd[4] <= mi2).mean()), 3), "P(null depth all ≤ observed)": round(float((nd[5] <= miall).mean()), 3),
            "mRMR depth: first gene in ≥2 lists": "none within top-2000" if np.isinf(mr2) else int(mr2), "mRMR depth: first gene in all lists": "none within top-2000" if np.isinf(mrall) else int(mrall),
            "Genes in top-100 of ≥2 cohorts": supp[100][0], "Expected if rankings independent, top-100 ≥2": round(expct(N, len(tr), 100)[0], 1), "Genes in top-100 of all": supp[100][1], "Genes in top-2000 of ≥2 cohorts": supp[2000][0], "Expected if independent, top-2000 ≥2": round(expct(N, len(tr), 2000)[0], 1),
            "Genes in top-2000 of all": supp[2000][1], "Expected if independent, top-2000 all": round(expct(N, len(tr), 2000)[1], 1),
            "Candidates (union of pools)": len(set().union(*pools.values())), "Panel genes from seed (top-5 summed rank)": int(P.in_seed.sum()),
            "Panel genes in ≥2 pools": int((P.n_home_pools >= 2).sum()),
            **{f"Panel genes ranked within MI top-{k} of another training cohort (or in ≥2 pools)": f"{int(P[f'support_top{k}'].sum())}/20" for k in (20, 100, 500, 2000)},
            "Panel genes from forward selection with no support in any other training cohort's top-100": f"{int(((~P.support_top100) & (~P.in_seed)).sum())}/20",
            "Panel genes with no support in top-2000": f"{int((~P.support_top2000).sum())}/20"})
C = pd.DataFrame(cons); C.to_csv(R / "Table_consensus_failure.csv", index=False); pd.DataFrame(prov).to_csv(R / "consensus_panel_provenance.csv", index=False)
pd.DataFrame(pairs).to_csv(R / "consensus_pool_pairwise.csv", index=False)
chk = pd.read_json(R / "rank/rank_check.jsonl", lines=True); chk.to_csv(R / "rank_reproduces_stage5_pools.csv", index=False)
# ================================================================= 3. relatedness + sibling
Rl = pd.read_json(R / "related/pairs.jsonl", lines=True)
Rl["platform_a"] = Rl.cohort_a.map(PLAT); Rl["platform_b"] = Rl.cohort_b.map(PLAT); Rl["same_platform"] = Rl.platform_a == Rl.platform_b
Rl["same_study_subseries"] = Rl.apply(lambda r: {r.cohort_a, r.cohort_b} == {"GSE19439", "GSE19444"}, axis=1)
Rl["cross_minus_within_r"] = (Rl.median_cross_sample_r - (Rl.median_within_a_r + Rl.median_within_b_r) / 2).round(4)
Rl.to_csv(R / "cohort_relatedness.csv", index=False)
S5A = pd.read_csv(S5 / "results/per_fold_auc.csv"); rnd, cur = {}, []
for r in map(json.loads, open(R / "sibling/sib_records.jsonl")):
    h = r["heldout"]; y = Y["LTB_vs_CON"][h]
    if r["kind"] == "rnd":
        for m, v in r["scores"].items(): rnd.setdefault((h, r["panel"], m), []).append(roc_auc_score(y, np.nan_to_num(v[h])))
    else: cur.append(r)
sib = []
for r in cur:
    h, p = r["heldout"], r["panel"]; y = Y["LTB_vs_CON"][h]
    for m, v in r["scores"].items():
        s = np.asarray(v[h]); a = roc_auc_score(y, s); lo, hi = boot_ci(y, s); ra = np.asarray(rnd[(h, p, "MLP" if m == "MLP_alpha" else m)])
        p5 = "Native" if p.startswith("Native") else p; o = S5A[(S5A.task == "LTB_vs_CON") & (S5A.heldout == h) & (S5A.panel == p5) & (S5A.method == m)]
        sib.append(dict(heldout=h, excluded_sibling=r["excluded_sibling"], training=";".join(r["training"]), panel=p, method=m, n_features=r["diag"]["n_features"],
            AUC=round(a, 3), CI_low=round(lo, 3), CI_high=round(hi, 3), rand_median=round(float(np.median(ra)), 3), rand_p2_5=round(float(np.percentile(ra, 2.5)), 3),
            rand_p97_5=round(float(np.percentile(ra, 97.5)), 3), prop_random_ge=round(float((ra >= a).mean()), 3),
            MLP_alpha=r["diag"].get("MLP_alpha_chosen") if m == "MLP_alpha" else None,
            stage5_AUC_with_sibling=float(o.AUC.iloc[0]) if len(o) else None, stage5_prop_random_ge=float(o.prop_random_ge_curated.iloc[0]) if len(o) else None))
SB = pd.DataFrame(sib); SB["delta_vs_stage5"] = (SB.AUC - SB.stage5_AUC_with_sibling).round(3); SB.to_csv(R / "sibling_exclusion.csv", index=False)
print("alpha rows", len(AL), "| consensus rows", len(C), "| relatedness pairs", len(Rl), "| sibling rows", len(SB))
