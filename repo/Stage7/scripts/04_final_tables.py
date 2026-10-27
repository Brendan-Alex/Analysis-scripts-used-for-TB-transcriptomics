"""Final consolidated supporting tables (Stage7). ATB_vs_LTB and LTB_vs_CON rows are taken from Stage5/Stage6 (verified identical in
Stage7); ATB_vs_OD rows are recomputed with the rebuilt GSE83456. Same definitions as Stage5/Stage6 (code copied, logic unchanged)."""
import json, re, itertools, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import binom
S7 = Path(__file__).resolve().parents[1]; MT = S7.parent; S5 = MT / "Stage5"; S6 = MT / "Stage6"; CR = MT / "Clean_rebuild"; R = S7 / "results"; SEED = 42
TASKS = {"ATB_vs_LTB": ["GSE19491", "GSE37250", "GSE73408", "GSE101705", "GSE107994"], "ATB_vs_OD": ["GSE37250", "GSE73408", "GSE42834", "GSE83456"],
         "LTB_vs_CON": ["GSE19439", "GSE19444", "GSE28623", "GSE107994"]}
src = lambda t: R if t == "ATB_vs_OD" else S5 / "results"           # native pools/panels
rsrc = lambda t: R if t == "ATB_vs_OD" else S6 / "results"          # full rankings
SEX = set(pd.read_csv(S5 / "data/sex_chromosome_genes.csv").Gene.astype(str))
# ---------------- consensus (Stage6 definitions)
def depth(M): S = np.sort(M, axis=1); return float(S[:, 1].min()), float(S[:, -1].min())
rng = np.random.default_rng(SEED); NULL = {}
def null_depth(N, m, reps=1000):
    if (N, m) not in NULL:
        d2, da = [], []
        for _ in range(reps):
            M = np.stack([rng.permutation(N) + 1 for _ in range(m)], 1); a, b = depth(M); d2.append(a); da.append(b)
        NULL[(N, m)] = (np.array(d2), np.array(da))
    return NULL[(N, m)]
def expct(N, m, k): q = k / N; return N * binom.sf(1, m, q), N * q ** m
seeds = {(r["task"], r["heldout"]): r["seed"] for r in map(json.loads, open(S6 / "results/seeds/seeds.jsonl"))}
seeds.update({(r["task"], r["heldout"]): r["seed"] for r in map(json.loads, open(R / "seeds/seeds_ATB_vs_OD.jsonl"))})
cons, prov = [], []
for t, coh in TASKS.items():
    rk = {c: pd.read_csv(rsrc(t) / f"rank/{t}__{c}.csv.gz").set_index("Gene") for c in coh}
    for h in coh:
        tr = [c for c in coh if c != h]; U = rk[tr[0]].index; N = len(U)
        MI = np.stack([rk[c].loc[U, "MI_rank"].to_numpy(float) for c in tr], 1)
        pools = {c: set(json.load(open(src(t) / f"native/{t}__{c}__cohort.json"))["mrmr_top20"]) for c in tr}
        panel = json.load(open(src(t) / f"native/{t}__fold_{h}.json"))["panel"]
        mi2, miall = depth(MI); d2, da = null_depth(N, len(tr)); po = [len(pools[a] & pools[b]) for a, b in itertools.combinations(tr, 2)]
        pr = []
        for g in panel:
            ranks = {c: int(rk[c].loc[g, "MI_rank"]) for c in tr}; home = [c for c in tr if g in pools[c]]; other = [ranks[c] for c in tr if c not in home]
            bo = min(other) if other else None
            pr.append(dict(task=t, heldout=h, gene=g, home_pools=";".join(home), in_seed=g in seeds[(t, h)], best_MI_rank_other_training_cohort=bo,
                           **{f"support_top{k}": bool(len(home) >= 2 or (bo is not None and bo <= k)) for k in (20, 100, 500, 2000)}))
        prov += pr; Pp = pd.DataFrame(pr)
        cons.append({"Task": t.replace("_vs_", " vs "), "Held-out cohort": h, "Training cohorts": ";".join(tr), "Genes ranked (task universe)": N,
            "Top-20 shortlist overlap between training cohorts, min–max genes": f"{min(po)}–{max(po)}", "Training-cohort pairs sharing any top-20 gene": f"{sum(x > 0 for x in po)}/{len(po)}",
            "Rank depth until a gene is in ≥2 cohorts' lists": int(mi2), "Same, random-ranking median [95% range]": f"{np.median(d2):.0f} [{np.percentile(d2, 2.5):.0f}–{np.percentile(d2, 97.5):.0f}]",
            "P (random depth ≤ observed), ≥2 lists": round(float((d2 <= mi2).mean()), 3),
            "Rank depth until a gene is in every cohort's list": int(miall), "Same, random-ranking median [95% range] (all)": f"{np.median(da):.0f} [{np.percentile(da, 2.5):.0f}–{np.percentile(da, 97.5):.0f}]",
            "P (random depth ≤ observed), all lists": round(float((da <= miall).mean()), 3),
            "Genes in every cohort's top-2000": int(((MI <= 2000).sum(1) == len(tr)).sum()), "Expected by chance (independent rankings), top-2000 all": round(expct(N, len(tr), 2000)[1], 1),
            "Genes in ≥2 cohorts' top-100": int(((MI <= 100).sum(1) >= 2).sum()), "Expected by chance, top-100 ≥2": round(expct(N, len(tr), 100)[0], 1),
            "Native panel genes from seed": int(Pp.in_seed.sum()), "Native panel genes in ≥2 cohorts' top-20 shortlists": int((Pp.home_pools.str.count(";") >= 1).sum()),
            **{f"Native panel genes within MI top-{k} of another training cohort": f"{int(Pp[f'support_top{k}'].sum())}/20" for k in (20, 100, 500, 2000)},
            "Native panel genes added by forward selection without top-100 support elsewhere": f"{int(((~Pp.support_top100) & (~Pp.in_seed)).sum())}/20"})
pd.DataFrame(cons).to_csv(R / "FINAL_Table_consensus_failure.csv", index=False); pd.DataFrame(prov).to_csv(R / "FINAL_consensus_panel_provenance.csv", index=False)
# ---------------- Native stability (Stage5 definitions)
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
t1, t2 = [], []
for t, coh in TASKS.items():
    P = {h: json.load(open(src(t) / f"native/{t}__fold_{h}.json"))["panel"] for h in coh}; cnt = pd.Series([g for v in P.values() for g in v]).value_counts()
    pw = [(len(set(P[a]) & set(P[b])), len(set(P[a]) & set(P[b])) / len(set(P[a]) | set(P[b]))) for a, b in itertools.combinations(coh, 2)]
    cats = {g: category(g) for g in cnt.index}
    t1.append({"Task": t.replace("_vs_", " vs "), "Folds": len(coh), "Genes per fold": 20, "Distinct genes (all folds)": len(cnt), "Genes in every fold": int((cnt == len(coh)).sum()),
               "Genes in ≥ half of folds": int((cnt >= len(coh) / 2).sum()), "Genes in one fold only": int((cnt == 1).sum()),
               "Pairwise fold overlap, genes (min–max)": f"{min(x for x, _ in pw)}–{max(x for x, _ in pw)}", "Pairwise Jaccard (min–max)": f"{min(j for _, j in pw):.2f}–{max(j for _, j in pw):.2f}",
               "Protein-coding": sum(c[0] == "gene with protein product" for c in cats.values()), "Olfactory receptor": sum("olfactory" in c[1] for c in cats.values()),
               "microRNA": sum("microRNA" in c[1] for c in cats.values()), "Pseudogene": sum("pseudogene" in c[1] for c in cats.values()), "lncRNA": sum("lncRNA" in c[1] for c in cats.values()),
               "X/Y-linked": sum(g in SEX for g in cnt.index)})
    for g in sorted(cnt.index, key=lambda g: (-cnt[g], g)):
        t2.append({"Task": t.replace("_vs_", " vs "), "Gene": g, **{f"held-out {c}": "●" if g in P[c] else "" for c in coh}, "Folds selected": f"{cnt[g]}/{len(coh)}",
                   "HGNC locus type": cats[g][0], "Flag": cats[g][1]})
pd.DataFrame(t1).to_csv(R / "FINAL_Table_native_stability_summary.csv", index=False); pd.DataFrame(t2).to_csv(R / "FINAL_Table_native_stability_genes.csv", index=False)
# ---------------- task difficulty (Stage5 definitions) from the master table
M = pd.read_csv(R / "MASTER_results_table.csv"); S = pd.read_csv(R / "gene_level_separability.csv"); S = S[S.scope == "all_measured"]; td = []
for t, coh in TASKS.items():
    for c in coh:
        s = S[(S.task == t) & (S.cohort == c)].iloc[0]; sub = M[(M.Task == t.replace("_vs_", " vs ")) & (M["Held-out cohort"] == c)]
        r = {"Task": t.replace("_vs_", " vs "), "Cohort": c, "Positive n": s.n_pos, "Negative n": s.n_neg, "Genes measured": s.n_genes,
             "% genes individually AUC > 0.8": round(100 * s.prop_AUC_gt_0_8, 2), "% genes individually AUC < 0.2": round(100 * s.prop_AUC_lt_0_2, 2),
             "% genes |AUC − 0.5| > 0.3": round(100 * s.prop_either_dev_gt_0_3, 2)}
        for p in ["Sweeney3", "Zak16", "Native", "Berry86_noTLR7"]:
            q = sub[(sub.Panel == p) & (sub.Method == "Logistic regression")].iloc[0]; r[f"{p} genes used"] = int(q["Panel genes used"]); r[f"Random-panel LR median ({p} size-matched)"] = q["Random panel AUC median"]
            r[f"{p} LR AUC"] = q.AUC
        td.append(r)
pd.DataFrame(td).to_csv(R / "FINAL_Table_task_difficulty.csv", index=False)
# ---------------- alpha: fixed 10 vs tuned (from master)
ml = M[M.Method == "MLP (alpha fixed at 10)"][["Task", "Held-out cohort", "Panel", "AUC", "AUC 95% CI lower", "AUC 95% CI upper", "MLP tuned alpha (inner LOCO, grid 1e-4..1000)",
       "MLP tuned-alpha AUC", "MLP tuned-alpha AUC 95% CI lower", "MLP tuned-alpha AUC 95% CI upper", "MLP tuned alpha gives near-constant predictions (held-out SD < 1e-3)",
       "Tuned-alpha near-constant in Stage6 (11-cell list)", "MLP converged (alpha 10)"]].copy()
ml["Tuned minus fixed-10 AUC"] = (ml["MLP tuned-alpha AUC"] - ml.AUC).round(3); ml.to_csv(R / "FINAL_Table_MLP_fixed_vs_tuned_alpha.csv", index=False)
# ---------------- relatedness (Stage6 + Stage7 ATB_vs_OD)
Rl6 = pd.read_csv(S6 / "results/cohort_relatedness.csv"); Rl7 = pd.read_csv(R / "related/pairs_ATB_vs_OD.csv")
Rl = pd.concat([Rl6[Rl6.task != "ATB_vs_OD"], Rl7.assign(platform_a=Rl7.cohort_a.map(dict(zip(Rl6.cohort_a, Rl6.platform_a)) | dict(zip(Rl6.cohort_b, Rl6.platform_b))),
                platform_b=Rl7.cohort_b.map(dict(zip(Rl6.cohort_a, Rl6.platform_a)) | dict(zip(Rl6.cohort_b, Rl6.platform_b))))])
Rl["same_platform"] = Rl.platform_a == Rl.platform_b; Rl.to_csv(R / "FINAL_cohort_relatedness.csv", index=False)
print(pd.DataFrame(t1).T.to_string()); print(pd.DataFrame(cons)[pd.DataFrame(cons).Task == "ATB vs OD"].T.to_string())
print(ml.groupby("Task").agg(collapsed=("MLP tuned alpha gives near-constant predictions (held-out SD < 1e-3)", "sum"), s6=("Tuned-alpha near-constant in Stage6 (11-cell list)", "sum"),
      med=("Tuned minus fixed-10 AUC", "median"), mn=("Tuned minus fixed-10 AUC", "min"), mx=("Tuned minus fixed-10 AUC", "max")))
