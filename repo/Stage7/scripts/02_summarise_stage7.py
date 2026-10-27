"""Stage7 summaries: per-fold AUCs (MLP at fixed alpha 10 primary, tuned alpha as comparison), before/after GSE83456 rebuild,
identity checks for unaffected results, and the master results table. Per fold only; nothing averaged across folds."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
S7 = Path(__file__).resolve().parents[1]; MT = S7.parent; S5 = MT / "Stage5"; S6 = MT / "Stage6"; R = S7 / "results"; SEED = 42; COLLAPSE = 1e-3
TASKS = {"ATB_vs_LTB": ["GSE19491", "GSE37250", "GSE73408", "GSE101705", "GSE107994"], "ATB_vs_OD": ["GSE37250", "GSE73408", "GSE42834", "GSE83456"],
         "LTB_vs_CON": ["GSE19439", "GSE19444", "GSE28623", "GSE107994"]}
CLS = {"ATB_vs_LTB": ("ATB", "LTB"), "ATB_vs_OD": ("ATB", "OD"), "LTB_vs_CON": ("LTB", "CON")}
USER = {"Berry86": {"GSE19444", "GSE19439"}, "Kaforou_ATB_v_LTB": {"GSE37250", "GSE19491"}, "Kaforou_ATB_v_OD": {"GSE37250", "GSE19491"},
        "Zak16": {"GSE19439", "GSE19444", "GSE19442", "GSE37250", "GSE42834"}, "Sweeney3": {"GSE19491", "GSE37250", "GSE42834", "GSE28623"}}
ADDED = {"Berry86": {"GSE19491"}, "Zak16": {"GSE19491"}, "Sweeney3": {"GSE19439", "GSE19444"}}
for d in (USER, ADDED): d["Berry86_noTLR7"] = d.get("Berry86", set())
REL = {("Berry86", "GSE19444"): "derivation", ("Berry86", "GSE19439"): "validation", ("Berry86", "GSE19491"): "contains derivation/validation sub-series",
       ("Kaforou_ATB_v_LTB", "GSE37250"): "derivation", ("Kaforou_ATB_v_LTB", "GSE19491"): "validation", ("Kaforou_ATB_v_OD", "GSE37250"): "derivation",
       ("Zak16", "GSE19439"): "parameterisation", ("Zak16", "GSE19444"): "testing", ("Zak16", "GSE37250"): "testing", ("Zak16", "GSE42834"): "testing",
       ("Zak16", "GSE19491"): "contains tested sub-series", ("Sweeney3", "GSE19491"): "discovery", ("Sweeney3", "GSE37250"): "discovery", ("Sweeney3", "GSE42834"): "discovery",
       ("Sweeney3", "GSE28623"): "validation", ("Sweeney3", "GSE19439"): "sub-series of discovery dataset", ("Sweeney3", "GSE19444"): "sub-series of discovery dataset"}
for (p, h), v in list(REL.items()):
    if p == "Berry86": REL[("Berry86_noTLR7", h)] = v
LAB = json.load(open(S7 / "data/labels.json")); Y = {t: {c: np.array(v[1]) for c, v in LAB[t].items()} for t in LAB}
F = pd.read_csv(S7 / "qc/folds.csv"); G = pd.read_csv(R / "global_shift_heldout.csv")
def boot_ci(y, s):
    rng = np.random.default_rng(SEED); I = np.stack([rng.integers(0, len(y), len(y)) for _ in range(1000)]); Yb, Sb = y[I], s[I]
    Rk = rankdata(Sb, axis=1); n1 = Yb.sum(1); n0 = Yb.shape[1] - n1; ok = (n1 > 0) & (n0 > 0)
    b = ((Rk * Yb).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1); return np.percentile(b[ok], [2.5, 97.5])
def load_recs(path):
    cur, rnd = [], {}
    for line in open(path):
        r = json.loads(line)
        if r["kind"] == "cur": cur.append(r)
        else: rnd.setdefault((r["heldout"], r["panel"]), []).append(r)
    return cur, rnd
rows = []; S6A = pd.read_csv(S6 / "results/alpha_extended.csv"); S6set = set(map(tuple, S6A[S6A.chosen_prediction_collapsed][["task", "heldout", "panel"]].values))
for t, coh in TASKS.items():
    cur, rnd = load_recs(R / f"bench/{t}__records.jsonl"); y = Y[t]
    for r in cur:
        h, p, d = r["heldout"], r["panel"], r["diag"]; f = F[(F.task == t) & (F.heldout == h)].iloc[0]; gs = G[(G.task == t) & (G.heldout == h)].iloc[0]
        ra = {m: np.array([roc_auc_score(y[h], np.nan_to_num(x["scores"][m][h])) for x in rnd[(h, p)]]) for m in ["LR", "RF", "MLP", "singscore", "DirAvg"]}
        tun = np.asarray(r["scores"]["MLP_alpha"][h]); ta = roc_auc_score(y[h], tun); tlo, thi = boot_ci(y[h], tun)
        for m, mlab in [("LR", "Logistic regression"), ("RF", "Random forest"), ("MLP", "MLP (alpha fixed at 10)"), ("singscore", "singscore (rank-based, untrained)"), ("DirAvg", "Directional average (untrained)")]:
            s = np.asarray(r["scores"][m][h]); a = roc_auc_score(y[h], s); lo, hi = boot_ci(y[h], s); rr = ra[m]
            row = {"Task": t.replace("_vs_", " vs "), "Positive class": CLS[t][0], "Negative class": CLS[t][1], "Held-out cohort": h,
                   "Training cohorts": ";".join(r["training"]), "Training n": int(f.n_train), f"Training n {CLS[t][0]}": None, "Training positive n": int(f.train_pos), "Training negative n": int(f.train_neg),
                   "Test n": int(f.n_test), "Test positive n": int(f.test_pos), "Test negative n": int(f.test_neg),
                   "Panel": p, "Panel genes used": int(d["n_features"]), "Panel genes zero-filled in held-out cohort": int(d["zero_filled_in_test"][h]),
                   "Method": mlab, "AUC": round(a, 3), "AUC 95% CI lower": round(lo, 3), "AUC 95% CI upper": round(hi, 3),
                   "Random panels (n)": len(rr), "Random panel AUC median": round(float(np.median(rr)), 3), "Random panel AUC 2.5th pct": round(float(np.percentile(rr, 2.5)), 3),
                   "Random panel AUC 97.5th pct": round(float(np.percentile(rr, 97.5)), 3), "Proportion of random panels with AUC >= panel": round(float((rr >= a).mean()), 3),
                   "Derivation overlap with held-out cohort (original list)": h in USER.get(p, set()), "Derivation overlap via sub-series containment (added, accepted)": h in ADDED.get(p, set()),
                   "Derivation relationship": REL.get((p, h), ""), "Optimistic (any derivation overlap)": h in USER.get(p, set()) or h in ADDED.get(p, set()),
                   "Sibling-cohort flag (GSE19439 fold: panel selection included sister sub-series GSE19444)": t == "LTB_vs_CON" and h == "GSE19439" and p == "Native",
                   "Held-out all-gene mean AUC (global shift)": gs.allgene_mean_AUC, "Held-out cohort quantile-normalised": bool(gs.quantile_normalised),
                   "GSE83456 rebuilt from raw in Stage7": "GSE83456" in [h] + r["training"] and t == "ATB_vs_OD"}
            del row[f"Training n {CLS[t][0]}"]
            if m == "MLP":
                row.update({"MLP tuned alpha (inner LOCO, grid 1e-4..1000)": d["MLP_alpha_chosen"], "MLP tuned-alpha AUC": round(ta, 3), "MLP tuned-alpha AUC 95% CI lower": round(tlo, 3),
                            "MLP tuned-alpha AUC 95% CI upper": round(thi, 3), "MLP tuned alpha gives near-constant predictions (held-out SD < 1e-3)": bool(np.std(tun) < COLLAPSE),
                            "Tuned-alpha near-constant in Stage6 (11-cell list)": (t, h, p) in S6set, "MLP converged (alpha 10)": d["MLP_converged"]})
            rows.append(row)
M = pd.DataFrame(rows); M.to_csv(R / "MASTER_results_table.csv", index=False)
# ---------------- before / after and identity
B5 = pd.read_csv(S5 / "results/per_fold_auc.csv"); key = ["task", "heldout", "panel", "method"]
Mk = M.assign(task=M.Task.str.replace(" vs ", "_vs_"), heldout=M["Held-out cohort"], panel=M.Panel,
              method=M.Method.map({"Logistic regression": "LR", "Random forest": "RF", "MLP (alpha fixed at 10)": "MLP10", "singscore (rank-based, untrained)": "singscore", "Directional average (untrained)": "DirAvg"}))
cmp = Mk[Mk.method != "MLP10"].merge(B5[key + ["AUC", "CI_low", "CI_high", "rand_median", "prop_random_ge_curated"]], on=key, how="left")
cmp = cmp.rename(columns={"AUC_x": "AUC_stage7", "AUC_y": "AUC_stage5"})
cmp["delta"] = (cmp.AUC_stage7 - cmp.AUC_stage5).round(3)
a10 = S6A.set_index(["task", "heldout", "panel"])
mlp = Mk[Mk.method == "MLP10"].copy(); mlp["AUC_stage6_alpha10"] = [a10.loc[(a, b, c), "heldout_AUC_a10.0"] for a, b, c in zip(mlp.task, mlp.heldout, mlp.panel)]
mlp["AUC_stage6_tuned"] = [a10.loc[(a, b, c), "heldout_AUC_chosen"] for a, b, c in zip(mlp.task, mlp.heldout, mlp.panel)]
mlp["alpha_stage6_tuned"] = [a10.loc[(a, b, c), "chosen_alpha_stage6"] for a, b, c in zip(mlp.task, mlp.heldout, mlp.panel)]
mlp["delta_alpha10"] = (mlp.AUC - mlp.AUC_stage6_alpha10).round(3); mlp["delta_tuned"] = (mlp["MLP tuned-alpha AUC"] - mlp.AUC_stage6_tuned).round(3)
out = pd.concat([cmp[key + ["AUC_stage5", "AUC_stage7", "delta", "CI_low", "CI_high", "AUC 95% CI lower", "AUC 95% CI upper", "rand_median", "Random panel AUC median",
                            "prop_random_ge_curated", "Proportion of random panels with AUC >= panel"]].rename(columns={"CI_low": "CI_low_stage5", "CI_high": "CI_high_stage5",
                            "AUC 95% CI lower": "CI_low_stage7", "AUC 95% CI upper": "CI_high_stage7", "rand_median": "rand_median_stage5", "Random panel AUC median": "rand_median_stage7",
                            "prop_random_ge_curated": "prop_random_stage5", "Proportion of random panels with AUC >= panel": "prop_random_stage7"}),
                 mlp[key + ["AUC_stage6_alpha10", "AUC", "delta_alpha10", "alpha_stage6_tuned", "MLP tuned alpha (inner LOCO, grid 1e-4..1000)", "AUC_stage6_tuned", "MLP tuned-alpha AUC", "delta_tuned"]]
                 .rename(columns={"AUC": "AUC_stage7"})])
out.to_csv(R / "before_after_vs_stage5_stage6.csv", index=False)
idn = out[out.task != "ATB_vs_OD"]
ident = dict(cells_checked_LR_RF_singscore_DirAvg=int(idn.delta.notna().sum()), nonzero=int((idn.delta.abs() > 0).sum()),
             MLP10_cells=int(idn.delta_alpha10.notna().sum()), MLP10_nonzero=int((idn.delta_alpha10.abs() > 0).sum()),
             tuned_cells=int(idn.delta_tuned.notna().sum()), tuned_nonzero=int((idn.delta_tuned.abs() > 0).sum()))
pd.DataFrame([ident]).to_csv(R / "identity_check_unaffected_tasks.csv", index=False); print("identity (ATB_vs_LTB, LTB_vs_CON):", ident)
# GSE83456-specific before/after
gs5 = pd.read_csv(S5 / "results/global_shift_heldout.csv"); sep5 = pd.read_csv(S5 / "results/gene_level_separability.csv"); sep7 = pd.read_csv(R / "gene_level_separability.csv")
g83 = pd.concat([gs5[(gs5.task == "ATB_vs_OD") & (gs5.heldout == "GSE83456")].assign(version="Stage5 (GEO gene-median-centred)"),
                 G[(G.task == "ATB_vs_OD") & (G.heldout == "GSE83456")].assign(version="Stage7 (rebuilt from raw)")])
g83.to_csv(R / "gse83456_global_shift_before_after.csv", index=False)
pd.concat([sep5[(sep5.cohort == "GSE83456")].assign(version="Stage5"), sep7[(sep7.cohort == "GSE83456")].assign(version="Stage7")]).to_csv(R / "gse83456_gene_separability_before_after.csv", index=False)
nat = []
for h in TASKS["ATB_vs_OD"]:
    a5 = json.load(open(S5 / f"results/native/ATB_vs_OD__fold_{h}.json")); a7 = json.load(open(R / f"native/ATB_vs_OD__fold_{h}.json"))
    nat.append(dict(heldout=h, GSE83456_role="held out" if h == "GSE83456" else "training", genes_shared_stage5_stage7=len(set(a5["panel"]) & set(a7["panel"])),
                    panel_identical=a5["panel"] == a7["panel"], train_cross_AUC_stage5=round(a5["mean_cross_auc"], 4), train_cross_AUC_stage7=round(a7["mean_cross_auc"], 4)))
p5 = json.load(open(S5 / "results/native/ATB_vs_OD__GSE83456__cohort.json"))["mrmr_top20"]; p7 = json.load(open(R / "native/ATB_vs_OD__GSE83456__cohort.json"))["mrmr_top20"]
pd.DataFrame(nat).assign(GSE83456_top20_pool_overlap_stage5_vs_stage7=len(set(p5) & set(p7))).to_csv(R / "atb_vs_od_native_before_after.csv", index=False)
print(pd.DataFrame(nat).to_string(), "\nGSE83456 pool overlap", len(set(p5) & set(p7)))
