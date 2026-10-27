"""Stage7: generalisation gap (all tasks; ATB_vs_OD now with rebuilt GSE83456) and ATB_vs_OD alpha-grid profile, from the Stage7
benchmark records (curated jobs scored the held-out cohort and every training cohort; MLP_alpha diag holds inner and held-out AUC at
every grid alpha). No new model fitting. Per fold only."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
S7 = Path(__file__).resolve().parents[1]; MT = S7.parent; R = S7 / "results"
LAB = json.load(open(S7 / "data/labels.json")); Y = {t: {c: np.array(v[1]) for c, v in LAB[t].items()} for t in LAB}
MN = {"LR": "Logistic regression", "RF": "Random forest", "MLP": "MLP (alpha fixed at 10)", "MLP_alpha": "MLP (tuned alpha)", "singscore": "singscore", "DirAvg": "Directional average"}
gap, prof = [], []
for t in ["ATB_vs_LTB", "ATB_vs_OD", "LTB_vs_CON"]:
    for line in open(R / f"bench/{t}__records.jsonl"):
        r = json.loads(line)
        if r["kind"] != "cur": continue
        h, tr, y = r["heldout"], r["training"], Y[t]
        for m, v in r["scores"].items():
            pc = {c: roc_auc_score(y[c], v[c]) for c in tr}; pooled = roc_auc_score(np.concatenate([y[c] for c in tr]), np.concatenate([v[c] for c in tr])); a = roc_auc_score(y[h], v[h])
            gap.append({"Task": t.replace("_vs_", " vs "), "Held-out cohort": h, "Panel": r["panel"], "Method": MN[m], "Training AUC, pooled resubstitution": round(pooled, 3),
                        "Training AUC, lowest cohort": round(min(pc.values()), 3), "Training AUC, highest cohort": round(max(pc.values()), 3), "Held-out AUC": round(a, 3),
                        "Gap (pooled training minus held-out)": round(pooled - a, 3), "Training AUC per cohort": ";".join(f"{c}={x:.3f}" for c, x in pc.items())})
        if t == "ATB_vs_OD":
            d = r["diag"]; al = list(d["MLP_alpha_inner_AUC"]); inn = np.array([d["MLP_alpha_inner_AUC"][a] for a in al]); ho = np.array([d["MLP_alpha_grid_heldout_AUC"][a] for a in al])
            prof.append({"Held-out cohort": h, "Panel": r["panel"], "Tuned alpha": d["MLP_alpha_chosen"], **{f"Held-out AUC, alpha {a}": round(x, 3) for a, x in zip(al, ho)},
                         **{f"Inner LOCO AUC, alpha {a}": round(x, 3) for a, x in zip(al, inn)}, "Held-out-best alpha": float(al[int(np.argmax(ho))]),
                         "Tuned minus held-out-best": round(float(ho[al.index(str(d["MLP_alpha_chosen"]))] - ho.max()), 3), "Tuned minus alpha 10": round(float(ho[al.index(str(d["MLP_alpha_chosen"]))] - ho[al.index("10.0")]), 3),
                         "Spearman inner vs held-out across grid": round(float(spearmanr(inn, ho)[0]), 3) if np.ptp(inn) > 0 and np.ptp(ho) > 0 else None})
G = pd.DataFrame(gap); G.to_csv(R / "FINAL_generalisation_gap.csv", index=False)
Pf = pd.DataFrame(prof); Pf.to_csv(R / "atb_vs_od_alpha_profile_rebuilt.csv", index=False)
g = G.groupby(["Task", "Method"])["Gap (pooled training minus held-out)"].agg(["min", "median", "max"]).round(3); print(g.to_string())
cols = [c for c in Pf.columns if c.startswith("Held-out AUC, alpha")]; print("ATB_vs_OD median held-out AUC per alpha:", Pf[cols].median().round(3).to_dict())
print(Pf[["Tuned minus held-out-best", "Tuned minus alpha 10", "Spearman inner vs held-out across grid"]].describe().round(3).to_string()); print(Pf["Tuned alpha"].value_counts().to_dict())
o = pd.read_csv(MT / "Stage5/results/generalisation_gap.csv"); o = o[o.task == "ATB_vs_OD"]
print("Stage5 ATB_vs_OD gap by method (min/max):", o.groupby("method").gap_pooled_minus_heldout.agg(["min", "max"]).round(3).to_dict())
