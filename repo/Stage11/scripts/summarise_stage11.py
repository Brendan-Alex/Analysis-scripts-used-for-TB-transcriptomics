"""Stage11 summary. Per fold: chosen k (inner LOCO, training cohorts only), held-out AUC at chosen k with bootstrap CI, size-matched random
panels at the same k, and k = 20 for comparison. Descriptive curves by k. Final signatures with fold stability."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
S11 = Path(__file__).resolve().parents[1]; MT = S11.parent; R = S11 / "results"; SEED = 42
LAB = json.load(open(MT / "Stage7/data/labels.json")); Y = {t: {c: np.array(v[1]) for c, v in LAB[t].items()} for t in LAB}
def boot_ci(y, s):
    rng = np.random.default_rng(SEED); I = np.stack([rng.integers(0, len(y), len(y)) for _ in range(1000)]); Yb, Sb = y[I], s[I]
    Rk = rankdata(Sb, axis=1); n1 = Yb.sum(1); n0 = Yb.shape[1] - n1; ok = (n1 > 0) & (n0 > 0)
    b = ((Rk * Yb).sum(1) - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1); return np.percentile(b[ok], [2.5, 97.5])
RND = {(r["task"], r["heldout"], r["k"]): r["auc"] for r in map(json.loads, open(R / "random_by_size.jsonl"))}
FC = [json.loads(l) for l in open(R / "fold_curves.jsonl")]
rows, curve = [], []
for r in FC:
    t, h, how = r["task"], r["heldout"], r["ranking"]; y = Y[t][h]
    for m, k in r["choice"].items():
        c = r["curve"][k - 1]; a = c["auc"][m]; lo, hi = boot_ci(y, np.asarray(c["scores"][m])); ra = np.asarray(RND[(t, h, k)][m]); r20 = np.asarray(RND[(t, h, 20)][m])
        a20 = r["curve"][19]["auc"][m]
        rows.append({"Task": t.replace("_vs_", " vs "), "Held-out cohort": h, "Ranking": how, "Method": m, "Chosen k (inner LOCO)": k, "Genes": ";".join(r["genes"][:k]),
                     "AUC at chosen k": round(a, 3), "CI lower": round(lo, 3), "CI upper": round(hi, 3), "Random median at k": round(float(np.median(ra)), 3),
                     "Proportion random >= at k": round(float((ra >= a).mean()), 3), "AUC at k=20": round(a20, 3), "Random median at 20": round(float(np.median(r20)), 3),
                     "Proportion random >= at 20": round(float((r20 >= a20).mean()), 3), "Chosen minus k=20": round(a - a20, 3),
                     "Sibling flag": t == "LTB_vs_CON" and h == "GSE19439"})
    for k in range(1, 21):
        for m, a in r["curve"][k - 1]["auc"].items():
            ra = np.asarray(RND[(t, h, k)][m])
            curve.append(dict(Task=t.replace("_vs_", " vs "), heldout=h, ranking=how, method=m, k=k, auc=a, inner=r["inner"][m][k - 1], rand_median=float(np.median(ra)),
                              beats=bool((ra >= a).mean() <= 0.05)))
T = pd.DataFrame(rows); T.to_csv(R / "chosen_k_per_fold.csv", index=False); C = pd.DataFrame(curve); C.to_csv(R / "ablation_curves_long.csv", index=False)
T["beats_k"] = T["Proportion random >= at k"] <= 0.05; T["beats_20"] = T["Proportion random >= at 20"] <= 0.05
print(T.groupby(["Task", "Ranking"]).agg(k_med=("Chosen k (inner LOCO)", "median"), k_min=("Chosen k (inner LOCO)", "min"), k_max=("Chosen k (inner LOCO)", "max"),
      auc_med=("AUC at chosen k", "median"), auc20_med=("AUC at k=20", "median"), d_med=("Chosen minus k=20", "median"), beats_k=("beats_k", "sum"), beats_20=("beats_20", "sum"),
      cells=("beats_k", "size")).round(3).to_string())
print(T[T.Ranking == "consistent"].pivot_table(index=["Task", "Held-out cohort"], columns="Method", values="Chosen k (inner LOCO)").to_string())
D = C[C.method == "DirAvg"].groupby(["Task", "ranking", "k"]).agg(auc=("auc", "median"), rnd=("rand_median", "median"), inner=("inner", "median")).round(3).reset_index()
D.to_csv(R / "median_curves_DirAvg.csv", index=False)
print(D[D.ranking == "consistent"].pivot_table(index="k", columns="Task", values=["auc", "rnd"]).round(3).to_string())
# final signatures + stability
F = [json.loads(l) for l in open(R / "final_signatures.jsonl")]; out = []
for f in F:
    t, how = f["task"], f["ranking"]; folds = [r for r in FC if r["task"] == t and r["ranking"] == how]
    for g in f["final_signature"]:
        out.append({"Task": t.replace("_vs_", " vs "), "Ranking": how, "Gene": g, "Rank in final": f["genes"].index(g) + 1, "Final k (DirAvg inner LOCO, all cohorts)": f["choice"]["DirAvg"],
                    "Final k by method": json.dumps(f["choice"]), "In fold top-20": f"{sum(g in r['genes'] for r in folds)}/{len(folds)}",
                    "In fold chosen-k panel (DirAvg)": f"{sum(g in r['genes'][:r['choice']['DirAvg']] for r in folds)}/{len(folds)}"})
FS = pd.DataFrame(out); FS.to_csv(R / "final_signatures.csv", index=False); print(FS.to_string(index=False))
c = {f["task"]: f["final_signature"] for f in F if f["ranking"] == "consistent"}
print("consistent final: LTB", len(c["ATB_vs_LTB"]), "OD", len(c["ATB_vs_OD"]), "overlap", sorted(set(c["ATB_vs_LTB"]) & set(c["ATB_vs_OD"])), "union", len(set(c["ATB_vs_LTB"]) | set(c["ATB_vs_OD"])))
