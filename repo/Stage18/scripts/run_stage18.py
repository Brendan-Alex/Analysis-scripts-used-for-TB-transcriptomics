"""Stage18: is each validation cohort actually independent of the discovery cohorts?

Stage12 checked GEO relations, parent SuperSeries and shared GSMs - all clean. Those catch literal
sample reuse and explicit GEO linkage. They do NOT catch two accessions deposited separately from the
same recruitment network. GSE101705 (discovery, ATB vs LTB) and GSE152218 (external validation, ATB vs
LTB) share a contributor (Johnson WE) and both are South India cohorts, and GEO lists the same two
downstream papers on both accessions. This stage runs the sample-level expression comparison the
manuscript already claims was done.

For every (validation cohort, discovery cohort of the same task) pair, on the genes shared by both,
it computes exactly the statistics of Stage6's related_job so the numbers are directly comparable with
Stage7/results/FINAL_cohort_relatedness.csv:
    mean_profile_spearman   Spearman r of per-gene mean log2 expression
    sd_profile_spearman     same for per-gene SD
    signal_spearman         Spearman r of per-gene class effect (pos-neg mean of label-free z / pooled SD)
    signal_top200_jaccard   overlap of the 200 largest-|effect| genes
    cohort_origin_AUC       5-fold CV logistic regression telling the two cohorts' samples apart
                            (1.0 = trivially separable, 0.5 = indistinguishable)
    max/median_cross_sample_r   Pearson r between samples of A and B on within-sample ranks (duplicate check)
    n_cross_pairs_r_gt_0_95     count of near-duplicate sample pairs
plus the within-cohort correlation medians, so cross-minus-within can be read off.

Two reference points are printed with the results, both from the discovery set:
    GSE19439 ~ GSE19444   max r 0.9585, median 0.9443, 161 pairs > 0.95   <- KNOWN related (sub-series of
                                                                             the same SuperSeries)
    GSE101705 ~ GSE107994 max r 0.9088, median 0.7950, 0 pairs > 0.95     <- unrelated, both RNA-seq
If GSE152218 ~ GSE101705 resembles the first, the independence claim fails. If it resembles the second,
it holds. Platform matters for these correlations, so compare like with like.

Reads only: Stage7/data (cache + universes), Stage12/expression, Stage12/tables, Stage5 code.
Writes only: Stage18/results/. Nothing upstream is modified.
Usage (from the Stage18 folder):  python scripts/run_stage18.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, pickle, warnings, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

S18 = Path(__file__).resolve().parents[1]; MT = S18.parent
S7, S12 = MT / "Stage7", MT / "Stage12"
OUT = S18 / "results"; OUT.mkdir(parents=True, exist_ok=True)
_sp = importlib.util.spec_from_file_location("s5", str(MT / "Stage5/scripts/run_stage5.py"))
s5 = importlib.util.module_from_spec(_sp); os.environ["STAGE5_OUT"] = str(OUT); _sp.loader.exec_module(s5)
s5.load = lambda task: pickle.load(open(S7 / "data/cache" / f"{task}.pkl", "rb"))
s5.universe = lambda task: pd.read_csv(S7 / "data" / f"universe_{task}.csv").Gene.astype(str).tolist()
TASKS = s5.TASKS; SEED = 42
VAL = {"GSE144127": "ATB_vs_OD", "GSE152218": "ATB_vs_LTB"}

def zscore(X):
    m = X.mean(1); s = X.std(1).replace(0, np.nan)
    return X.sub(m, axis=0).div(s, axis=0).fillna(0.0)

def load_val(G):
    """same construction as Stage13 build_val: label-free z over all filter-passing samples"""
    X = pd.read_csv(S12 / "expression" / f"{G}_log2_genelevel.csv.gz", index_col=0)
    X.index = X.index.astype(str); X = X[~X.index.duplicated()].dropna()
    F = pd.read_csv(S12 / "tables" / f"{G}_filter_decisions.csv")
    F["included"] = F["included"].astype(str).str.lower().eq("true")
    inc = F[F["included"]].set_index("GSM")["class"]
    keep = [s for s in X.columns if s in inc.index]
    t = TASKS[VAL[G]]; cl = inc.reindex(keep)
    y = (cl == t["pos"]).astype(int)
    return dict(L=X[keep], Z=zscore(X)[keep], y=y.to_numpy())

def prof(L, Z, y, U):
    from scipy.stats import rankdata
    L = L.loc[U]; Z = Z.loc[U]
    e = (Z.loc[:, y == 1].mean(1) - Z.loc[:, y == 0].mean(1)) / np.sqrt(
        (Z.loc[:, y == 1].var(1) + Z.loc[:, y == 0].var(1)) / 2).replace(0, np.nan)
    R = pd.DataFrame(rankdata(L.to_numpy(), axis=0) / L.shape[0], index=U, columns=L.columns)
    return dict(mean=L.mean(1), sd=L.std(1), eff=e.fillna(0), R=R)

def compare(a_name, A, b_name, B, task, U):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    top = lambda e: set(e.abs().sort_values(ascending=False).index[:200])
    RR = pd.concat([A["R"], B["R"]], axis=1); v = RR.var(1).sort_values(ascending=False).index[:2000]
    X = RR.loc[v].T.to_numpy(); yy = np.r_[np.zeros(A["R"].shape[1]), np.ones(B["R"].shape[1])]
    pr = cross_val_predict(LogisticRegression(max_iter=2000), X, yy,
                           cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    Xa, Xb = A["R"].to_numpy(), B["R"].to_numpy()
    Xa = (Xa - Xa.mean(0)) / Xa.std(0); Xb = (Xb - Xb.mean(0)) / Xb.std(0)
    cr = Xa.T @ Xb / Xa.shape[0]
    i, j = np.unravel_index(np.argmax(cr), cr.shape)
    wa = Xa.T @ Xa / Xa.shape[0]; wb = Xb.T @ Xb / Xb.shape[0]
    iu = lambda w: w[np.triu_indices(w.shape[0], 1)]
    row = dict(task=task, validation_cohort=a_name, discovery_cohort=b_name,
               n_validation=A["R"].shape[1], n_discovery=B["R"].shape[1], genes=len(U),
               shared_GSMs=len(set(A["R"].columns) & set(B["R"].columns)),
               mean_profile_spearman=round(float(spearmanr(A["mean"], B["mean"])[0]), 4),
               sd_profile_spearman=round(float(spearmanr(A["sd"], B["sd"])[0]), 4),
               signal_spearman=round(float(spearmanr(A["eff"], B["eff"])[0]), 4),
               signal_top200_jaccard=round(len(top(A["eff"]) & top(B["eff"])) / len(top(A["eff"]) | top(B["eff"])), 4),
               cohort_origin_AUC=round(float(roc_auc_score(yy, pr)), 4),
               max_cross_sample_r=round(float(cr.max()), 4),
               max_pair=f"{A['R'].columns[i]}~{B['R'].columns[j]}",
               median_cross_sample_r=round(float(np.median(cr)), 4),
               median_within_validation_r=round(float(np.median(iu(wa))), 4),
               median_within_discovery_r=round(float(np.median(iu(wb))), 4),
               n_cross_pairs_r_gt_0_95=int((cr > 0.95).sum()))
    row["cross_minus_within_r"] = round(row["median_cross_sample_r"]
                                        - (row["median_within_validation_r"] + row["median_within_discovery_r"]) / 2, 4)
    dup = [dict(task=task, validation_cohort=a_name, discovery_cohort=b_name,
                validation_GSM=A["R"].columns[p], discovery_GSM=B["R"].columns[q], r=round(float(cr[p, q]), 4))
           for p, q in zip(*np.where(cr > 0.95))]
    return row, dup

def main():
    warnings.filterwarnings("ignore")
    rows, dups = [], []
    for G, task in VAL.items():
        print(f"[{G}] {task}", flush=True)
        V = load_val(G); D = s5.load(task); U0 = s5.universe(task)
        U = [g for g in U0 if g in V["L"].index]
        print(f"  task universe {len(U0)}; shared with {G}: {len(U)}", flush=True)
        A = prof(V["L"], V["Z"], V["y"], U)
        for c in TASKS[task]["cohorts"]:
            B = prof(D[c]["L"], D[c]["Z"], D[c]["y"].to_numpy(), U)
            r, d = compare(G, A, c, B, task, U); rows.append(r); dups += d
            print(f"  vs {c:10s} origin AUC {r['cohort_origin_AUC']:.4f}  max r {r['max_cross_sample_r']:.4f}"
                  f"  median r {r['median_cross_sample_r']:.4f}  pairs>0.95 {r['n_cross_pairs_r_gt_0_95']}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "validation_relatedness.csv", index=False)
    pd.DataFrame(dups).to_csv(OUT / "near_duplicate_pairs.csv", index=False)

    ref = pd.DataFrame([
        dict(pair="GSE19439 ~ GSE19444", relation="KNOWN related: sub-series of the same SuperSeries",
             max_cross_sample_r=0.9585, median_cross_sample_r=0.9443, n_cross_pairs_r_gt_0_95=161, cohort_origin_AUC=1.0),
        dict(pair="GSE101705 ~ GSE107994", relation="unrelated, both RNA-seq",
             max_cross_sample_r=0.9088, median_cross_sample_r=0.7950, n_cross_pairs_r_gt_0_95=0, cohort_origin_AUC=1.0),
        dict(pair="GSE19491 ~ GSE37250", relation="unrelated, both microarray",
             max_cross_sample_r=0.9365, median_cross_sample_r=0.8794, n_cross_pairs_r_gt_0_95=0, cohort_origin_AUC=1.0)])
    def md(d):
        return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |",
                          "|" + "|".join(["---"] * len(d.columns)) + "|"]
                         + ["| " + " | ".join(map(str, r)) + " |" for r in d.values])
    worst = R.sort_values("max_cross_sample_r", ascending=False).iloc[0]
    rep = ["# Stage18: validation-cohort independence", "",
           "## Pairwise comparison (validation cohort vs each discovery cohort of its task)", "", md(R), "",
           "## Reference pairs from the discovery set (Stage7/results/FINAL_cohort_relatedness.csv)", "", md(ref), "",
           "## Near-duplicate sample pairs (cross-cohort r > 0.95)", "",
           (md(pd.DataFrame(dups)) if dups else "None. No validation sample resembles any discovery sample "
            "more closely than samples within a cohort resemble each other."), "",
           "## Read this", "",
           f"- Shared GSMs across every pair: {int(R.shared_GSMs.sum())}.",
           f"- Highest cross-sample correlation anywhere: {worst.max_cross_sample_r} "
           f"({worst.validation_cohort} ~ {worst.discovery_cohort}, pair {worst.max_pair}).",
           "- Compare that with the reference table above, matching platform where possible. Resembling the "
           "GSE19439~GSE19444 row means the cohorts are not independent; resembling the other two rows means they are.",
           "- cohort_origin_AUC near 1.0 is expected and benign: any two cohorts are trivially separable by batch. "
           "It is informative only if it comes out LOW, which would mean the two cohorts are indistinguishable.",
           "- This tests expression, not provenance. It cannot rule out shared recruitment, shared investigators or "
           "a shared parent study, which for GSE101705 and GSE152218 is already evident from their GEO records and "
           "should be declared in Limitations whatever these numbers say."]
    (OUT / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    print("\nwrote results/validation_relatedness.csv, near_duplicate_pairs.csv, REPORT.md")
    print(f"near-duplicate pairs found: {len(dups)}")

if __name__ == "__main__":
    main()
