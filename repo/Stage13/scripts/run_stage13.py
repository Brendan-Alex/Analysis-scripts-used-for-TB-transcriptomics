"""Stage13: external validation on GSE144127 (ATB vs OD) and GSE152218 (ATB vs LTB).
Locked models: every signature/panel is trained on ALL training cohorts of its task (Stage7 caches, per-cohort label-free z-scores),
then scored ONCE on the validation cohort. Nothing is selected, tuned or re-fitted on validation data.
Pre-specified primary analysis: Stage11 'consistent' signature, DirAvg score (Stage11 primary method).
Run (laptop):  cd "<repo root>\\Stage13"   then   python scripts\\run_stage13.py --workers 10
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_v, "1")
import sys, json, pickle, argparse, warnings, importlib.util, shutil, time
from pathlib import Path
import numpy as np, pandas as pd
S13 = Path(__file__).resolve().parents[1]; MT = S13.parent
S7, S11, S12 = MT / "Stage7", MT / "Stage11", MT / "Stage12"
for _d in ("data", "results", "logs"): (S13 / _d).mkdir(parents=True, exist_ok=True)
if not (S13 / "data/sex_chromosome_genes.csv").exists(): shutil.copy(S7 / "data/sex_chromosome_genes.csv", S13 / "data/sex_chromosome_genes.csv")
os.environ["STAGE5_OUT"] = str(S13)
_sp = importlib.util.spec_from_file_location("s5", str(MT / "Stage5/scripts/run_stage5.py")); s5 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(s5)
s5.load = lambda task: pickle.load(open(S7 / "data/cache" / f"{task}.pkl", "rb"))
s5.universe = lambda task: pd.read_csv(S7 / "data" / f"universe_{task}.csv").Gene.astype(str).tolist()
TASKS = s5.TASKS; TL = list(TASKS); METHODS = s5.RND; A10 = 10.0; NRAND = 50; NBOOT = 1000; SEED = 42
VAL = {"GSE144127": "ATB_vs_OD", "GSE152218": "ATB_vs_LTB"}; VL = list(VAL)
PRIMARY = ("S11_consistent", "DirAvg")
SIG = pd.read_csv(S11 / "results/final_signatures.csv")

def zscore(X): return X.sub(X.mean(1), axis=0).div(X.std(1, ddof=0).replace(0, np.nan), axis=0).fillna(0.0)
def fauc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def ci(y, s):
    y, s = np.asarray(y), np.asarray(s); rng = np.random.default_rng(SEED); I = rng.integers(0, len(y), (NBOOT, len(y)))
    a = [fauc(y[i], s[i]) for i in I if 0 < y[i].sum() < len(i)]; return [round(float(x), 4) for x in np.percentile(a, [2.5, 97.5])]

def build_val(G):
    """identical to Stage5 step_prepare: z-score over all filter-passing samples (label-free), keep task classes, within-sample ranks for singscore"""
    from scipy.stats import rankdata
    t = TASKS[VAL[G]]
    X = pd.read_csv(S12 / "expression" / f"{G}_log2_genelevel.csv.gz", index_col=0); X.index = X.index.astype(str)
    n_dup = int(X.index.duplicated().sum()); X = X[~X.index.duplicated()]
    n_nan = int(X.isna().any(axis=1).sum()); X = X.dropna()
    F = pd.read_csv(S12 / "tables" / f"{G}_filter_decisions.csv"); F["included"] = F["included"].astype(str).str.lower().eq("true")
    inc = F[F["included"]].set_index("GSM")["class"]
    miss = [s for s in inc.index if s not in X.columns]; assert not miss, f"{G}: included GSMs missing from expression: {miss[:5]}"
    Zall = zscore(X); keep = [s for s in X.columns if s in inc.index]; cl = inc.reindex(keep)
    assert set(cl) <= {t["pos"], t["neg"]}, f"{G}: unexpected classes {set(cl)}"
    y = (cl == t["pos"]).astype(int); L = X[keep]; R = pd.DataFrame(rankdata(L.to_numpy(), axis=0), index=L.index, columns=keep)
    info = dict(cohort=G, task=VAL[G], genes=X.shape[0], genes_dropped_any_NaN=n_nan, duplicate_symbols_dropped=n_dup,
                zscore_samples=X.shape[1], n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), z_values_are_finite=bool(np.isfinite(Zall[keep].to_numpy()).all()))
    return dict(Z=Zall[keep], L=L, R=R, y=y, n_cohort_all=X.shape[1]), info

_C = {}
def getDV(G):
    if G not in _C:
        _C.clear(); D = dict(s5.load(VAL[G])); V, info = build_val(G)
        tr = TASKS[VAL[G]]["cohorts"]; assert G not in tr
        assert not any(set(D[c]["Z"].columns) & set(V["Z"].columns) for c in tr), "validation GSM present in a training cohort"
        D[G] = V; _C[G] = (D, info)
    return _C[G]
def s11(task, how):
    d = SIG[(SIG.Task == task.replace("_", " ")) & (SIG.Ranking == how)]; return d.sort_values("Rank in final").Gene.astype(str).tolist()
def panels(task):
    p = {"S11_consistent": s11(task, "consistent"), "S11_MI_meanrank": s11(task, "MI_meanrank"),
         "Combined8": list(dict.fromkeys(s11("ATB_vs_LTB", "consistent") + s11("ATB_vs_OD", "consistent")))}
    for q in TASKS[task]["panels"]:
        if q != "Native": p[q] = s5.panel(q)
    return p
RAND_PANELS = ("S11_consistent", "S11_MI_meanrank", "Combined8")
def evaluate(D, G, genes):
    out, diag, feats = s5.fit_eval(D, TASKS[VAL[G]]["cohorts"], [G], genes, METHODS, alpha_fixed=A10)
    return {m: np.asarray(v[G], dtype=float) for m, v in out.items()}, diag, feats
def rpool(D, G): Zi = set(D[G]["Z"].index); return [g for g in s5.universe(VAL[G]) if g in Zi]
def random_job(G, k, d):
    warnings.filterwarnings("ignore")
    D, _ = getDV(G); task = VAL[G]
    rng = np.random.default_rng([SEED, TL.index(task), 100 + VL.index(G), k, d]); g = [str(x) for x in rng.choice(rpool(D, G), size=k, replace=False)]
    sc, _, _ = evaluate(D, G, g); y = D[G]["y"].to_numpy()
    return dict(cohort=G, k=k, draw=d, genes=g, auc={m: s5.auc(y, sc[m]) for m in METHODS})

def subgroups(G, gsm):
    """masks over the validation samples (order = gsm); each keeps all positives of the subgroup + the selected negatives"""
    M = pd.read_csv(S12 / "tables" / f"{G}_samples_long.csv", dtype=str).set_index("GSM").reindex(gsm)
    cl = pd.read_csv(S12 / "tables" / f"{G}_filter_decisions.csv").set_index("GSM")["class"].reindex(gsm); pos = cl.eq("ATB").to_numpy()
    out = {}
    if G == "GSE144127":
        dx = M["ch:diagnosis"].fillna("")
        for nm, rx in [("OD = pneumonia", "Pneumonia"), ("OD = sarcoidosis", "Sarcoidosis"), ("OD = cancer", "Cancer"),
                       ("OD = LRTI/chest infection/URTI", "LRTI|Chest Infection|URTI"), ("OD = LTBI (treatment indicated)", "LTBI")]:
            out[nm] = pos | (~pos & dx.str.contains(rx).to_numpy())
        out["OD = 'Other' only"] = pos | (~pos & dx.eq("Other").to_numpy())
        out["OD excluding LTBI"] = pos | (~pos & ~dx.str.contains("LTBI").to_numpy())
        for s in ("Male", "Female"): out[f"sex = {s}"] = M["ch:gender"].eq(s).to_numpy()
        for e in ("Indian", "Black", "White"): out[f"ethnicity = {e}"] = M["ch:ethnicity1"].eq(e).to_numpy()
    else:
        bmi = pd.to_numeric(M["ch:bmi"], errors="coerce").to_numpy()
        out["LTB BMI<16 only (malnourished vs malnourished)"] = pos | (~pos & (bmi < 16))
        out["LTB BMI>=18.5 only"] = pos | (~pos & (bmi >= 18.5))
        out["ATB BMI<16 vs LTB BMI<16, males only"] = (pos | (bmi < 16)) & M["ch:sex"].eq("male").to_numpy() & (bmi < 16)
        for s in ("male", "female"): out[f"sex = {s}"] = M["ch:sex"].eq(s).to_numpy()
    return out

def md(df):
    try: return df.to_markdown(index=False)
    except Exception: return "```\n" + df.to_string(index=False) + "\n```"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=10); a = ap.parse_args()
    warnings.filterwarnings("ignore"); t0 = time.time(); R = S13 / "results"; rep = ["# Stage13 external validation report (auto-generated)", "",
        "Locked models: trained on all training cohorts of the task, scored once on the validation cohort; no selection, tuning or refitting on validation data.",
        f"Pre-specified primary analysis: {PRIMARY[0]} signature, {PRIMARY[1]} score. Bootstrap CI: {NBOOT} resamples, seed {SEED}. Random baseline: {NRAND} size-matched panels from the task universe genes measured in the validation cohort.", ""]
    rows, scl, pg, infos, canary = [], [], [], [], []
    for G in VL:
        task = VAL[G]; D, info = getDV(G); infos.append(info); y = D[G]["y"].to_numpy(); gsm = list(D[G]["Z"].columns)
        s5.log("validation", G, task, info)
        P = panels(task)
        for nm, genes in P.items():
            sc, diag, feats = evaluate(D, G, genes)
            for m in METHODS:
                lo, hi = ci(y, sc[m])
                rows.append(dict(cohort=G, task=task, panel=nm, method=m, n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), AUC=round(s5.auc(y, sc[m]), 4), CI_lo=lo, CI_hi=hi,
                                 panel_genes=len(genes), features_in_all_training=len(feats), features_zero_filled_in_validation=diag["zero_filled_in_test"][G],
                                 primary=(nm, m) == PRIMARY))
                scl.append(pd.DataFrame(dict(cohort=G, panel=nm, method=m, GSM=gsm, y=y, score=sc[m])))
            s5.log("  ", nm, {m: round(s5.auc(y, sc[m]), 3) for m in METHODS})
        # per-gene direction check for the Stage11 signatures
        tr = TASKS[task]["cohorts"]
        for nm in ("S11_consistent", "S11_MI_meanrank"):
            dirn = s5.direction(D, tr, P[nm])
            for g in P[nm]:
                ga = {c: round(fauc(D[c]["y"].to_numpy(), D[c]["Z"].loc[g].to_numpy()), 3) if g in D[c]["Z"].index else None for c in tr}
                va = fauc(y, D[G]["Z"].loc[g].to_numpy()) if g in D[G]["Z"].index else np.nan
                pg.append(dict(cohort=G, panel=nm, gene=g, training_direction="up in ATB" if dirn[g] > 0 else "down in ATB", training_AUC_per_cohort=json.dumps(ga),
                               validation_AUC=round(va, 4), same_direction_in_validation=bool((va - 0.5) * dirn[g] > 0)))
        # canary: validation expression replaced by noise (labels kept) -> primary AUC should be ~0.5
        Dn = dict(D); V = dict(D[G]); rng = np.random.default_rng(7); Ln = pd.DataFrame(rng.normal(size=V["L"].shape), index=V["L"].index, columns=V["L"].columns)
        from scipy.stats import rankdata
        V.update(Z=zscore(Ln), L=Ln, R=pd.DataFrame(rankdata(Ln.to_numpy(), axis=0), index=Ln.index, columns=Ln.columns)); Dn[G] = V
        scn, _, _ = evaluate(Dn, G, P["S11_consistent"])
        canary.append(dict(cohort=G, **{f"noise_AUC_{m}": round(s5.auc(y, scn[m]), 4) for m in METHODS}))
    res = pd.DataFrame(rows); res.to_csv(R / "external_auc.csv", index=False)
    S = pd.concat(scl); S.to_csv(R / "external_scores_long.csv.gz", index=False)
    pd.DataFrame(pg).to_csv(R / "per_gene_direction.csv", index=False); pd.DataFrame(canary).to_csv(R / "canary_noise.csv", index=False)
    pd.DataFrame(infos).to_csv(R / "validation_cohort_info.csv", index=False)
    # random baseline (parallel)
    jobs = [(G, k, d) for G in VL for k in sorted({len(panels(VAL[G])[p]) for p in RAND_PANELS}) for d in range(NRAND)]
    s5.log("random panels:", len(jobs), "jobs on", a.workers, "workers")
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex: rnd = list(ex.map(random_job, *zip(*jobs), chunksize=5))
    with open(R / "random_panels.jsonl", "w") as fh:
        for r in rnd: fh.write(json.dumps(r) + "\n")
    rr = []
    for r in res[res.panel.isin(RAND_PANELS)].itertuples():
        k = r.panel_genes; dist = np.array([x["auc"][r.method] for x in rnd if x["cohort"] == r.cohort and x["k"] == k]); nge = int((dist >= r.AUC).sum())
        rr.append(dict(cohort=r.cohort, panel=r.panel, method=r.method, k=k, AUC=r.AUC, random_median=round(float(np.median(dist)), 4),
                       random_95th=round(float(np.percentile(dist, 95)), 4), n_random=len(dist), n_random_ge_observed=nge, empirical_p=round((1 + nge) / (1 + len(dist)), 4)))
    rr = pd.DataFrame(rr); rr.to_csv(R / "vs_random.csv", index=False)
    # subgroups / sensitivity (scores from the full-cohort locked models, re-evaluated on subsets)
    sg = []
    for G in VL:
        D, _ = getDV(G); gsm = list(D[G]["Z"].columns)
        for nm, mask in subgroups(G, gsm).items():
            for (pn, m), d in S[(S.cohort == G) & S.panel.isin(["S11_consistent", "Combined8"] + [p for p in TASKS[VAL[G]]["panels"] if p != "Native"])].groupby(["panel", "method"]):
                d = d.set_index("GSM").reindex(gsm)[mask]; yy, ss = d.y.to_numpy(), d.score.to_numpy(); n1 = int(yy.sum()); n0 = len(yy) - n1
                ok = n1 >= 5 and n0 >= 5; lo, hi = ci(yy, ss) if ok else (np.nan, np.nan)
                sg.append(dict(cohort=G, subgroup=nm, panel=pn, method=m, n_pos=n1, n_neg=n0, AUC=round(fauc(yy, ss), 4) if ok else np.nan, CI_lo=lo, CI_hi=hi))
    sg = pd.DataFrame(sg); sg.to_csv(R / "subgroups.csv", index=False)
    # report
    rep += ["## Validation cohorts", md(pd.DataFrame(infos)), "", "## Leakage canary (validation expression replaced by noise, primary signature)", md(pd.DataFrame(canary)), ""]
    for G in VL:
        r = res[res.cohort == G]; pr = r[r.primary].iloc[0]
        rep += [f"## {G} ({VAL[G]}): {pr.n_pos} ATB vs {pr.n_neg} {VAL[G].split('_vs_')[1]}", "",
                f"**PRIMARY ({PRIMARY[0]}, {PRIMARY[1]}): AUC {pr.AUC:.3f} (95% CI {pr.CI_lo:.3f}-{pr.CI_hi:.3f})**", "",
                "### All panels x methods (AUC [95% CI])"]
        w = r.assign(v=r.apply(lambda x: f"{x.AUC:.3f} [{x.CI_lo:.3f}-{x.CI_hi:.3f}]", axis=1)).pivot(index="panel", columns="method", values="v")[METHODS].reset_index()
        g = r.drop_duplicates("panel")[["panel", "panel_genes", "features_in_all_training", "features_zero_filled_in_validation"]]
        rep += [md(w.merge(g, on="panel")), "", "### Stage11 signatures vs size-matched random panels", md(rr[rr.cohort == G].drop(columns="cohort")), "",
                "### Per-gene direction (training vs validation)", md(pd.DataFrame(pg).query("cohort == @G").drop(columns="cohort")), "",
                "### Subgroups / sensitivity (DirAvg; all methods in results/subgroups.csv)"]
        q = sg[(sg.cohort == G) & (sg.method == "DirAvg")]
        q = q.assign(v=q.apply(lambda x: "n<5" if pd.isna(x.AUC) else f"{x.AUC:.3f} [{x.CI_lo:.3f}-{x.CI_hi:.3f}]", axis=1))
        n = q.drop_duplicates("subgroup")[["subgroup", "n_pos", "n_neg"]]
        rep += [md(q.pivot(index="subgroup", columns="panel", values="v").reset_index().merge(n, on="subgroup")), ""]
    rep += [f"Runtime {time.time() - t0:.0f} s. Files: results/external_auc.csv, vs_random.csv, per_gene_direction.csv, subgroups.csv, canary_noise.csv, external_scores_long.csv.gz, random_panels.jsonl"]
    open(S13 / "REPORT.md", "w", encoding="utf-8").write("\n".join(rep)); s5.log("done", f"{time.time() - t0:.0f}s -> REPORT.md")
if __name__ == "__main__": main()
