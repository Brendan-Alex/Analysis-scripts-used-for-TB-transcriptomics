"""Stage8 working figures (PNG, 150 dpi) from Stage7 outputs only - no refitting. Usage: python figs.py 1 2 3 ... (figure numbers)."""
import sys, json, re, itertools, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
S8 = Path(__file__).resolve().parents[1]; MT = S8.parent; S7 = MT / "Stage7"; R7 = S7 / "results"; OUT = S8 / "figures"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b29", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
INK, INK2, GREY, GREYD = "#0b0b0b", "#52514e", "#d5d4cf", "#a3a29c"
PCOL = {"Native": "#2a78d6", "Zak16": "#eb6834", "Kaforou_ATB_v_LTB": "#1baf7a", "Kaforou_ATB_v_OD": "#1baf7a", "Berry86": "#eda100", "Berry86_noTLR7": "#e87ba4", "Sweeney3": "#008300"}
PLAB = {"Native": "Native", "Zak16": "Zak16", "Kaforou_ATB_v_LTB": "Kaforou", "Kaforou_ATB_v_OD": "Kaforou", "Berry86": "Berry86", "Berry86_noTLR7": "Berry86 −TLR7", "Sweeney3": "Sweeney3"}
MCOL = {"LR": "#2a78d6", "RF": "#eb6834", "MLP": "#6250d6", "singscore": "#1baf7a", "DirAvg": "#e34948"}
MNAME = {"Logistic regression": "LR", "Random forest": "RF", "MLP (alpha fixed at 10)": "MLP", "singscore (rank-based, untrained)": "singscore", "Directional average (untrained)": "DirAvg",
         "MLP (tuned alpha)": "MLP tuned", "singscore": "singscore", "Directional average": "DirAvg"}
METHODS = ["LR", "RF", "MLP", "singscore", "DirAvg"]
TASKS = {"ATB vs LTB": ["GSE19491", "GSE37250", "GSE73408", "GSE101705", "GSE107994"], "ATB vs OD": ["GSE37250", "GSE73408", "GSE42834", "GSE83456"],
         "LTB vs CON": ["GSE19439", "GSE19444", "GSE28623", "GSE107994"]}
PANELS = {"ATB vs LTB": ["Native", "Zak16", "Kaforou_ATB_v_LTB", "Berry86", "Berry86_noTLR7", "Sweeney3"], "ATB vs OD": ["Native", "Zak16", "Kaforou_ATB_v_OD", "Berry86", "Berry86_noTLR7", "Sweeney3"],
          "LTB vs CON": ["Native", "Zak16", "Berry86", "Berry86_noTLR7", "Sweeney3"]}
M = pd.read_csv(R7 / "MASTER_results_table.csv"); M["m"] = M.Method.map(MNAME)
FL = "Sibling-cohort flag (GSE19439 fold: panel selection included sister sub-series GSE19444)"
M["flag"] = np.where(M["Optimistic (any derivation overlap)"], "derivation", np.where(M[FL], "sibling", "clean"))
M["beats"] = M["Proportion of random panels with AUC >= panel"] <= 0.05
def cap(name, text): (OUT / f"{name}_caption.txt").write_text(text.strip() + "\n", encoding="utf-8")
def save(fig, name): fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white"); plt.close(fig); print("saved", name)
# =============================================================== Figure 1
def fig1():
    RA = pd.read_csv(S8 / "data/random_auc_long.csv")
    chk = RA.groupby(["Task", "Held-out cohort", "Panel", "method"]).AUC.median().round(3)
    mm = M.set_index(["Task", "Held-out cohort", "Panel", "m"])["Random panel AUC median"]; mm.index.names = chk.index.names
    assert (chk.reindex(mm.index) - mm).abs().max() < 1e-9, "random medians do not reproduce the master table"
    for ti, (task, folds) in enumerate(TASKS.items()):
        P = PANELS[task]; fig, axs = plt.subplots(len(METHODS), len(folds), figsize=(2.9 * len(folds) + 1, 11.5), sharey=True, squeeze=False)
        for i, m in enumerate(METHODS):
            for j, h in enumerate(folds):
                ax = axs[i, j]; ax.axhline(0.5, color=GREYD, lw=0.8, ls=":")
                for k, p in enumerate(P):
                    r = RA[(RA.Task == task) & (RA["Held-out cohort"] == h) & (RA.Panel == p) & (RA.method == m)].AUC.to_numpy()
                    lo, q1, md, q3, hi = np.percentile(r, [2.5, 25, 50, 75, 97.5])
                    ax.add_patch(plt.Rectangle((k - 0.32, lo), 0.64, hi - lo, color=GREY, lw=0, alpha=0.6))
                    ax.add_patch(plt.Rectangle((k - 0.32, q1), 0.64, q3 - q1, color=GREYD, lw=0, alpha=0.7))
                    ax.plot([k - 0.32, k + 0.32], [md, md], color="white", lw=1.5)
                    c = M[(M.Task == task) & (M["Held-out cohort"] == h) & (M.Panel == p) & (M.m == m)].iloc[0]
                    col = PCOL[p]; fl = c.flag
                    ax.plot([k, k], [c["AUC 95% CI lower"], c["AUC 95% CI upper"]], color=col, lw=1.6, zorder=3)
                    ax.plot(k, c.AUC, marker="o" if fl == "clean" else ("D" if fl == "derivation" else "s"), ms=7, mfc=col if fl == "clean" else "white",
                            mec=col, mew=1.8, zorder=4)
                    if c.beats and fl == "clean": ax.text(k, min(c["AUC 95% CI upper"] + 0.02, 1.04), "*", ha="center", va="bottom", fontsize=11, color=INK)
                ax.set_xlim(-0.6, len(P) - 0.4); ax.set_ylim(0.2, 1.08); ax.set_xticks(range(len(P)))
                ax.set_xticklabels([PLAB[p] for p in P] if i == len(METHODS) - 1 else [], rotation=55, ha="right", fontsize=8)
                ax.grid(axis="x", visible=False)
                if i == 0: ax.set_title(f"held out: {h}", fontsize=9.5, color=INK)
                if j == 0: ax.set_ylabel(f"{m}\nheld-out AUC", fontsize=9)
        hand = [Patch(color=GREY, alpha=0.6, label="random panels, 2.5–97.5%"), Patch(color=GREYD, alpha=0.7, label="random panels, IQR (white = median)"),
                Line2D([], [], marker="o", color=INK2, mfc=INK2, ls="-", label="curated panel, clean (AUC ± 95% CI)"),
                Line2D([], [], marker="D", color=INK2, mfc="white", mew=1.8, ls="", label="derivation-overlap flag (optimistic)"),
                Line2D([], [], marker="s", color=INK2, mfc="white", mew=1.8, ls="", label="GSE19439 sibling flag"),
                Line2D([], [], marker="$*$", color=INK, ls="", ms=9, label="clean and ≤5% of random panels reach it")]
        fig.legend(handles=hand, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.035), fontsize=8.5)
        fig.suptitle(f"Figure 1{'abc'[ti]}. {task}: curated panels against size-matched random panels, every fold and method", y=1.065, fontsize=11, color=INK)
        fig.tight_layout(); save(fig, f"Fig1{'abc'[ti]}_{task.replace(' ', '_')}_curated_vs_random")
    s = M.groupby(["Task", "flag"]).agg(cells=("AUC", "size"), beat=("beats", "sum")).reset_index()
    txt = "; ".join(f"{r.Task} {r.flag}: {int(r.beat)}/{int(r.cells)} beat random" for r in s.itertuples())
    cap("Fig1_curated_vs_random", f"""
Figure 1a-c (one file per task). Held-out AUC of every curated panel (coloured points, vertical line = bootstrap 95% CI, 1,000 resamples,
seed 42) in every fold (columns = held-out cohort) and method (rows: LR, RF, MLP at alpha 10, singscore, directional average), drawn over
the distribution of 50 size-matched random gene panels for the same fold, panel size and method (light grey = 2.5-97.5 percentile range,
dark grey = interquartile range, white tick = median; dotted line = AUC 0.5). Filled circles = clean cells; open diamonds = cell carries a
derivation-overlap flag (your list or accepted sub-series containment); open squares = GSE19439 sibling flag (Native panel, LTB vs CON).
Asterisk = clean cell where <=5% of random panels reach the curated AUC.
Counts: {txt}.
Sources: Stage7/results/MASTER_results_table.csv (AUC, CI, flags, proportion of random panels >= panel); random distributions from
Stage7/results/bench/<task>__records.jsonl via Stage8/data/random_auc_long.csv (medians verified identical to the master table).""")
# =============================================================== Figure 2
def fig2():
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True, gridspec_kw={"width_ratios": [6, 6, 5]})
    rows = []
    for ax, (task, folds) in zip(axs, TASKS.items()):
        P = PANELS[task]; s = M[M.Task == task]
        for k, p in enumerate(P):
            a = s[s.Panel == p]; c = a[a.flag == "clean"]
            fc = c.groupby("Held-out cohort").beats.sum(); nf = int((fc >= 3).sum())
            pc = c.beats.mean() if len(c) else np.nan; pa = a.beats.mean()
            rows.append(dict(Task=task, Panel=p, clean_cells=len(c), clean_beat=int(c.beats.sum()), all_cells=len(a), all_beat=int(a.beats.sum()),
                             clean_folds=int(c["Held-out cohort"].nunique()), clean_folds_ge3of5=nf))
            ax.bar(k, pc if len(c) else 0, width=0.62, color=PCOL[p], edgecolor="white", lw=2)
            ax.bar(k, pa, width=0.62, fill=False, edgecolor=INK2, lw=1, ls="--")
            ax.text(k, (pc if len(c) else 0) + 0.03, f"{int(c.beats.sum())}/{len(c)}" if len(c) else "0 clean", ha="center", fontsize=9, color=INK, fontweight="bold")
        TT = pd.DataFrame([r for r in rows if r["Task"] == task])
        ax.set_xticks(range(len(P))); ax.set_xticklabels([f"{PLAB[r.Panel]}\nclean folds {r.clean_folds_ge3of5}/{r.clean_folds}\nall cells {r.all_beat}/{r.all_cells}" for r in TT.itertuples()], fontsize=7.8)
        ax.set_ylim(0, 1.1); ax.set_title(task, fontsize=10.5, color=INK); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("share of CLEAN cells beating random\n(≤5% of 50 random panels reach the panel)")
    fig.legend(handles=[Patch(color=GREYD, label="clean cells (bar, bold label)"), Patch(fill=False, edgecolor=INK2, ls="--", label="all cells incl. flagged (dashed outline)")],
               loc="upper right", ncol=2, fontsize=8.5, bbox_to_anchor=(1, 1.04))
    fig.suptitle("Figure 2. How often each curated panel beats size-matched random panels (cell = fold × method)", x=0.01, ha="left", y=1.04, fontsize=11, color=INK)
    fig.tight_layout(); save(fig, "Fig2_beats_random_summary")
    T = pd.DataFrame(rows); T.to_csv(OUT / "Fig2_data.csv", index=False)
    KV = "; ".join(f"{r.Task} {PLAB[r.Panel]} {r.clean_beat}/{r.clean_cells} ({r.all_beat}/{r.all_cells})" for r in T.itertuples())
    cap("Fig2_beats_random_summary", f"""
Figure 2. Proportion of cells (cell = one fold x one method; 5 methods per fold) in which each curated panel beats its size-matched random
distribution (<=5% of 50 random panels reach the panel's AUC), clean cells only (bars, bold k/n). Dashed outlines = all cells including those
with a derivation-overlap or sibling flag. Tick labels: clean folds in which >=3 of 5 methods beat random, and the all-cells count.
Counts (clean beat / clean cells; all cells): {KV}.
Sources: Stage7/results/MASTER_results_table.csv; per-panel counts in Stage8/figures/Fig2_data.csv.""")
    return T

# =============================================================== Figure 3
def fig3():
    s = M[M.Task == "ATB vs LTB"].copy(); s["gap"] = s.AUC - s["Random panel AUC median"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    rng = np.random.default_rng(0); order = sorted(PANELS["ATB vs LTB"], key=lambda p: (s[s.Panel == p]["Panel genes used"].median(), p)); pos = {p: i for i, p in enumerate(order)}
    for p in PANELS["ATB vs LTB"]:
        a = s[s.Panel == p]; x = pos[p] + rng.uniform(-0.22, 0.22, len(a))
        axs[0].scatter(x, a.AUC, s=22, color=PCOL[p], edgecolor="white", lw=0.6, label=PLAB[p], zorder=3)
        axs[0].scatter(x, a["Random panel AUC median"], s=22, marker="_", color=INK2, lw=1.2, zorder=2)
        axs[1].scatter(x, a.gap, s=22, color=PCOL[p], edgecolor="white", lw=0.6, zorder=3)
    g = s.groupby("Panel").agg(size=("Panel genes used", lambda v: f"{int(v.min())}" if v.min() == v.max() else f"{int(v.min())}-{int(v.max())}"), med=("gap", "median"),
                               lo=("gap", lambda v: v.quantile(.1)), hi=("gap", lambda v: v.quantile(.9))).reindex(order).reset_index()
    axs[1].plot(range(len(order)), g.med, color=INK, lw=1.5, marker="o", ms=5, zorder=4, label="median per panel")
    for ax in axs: ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{PLAB[p]}\n{z} genes" for p, z in zip(g.Panel, g["size"])], fontsize=8); ax.set_xlabel("panel (ordered by genes used)"); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("held-out AUC"); axs[0].set_title("Curated (colour) vs random-panel median (grey tick), every fold × method", fontsize=9.5)
    axs[1].axhline(0, color=GREYD, lw=1); axs[1].set_ylabel("curated AUC − random-panel median")
    axs[1].set_title("Advantage over size-matched random panels", fontsize=9.5)
    axs[0].legend(fontsize=8, loc="lower right", ncol=2); axs[1].legend(fontsize=8, loc="upper right")
    for i, r in enumerate(g.itertuples()): axs[1].annotate(f"{r.med:+.3f}", (i, r.med), xytext=(14, 0), textcoords="offset points", fontsize=8.5, color=INK, va="center", fontweight="bold")
    fig.suptitle("Figure 3. ATB vs LTB: curated and random panels converge from ~16 genes; separation only at 3 genes (Sweeney3)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); save(fig, "Fig3_ATBvsLTB_panel_size")
    cap("Fig3_ATBvsLTB_panel_size", f"""
Figure 3. ATB vs LTB, all 150 cells (5 folds x 6 panels x 5 methods). Left: curated-panel held-out AUC (colour) and the median AUC of 50
size-matched random panels in the same cell (grey tick), panels ordered by genes used (small horizontal jitter).
Right: curated minus random-panel median; black line = median per panel.
Median advantage by panel: {"; ".join(f"{PLAB[r.Panel]} ({r.size} genes) {r.med:+.3f} (10-90%: {r.lo:+.3f} to {r.hi:+.3f})" for r in g.itertuples())}.
Sources: Stage7/results/MASTER_results_table.csv (columns AUC, Random panel AUC median, Panel genes used). Per-cohort gene-level separability
and random medians: Stage7/results/FINAL_Table_task_difficulty.csv.""")
# =============================================================== Figure 4
def fig4():
    pw = []
    for task, coh in TASKS.items():
        t = task.replace(" vs ", "_vs_"); src = S7 / "results/native" if t == "ATB_vs_OD" else MT / "Stage5/results/native"
        pools = {c: set(json.load(open(src / f"{t}__{c}__cohort.json"))["mrmr_top20"]) for c in coh}
        for a, b in itertools.combinations(coh, 2): pw.append(dict(Task=task, a=a, b=b, overlap=len(pools[a] & pools[b])))
    PW = pd.DataFrame(pw); PW.to_csv(OUT / "Fig4_pairwise_overlap_data.csv", index=False)
    C = pd.read_csv(R7 / "FINAL_Table_consensus_failure.csv")
    fig = plt.figure(figsize=(15, 5.2)); gs = fig.add_gridspec(1, 5, width_ratios=[1.1, 1, 1, 0.15, 2.4])
    for k, (task, coh) in enumerate(TASKS.items()):
        ax = fig.add_subplot(gs[0, k]); n = len(coh); Mx = np.full((n, n), np.nan)
        for r in PW[PW.Task == task].itertuples(): i, j = coh.index(r.a), coh.index(r.b); Mx[j, i] = r.overlap
        im = ax.imshow(Mx, cmap="Blues", vmin=0, vmax=20)
        for i in range(n):
            for j in range(n):
                if not np.isnan(Mx[i, j]): ax.text(j, i, int(Mx[i, j]), ha="center", va="center", fontsize=10, color=INK if Mx[i, j] < 10 else "white", fontweight="bold")
        ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(coh, rotation=60, ha="right", fontsize=7.5); ax.set_yticklabels(coh, fontsize=7.5)
        ax.grid(False); ax.set_title(f"{task}\ntop-20 shortlist overlap (genes of 20)", fontsize=9)
        for sp in ax.spines.values(): sp.set_visible(False)
    ax = fig.add_subplot(gs[0, 4]); y = 0; yt, yl = [], []
    tc = {"ATB vs LTB": "#2a78d6", "ATB vs OD": "#eb6834", "LTB vs CON": "#008300"}
    for task in TASKS:
        for r in C[C.Task == task].itertuples():
            nm = re.match(r"(\d+) \[(\d+)–(\d+)\]", r[C.columns.get_loc("Same, random-ranking median [95% range] (all)") + 1]); md, lo, hi = map(float, nm.groups())
            obs = r[C.columns.get_loc("Rank depth until a gene is in every cohort's list") + 1]; pv = r[C.columns.get_loc("P (random depth ≤ observed), all lists") + 1]
            ax.plot([lo, hi], [y, y], color=GREY, lw=7, solid_capstyle="butt", zorder=1); ax.plot(md, y, "|", color=INK2, ms=12, mew=2, zorder=2)
            ax.plot(obs, y, "o", color=tc[task], ms=8, mec="white", mew=1, zorder=3)
            ax.text(2600, y, f"{obs}  (p{'<0.001' if pv == 0 else '=' + format(pv, '.3f')})", va="center", fontsize=7.5, color=INK)
            yt.append(y); yl.append(f"{task} | held out {r[2]}"); y -= 1
        y -= 0.6
    ax.set_xscale("log"); ax.set_xlim(8, 6000); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.5); ax.set_xlabel("rank depth until a gene appears in EVERY training cohort's MI list (log)")
    ax.set_title("Observed depth (dot) vs independent random rankings\n(grey = 95% range, tick = median; p = share of random ≤ observed)", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 4. Consensus-selection failure: per-cohort shortlists do not overlap, and only ATB vs LTB rankings agree far beyond chance", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); save(fig, "Fig4_consensus_failure")
    s = PW.groupby("Task").overlap.agg(["min", "max", lambda v: int((v > 0).sum()), "size"])
    cap("Fig4_consensus_failure", f"""
Figure 4. Left three panels: number of genes shared between the per-cohort top-20 Native shortlists (MI top-2000 -> mRMR top-20) for every
cohort pair in each task (lower triangle; per-cohort shortlists are fold-independent). Pairs with any overlap: {"; ".join(f"{t} {int(r.iloc[2])}/{int(r.iloc[3])} (range {int(r['min'])}-{int(r['max'])})" for t, r in s.iterrows())}.
Right: for each fold, the rank depth at which a gene first appears in every training cohort's mutual-information ranking (dot) against
1,000 independent random rankings of the same universe (grey bar = 95% range, tick = median); p = share of random draws at or below the
observed depth. ATB vs LTB 15-62 vs random median 984 (p < 0.001); ATB vs OD 82-238 vs 615 (p 0.003-0.037); LTB vs CON 209-734 vs 439
(p 0.08-0.95, i.e. chance).
Sources: Stage7/results/FINAL_Table_consensus_failure.csv; shortlists Stage5/results/native/*__cohort.json (ATB vs LTB, LTB vs CON) and
Stage7/results/native/ATB_vs_OD__*__cohort.json (rebuilt GSE83456); pair counts in Stage8/figures/Fig4_pairwise_overlap_data.csv.""")
# =============================================================== Figure 5
def fig5():
    G = pd.read_csv(R7 / "FINAL_Table_native_stability_genes.csv"); FC = {"olfactory receptor": "#6250d6", "microRNA": "#e34948", "pseudogene": "#eda100"}
    fig, axs = plt.subplots(1, 3, figsize=(15, 13), gridspec_kw={"width_ratios": [5, 4, 4]})
    for ax, (task, coh) in zip(axs, TASKS.items()):
        g = G[G.Task == task].reset_index(drop=True); n = len(g)
        for i, r in g.iterrows():
            for j, c in enumerate(coh):
                on = r[f"held-out {c}"] == "●"
                ax.add_patch(plt.Rectangle((j + 0.06, n - 1 - i + 0.06), 0.88, 0.88, color="#2a78d6" if on else "#f1f0ec", lw=0))
            fl = r.Flag if isinstance(r.Flag, str) else ""
            ax.text(-0.15, n - 1 - i + 0.5, r.Gene, ha="right", va="center", fontsize=7, color=FC.get(fl.split(";")[0], INK), fontweight="bold" if fl else "normal")
            ax.text(len(coh) + 0.15, n - 1 - i + 0.5, r["Folds selected"] + (f"  [{fl}]" if fl else ""), ha="left", va="center", fontsize=6.5, color=FC.get(fl.split(";")[0], INK2))
        ax.set_xlim(0, len(coh)); ax.set_ylim(0, n); ax.set_xticks(np.arange(len(coh)) + 0.5); ax.set_xticklabels([f"held out\n{c}" for c in coh], fontsize=7.5)
        ax.xaxis.tick_top(); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)
        shared = int((g["Folds selected"].str.split("/").str[0].astype(int) == len(coh)).sum())
        ax.set_title(f"{task}: {n} distinct genes, {shared} in every fold", fontsize=10, pad=34, color=INK)
    fig.legend(handles=[Patch(color="#2a78d6", label="selected in that fold's Native panel")] + [Patch(color=v, label=f"{k} (gene name coloured)") for k, v in FC.items()],
               loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Figure 5. Native panel instability: gene × fold presence (20 genes per fold; X/Y genes excluded)", x=0.01, ha="left", y=1.0, fontsize=11, color=INK)
    fig.tight_layout(rect=(0.03, 0.02, 1, 0.98)); save(fig, "Fig5_native_instability")
    S = pd.read_csv(R7 / "FINAL_Table_native_stability_summary.csv")
    cap("Fig5_native_instability", "Figure 5. Presence (blue) of each gene in each fold's 20-gene Native panel, per task; genes sorted by number of folds. Gene names coloured "
        "and labelled by HGNC-derived flag (olfactory receptor, microRNA, pseudogene). Summary: " + "; ".join(
        f"{r.Task}: {r['Distinct genes (all folds)']} distinct genes, {r['Genes in every fold']} in every fold, pairwise overlap {r['Pairwise fold overlap, genes (min–max)']}, "
        f"OR {r['Olfactory receptor']}, miRNA {r['microRNA']}, pseudogene {r['Pseudogene']}" for _, r in S.iterrows()) +
        ".\nSources: Stage7/results/FINAL_Table_native_stability_genes.csv and FINAL_Table_native_stability_summary.csv (ATB vs OD with rebuilt GSE83456).")
# =============================================================== Figure 6
def fig6():
    G = pd.read_csv(R7 / "FINAL_generalisation_gap.csv"); G["m"] = G.Method.map(MNAME); G = G[G.m.isin(METHODS)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax, task in zip(axs, TASKS):
        g = G[G.Task == task]
        for m in METHODS:
            a = g[g.m == m]; ax.scatter(a["Training AUC, pooled resubstitution"], a["Held-out AUC"], s=18, color=MCOL[m], alpha=0.8, edgecolor="white", lw=0.4,
                                         label=f"{m}: gap {a['Gap (pooled training minus held-out)'].min():+.2f} to {a['Gap (pooled training minus held-out)'].max():+.2f}", zorder=3)
        ax.plot([0.3, 1.02], [0.3, 1.02], color=GREYD, lw=1, ls="--", zorder=1); ax.set_xlim(0.3, 1.02); ax.set_ylim(0.3, 1.02); ax.set_aspect("equal")
        ax.set_title(task, fontsize=10.5); ax.set_xlabel("training AUC (pooled resubstitution)"); ax.legend(fontsize=7.5, loc="upper left", title="gap = training − held-out", title_fontsize=7.5)
    axs[0].set_ylabel("held-out AUC")
    fig.suptitle("Figure 6. Generalisation gap per fold × panel: trained models (LR, RF, MLP) fall furthest below the diagonal; untrained scores (singscore, DirAvg) sit closer to it",
                 x=0.01, ha="left", y=1.03, fontsize=11, color=INK)
    fig.tight_layout(); save(fig, "Fig6_generalisation_gap")
    s = G.groupby(["Task", "m"])["Gap (pooled training minus held-out)"].agg(["min", "median", "max"]).round(3)
    cap("Fig6_generalisation_gap", "Figure 6. Each point = one fold x panel x method: pooled training-cohort resubstitution AUC (x) vs held-out AUC (y); "
        "distance below the dashed diagonal = generalisation gap. MLP at fixed alpha 10.\nGap min / median / max by task and method:\n" +
        "\n".join(f"  {t} {m}: {r['min']:+.3f} / {r['median']:+.3f} / {r['max']:+.3f}" for (t, m), r in s.iterrows()) +
        "\nSource: Stage7/results/FINAL_generalisation_gap.csv (all tasks; ATB vs OD with rebuilt GSE83456).")
# =============================================================== Figure 7
def fig7():
    X = pd.read_csv(R7 / "FINAL_sex_gene_exclusion.csv"); s3 = pd.read_csv(MT / "Stage3/results/sex_balance.csv"); s7 = pd.read_csv(R7 / "sex_balance_ATB_vs_OD.csv")
    SB = pd.concat([s3.rename(columns={"male_frac_pos": "pos", "male_frac_neg": "neg"})[["task", "cohort", "pos", "neg", "sex_source"]],
                    s7.rename(columns={"male_frac_ATB": "pos", "male_frac_OD": "neg"})[["task", "cohort", "pos", "neg", "sex_source"]]])
    Y = {"RPS4Y1", "TXLNGY", "BCORP1", "TTTY7"}; genes = sorted({g for v in X.sex_genes_in_withsex_panel.dropna() for g in v.split(";")}, key=lambda g: (g not in Y, g))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]; rows = [(r.task.replace("_vs_", " vs "), r.heldout, str(r.sex_genes_in_withsex_panel) if isinstance(r.sex_genes_in_withsex_panel, str) else "") for r in X.itertuples()]
    for i, (t, h, gs) in enumerate(rows):
        for j, g in enumerate(genes):
            on = g in gs.split(";"); ax.add_patch(plt.Rectangle((j + 0.06, len(rows) - 1 - i + 0.06), 0.88, 0.88, color=("#2a78d6" if g in Y else "#e87ba4") if on else "#f1f0ec", lw=0))
    ax.set_xlim(0, len(genes)); ax.set_ylim(0, len(rows)); ax.set_xticks(np.arange(len(genes)) + 0.5); ax.set_xticklabels([f"{g} ({'Y' if g in Y else 'X'})" for g in genes], rotation=45, ha="left", fontsize=8)
    ax.xaxis.tick_top(); ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels([f"{t} | held out {h}" for t, h, _ in rows][::-1], fontsize=8); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("X/Y genes entering the Native panel when allowed (9 of 13 folds)", fontsize=9.5, pad=8, y=1.18)
    ax.legend(handles=[Patch(color="#2a78d6", label="Y-linked"), Patch(color="#e87ba4", label="X-linked")], loc="lower right", bbox_to_anchor=(1, -0.12), ncol=2, fontsize=8)
    ax = axs[1]; tc = {"ATB_vs_LTB": "#2a78d6", "ATB_vs_OD": "#eb6834", "LTB_vs_CON": "#008300"}; yl = []
    for i, r in enumerate(SB.itertuples()):
        y = len(SB) - 1 - i; ax.plot([r.neg, r.pos], [y, y], color=GREYD, lw=2, zorder=1)
        ax.plot(r.pos, y, "o", color=tc[r.task], ms=8, zorder=3); ax.plot(r.neg, y, "o", mfc="white", mec=tc[r.task], mew=1.8, ms=8, zorder=3)
        ax.text(1.02, y, f"{r.pos - r.neg:+.2f}", va="center", fontsize=8, color=INK); yl.append(f"{r.task.replace('_vs_', ' vs ')} | {r.cohort}")
    ax.set_yticks(range(len(SB))); ax.set_yticklabels(yl[::-1], fontsize=8); ax.set_xlim(0.2, 1.0); ax.axvline(0.5, color=GREYD, lw=0.8, ls=":")
    ax.set_xlabel("fraction male\n(filled = ATB, or LTB in LTB vs CON; open = comparison class)"); ax.set_title("Sex balance by class per cohort (difference at right)", fontsize=9.5)
    fig.suptitle("Figure 7. Sex-chromosome genes are selected where ATB is male-enriched (ATB tasks) and not in the sex-balanced LTB vs CON task", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); save(fig, "Fig7_sex_genes")
    cap("Fig7_sex_genes", "Figure 7. Left: X/Y-linked genes present in each fold's Native panel when X/Y genes are allowed in selection (final universe, comparison run "
        "only; primary panels exclude them): ATB vs LTB 5/5 folds (TXLNGY 4, RPS4Y1 3, BCORP1 3), ATB vs OD 4/4 (TTTY7 3, SPANXA2-OT1 3, MAGEB5, CDX4), "
        "LTB vs CON 0/4. Right: fraction male in the positive (filled) and comparison (open) class per cohort; sex from GEO annotation or inferred from RPS4Y1.\n"
        "Sources: Stage7/results/FINAL_sex_gene_exclusion.csv; Stage3/results/sex_balance.csv (ATB vs LTB, LTB vs CON); Stage7/results/sex_balance_ATB_vs_OD.csv "
        "(GSE42834: 4 samples without sex annotation counted as not male).")
# =============================================================== Figure 8
def fig8():
    G = pd.read_csv(R7 / "global_shift_heldout.csv"); tc = {"ATB_vs_LTB": "#2a78d6", "ATB_vs_OD": "#eb6834", "LTB_vs_CON": "#008300"}
    fig, ax = plt.subplots(figsize=(9, 5.2)); yl = []
    for i, r in enumerate(G.itertuples()):
        y = len(G) - 1 - i; lo, hi = map(float, r.CI.split("-")); q = bool(r.quantile_normalised) or r.heldout == "GSE83456"   # Stage7 rebuild is quantile-normalised (flag column predates it)
        ax.plot([lo, hi], [y, y], color=tc[r.task], lw=2, alpha=0.6); ax.plot(r.allgene_mean_AUC, y, "o" if not q else "D", ms=8, color=tc[r.task], mfc="white" if q else tc[r.task], mew=1.8)
        ax.text(1.0, y, f"{r.allgene_mean_AUC:.3f}   genes |AUC−0.5|>0.2: {r.share_genes_AUCdev_gt_0_2:.1%}", va="center", fontsize=8, color=INK)
        yl.append(f"{r.task.replace('_vs_', ' vs ')} | {r.heldout}")
    ax.axvline(0.5, color=GREYD, lw=1, ls=":"); ax.set_yticks(range(len(G))); ax.set_yticklabels(yl[::-1], fontsize=8.5); ax.set_xlim(0, 1.0)
    ax.set_xlabel("AUC of the per-sample all-gene mean for the class (0.5 = no shift; 95% bootstrap CI)")
    ax.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="", label="per-cohort normalised as deposited"),
                       Line2D([], [], marker="D", color=INK2, mfc="white", ls="", label="quantile-normalised (diagnostic squashed by construction)")], loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, fontsize=8)
    fig.suptitle("Figure 8. Global-shift diagnostic per cohort: sign-inconsistent; GSE107994 (ATB vs LTB) extreme at 0.19", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); save(fig, "Fig8_global_shift")
    cap("Fig8_global_shift", "Figure 8. For each cohort within each task, AUC of the per-sample mean over all measured genes (log2) for the positive class, with 1,000-resample "
        "bootstrap 95% CI; right-hand text = value and share of genes individually separating the classes at |AUC − 0.5| > 0.2. Open diamonds = quantile-normalised "
        "cohorts (GSE19491, GSE42834, GSE83456 rebuilt), where the all-gene mean is near-constant by construction. Range 0.193 (GSE107994, ATB vs LTB) to 0.657 "
        "(GSE37250, ATB vs OD).\nSource: Stage7/results/global_shift_heldout.csv (GSE83456 recomputed from the rebuilt file; before/after in gse83456_global_shift_before_after.csv).")

if __name__ == "__main__":
    for a in sys.argv[1:]: globals()[f"fig{a}"]()
