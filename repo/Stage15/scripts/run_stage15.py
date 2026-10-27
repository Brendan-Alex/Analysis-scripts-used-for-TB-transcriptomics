"""Stage15: principal component analysis of cohort structure, per task.
Two views per task, computed on the task's gene universe over the samples that entered that task:
  (A) log2 expression as used before normalisation, coloured by cohort  -> shows between-cohort offsets
  (B) after the per-cohort, label-free z-scoring used by the models, coloured by class -> shows what the models see
Nothing is refitted and no selection is performed; this is a description of the input representation.
Reads only: Clean_rebuild/expression/*.csv.gz (GSE19491 from Stage3), Stage3/data/sample_manifest_stage3.csv,
Stage7/data/universe_<task>.csv.  Writes Stage15/figures (PNG preview + LZW TIFF + caption) and Stage15/results.
Run (laptop, Arial required):  cd "...\\MTB ML\\Stage15"  then  python scripts\\run_stage15.py
Options: --dpi 450  --draft (allow Liberation Sans for layout previews only)
"""
import argparse, sys, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.lines import Line2D
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--out"); ap.add_argument("--dpi", type=int, default=450)
ap.add_argument("--draft", action="store_true"); A = ap.parse_args()
S15 = Path(A.out) if A.out else Path(__file__).resolve().parents[1]; MT = Path(A.root) if A.root else S15.parent
CR, S3, S7 = MT / "Clean_rebuild", MT / "Stage3", MT / "Stage7"
FIG, RES = S15 / "figures", S15 / "results"
for d in (FIG, RES): d.mkdir(parents=True, exist_ok=True)

FAM = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Arial" if "Arial" in FAM else ("Liberation Sans" if A.draft and "Liberation Sans" in FAM else None)
if FONT is None: sys.exit("Arial is not installed - PLOS requires Arial. (--draft is for layout previews only.)")
CM = 1 / 2.54
plt.rcParams.update({"font.family": FONT, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "legend.title_fontsize": 8, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelpad": 2,
                     "axes.edgecolor": "#555555", "axes.labelcolor": "#111111", "xtick.color": "#333333", "ytick.color": "#333333",
                     "text.color": "#111111", "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.4,
                     "legend.columnspacing": 0.9, "legend.borderaxespad": 0.2, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})
TASKS = {"ATB_vs_LTB": dict(cohorts=["GSE19491", "GSE37250", "GSE73408", "GSE101705", "GSE107994"], cls=["ATB", "LTB"]),
         "ATB_vs_OD":  dict(cohorts=["GSE37250", "GSE73408", "GSE42834", "GSE83456"], cls=["ATB", "OD"]),
         "LTB_vs_CON": dict(cohorts=["GSE19439", "GSE19444", "GSE28623", "GSE107994"], cls=["LTB", "CON"])}
CCOL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]          # colour-blind safe, max 5 cohorts
KCOL = {"ATB": "#D55E00", "LTB": "#0072B2", "CON": "#009E73", "OD": "#CC79A7"}

def expr(c):
    f = S3 / "data/GSE19491_log2_genelevel_stage3.csv.gz" if c == "GSE19491" else CR / f"expression/{c}_log2_genelevel.csv.gz"
    X = pd.read_csv(f, index_col=0); X.index = X.index.astype(str); return X
def pca(M, n=2):
    """M: samples x genes. Returns scores, percent variance explained per PC (of total variance in M)."""
    Xc = M - M.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2; return U[:, :n] * S[:n], 100 * var[:n] / var.sum()

man = pd.read_csv(S3 / "data/sample_manifest_stage3.csv"); inc = man[man.included_stage3]
rows, store = [], {}
for task, t in TASKS.items():
    U = pd.read_csv(S7 / f"data/universe_{task}.csv").Gene.astype(str).tolist()
    raw, zed, meta = [], [], []
    for c in t["cohorts"]:
        X = expr(c); m = inc[inc.cohort == c].set_index("GSM")
        keep = [s for s in X.columns if s in m.index]                      # cohort's included samples, any class
        Xa = X[keep]
        Z = Xa.sub(Xa.mean(1), axis=0).div(Xa.std(1, ddof=0).replace(0, np.nan), axis=0).fillna(0.0)
        sel = [s for s in keep if m.loc[s, "class_label"] in t["cls"]]      # task samples only
        g = [x for x in U if x in Xa.index]
        assert len(g) == len(U), f"{task} {c}: {len(U) - len(g)} universe genes missing"
        raw.append(Xa.loc[U, sel].T); zed.append(Z.loc[U, sel].T)
        meta += [(s, c, m.loc[s, "class_label"]) for s in sel]
    R = pd.concat(raw); Zc = pd.concat(zed); M = pd.DataFrame(meta, columns=["GSM", "cohort", "cls"]).set_index("GSM")
    assert list(R.index) == list(M.index) == list(Zc.index) and not R.index.duplicated().any()
    sr, vr = pca(R.to_numpy()); sz, vz = pca(Zc.to_numpy())
    store[task] = (sr, vr, sz, vz, M)
    # how much of PC1-2 is explained by cohort vs class: one-way R^2 per grouping
    def r2(scores, lab):
        y = scores; grand = y.mean(0); ss = ((y - grand) ** 2).sum()
        within = sum(((y[lab == l] - y[lab == l].mean(0)) ** 2).sum() for l in np.unique(lab))
        return float(1 - within / ss)
    for lab, nm in [(M.cohort.to_numpy(), "cohort"), (M.cls.to_numpy(), "class")]:
        rows.append(dict(task=task.replace("_vs_", " vs "), view="log2 (pre-normalisation)", grouping=nm,
                         R2_PC1_PC2=round(r2(sr, lab), 3), PC1_pct=round(vr[0], 1), PC2_pct=round(vr[1], 1), n=len(M), genes=len(U)))
        rows.append(dict(task=task.replace("_vs_", " vs "), view="per-cohort z-score", grouping=nm,
                         R2_PC1_PC2=round(r2(sz, lab), 3), PC1_pct=round(vz[0], 1), PC2_pct=round(vz[1], 1), n=len(M), genes=len(U)))
T = pd.DataFrame(rows); T.to_csv(RES / "pca_variance_and_structure.csv", index=False)
print(T.to_string(index=False))
pd.concat([store[t][4].assign(task=t, PC1_log2=store[t][0][:, 0], PC2_log2=store[t][0][:, 1],
                              PC1_z=store[t][2][:, 0], PC2_z=store[t][2][:, 1]) for t in TASKS]).to_csv(RES / "pca_scores.csv")

fig = plt.figure(figsize=(19.05 * CM, 20.0 * CM))
W, H, L, TOP, GX, GY = 0.37, 0.235, 0.085, 0.955, 0.10, 0.085
for i, (task, t) in enumerate(TASKS.items()):
    sr, vr, sz, vz, M = store[task]
    for j, (sc, v, by) in enumerate([(sr, vr, "cohort"), (sz, vz, "class")]):
        ax = fig.add_axes([L + j * (W + GX), TOP - (i + 1) * H - i * GY, W, H])
        if by == "cohort":
            for k, c in enumerate(t["cohorts"]):
                m = (M.cohort == c).to_numpy(); ax.scatter(sc[m, 0], sc[m, 1], s=5, color=CCOL[k], lw=0, alpha=0.75, label=c)
        else:
            for c in t["cls"]:
                m = (M.cls == c).to_numpy(); ax.scatter(sc[m, 0], sc[m, 1], s=5, color=KCOL[c], lw=0, alpha=0.75, label=c)
        ax.set_xlabel(f"PC1 ({v[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({v[1]:.1f}%)")
        ax.set_title(("log2 expression, by cohort" if by == "cohort" else "after per-cohort z-scoring, by class"), loc="left", pad=3)
        ax.legend(loc="best", ncol=2 if by == "cohort" else 1, markerscale=1.6, handletextpad=0.2)
        ax.tick_params(labelsize=8)
    fig.text(0.005, TOP - i * (H + GY) + 0.035, "ABC"[i], fontsize=10, fontweight="bold", va="top")
    fig.text(L, TOP - i * (H + GY) + 0.032, task.replace("_vs_", " vs "), fontsize=8, fontweight="bold", va="top")

fig.canvas.draw(); R = fig.canvas.get_renderer()
for tx in fig.findobj(Text):
    s = tx.get_text()
    if not tx.get_visible() or not s.strip(): continue
    if not (8 - 1e-6 <= tx.get_fontsize() <= 12 + 1e-6): sys.exit(f"font size {tx.get_fontsize()} on '{s[:30]}'")
    if font_manager.get_font(font_manager.findfont(tx.get_fontproperties())).family_name != FONT: sys.exit(f"font on '{s[:30]}'")
png, tif = FIG / "Fig_PCA.png", FIG / "Fig_PCA.tif"
fig.savefig(png, dpi=A.dpi, facecolor="white"); plt.close(fig)
im = Image.open(png).convert("RGB"); im.save(png, dpi=(A.dpi, A.dpi)); im.save(tif, compression="tiff_lzw", dpi=(A.dpi, A.dpi))
w, h = im.size[0] / A.dpi * 2.54, im.size[1] / A.dpi * 2.54; mb = tif.stat().st_size / 1e6
ok = dict(TIFF=im.format == "TIFF", RGB=im.mode == "RGB", no_alpha="A" not in im.getbands(), LZW=True,
          dpi_300_600=300 <= A.dpi <= 600, width_ok=w <= 19.05 + 0.01, height_ok=h <= 22.23 + 0.01, size_ok=mb < 10)
pd.DataFrame([dict(file=tif.name, px=f"{im.size[0]}x{im.size[1]}", dpi=A.dpi, width_cm=round(w, 2), height_cm=round(h, 2),
                   MB=round(mb, 2), font=FONT, **ok, PASS=all(ok.values()))]).to_csv(FIG / "preflight_check.csv", index=False)
def g(task, view, grouping): return T[(T.task == task.replace("_vs_", " vs ")) & (T.view == view) & (T.grouping == grouping)].iloc[0]
cap = ["Fig X. Cohort and class structure before and after per-cohort normalisation.",
       "Principal component analysis of the samples entering each task, computed on that task's gene universe (genes measured in every cohort of the task, "
       "excluding sex-chromosome genes and uncharacterised loci). Left column: log2 expression as used before normalisation, coloured by cohort. Right column: the same "
       "samples after the label-free per-cohort z-scoring used by all models, coloured by class. Axis labels give the percentage of total variance carried by each component. "
       "(A) ATB vs LTB, (B) ATB vs OD, (C) LTB vs CON.",
       "Proportion of variance in the first two components attributable to cohort and to class (one-way R2):"]
for task in TASKS:
    cap.append("  " + task.replace("_vs_", " vs ") + ": log2 cohort R2 = {:.2f}, class R2 = {:.2f}; after z-scoring cohort R2 = {:.2f}, class R2 = {:.2f} (n = {}, {} genes)".format(
        g(task, "log2 (pre-normalisation)", "cohort").R2_PC1_PC2, g(task, "log2 (pre-normalisation)", "class").R2_PC1_PC2,
        g(task, "per-cohort z-score", "cohort").R2_PC1_PC2, g(task, "per-cohort z-score", "class").R2_PC1_PC2,
        int(g(task, "per-cohort z-score", "class").n), int(g(task, "per-cohort z-score", "class").genes)))
cap.append("Sources: Clean_rebuild/expression (GSE19491 from Stage3/data), Stage3/data/sample_manifest_stage3.csv, Stage7/data/universe_<task>.csv. "
           "Values in Stage15/results/pca_variance_and_structure.csv; per-sample scores in pca_scores.csv.")
(FIG / "Fig_PCA_caption.txt").write_text("\n".join(cap) + "\n", encoding="utf-8")
print("\n".join(cap)); print(f"\nsaved {png.name} / {tif.name}  {w:.2f} x {h:.2f} cm, {mb:.2f} MB")
