"""Stage16: final publication figures (PLOS ONE spec) for the TB benchmarking paper.
Rebuilds the agreed eight main figures and the supporting set from saved Stage3/7/8/11/12/13/15 outputs.
Nothing is refitted, re-selected or re-derived. Berry86 is used as published (the TLR7-removed variant is
dropped from all figures and from the DeLong comparator family, which is recomputed here).

Per figure: <name>.png (preview, same pixels/dpi as the TIFF), <name>.tif (RGB, LZW), <name>_caption.txt.
Also writes results/delong_paired.csv (six comparators) and figures/preflight_check.csv.

Run (laptop, Arial required):  cd "...\\MTB ML\\Stage16"  then  python scripts\\run_stage16.py
Options: --only Fig3 Fig6 ...   --dpi 450   --draft (metric-identical Liberation Sans; previews only)
"""
import argparse, json, re, sys, gzip, itertools
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch, FancyArrowPatch
from scipy.stats import norm
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--out"); ap.add_argument("--only", nargs="*")
ap.add_argument("--dpi", type=int, default=450); ap.add_argument("--draft", action="store_true"); A = ap.parse_args()
S16 = Path(A.out) if A.out else Path(__file__).resolve().parents[1]; MT = Path(A.root) if A.root else S16.parent
S3, S7, S8, S11, S12, S13, S15 = (MT / f"Stage{i}" for i in (3, 7, 8, 11, 12, 13, 15))
FIG, RES = S16 / "figures", S16 / "results"
for d in (FIG, RES): d.mkdir(parents=True, exist_ok=True)
DPI = A.dpi
FAM = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Arial" if "Arial" in FAM else ("Liberation Sans" if A.draft and "Liberation Sans" in FAM else None)
if FONT is None: sys.exit("Arial is not installed - PLOS requires Arial. Use --draft only for layout previews.")
CM = 1 / 2.54; WMAX, HMAX = 19.05, 22.23
plt.rcParams.update({"font.family": FONT, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "legend.title_fontsize": 8, "figure.titlesize": 9, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2,
    "axes.edgecolor": "#555555", "axes.labelcolor": "#111111", "xtick.color": "#333333", "ytick.color": "#333333",
    "text.color": "#111111", "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0, "legend.borderaxespad": 0.2, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "lines.linewidth": 1.0, "pdf.fonttype": 42})
INK, INK2, GREY, GREYD = "#111111", "#555555", "#dddddd", "#b3b3b3"
PCOL = {"Nat": "#D55E00", "Native": "#0072B2", "Zak16": "#E69F00", "Kaforou_ATB_v_LTB": "#009E73",
        "Kaforou_ATB_v_OD": "#009E73", "Berry86": "#CC79A7", "Sweeney3": "#8C564B"}
PLAB = {"Nat": "Nat", "Native": "Nat(mRMR)", "Zak16": "Zak16", "Kaforou_ATB_v_LTB": "Kaforou",
        "Kaforou_ATB_v_OD": "Kaforou", "Berry86": "Berry86", "Sweeney3": "Sweeney3"}
METHODS = ["LR", "RF", "MLP", "singscore", "DirAvg"]
MLAB = {"LR": "Logistic regression", "RF": "Random forest", "MLP": "MLP (α = 10)", "singscore": "singscore", "DirAvg": "DirAvg"}
MCOL = {"LR": "#0072B2", "RF": "#E69F00", "MLP": "#CC79A7", "singscore": "#009E73", "DirAvg": "#D55E00"}
MNAME = {"Logistic regression": "LR", "Random forest": "RF", "MLP (alpha fixed at 10)": "MLP",
         "singscore (rank-based, untrained)": "singscore", "Directional average (untrained)": "DirAvg",
         "singscore": "singscore", "Directional average": "DirAvg"}
TASKS = {"ATB vs LTB": ["GSE19491", "GSE37250", "GSE73408", "GSE101705", "GSE107994"],
         "ATB vs OD": ["GSE37250", "GSE73408", "GSE42834", "GSE83456"],
         "LTB vs CON": ["GSE19439", "GSE19444", "GSE28623", "GSE107994"]}
PANELS = {"ATB vs LTB": ["Nat", "Native", "Zak16", "Kaforou_ATB_v_LTB", "Berry86", "Sweeney3"],
          "ATB vs OD": ["Nat", "Native", "Zak16", "Kaforou_ATB_v_OD", "Berry86", "Sweeney3"],
          "LTB vs CON": ["Nat", "Native", "Zak16", "Berry86", "Sweeney3"]}
TCOL = {"ATB vs LTB": "#0072B2", "ATB vs OD": "#D55E00", "LTB vs CON": "#009E73"}
QN = {"GSE19491", "GSE42834", "GSE83456"}
NAT = {"ATB vs LTB": ["ANKRD22", "FCGR1A", "PSTPIP2"], "ATB vs OD": ["APOL2", "WDFY1", "BATF2", "POLB", "GBP2"]}
TKEY = lambda t: t.replace(" vs ", "_vs_")
MADE = []

# ---------------------------------------------------------------- inputs
M = pd.read_csv(S7 / "results/MASTER_results_table.csv"); M["m"] = M.Method.map(MNAME)
FL = "Sibling-cohort flag (GSE19439 fold: panel selection included sister sub-series GSE19444)"
M["flag"] = np.where(M["Optimistic (any derivation overlap)"], "derivation", np.where(M[FL], "sibling", "clean"))
RA = pd.read_csv(S8 / "data/random_auc_long.csv")
CK = pd.read_csv(S11 / "results/chosen_k_per_fold.csv"); CK = CK[CK.Ranking == "consistent"]
AB = pd.read_csv(S11 / "results/ablation_curves_long.csv"); AB = AB[AB.ranking == "consistent"]
RBS = {(r["task"], r["heldout"], r["k"]): r["auc"] for r in map(json.loads, open(S11 / "results/random_by_size.jsonl"))}
FS = {(r["task"].replace("_vs_", " vs "), r["ranking"]): r for r in map(json.loads, open(S11 / "results/final_signatures.jsonl"))}
LAB = json.load(open(S7 / "data/labels.json"))
for t, g in NAT.items():
    if FS[(t, "consistent")]["final_signature"] != g: sys.exit(f"STOP: Nat list for {t} disagrees with Stage11")

def roc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    o = np.argsort(-s, kind="mergesort"); y = y[o]; s = s[o]
    d = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[d]; fp = np.cumsum(1 - y)[d]
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]
def fauc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

_SC = {}
def scores_for(task, held, panel, method="DirAvg"):
    """held-out predictions: curated panels and Nat(mRMR) from Stage7 bench records; Nat from Stage11 fold curves."""
    k = TKEY(task)
    if panel == "Nat":
        if ("nat", k) not in _SC:
            d = {}
            for line in open(S11 / "results/fold_curves.jsonl"):
                r = json.loads(line)
                if r["task"] == k and r["ranking"] == "consistent":
                    d[r["heldout"]] = r
            _SC[("nat", k)] = d
        r = _SC[("nat", k)][held]; kk = r["choice"][method]
        c = [x for x in r["curve"] if x["k"] == kk][0]; return np.array(c["scores"][method], dtype=float), kk
    if ("cur", k) not in _SC:
        d = {}
        for line in open(S7 / f"results/bench/{k}__records.jsonl"):
            r = json.loads(line)
            if r.get("kind") == "cur": d[(r["heldout"], r["panel"])] = r
        _SC[("cur", k)] = d
    r = _SC[("cur", k)][(held, panel)]
    return np.array(r["scores"][method][held], dtype=float), len(r["features"])
def labels_for(task, held):
    gsm, y = LAB[TKEY(task)][held]; return np.array(y, dtype=int), gsm

# ---------------------------------------------------------------- helpers
def fig_new(w, h):
    assert w <= WMAX + 1e-9 and h <= HMAX + 1e-9, (w, h); return plt.figure(figsize=(w * CM, h * CM))
def letter(fig, x, y, s): fig.text(x, y, s, fontsize=10, fontweight="bold", va="top", ha="left")
def check(fig, name):
    fig.canvas.draw(); R = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height; bad, warn, boxes = [], [], []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not t.get_visible() or not s.strip(): continue
        if not (8 - 1e-6 <= t.get_fontsize() <= 12 + 1e-6): bad.append(f"size {t.get_fontsize():.1f}: '{s[:30]}'")
        if font_manager.get_font(font_manager.findfont(t.get_fontproperties())).family_name != FONT: bad.append(f"font: '{s[:30]}'")
        if re.search(r"(?<![/_])\bnative", s, re.I): bad.append(f"retired name: '{s[:40]}'")
        bb = t.get_window_extent(R)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1: warn.append(f"outside canvas: '{s[:30]}'")
        boxes.append((bb, s))
    for (b1, s1), (b2, s2) in itertools.combinations(boxes, 2):
        ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0); iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ix > 1 and iy > 1 and ix * iy > 0.15 * min(b1.width * b1.height, b2.width * b2.height):
            warn.append(f"overlap: '{s1[:18]}' x '{s2[:18]}'")
    if bad: sys.exit(f"{name}: text check FAILED\n  " + "\n  ".join(bad[:20]))
    return warn
def save(fig, name, caption):
    w, h = fig.get_size_inches() / CM
    warn = check(fig, name); png, tif = FIG / f"{name}.png", FIG / f"{name}.tif"
    fig.savefig(png, dpi=DPI, facecolor="white"); plt.close(fig)
    im = Image.open(png).convert("RGB"); im.save(png, dpi=(DPI, DPI)); im.save(tif, compression="tiff_lzw", dpi=(DPI, DPI))
    if re.search(r"(?<![/_])\bnative", caption, re.I): sys.exit(f"{name}: retired name in caption")
    (FIG / f"{name}_caption.txt").write_text(caption.strip() + "\n", encoding="utf-8"); MADE.append(name)
    print(f"saved {name}  {w:.2f} x {h:.2f} cm" + "".join(f"\n   WARN {x}" for x in warn[:12]))
def boxrand(ax, x, d, w=0.36):
    lo, q1, md, q3, hi = np.percentile(d, [2.5, 25, 50, 75, 97.5])
    ax.add_patch(Rectangle((x - w, lo), 2 * w, hi - lo, color=GREY, lw=0, zorder=1))
    ax.add_patch(Rectangle((x - w, q1), 2 * w, q3 - q1, color=GREYD, lw=0, zorder=1))
    ax.plot([x - w, x + w], [md, md], color="white", lw=1.0, zorder=2)

def bench_rows():
    rows = []
    for task, folds in TASKS.items():
        for h in folds:
            for m in METHODS:
                for p in PANELS[task]:
                    if p == "Nat":
                        c = CK[(CK.Task == task) & (CK["Held-out cohort"] == h) & (CK.Method == m)].iloc[0]
                        k = int(c["Chosen k (inner LOCO)"]); d = np.array(RBS[(TKEY(task), h, k)][m])
                        rows.append(dict(task=task, fold=h, m=m, panel=p, auc=c["AUC at chosen k"], lo=c["CI lower"],
                                         hi=c["CI upper"], k=k, flag="sibling" if c["Sibling flag"] else "clean",
                                         prop=c["Proportion random >= at k"], dist=d))
                    else:
                        c = M[(M.Task == task) & (M["Held-out cohort"] == h) & (M.Panel == p) & (M.m == m)].iloc[0]
                        d = RA[(RA.Task == task) & (RA["Held-out cohort"] == h) & (RA.Panel == p) & (RA.method == m)].AUC.to_numpy()
                        assert len(d) == 50 and abs(np.median(d) - c["Random panel AUC median"]) < 6e-4
                        rows.append(dict(task=task, fold=h, m=m, panel=p, auc=c.AUC, lo=c["AUC 95% CI lower"],
                                         hi=c["AUC 95% CI upper"], k=int(c["Panel genes used"]), flag=c.flag,
                                         prop=c["Proportion of random panels with AUC >= panel"], dist=d))
    B = pd.DataFrame(rows); B["beats"] = B.prop <= 0.05; return B
B = bench_rows()

# ---------------------------------------------------------------- DeLong, six comparators
def midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2
def delong(y, s1, s2):
    y = np.asarray(y).astype(bool); P = np.vstack([s1, s2]); m, n = y.sum(), (~y).sum()
    X, Yn = P[:, y], P[:, ~y]
    tx = np.array([midrank(r) for r in X]); ty = np.array([midrank(r) for r in Yn]); tz = np.array([midrank(r) for r in P])
    auc = (tz[:, y].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, y] - tx) / n; v10 = 1 - (tz[:, ~y] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n; d = auc[0] - auc[1]; se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    z = d / se if se > 0 else 0.0; return auc[0], auc[1], d, se, z, 2 * norm.sf(abs(z)), (d - 1.96 * se, d + 1.96 * se)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj
SX = pd.read_csv(S13 / "results/external_scores_long.csv.gz")
EX = pd.read_csv(S13 / "results/external_auc.csv")
DROP = {"Berry86_noTLR7"}
rows = []
for G, g in SX.groupby("cohort"):
    for m, gm in g.groupby("method"):
        W = gm.pivot(index="GSM", columns="panel", values="score")
        y = gm.drop_duplicates("GSM").set_index("GSM").y.reindex(W.index).to_numpy()
        for q in [c for c in W.columns if c != "S11_consistent" and c not in DROP]:
            a1, a2, d, se, z, p, (lo, hi) = delong(y, W["S11_consistent"].to_numpy(), W[q].to_numpy())
            rows.append(dict(cohort=G, method=m, signature="Nat", comparator=q, AUC_Nat=round(a1, 4),
                             AUC_comparator=round(a2, 4), AUC_diff=round(d, 4), diff_CI_lo=round(lo, 4),
                             diff_CI_hi=round(hi, 4), z=round(z, 3), p=p))
DL = pd.DataFrame(rows)
DL["p_Holm"] = DL.groupby(["cohort", "method"]).p.transform(lambda v: holm(v.to_numpy()))
DL["p"] = DL.p.round(5); DL["p_Holm"] = DL.p_Holm.round(5)
DL.to_csv(RES / "delong_paired.csv", index=False)
print(f"DeLong recomputed with {DL.comparator.nunique()} comparators -> results/delong_paired.csv")

# ================================================================ Fig 1: data, preprocessing, design
def fig1():
    man = pd.read_csv(S3 / "data/sample_manifest_stage3.csv")
    def cat(r):
        e = str(r.exclusion_reason) if isinstance(r.exclusion_reason, str) else ""
        s = str(r.stage3_exclusion) if isinstance(r.stage3_exclusion, str) else ""
        for rx, lab in [(r"HIV", "HIV positive"), (r"not whole blood", "Not whole blood"), (r"extrapulmonary", "Extrapulmonary TB"),
                        (r"on treatment", "On treatment"), (r"GSE22098", "Outside class definitions"), (r"sarcoid", "Non-active sarcoidosis"),
                        (r"progressor|baseline", "LTBI progressor / not baseline"), (r"QC flag", "Submitter QC outlier"),
                        (r"GSE42832", "Not linkable to GSM"), (r"duplicate", "Duplicate array")]:
            if re.search(rx, e): return lab
        if e: return "Other"
        if s: return "Batch-confounded OD (GSE19491)" if "OD dropped" in s else "ATB without LTB in sub-series"
        return None
    man["cat"] = man.apply(cat, axis=1); incl = man[man.included_stage3]; exc = man[~man.included_stage3]
    ec = exc["cat"].value_counts(); n_all, n_inc = len(man), len(incl)
    tn = {t: M[M.Task == t].drop_duplicates("Held-out cohort") for t in TASKS}
    fig = fig_new(19.05, 21.0); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 19.05); ax.set_ylim(0, 21.0); ax.axis("off")
    def box(x, y, w, h, fc="#f4f4f4", ec_="#999999", lw=0.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec_, lw=lw))
    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK2))
    # A: cohorts and exclusions
    ax.text(0.3, 20.8, "A", fontsize=10, fontweight="bold", va="top")
    ax.text(0.85, 20.75, f"Discovery cohorts: 10 GEO series, {n_all:,} samples", fontsize=9, fontweight="bold", va="top")
    y0 = 20.0; box(0.25, 14.6, 10.25, 5.7, fc="white")
    cols = [(0.4, "Cohort", "left"), (2.55, "Data", "left"), (4.75, "Samples", "right"), (6.15, "Excluded", "right"),
            (7.3, "ATB", "right"), (8.3, "LTB", "right"), (9.3, "OD", "right"), (10.3, "CON", "right")]
    for x, s, ha in cols: ax.text(x, y0, s, fontsize=8, fontweight="bold", ha=ha, va="center")
    order = ["GSE19491", "GSE19439", "GSE19444", "GSE37250", "GSE73408", "GSE42834", "GSE83456", "GSE28623", "GSE101705", "GSE107994"]
    cls = incl.groupby(["cohort", "class_label"]).size().unstack(fill_value=0)
    used = {}
    for task, folds in TASKS.items():
        for c in folds:
            y, gsm = labels_for(task, c)
            pos, neg = task.split(" vs ")
            for g, v in zip(gsm, y): used.setdefault(c, {})[g] = pos if v == 1 else neg
    for i, c in enumerate(order):
        yy = y0 - 0.42 * (i + 1); u = pd.Series(used.get(c, {}), dtype=object)
        tech = "RNA-seq" if c in ("GSE101705", "GSE107994") else "Microarray"
        vals = [c, tech, f"{int((man.cohort == c).sum())}", f"{int((exc.cohort == c).sum())}"] + \
               [str(int((u == k).sum())) if (u == k).sum() else "–" for k in ("ATB", "LTB", "OD", "CON")]
        for (x, _, ha), v in zip(cols, vals): ax.text(x, yy, v, fontsize=8, ha=ha, va="center")
    yt = y0 - 0.42 * 11; allu = pd.Series({g: k for c in used for g, k in used[c].items()}, dtype=object)
    ax.text(0.4, yt, "Total (unique)", fontsize=8, fontweight="bold", va="center")
    for (x, _, ha), v in zip(cols[2:], [f"{n_all:,}", f"{len(exc)}"] + [str(int((allu == k).sum())) for k in ("ATB", "LTB", "OD", "CON")]):
        ax.text(x, yt, v, fontsize=8, fontweight="bold", ha=ha, va="center")
    bx, bw = 10.95, 7.85; box(bx, 14.6, bw, 5.7, fc="#fbf3ec", ec_="#D55E00")
    ax.text(bx + 0.2, y0, f"Excluded by sample: {len(exc)} (included {n_inc:,})", fontsize=8, fontweight="bold", va="center")
    for i, (lab, n) in enumerate(ec.items()):
        ax.text(bx + 0.25, y0 - 0.42 * (i + 1), lab, fontsize=8, va="center")
        ax.text(bx + bw - 0.2, y0 - 0.42 * (i + 1), str(n), fontsize=8, va="center", ha="right")
    # B: preprocessing
    yb = 14.1; ax.text(0.3, yb, "B", fontsize=10, fontweight="bold", va="top")
    ax.text(0.85, yb - 0.05, "Preprocessing (per cohort, no samples mixed, no labels used)", fontsize=9, fontweight="bold", va="top")
    steps = [("Download from GEO", "checksummed;\nsample labels from\nGEO metadata"), ("To gene level", "log2; probes → genes\nby highest mean;\ncurrent HGNC symbols"),
             ("Exclusions", "HIV, non-blood,\nEPTB, on-treatment,\nQC, duplicates"), ("Per-gene z-score", "within each cohort,\nover all included\nsamples, label-free"),
             ("Gene universe", "measured in every\ncohort of the task;\nno X/Y, no LOC/orf")]
    sw = 3.45
    for i, (a, b) in enumerate(steps):
        x = 0.3 + i * (sw + 0.33); box(x, yb - 2.55, sw, 1.95, fc="#f4f4f4")
        ax.text(x + sw / 2, yb - 0.95, a, fontsize=8, fontweight="bold", ha="center", va="center")
        ax.text(x + sw / 2, yb - 1.75, b, fontsize=8, ha="center", va="center", linespacing=1.2)
        if i: arrow(x - 0.31, yb - 1.55, x - 0.02, yb - 1.55)
    # C: tasks and LOCO
    yc = 11.1; ax.text(0.3, yc, "C", fontsize=10, fontweight="bold", va="top")
    ax.text(0.85, yc - 0.05, "Three tasks, rotating leave-one-cohort-out", fontsize=9, fontweight="bold", va="top")
    for k, (t, folds) in enumerate(TASKS.items()):
        x0 = 0.3 + k * 6.25; yb2 = yc - 0.6; w = 6.0; box(x0, yb2 - 4.05, w, 3.95, fc="white", ec_=TCOL[t], lw=0.8)
        d = tn[t]
        ax.text(x0 + 0.2, yb2 - 0.3, f"{t}: {len(folds)} cohorts, n = {int(d['Test n'].sum())}", fontsize=8, fontweight="bold", va="center", color=TCOL[t])
        ax.text(x0 + 0.2, yb2 - 0.72, f"{int(d['Test positive n'].sum())} {d['Positive class'].iloc[0]} vs {int(d['Test negative n'].sum())} {d['Negative class'].iloc[0]}", fontsize=8, va="center")
        cw = (w - 0.4) / len(folds)
        for i in range(len(folds)):
            yy = yb2 - 1.2 - 0.33 * i
            for j in range(len(folds)):
                ax.add_patch(Rectangle((x0 + 0.2 + j * cw + 0.03, yy - 0.13), cw - 0.06, 0.26, color=TCOL[t] if i == j else "#e4e4e4", lw=0))
        for j, c in enumerate(folds):
            ax.text(x0 + 0.2 + (j + 0.5) * cw, yb2 - 1.2 - 0.33 * len(folds) - 0.1, c.replace("GSE", ""), fontsize=8, ha="center", va="center")
        ax.text(x0 + 0.2, yb2 - 3.3, "Coloured = held out; grey = training", fontsize=8, va="center", color=INK2)
        ax.text(x0 + 0.2, yb2 - 3.7, "Panels and k from training only", fontsize=8, va="center", color=INK2)
    # D: validation
    yd = 5.9; ax.text(0.3, yd, "D", fontsize=10, fontweight="bold", va="top")
    ax.text(0.85, yd - 0.05, "Independent validation: models trained on all cohorts of the task, scored once", fontsize=9, fontweight="bold", va="top")
    F1 = pd.read_csv(S12 / "tables/GSE144127_filter_decisions.csv"); F2 = pd.read_csv(S12 / "tables/GSE152218_filter_decisions.csv")
    def flow(F, x, title, steps_, tail, col):
        keep = pd.Series(True, index=F.index); lines = [f"{len(F)} samples"]
        for c, lab in steps_:
            rm = int((keep & F[c].astype(str).str.lower().eq("true")).sum()); keep &= ~F[c].astype(str).str.lower().eq("true")
            lines.append(f"−{rm}  {lab}")
        vc = F[F.included.astype(str).str.lower().eq("true")]["class"].value_counts()
        box(x, 1.35, 9.1, yd - 2.05, fc="white", ec_=col, lw=0.8)
        ax.text(x + 0.2, yd - 0.95, title, fontsize=8, fontweight="bold", va="center", color=col)
        for i, s in enumerate(lines): ax.text(x + 0.3, yd - 1.45 - 0.38 * i, s, fontsize=8, va="center")
        ax.text(x + 0.3, yd - 1.45 - 0.38 * len(lines) - 0.1, tail(vc), fontsize=8, fontweight="bold", va="center")
        return vc
    v1 = flow(F1, 0.3, "GSE144127 (UK, microarray) → ATB vs OD",
              [("hiv_positive", "HIV positive"), ("eptb", "EPTB or PTB+EPTB"), ("paediatric", "Age < 18"),
               ("atb_not_culture_confirmed", "ATB not culture-confirmed")],
              lambda v: f"{v['ATB']} ATB vs {v['OD']} OD (incl. 16 LTBI)", TCOL["ATB vs OD"])
    v2 = flow(F2, 9.65, "GSE152218 (India, RNA-seq) → ATB vs LTB", [("paediatric", "Age < 18")],
              lambda v: f"{v['ATB']} ATB vs {v['LTB']} LTB", TCOL["ATB vs LTB"])
    ax.text(9.95, 2.85, "No HIV or treatment field. All ATB have BMI < 16;\nsex ATB 14 M / 2 F vs LTB 11 M / 21 F.", fontsize=8, va="top", color=INK2, linespacing=1.2)
    ax.text(0.6, 1.75, "No treatment field; only GEO-normalised values available.", fontsize=8, va="center", color=INK2)
    lines = "\n".join(f"  {k}: {v}" for k, v in ec.items())
    save(fig, "Fig1", f"""
Fig 1. Study design, preprocessing and cohort flow.
(A) Discovery cohorts: samples per GEO series, samples excluded, and the samples used in at least one task, by class. {n_all:,} samples,
{len(exc)} excluded, {n_inc:,} retained. Exclusions by reason (one reason per sample):
{lines}
Totals count unique samples: GSE19439 and GSE19444 are sub-series of GSE19491, so their 38 LTB samples appear in two rows but once in the total.
(B) Preprocessing. Each cohort was processed independently; no samples were mixed across cohorts and no class labels were used at any step.
(C) The three tasks and rotating leave-one-cohort-out validation. Each row is one fold; the coloured cohort is held out and all gene selection,
model fitting and the choice of panel size use the grey training cohorts only.
(D) Independent validation cohorts, with filters applied in the order shown: GSE144127 {v1['ATB']} culture-confirmed pulmonary ATB vs {v1['OD']} other
diseases (the 16 patients with latent infection and a second diagnosis counted as OD), and GSE152218 {v2['ATB']} ATB vs {v2['LTB']} LTB.
Sources: Stage3/data/sample_manifest_stage3.csv; Stage7/data/labels.json; Stage7/results/MASTER_results_table.csv;
Stage12/tables/GSE144127_filter_decisions.csv and GSE152218_filter_decisions.csv.""")

# ================================================================ Fig 2: feature selection + derivation
def fig2():
    MC = pd.read_csv(S11 / "results/median_curves_DirAvg.csv")
    fig = fig_new(19.05, 15.6); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 19.05); ax.set_ylim(0, 15.6); ax.axis("off")
    def box(x, y, w, h, fc="#f4f4f4", ec_="#999999", lw=0.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec_, lw=lw))
    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK2))
    ax.text(0.3, 15.4, "A", fontsize=10, fontweight="bold", va="top")
    ax.text(0.85, 15.35, "Nat(mRMR): relevance and redundancy, then a wrapper", fontsize=9, fontweight="bold", va="top", color=PCOL["Native"])
    a = [("Per cohort", "mutual information\nwith class label;\ntop 2,000 genes"), ("Per cohort", "mRMR to 20 genes\nrelevance = MI\nredundancy = mean |r|"),
         ("Candidate set", "union of the training\ncohorts' lists;\nranks summed"), ("Forward selection", "add the gene with best\nmean cross-cohort AUC\nuntil 20 genes"),
         ("Swap refinement", "exchange panel genes\nwhile mean cross-cohort\nAUC improves")]
    sw = 3.45; ytop = 14.7
    for i, (t1, t2) in enumerate(a):
        x = 0.3 + i * (sw + 0.33); box(x, ytop - 2.0, sw, 1.95, fc="#eaf2fa", ec_=PCOL["Native"])
        ax.text(x + sw / 2, ytop - 0.4, t1, fontsize=8, fontweight="bold", ha="center", va="center")
        ax.text(x + sw / 2, ytop - 1.2, t2, fontsize=8, ha="center", va="center", linespacing=1.2)
        if i: arrow(x - 0.31, ytop - 1.0, x - 0.02, ytop - 1.0)
    ax.text(0.3, 12.2, "B", fontsize=10, fontweight="bold", va="top")
    ax.text(0.85, 12.15, "Nat: direction consistency, size chosen by an inner loop", fontsize=9, fontweight="bold", va="top", color=PCOL["Nat"])
    b = [("Per gene", "AUC for the positive\nclass in each training\ncohort"), ("Keep", "genes deviating from\n0.5 in the same\ndirection in all cohorts"),
         ("Rank", "by the smallest\ndeviation across\ntraining cohorts"), ("Choose k", "inner leave-one-cohort-out;\nsmallest k within 0.01\nof the best"),
         ("Panel", "top k genes\n(k ≤ 20, the PCR\nassay ceiling)")]
    ytop2 = 11.5
    for i, (t1, t2) in enumerate(b):
        x = 0.3 + i * (sw + 0.33); box(x, ytop2 - 2.0, sw, 1.95, fc="#fdf0e6", ec_=PCOL["Nat"])
        ax.text(x + sw / 2, ytop2 - 0.4, t1, fontsize=8, fontweight="bold", ha="center", va="center")
        ax.text(x + sw / 2, ytop2 - 1.2, t2, fontsize=8, ha="center", va="center", linespacing=1.2)
        if i: arrow(x - 0.31, ytop2 - 1.0, x - 0.02, ytop2 - 1.0)
    ax.text(0.3, 9.0, "Both panels are rebuilt inside every fold, from that fold's training cohorts only. Final signatures use all cohorts of the task.",
            fontsize=8, va="top", color=INK2)
    letter(fig, 0.005, 0.545, "C")
    fig.text(0.045, 0.54, "Held-out AUC against panel size (directional average; median over folds, shaded band = fold range)", fontsize=8, va="top")
    L, W, GX, HH = 0.075, 0.26, 0.055, 0.42
    for i, task in enumerate(TASKS):
        axp = fig.add_axes([L + i * (W + GX), 0.085, W, HH])
        a_ = AB[(AB.Task == task) & (AB.method == "DirAvg")]
        piv = a_.pivot_table(index="k", columns="heldout", values="auc")
        axp.fill_between(piv.index, piv.min(1), piv.max(1), color=TCOL[task], alpha=0.15, lw=0)
        axp.plot(piv.index, piv.median(1), color=TCOL[task], lw=1.2, marker="o", ms=2)
        ck = CK[(CK.Task == task) & (CK.Method == "DirAvg")]
        kk = FS[(task, "consistent")]["choice"]["DirAvg"]
        if task in NAT:
            axp.axvline(kk, color=INK, lw=0.6, ls=":")
            axp.text(kk + 0.5, 0.33, f"final k = {kk}", fontsize=8, va="bottom", ha="left")
        else:
            axp.text(10.5, 0.33, "no signature supported", fontsize=8, va="bottom", ha="center")
        axp.axhline(0.5, color=GREYD, lw=0.5, ls=":")
        axp.set_xlim(0.5, 20.5); axp.set_ylim(0.3, 1.02); axp.set_xticks([1, 5, 10, 15, 20])
        axp.set_xlabel("Genes in panel (k)"); axp.set_title(task, loc="left", pad=3, color=TCOL[task])
        if i == 0: axp.set_ylabel("Held-out AUC")
        else: axp.set_yticklabels([])
    sig = "; ".join(f"{t}: {', '.join(g)} (k = {len(g)})" for t, g in NAT.items())
    save(fig, "Fig2", f"""
Fig 2. Derivation of the two internally generated panels.
(A) Nat(mRMR), a conventional filter-and-wrapper pipeline: mutual-information filtering and a minimum-redundancy-maximum-relevance criterion
within each training cohort, then forward selection and swap refinement on mean cross-cohort AUC. Relevance is mutual information
(approximately 0.1-0.6 in these data) and redundancy the mean absolute correlation with the genes already selected (0-1).
(B) Nat, selected by direction consistency: genes whose AUC for the positive class deviates from 0.5 in the same direction in every training
cohort, ranked by their smallest deviation, truncated at a size k chosen by an inner leave-one-cohort-out loop over the training cohorts
(smallest k within 0.01 of the best mean inner AUC, chosen separately for each scoring method).
(C) Held-out AUC of Nat against panel size under the directional average: median across folds with the fold range shaded, and the final
all-cohort k marked. Final signatures: {sig}; no signature was supported for LTB vs CON.
Sources: Stage5/scripts/run_stage5.py and Stage11/scripts/run_stage11.py (procedures); Stage11/results/ablation_curves_long.csv,
chosen_k_per_fold.csv and final_signatures.jsonl (panel C).""")

# ================================================================ Fig 3: ROC by held-out cohort
def fig3():
    fig = fig_new(19.05, 14.6); L, W, GX, GY = 0.075, 0.163, 0.027, 0.075
    rows_top = [0.945, 0.635, 0.325]; cap_rows = []
    for r, (task, folds) in enumerate(TASKS.items()):
        hh = W * 19.05 / 14.6
        for j, h in enumerate(folds):
            axp = fig.add_axes([L + j * (W + GX), rows_top[r] - hh, W, hh])
            axp.plot([0, 1], [0, 1], color=GREYD, lw=0.5, ls=":")
            txt = []
            for p in PANELS[task]:
                s, k = scores_for(task, h, p); y, _ = labels_for(task, h)
                fx, tx = roc(y, s); b = B[(B.task == task) & (B.fold == h) & (B.m == "DirAvg") & (B.panel == p)].iloc[0]
                axp.plot(fx, tx, color=PCOL[p], lw=1.5 if p == "Nat" else 0.9,
                         ls="--" if b.flag != "clean" else "-", zorder=5 if p == "Nat" else 3)
                txt.append(f"{PLAB[p]} {b.auc:.3f}{'*' if b.beats and b.flag == 'clean' else ''}{' (d)' if b.flag != 'clean' else ''}")
            axp.set_xlim(-0.02, 1.02); axp.set_ylim(-0.02, 1.02); axp.set_xticks([0, 0.5, 1]); axp.set_yticks([0, 0.5, 1])
            axp.set_aspect("equal"); axp.set_title(h, pad=2, fontweight="normal")
            if j == 0: axp.set_ylabel("Sensitivity")
            else: axp.set_yticklabels([])
            if r == 2: axp.set_xlabel("1 − specificity")
            cap_rows.append(f"  {task}, {h}: " + "; ".join(txt))
        fig.text(0.005, rows_top[r] + 0.04, "ABC"[r], fontsize=10, fontweight="bold", va="top")
        fig.text(L, rows_top[r] + 0.036, task, fontsize=8, fontweight="bold", color=TCOL[task], va="top")
    hs = [Line2D([], [], color=PCOL[p], lw=1.5 if p == "Nat" else 0.9, label=PLAB[p]) for p in PANELS["ATB vs LTB"]]
    hs += [Line2D([], [], color=INK2, lw=0.9, ls="--", label="Derivation overlap")]
    lax = fig.add_axes([L + 4 * (W + GX), rows_top[1] - W * 19.05 / 14.6, W, W * 19.05 / 14.6]); lax.axis("off")
    lax.legend(handles=hs, loc="center left", handlelength=1.4, labelspacing=0.9)
    nb = B[(B.m == "DirAvg") & (B.flag == "clean")].groupby(["task", "panel"]).beats.agg(["sum", "size"])
    save(fig, "Fig3", f"""
Fig 3. Receiver operating characteristic curves in every held-out cohort (directional average).
Each panel is one fold: the cohort named above it was withheld in full, panels were derived and models trained on the remaining cohorts of
that task, and the withheld cohort was scored once. Curves are not pooled across folds. Nat is drawn heavier. Dashed curves mark cells in
which the panel's own derivation included the held-out cohort or one of its sub-series, or, for the GSE19439 fold, in which the training
cohorts included the sister sub-series GSE19444; these are optimistic and are not counted in the summaries below.
(A) ATB vs LTB, (B) ATB vs OD, (C) LTB vs CON. Kaforou is task-specific and is not evaluated for LTB vs CON.
Area under the curve by fold ("*" = at most 5% of 50 size-matched random gene panels reached that value; "(d)" = derivation overlap):
{chr(10).join(cap_rows)}
Clean cells beating size-matched random panels, by panel: {"; ".join(f"{t} {PLAB[p]} {int(r['sum'])}/{int(r['size'])}" for (t, p), r in nb.iterrows())}.
Berry86 is used as published and contains the X-linked gene TLR7, which our own selection would have excluded; removing it changed held-out
AUC by at most 0.024 under logistic regression, singscore and the directional average.
Sources: predictions from Stage7/results/bench/<task>__records.jsonl (published panels and Nat(mRMR)) and
Stage11/results/fold_curves.jsonl (Nat at its chosen k); labels Stage7/data/labels.json; AUCs and flags
Stage7/results/MASTER_results_table.csv and Stage11/results/chosen_k_per_fold.csv.""")

# ================================================================ Fig 4: panel size
def fig4():
    t = "ATB vs LTB"; b = B[(B.task == t)].copy(); b["adv"] = b.auc - b.dist.map(np.median); b["rmed"] = b.dist.map(np.median)
    rng = np.random.default_rng(1); b["xj"] = b.k * np.exp(rng.uniform(-0.06, 0.06, len(b)))
    fig = fig_new(19.05, 9.0); axs = [fig.add_axes([0.07, 0.25, 0.41, 0.66]), fig.add_axes([0.575, 0.25, 0.41, 0.66])]
    for p in PANELS[t]:
        q = b[b.panel == p]
        axs[0].scatter(q.xj, q.auc, s=7, color=PCOL[p], lw=0, alpha=0.85, zorder=3)
        axs[1].scatter(q.xj, q.adv, s=7, color=PCOL[p], lw=0, alpha=0.85, zorder=3)
    axs[0].scatter(b.xj, b.rmed, s=10, marker="_", color=INK2, lw=0.8, zorder=2)
    med = b.groupby("panel").agg(k=("k", "median"), adv=("adv", "median"),
                                 lo=("adv", lambda v: np.percentile(v, 10)), hi=("adv", lambda v: np.percentile(v, 90)))
    for p, r in med.iterrows(): axs[1].plot([r.k / 1.12, r.k * 1.12], [r.adv, r.adv], color=INK, lw=1.6, zorder=4)
    for axx in axs:
        axx.set_xscale("log"); axx.set_xticks([1, 3, 5, 10, 16, 20, 50, 74])
        axx.set_xticklabels(["1", "3", "5", "10", "16", "20", "50", "74"]); axx.minorticks_off()
        axx.set_xlim(0.8, 100); axx.set_xlabel("Genes in panel (log scale)")
    axs[0].set_ylabel("Held-out AUC"); axs[0].set_ylim(0.4, 1.02)
    axs[1].set_ylabel("AUC − random-panel median"); axs[1].axhline(0, color=GREYD, lw=0.6, ls=":")
    axs[0].set_title("Panel AUC (colour) and random-panel median (grey)", loc="left", pad=3)
    axs[1].set_title("Advantage over random (black = median per panel)", loc="left", pad=3)
    letter(fig, 0.005, 0.985, "A"); letter(fig, 0.51, 0.985, "B")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=PCOL[p], ms=4, label=PLAB[p]) for p in PANELS[t]] +
               [Line2D([], [], marker="_", ls="", color=INK2, ms=6, label="Random median")],
               loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.0))
    srt = med.sort_values("k")
    save(fig, "Fig4", f"""
Fig 4. Panel size and the advantage over random gene sets, ATB vs LTB.
All {len(b)} cells (5 folds x {len(PANELS[t])} panels x 5 scoring methods). (A) Held-out AUC of each panel (colour) and the median AUC of the 50
size-matched random panels drawn for that fold and size (grey), against the number of genes used (log scale, small horizontal jitter).
(B) Panel AUC minus the random-panel median; the black bar is the median per panel, drawn at its median size. Nat is plotted at the size
chosen in each fold and for each method (range {b[b.panel == 'Nat'].k.min()}-{b[b.panel == 'Nat'].k.max()} genes).
Median advantage over size-matched random panels (10th-90th percentile):
{chr(10).join(f"  {PLAB[p]} ({r.k:g} genes): {r.adv:+.3f} ({r.lo:+.3f} to {r.hi:+.3f})" for p, r in srt.iterrows())}
Random panels improve steeply with size in this task, so the advantage of a curated panel falls as the panel grows.
Sources: Stage7/results/MASTER_results_table.csv; Stage8/data/random_auc_long.csv; Stage11/results/chosen_k_per_fold.csv
and random_by_size.jsonl (Nat).""")

# ================================================================ Fig 5: selector failure
def fig5():
    PW = pd.read_csv(S8 / "figures/Fig4_pairwise_overlap_data.csv"); C = pd.read_csv(S7 / "results/FINAL_Table_consensus_failure.csv")
    G = pd.read_csv(S7 / "results/FINAL_Table_native_stability_genes.csv")
    G["nf"] = G["Folds selected"].str.split("/").str[0].astype(int)
    fig = fig_new(19.05, 21.5)
    cm_, cd, cp = "Same, random-ranking median [95% range] (all)", "Rank depth until a gene is in every cohort's list", "P (random depth ≤ observed), all lists"
    for k, (task, coh) in enumerate(TASKS.items()):
        n = len(coh); sz = 0.040 * (n - 1)
        axp = fig.add_axes([0.085 + k * 0.315, 0.925 - sz * 19.05 / 21.5, sz, sz * 19.05 / 21.5])
        X = np.full((n - 1, n - 1), np.nan)
        for r in PW[PW.Task == task].itertuples(): i, j = coh.index(r.a), coh.index(r.b); X[j - 1, i] = r.overlap
        axp.imshow(np.ma.masked_invalid(X), cmap="Blues", vmin=0, vmax=10, aspect="auto")
        for i in range(n - 1):
            for j in range(n - 1):
                if not np.isnan(X[i, j]): axp.text(j, i, int(X[i, j]), ha="center", va="center", fontsize=8, color=INK)
        axp.set_xticks(range(n - 1)); axp.set_yticks(range(n - 1))
        axp.set_xticklabels([c.replace("GSE", "") for c in coh[:-1]], rotation=90)
        axp.set_yticklabels([c.replace("GSE", "") for c in coh[1:]]); axp.tick_params(length=0)
        for sp in axp.spines.values(): sp.set_visible(False)
        axp.set_title(task, pad=3, color=TCOL[task])
    letter(fig, 0.005, 0.99, "A")
    fig.text(0.085, 0.985, "Genes shared between the per-cohort 20-gene mRMR shortlists (cohort IDs without the GSE prefix)", fontsize=8, va="top")
    letter(fig, 0.005, 0.770, "B")
    axp = fig.add_axes([0.30, 0.545, 0.52, 0.195]); y = 0; yt, yl = [], []
    for task in TASKS:
        for _, r in C[C.Task == task].iterrows():
            md, lo, hi = map(float, re.match(r"(\d+) \[(\d+)–(\d+)\]", r[cm_]).groups())
            obs = r[cd]; pv = r[cp]
            axp.plot([lo, hi], [y, y], color=GREY, lw=4.5, solid_capstyle="butt", zorder=1)
            axp.plot(md, y, "|", color=INK2, ms=6, mew=1.1, zorder=2)
            axp.plot(obs, y, "o", color=TCOL[task], ms=4, mec="white", mew=0.5, zorder=3)
            axp.text(1.02, y, f"{obs}  (p {'< 0.001' if pv == 0 else '= ' + format(pv, '.3f')})",
                     transform=axp.get_yaxis_transform(), va="center", fontsize=8)
            yt.append(y); yl.append(f"{task} | {r['Held-out cohort']}"); y -= 1
        y -= 0.6
    axp.set_xscale("log"); axp.set_xlim(8, 3000); axp.set_xticks([10, 30, 100, 300, 1000, 3000])
    axp.set_xticklabels(["10", "30", "100", "300", "1,000", "3,000"]); axp.minorticks_off()
    axp.set_yticks(yt); axp.set_yticklabels(yl); axp.set_ylim(y + 0.4, 0.8)
    axp.set_xlabel("Rank depth until one gene is in every training cohort's MI ranking (log scale)")
    axp.set_title("Observed depth (dot) vs 1,000 independent random rankings (bar = 95% range, tick = median)", loc="left", pad=3)
    letter(fig, 0.005, 0.500, "C")
    fig.text(0.045, 0.495, "Genes selected in two or more folds of the Nat(mRMR) panel (full matrix in S7 Fig)", fontsize=8, va="top")
    FC = {"olfactory receptor": "#6a3d9a", "microRNA": "#D55E00", "pseudogene": "#b8860b"}
    SY = {"olfactory receptor": " (OR)", "microRNA": " (miR)", "pseudogene": " (ψ)"}
    sub = {t: G[(G.Task == t) & (G.nf >= 2)].reset_index(drop=True) for t in TASKS}
    nmax = max(len(v) for v in sub.values()); top, bot = 0.375, 0.035; rh = (top - bot) / nmax
    for k, (task, coh) in enumerate(TASKS.items()):
        g = sub[task]; n = len(g); cw = 0.030; x0 = [0.135, 0.465, 0.775][k]
        axp = fig.add_axes([x0, top - n * rh, cw * len(coh), n * rh])
        axp.set_xlim(0, len(coh)); axp.set_ylim(0, n); axp.axis("off")
        for i, r in g.iterrows():
            for j, c in enumerate(coh):
                axp.add_patch(Rectangle((j + 0.08, n - 1 - i + 0.1), 0.84, 0.8,
                                        color="#0072B2" if r[f"held-out {c}"] == "●" else "#efefef", lw=0))
            fl = r.Flag if isinstance(r.Flag, str) else ""
            axp.text(-0.2, n - 1 - i + 0.5, r.Gene + SY.get(fl, ""), ha="right", va="center", fontsize=8,
                     color=FC.get(fl, INK), fontweight="bold" if fl else "normal")
            axp.text(len(coh) + 0.2, n - 1 - i + 0.5, r["Folds selected"], ha="left", va="center", fontsize=8, color=INK2)
        for j, c in enumerate(coh): axp.text(j + 0.5, n + 0.3, c.replace("GSE", ""), rotation=90, ha="center", va="bottom", fontsize=8)
        fig.text(x0 - 0.045, 0.455, task, fontsize=8, fontweight="bold", color=TCOL[task], va="top")
    fig.legend(handles=[Patch(color="#0072B2", label="Selected in that fold")] + [Patch(color=v, label=f"{kk}{SY[kk]}") for kk, v in FC.items()],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    s = PW.groupby("Task").overlap.agg(["min", "max", lambda v: int((v > 0).sum()), "size"])
    S_ = pd.read_csv(S7 / "results/FINAL_Table_native_stability_summary.csv")
    POV = [c for c in S_.columns if c.startswith("Pairwise fold overlap, genes")][0]
    rmed = {t: re.match(r"(\d+)", C[C.Task == t][cm_].iloc[0]).group(1) for t in TASKS}
    dep = "; ".join("{}: {}-{} vs random median {} (p {:.3f}-{:.3f})".format(
        t, C[C.Task == t][cd].min(), C[C.Task == t][cd].max(), rmed[t], C[C.Task == t][cp].min(), C[C.Task == t][cp].max()) for t in TASKS)
    save(fig, "Fig5", f"""
Fig 5. The mRMR-derived panels do not agree between cohorts and are unstable between folds.
(A) Genes shared between the per-cohort 20-gene shortlists (mutual-information top 2,000 followed by mRMR) for every pair of cohorts in each
task; shortlists are computed per cohort and do not depend on the fold. Pairs sharing any gene:
{"; ".join(f"{t} {int(r.iloc[2])}/{int(r.iloc[3])} (range {int(r['min'])}-{int(r['max'])})" for t, r in s.iterrows())}.
(B) For each fold, the rank depth at which one gene first appears in every training cohort's mutual-information ranking (dot) against 1,000
independent random rankings of the same gene universe (grey bar = 95% range, tick = median); p is the share of random rankings at or below
the observed depth. {dep}.
(C) Genes selected in two or more folds of the Nat(mRMR) panel, by held-out cohort; the number at right is the count of folds selecting that
gene, and coloured names are HGNC-flagged entries. No gene was selected in every fold of any task; across folds the three tasks drew on
{"; ".join(f"{r.Task} {r['Distinct genes (all folds)']}" for _, r in S_.iterrows())} distinct genes, with pairwise fold overlap of
{"; ".join(f"{r.Task} {r[POV]}" for _, r in S_.iterrows())} genes. The full matrices are in S7 Fig.
Sources: Stage7/results/FINAL_Table_consensus_failure.csv, FINAL_Table_native_stability_genes.csv and
FINAL_Table_native_stability_summary.csv; pair counts Stage8/figures/Fig4_pairwise_overlap_data.csv.""")

# ================================================================ Fig 6: model comparison
def fig6():
    G = pd.read_csv(S7 / "results/FINAL_generalisation_gap.csv"); G["m"] = G.Method.map(MNAME); G = G[G.m.isin(METHODS)]
    gp = "Gap (pooled training minus held-out)"
    fig = fig_new(19.05, 13.6)
    axp = fig.add_axes([0.085, 0.69, 0.885, 0.255])
    xs = []
    for i, task in enumerate(TASKS):
        for j, m in enumerate(METHODS):
            x = i * 6 + j; q = B[(B.task == task) & (B.m == m)]
            axp.plot([x, x], [q.auc.quantile(0.1), q.auc.quantile(0.9)], color=MCOL[m], lw=1.0, zorder=2)
            axp.plot(x, q.auc.median(), "o", color=MCOL[m], ms=5, zorder=3)
            xs.append((x, m))
        axp.text(i * 6 + 2, 0.27, task, ha="center", va="bottom", fontsize=8, fontweight="bold", color=TCOL[task])
    axp.axhline(0.5, color=GREYD, lw=0.6, ls=":")
    SHORT = {"LR": "LR", "RF": "RF", "MLP": "MLP", "singscore": "singscore", "DirAvg": "DirAvg"}
    axp.set_xticks([x for x, _ in xs]); axp.set_xticklabels([SHORT[m] for _, m in xs], rotation=45, ha="right")
    axp.set_xlim(-1, 16); axp.set_ylim(0.25, 1.02); axp.set_ylabel("Held-out AUC")
    axp.set_title("Median held-out AUC by scoring method (bar = 10th-90th percentile over folds and panels)", loc="left", pad=3)
    letter(fig, 0.005, 0.985, "A"); letter(fig, 0.005, 0.505, "B")
    for k, task in enumerate(TASKS):
        axq = fig.add_axes([0.085 + k * 0.315, 0.085, 0.235, 0.235 * 19.05 / 13.6])
        g = G[G.Task == task]
        for m in METHODS:
            a = g[g.m == m]
            axq.scatter(a["Training AUC, pooled resubstitution"], a["Held-out AUC"], s=6, color=MCOL[m], lw=0, alpha=0.85, zorder=3)
        axq.plot([0.3, 1.0], [0.3, 1.0], color=GREYD, lw=0.6, ls="--")
        axq.set_xlim(0.3, 1.01); axq.set_ylim(0.3, 1.01); axq.set_aspect("equal")
        axq.set_title(task, pad=3, color=TCOL[task]); axq.set_xlabel("Training AUC (pooled)")
        if k == 0: axq.set_ylabel("Held-out AUC")
    fig.text(0.045, 0.50, "Training against held-out AUC; distance below the dashed line is the generalisation gap", fontsize=8, va="top")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=MCOL[m], ms=4, label=MLAB[m]) for m in METHODS],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    med = B.groupby(["task", "m"]).auc.median().unstack()
    gapmed = G.groupby("m")[gp].median()
    rng_ = {t: f"{med.loc[t].min():.3f}-{med.loc[t].max():.3f}" for t in TASKS}
    save(fig, "Fig6", f"""
Fig 6. Scoring method matters less than task, and fitting buys training fit rather than transfer.
(A) Median held-out AUC for each scoring method within each task, over all folds and panels, with the 10th-90th percentile range.
Medians span {rng_['ATB vs LTB']} in ATB vs LTB, {rng_['ATB vs OD']} in ATB vs OD and {rng_['LTB vs CON']} in LTB vs CON, so the
difference between methods within a task (at most {max(med.max(1) - med.min(1)):.2f}) is small beside the difference between tasks
(about {med.max(1).max() - med.min(1).min():.2f}).
(B) Pooled training-cohort resubstitution AUC against held-out AUC for every fold, panel and method; the distance below the dashed diagonal
is the generalisation gap. Median gaps across all tasks: {", ".join(f"{MLAB[m].split(' (')[0]} {gapmed[m]:+.3f}" for m in METHODS)}.
The two scores that fit no weights sit closest to the diagonal; random forest is furthest below it, reaching a median gap of
{G[(G.Task == 'LTB vs CON') & (G.m == 'RF')][gp].median():+.3f} in LTB vs CON.
Sources: Stage7/results/MASTER_results_table.csv and Stage11/results/chosen_k_per_fold.csv (panel A);
Stage7/results/FINAL_generalisation_gap.csv (panel B).""")

# ================================================================ Fig 7: external validation
def fig7():
    S = SX[SX.method == "DirAvg"]; E = EX[EX.method == "DirAvg"]; D = DL[DL.method == "DirAvg"]
    spec = {"GSE144127": ("ATB vs OD", ["S11_consistent", "Zak16", "Kaforou_ATB_v_OD", "Berry86", "Sweeney3"]),
            "GSE152218": ("ATB vs LTB", ["S11_consistent", "Zak16", "Kaforou_ATB_v_LTB", "Berry86", "Sweeney3"])}
    fig = fig_new(19.05, 9.2)
    for k, (G, (task, P)) in enumerate(spec.items()):
        axp = fig.add_axes([0.075 + k * 0.5, 0.115, 0.38, 0.38 * 19.05 / 9.2])
        axp.plot([0, 1], [0, 1], color=GREYD, lw=0.6, ls=":")
        for p in P[::-1]:
            q = S[(S.cohort == G) & (S.panel == p)]; e = E[(E.cohort == G) & (E.panel == p)].iloc[0]
            nm = "Nat" if p == "S11_consistent" else p
            lab = f"{PLAB[nm]}{' (' + str(len(NAT[task])) + ' genes)' if nm == 'Nat' else ''}  {e.AUC:.3f}"
            fx, tx = roc(q.y.to_numpy(), q.score.to_numpy())
            axp.plot(fx, tx, color=PCOL[nm], lw=1.8 if nm == "Nat" else 0.9, zorder=5 if nm == "Nat" else 3, label=lab)
        h_, l_ = axp.get_legend_handles_labels(); axp.legend(h_[::-1], l_[::-1], loc="lower right", title="DirAvg AUC", handlelength=1.6)
        axp.set_xlim(-0.01, 1.01); axp.set_ylim(-0.01, 1.01); axp.set_aspect("equal")
        axp.set_xlabel("1 − specificity"); axp.set_ylabel("Sensitivity")
        e = E[(E.cohort == G) & (E.panel == "S11_consistent")].iloc[0]; neg = task.split(" vs ")[1]
        axp.set_title(f"{G}, {task} ({e.n_pos} ATB vs {e.n_neg} {neg})", loc="left", pad=3)
        letter(fig, 0.005 + k * 0.5, 0.985, "AB"[k])
    def dl(G):
        d = D[D.cohort == G]
        return "; ".join(f"vs {PLAB.get(r.comparator, r.comparator)} {r.AUC_diff:+.3f} ({r.diff_CI_lo:+.3f} to {r.diff_CI_hi:+.3f}), "
                         f"p = {r.p:.3f}, Holm p = {r.p_Holm:.3f}" for r in d.itertuples())
    ci = lambda G: E[(E.cohort == G) & (E.panel == "S11_consistent")].iloc[0]
    z = D[(D.cohort == "GSE144127") & (D.comparator == "Zak16")].iloc[0]
    save(fig, "Fig7", f"""
Fig 7. External validation of the Nat signatures in two independent cohorts.
Locked models: each was trained on all cohorts of its task and scored once on the validation cohort; nothing was selected, tuned or
thresholded using validation data. Curves are the directional average. Nat is {', '.join(NAT['ATB vs OD'])} in (A) and
{', '.join(NAT['ATB vs LTB'])} in (B); areas under the curve are given in the legends.
(A) GSE144127, {ci('GSE144127').n_pos} culture-confirmed pulmonary ATB vs {ci('GSE144127').n_neg} other diseases: Nat AUC {ci('GSE144127').AUC:.3f}
(95% CI {ci('GSE144127').CI_lo:.3f}-{ci('GSE144127').CI_hi:.3f}).
(B) GSE152218, {ci('GSE152218').n_pos} ATB vs {ci('GSE152218').n_neg} LTB: Nat AUC {ci('GSE152218').AUC:.3f}
({ci('GSE152218').CI_lo:.3f}-{ci('GSE152218').CI_hi:.3f}).
Paired DeLong tests of Nat against each published panel on the same samples (Holm-adjusted within cohort and scoring method):
  GSE144127: {dl('GSE144127')}.
  GSE152218: {dl('GSE152218')}.
No difference is significant after adjustment, so comparability rather than superiority is claimed; against Zak16, a 16-gene signature, the
difference is {z.AUC_diff:+.3f} (95% CI {z.diff_CI_lo:+.3f} to {z.diff_CI_hi:+.3f}) using five genes. In GSE152218 random three-gene panels
reach a median AUC of 0.83 and every panel scores above 0.92, so that cohort shows replication rather than specificity.
Sources: Stage13/results/external_scores_long.csv.gz, external_auc.csv, vs_random.csv; Stage16/results/delong_paired.csv
(recomputed with Berry86 as published, six comparators).""")

# ================================================================ Fig 8: PCA
def fig8():
    P = pd.read_csv(S15 / "results/pca_scores.csv", index_col=0)
    V = pd.read_csv(S15 / "results/pca_variance_and_structure.csv")
    CC = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    KC = {"ATB": "#D55E00", "LTB": "#0072B2", "CON": "#009E73", "OD": "#CC79A7"}
    fig = fig_new(19.05, 20.0); W, H, L, TOP, GX, GY = 0.37, 0.235, 0.085, 0.955, 0.10, 0.085
    for i, (task, coh) in enumerate(TASKS.items()):
        q = P[P.task == TKEY(task)]
        vv = {(r.view, r.grouping): r for r in V[V.task == task].itertuples()}
        for j, (xa, ya, by) in enumerate([("PC1_log2", "PC2_log2", "cohort"), ("PC1_z", "PC2_z", "cls")]):
            axp = fig.add_axes([L + j * (W + GX), TOP - (i + 1) * H - i * GY, W, H])
            vrow = vv[("log2 (pre-normalisation)" if j == 0 else "per-cohort z-score", "cohort")]
            if by == "cohort":
                for k, c in enumerate(coh):
                    m = (q.cohort == c).to_numpy(); axp.scatter(q[xa][m], q[ya][m], s=5, color=CC[k], lw=0, alpha=0.75, label=c)
            else:
                for c in task.split(" vs "):
                    m = (q.cls == c).to_numpy(); axp.scatter(q[xa][m], q[ya][m], s=5, color=KC[c], lw=0, alpha=0.75, label=c)
            axp.set_xlabel(f"PC1 ({vrow.PC1_pct:.1f}%)"); axp.set_ylabel(f"PC2 ({vrow.PC2_pct:.1f}%)")
            axp.set_title("log2 expression, by cohort" if j == 0 else "after per-cohort z-scoring, by class", loc="left", pad=3)
            axp.legend(loc="best", ncol=2 if by == "cohort" else 1, markerscale=1.6, handletextpad=0.2)
        fig.text(0.005, TOP - i * (H + GY) + 0.035, "ABC"[i], fontsize=10, fontweight="bold", va="top")
        fig.text(L, TOP - i * (H + GY) + 0.032, task, fontsize=8, fontweight="bold", color=TCOL[task], va="top")
    def gg(task, view, grouping): return V[(V.task == task) & (V.view == view) & (V.grouping == grouping)].iloc[0].R2_PC1_PC2
    lines = "\n".join(f"  {t}: before, cohort R2 = {gg(t, 'log2 (pre-normalisation)', 'cohort'):.2f} and class R2 = "
                      f"{gg(t, 'log2 (pre-normalisation)', 'class'):.2f}; after, cohort R2 = {gg(t, 'per-cohort z-score', 'cohort'):.2f} "
                      f"and class R2 = {gg(t, 'per-cohort z-score', 'class'):.2f}" for t in TASKS)
    save(fig, "Fig8", f"""
Fig 8. Cohort structure before and after per-cohort normalisation.
Principal component analysis of the samples entering each task, computed on that task's gene universe. Left column: log2 expression before
normalisation, coloured by cohort. Right column: the same samples after the label-free per-cohort z-scoring used by every model, coloured by
class. Axis labels give the percentage of total variance carried by each component. (A) ATB vs LTB, (B) ATB vs OD, (C) LTB vs CON.
Share of the variance in the first two components attributable to cohort and to class (one-way R2):
{lines}
Cohort identity accounts for almost all of the leading variance before normalisation and almost none after it. Class is not a leading axis of
variance in any task, weakly so in ATB vs LTB; in ATB vs OD the class difference is absent from the first two components even though the
five-gene signature reaches an AUC of 0.855 in an independent cohort, so unsupervised structure does not track discriminability.
Sources: Stage15/results/pca_scores.csv and pca_variance_and_structure.csv.""")

# ================================================================ supporting information
def bench_axis(ax, task, m, flabels=True):
    P = PANELS[task]; folds = TASKS[task]; n = len(P) + 1.2
    ax.axhline(0.5, color=GREYD, lw=0.6, ls=":", zorder=0)
    MK = {"clean": "o", "derivation": "D", "sibling": "s"}
    for j, h in enumerate(folds):
        for i, p in enumerate(P):
            x = j * n + i; r = B[(B.task == task) & (B.fold == h) & (B.m == m) & (B.panel == p)].iloc[0]; col = PCOL[p]
            boxrand(ax, x, r.dist); ax.plot([x, x], [r.lo, r.hi], color=col, lw=1.0, zorder=3)
            ax.plot(x, r.auc, MK[r.flag], ms=3.6, color=col, mfc=col if r.flag == "clean" else "white", mew=0.9, zorder=4)
            if r.beats and r.flag == "clean": ax.text(x, min(r.hi + 0.01, 1.03), "*", ha="center", va="bottom", fontsize=8, zorder=5)
        if flabels: ax.text(j * n + (len(P) - 1) / 2, -0.035, h, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ax.set_xlim(-0.8, len(folds) * n - 1.4); ax.set_ylim(0.15, 1.09); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    for j in range(1, len(folds)): ax.axvline(j * n - 1.1, color="#eeeeee", lw=0.6, zorder=0)
    ax.set_ylabel("Held-out AUC")
def figS_bench():
    for si, task in enumerate(TASKS):
        fig = fig_new(19.05, 21.5); hrow, gap, top = 0.135, 0.048, 0.955
        for i, m in enumerate(METHODS):
            ax = fig.add_axes([0.075, top - (i + 1) * hrow - i * gap, 0.91, hrow]); bench_axis(ax, task, m)
            ax.set_title(MLAB[m], loc="left", pad=2)
        hs = [Line2D([], [], marker="o", color=PCOL[p], mfc=PCOL[p], ms=4, lw=1, label=PLAB[p]) for p in PANELS[task]]
        hs2 = [Patch(color=GREY, label="Random panels, 2.5–97.5%"), Patch(color=GREYD, label="Random panels, IQR"),
               Line2D([], [], marker="D", color=INK2, mfc="white", ms=3.6, ls="", label="Derivation overlap"),
               Line2D([], [], marker="s", color=INK2, mfc="white", ms=3.6, ls="", label="Sibling sub-series"),
               Line2D([], [], marker="$*$", color=INK, ls="", ms=5, label="Clean, ≤5% of random reach it")]
        fig.legend(handles=hs, loc="lower center", ncol=len(hs), bbox_to_anchor=(0.5, 0.030), handlelength=1.2)
        fig.legend(handles=hs2, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), handlelength=1.2)
        b = B[B.task == task]
        cnt = "; ".join(f"{PLAB[p]} {int(b[(b.panel == p) & (b.flag == 'clean')].beats.sum())}/{len(b[(b.panel == p) & (b.flag == 'clean')])} clean"
                        for p in PANELS[task])
        save(fig, f"S{1 + si}_Fig", f"""
S{1 + si} Fig. {task}: every panel against size-matched random panels, all five scoring methods.
Rows are scoring methods; groups along the x-axis are held-out cohorts. Each point is the held-out AUC with its bootstrap 95% confidence
interval (1,000 resamples, seed 42), drawn over the AUCs of the 50 random gene panels of the same size in the same fold (light grey =
2.5th-97.5th percentile, dark grey = interquartile range, white line = median). Filled circles are clean cells; open diamonds mark
derivation overlap with the held-out cohort and open squares the GSE19439 sibling fold. An asterisk marks clean cells where at most 5% of
random panels reached the panel's AUC. Nat is the fold-level panel at the size chosen for that fold and method.
Cells beating random, all methods: {cnt}.
Sources: Stage7/results/MASTER_results_table.csv; Stage8/data/random_auc_long.csv; Stage11/results/chosen_k_per_fold.csv and
random_by_size.jsonl.""")
def curve_axis(ax, task, h, m):
    t = TKEY(task); ks = np.arange(1, 21); D = np.array([RBS[(t, h, k)][m] for k in ks])
    lo, q1, md, q3, hi = (np.percentile(D, q, axis=1) for q in (2.5, 25, 50, 75, 97.5))
    ax.fill_between(ks, lo, hi, color=GREY, lw=0, zorder=1); ax.fill_between(ks, q1, q3, color=GREYD, lw=0, zorder=1)
    ax.plot(ks, md, color="white", lw=0.8, zorder=2)
    a = AB[(AB.Task == task) & (AB.heldout == h) & (AB.method == m)].sort_values("k")
    col = MCOL[m] if m != "DirAvg" else PCOL["Nat"]
    ax.plot(a.k, a.auc, color=col, lw=1.0, marker="o", ms=1.8, zorder=3)
    c = CK[(CK.Task == task) & (CK["Held-out cohort"] == h) & (CK.Method == m)].iloc[0]
    kc = int(c["Chosen k (inner LOCO)"])
    ax.plot(kc, c["AUC at chosen k"], "o", ms=5, mfc="none", mec=INK, mew=1.0, zorder=4)
    ax.axvline(kc, color=INK, lw=0.5, ls=":", zorder=0)
    ax.set_xlim(0.5, 20.5); ax.set_xticks([1, 5, 10, 15, 20]); ax.axhline(0.5, color=GREYD, lw=0.5, ls=":", zorder=0)
    return kc, c
YL = {"ATB vs LTB": (0.3, 1.02), "ATB vs OD": (0.3, 0.95), "LTB vs CON": (0.2, 0.95)}
def figS_curves():
    for si, (task, folds) in enumerate(TASKS.items()):
        nf = len(folds); fig = fig_new(19.05, 22.2); L, W = 0.085, 0.168; gx = (0.985 - L - 5 * W) / 4
        hh = 0.085 if nf == 5 else 0.105; gy = 0.035; top = 0.95
        for i, h in enumerate(folds):
            for j, m in enumerate(METHODS):
                ax = fig.add_axes([L + j * (W + gx), top - (i + 1) * hh - i * gy, W, hh])
                kc, c = curve_axis(ax, task, h, m); ax.set_ylim(*YL[task])
                ax.text(0.97, 0.05, f"k = {kc}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
                if i == 0: ax.set_title(MLAB[m], pad=2)
                if j == 0: ax.set_ylabel(f"{h}{' (sib.)' if c['Sibling flag'] else ''}\nheld-out AUC")
                else: ax.set_yticklabels([])
                if i < nf - 1: ax.set_xticklabels([])
                else: ax.set_xlabel("k")
        letter(fig, 0.005, 0.99, "A"); yb = top - nf * hh - (nf - 1) * gy - 0.1
        rec = FS[(task, "consistent")]; ax = fig.add_axes([L, 0.075, 0.9, yb - 0.075 - 0.02]); letter(fig, 0.005, yb + 0.02, "B")
        for m in METHODS:
            v = np.array(rec["inner"][m]); kk = rec["choice"][m]
            ax.plot(range(1, 21), v, color=MCOL[m], marker="o", ms=2, lw=1.0, label=f"{MLAB[m].split(' (')[0]} (k = {kk})")
            ax.plot(kk, v[kk - 1], "o", ms=5, mfc="none", mec=MCOL[m], mew=1.0)
        ax.set_xticks(range(1, 21)); ax.set_xticklabels(rec["genes"], rotation=90); ax.set_xlim(0.4, 20.6)
        ax.set_ylabel("Inner-LOCO AUC\n(training cohorts; not held-out)")
        ax.set_title(f"{task}: final all-cohort derivation, all methods", loc="left", pad=3); ax.legend(loc="lower right", ncol=3)
        cks = CK[CK.Task == task]
        note = (" In ATB vs OD the untrained scores need few genes (final k = 5 for both the directional average and singscore) while the "
                f"trained models prefer long panels (final k: logistic regression {rec['choice']['LR']}, random forest {rec['choice']['RF']}, "
                f"MLP {rec['choice']['MLP']}), so the plateau depends on the scoring method.") if task == "ATB vs OD" else ""
        save(fig, f"S{4 + si}_Fig", f"""
S{4 + si} Fig. {task}: derivation of Nat under all five scoring methods.
(A) Rows are held-out cohorts and columns scoring methods. The coloured line is the held-out AUC of the top k genes of that fold's
training-only consistency ranking, over the AUCs of 50 random panels of the same size in the same fold (shading as in S1-S3 Figs). The open
circle and dotted line mark the size chosen by the inner leave-one-cohort-out loop, which is chosen separately for each method.
(B) Final all-cohort derivation: mean inner leave-one-cohort-out AUC against the gene added at each step of the nested ranking. This is an
inner-loop estimate over training cohorts, not held-out performance, and is not comparable with panel A.{note}
Chosen k per fold: {' | '.join(f"{MLAB[m].split(' (')[0]}: " + ', '.join(f"{r['Held-out cohort']} {int(r['Chosen k (inner LOCO)'])}" for _, r in cks[cks.Method == m].iterrows()) for m in METHODS)}.
Sources: Stage11/results/ablation_curves_long.csv, chosen_k_per_fold.csv, random_by_size.jsonl and final_signatures.jsonl.""")
def figS_stability():
    G = pd.read_csv(S7 / "results/FINAL_Table_native_stability_genes.csv")
    FC = {"olfactory receptor": "#6a3d9a", "microRNA": "#D55E00", "pseudogene": "#b8860b"}
    SY = {"olfactory receptor": " (OR)", "microRNA": " (miR)", "pseudogene": " (ψ)"}
    fig = fig_new(19.05, 22.2); nmax = G.groupby("Task").size().max(); top, bot = 0.862, 0.055; rh = (top - bot) / nmax
    for k, (task, coh) in enumerate(TASKS.items()):
        g = G[G.Task == task].reset_index(drop=True); n = len(g); cw = 0.034; x0 = [0.115, 0.47, 0.755][k]
        ax = fig.add_axes([x0, top - n * rh, cw * len(coh), n * rh]); ax.set_xlim(0, len(coh)); ax.set_ylim(0, n); ax.axis("off")
        for i, r in g.iterrows():
            for j, c in enumerate(coh):
                ax.add_patch(Rectangle((j + 0.08, n - 1 - i + 0.1), 0.84, 0.8,
                                       color="#0072B2" if r[f"held-out {c}"] == "●" else "#efefef", lw=0))
            fl = r.Flag if isinstance(r.Flag, str) else ""
            ax.text(-0.2, n - 1 - i + 0.5, r.Gene + SY.get(fl, ""), ha="right", va="center", fontsize=8,
                    color=FC.get(fl, INK), fontweight="bold" if fl else "normal")
            ax.text(len(coh) + 0.2, n - 1 - i + 0.5, r["Folds selected"], ha="left", va="center", fontsize=8, color=INK2)
        for j, c in enumerate(coh): ax.text(j + 0.5, n + 0.3, c, rotation=90, ha="center", va="bottom", fontsize=8)
        shared = int((g["Folds selected"].str.split("/").str[0].astype(int) == len(coh)).sum())
        fig.text(x0 - 0.11, 0.99, "ABC"[k], fontsize=10, fontweight="bold", va="top")
        fig.text(x0 - 0.08, 0.988, f"{task}\n{n} genes, {shared} in every fold", fontsize=8, va="top", color=TCOL[task], fontweight="bold")
    fig.legend(handles=[Patch(color="#0072B2", label="Selected in that fold's Nat(mRMR) panel")] +
               [Patch(color=v, label=f"{kk}{SY[kk]}") for kk, v in FC.items()], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    S_ = pd.read_csv(S7 / "results/FINAL_Table_native_stability_summary.csv")
    POV = [c for c in S_.columns if c.startswith("Pairwise fold overlap, genes")][0]
    save(fig, "S7_Fig", "S7 Fig. Every gene selected by the mRMR-derived panel, by fold.\nPresence (blue) of each gene in each fold's 20-gene "
         "Nat(mRMR) panel; columns are held-out cohorts and the number at right is the count of folds selecting that gene. Sex-chromosome genes "
         "are excluded from selection. Coloured names are HGNC-flagged entries. Summary: " +
         "; ".join(f"{r.Task} {r['Distinct genes (all folds)']} distinct genes, {r['Genes in every fold']} in every fold, pairwise fold overlap {r[POV]} genes"
                   for _, r in S_.iterrows()) +
         ".\nSources: Stage7/results/FINAL_Table_native_stability_genes.csv and FINAL_Table_native_stability_summary.csv.")
def figS_shift():
    g = pd.read_csv(S7 / "results/global_shift_heldout.csv"); g["T"] = g.task.str.replace("_vs_", " vs ")
    g["key"] = g["T"] + "|" + g.heldout; order = [f"{t}|{c}" for t, cs in TASKS.items() for c in cs]
    g = g.set_index("key").reindex(order)
    fig = fig_new(16.0, 9.0); ax = fig.add_axes([0.25, 0.17, 0.5, 0.78]); n = len(g)
    for i, (key, r) in enumerate(g.iterrows()):
        y = n - 1 - i; lo, hi = map(float, r.CI.split("-")); q = r.heldout in QN; col = TCOL[r["T"]]
        ax.plot([lo, hi], [y, y], color=col, lw=1.0)
        ax.plot(r.allgene_mean_AUC, y, "D" if q else "o", ms=4, color=col, mfc="white" if q else col, mew=1.0)
        ax.text(1.03, y, f"{r.allgene_mean_AUC:.3f}  ({r.share_genes_AUCdev_gt_0_2:.3f})", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.text(1.03, n - 0.2, "AUC  (share)", transform=ax.get_yaxis_transform(), va="bottom", fontsize=8, fontweight="bold")
    ax.set_yticks(range(n)); ax.set_yticklabels([k.replace("|", " | ") for k in order][::-1])
    ax.axvline(0.5, color=GREYD, lw=0.6, ls=":"); ax.set_xlim(0, 1)
    ax.set_xlabel("AUC of the per-sample all-gene mean (positive class)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, ms=4, label="As deposited or rebuilt"),
                        Line2D([], [], marker="D", ls="", color=INK2, mfc="white", ms=4, label="Quantile-normalised")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    save(fig, "S8_Fig", "S8 Fig. Cohort-level offset diagnostic.\nAUC of the per-sample mean across all measured genes for the positive class, "
         "with a bootstrap 95% confidence interval (1,000 resamples). Values at right give the AUC and, in brackets, the share of genes "
         "individually separating the classes at |AUC - 0.5| > 0.2. A value of 0.5 indicates no cohort-level offset aligned with class. Open "
         "diamonds mark quantile-normalised cohorts (GSE19491 and GSE42834 as deposited, GSE83456 rebuilt in Stage7 with log2 and quantile "
         f"normalisation), where this diagnostic is near-constant by construction. Range {g.allgene_mean_AUC.min():.3f} to {g.allgene_mean_AUC.max():.3f}."
         "\nSource: Stage7/results/global_shift_heldout.csv.")

JOBS = {"Fig1": fig1, "Fig2": fig2, "Fig3": fig3, "Fig4": fig4, "Fig5": fig5, "Fig6": fig6, "Fig7": fig7, "Fig8": fig8,
        "S1": figS_bench, "S4": figS_curves, "S7": figS_stability, "S8": figS_shift}
for k, f in JOBS.items():
    if not A.only or k in A.only: f()
rows = []
for t in sorted(FIG.glob("*.tif")):
    im = Image.open(t); dpi = tuple(float(x) for x in im.info.get("dpi", (0, 0))); comp = im.info.get("compression")
    w, h = im.size[0] / dpi[0] * 2.54, im.size[1] / dpi[1] * 2.54; mb = t.stat().st_size / 1e6
    ok = dict(format=im.format == "TIFF", mode_RGB=im.mode == "RGB", no_alpha="A" not in im.getbands(), LZW=comp == "tiff_lzw",
              dpi_300_600=300 <= round(dpi[0]) <= 600, width_le_19_05=w <= 19.05 + 0.01, width_ge_6_68=w >= 6.68,
              height_le_22_23=h <= 22.23 + 0.01, size_lt_10MB=mb < 10, single_frame=getattr(im, "n_frames", 1) == 1)
    rows.append(dict(file=t.name, px=f"{im.size[0]}x{im.size[1]}", dpi=round(dpi[0]), width_cm=round(w, 2), height_cm=round(h, 2),
                     MB=round(mb, 2), font=FONT, **ok, PASS=all(ok.values())))
P = pd.DataFrame(rows); P.to_csv(FIG / "preflight_check.csv", index=False); print(P.to_string(index=False))
