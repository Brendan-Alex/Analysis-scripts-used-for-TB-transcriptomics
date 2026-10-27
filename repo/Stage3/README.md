# Stage3 (19 Oct 2026)

New folder; nothing existing was changed. Everything ran locally (2 cores, `n_jobs=-1`). No figures.

**Applied:**
- GSE19491's 59 OD dropped.
- The 7 GSE19435 ATB dropped.
- No per-sample age rule.
- GSE19491 renormalised from the GEO raw files over its 147 remaining samples (log2 + quantile, as in Stage2).

## 1. Task datasets (`qc/task_cohorts.csv`)

| Task | Role | Cohort | Positive | Negative | Both classes |
|---|---|---|---|---|---|
| ATB vs LTB | train | GSE19491 | 54 | 69 | yes |
| | train | GSE37250 | 97 | 83 | yes |
| | train | GSE73408 | 35 | 35 | yes |
| | test | GSE101705 | 28 | 16 | yes |
| | test | GSE107994 | 34 | 44 | yes |
| LTB vs CON | train | GSE19439 | 17 | 12 | yes |
| | train | GSE19444 | 21 | 12 | yes |
| | train | GSE28623 | 25 | 37 | yes |
| | test | GSE107994 | 44 | 49 | yes |
| ATB vs OD (held) | train | GSE19491 | 54 | **0** | **no** |
| | train | GSE37250 | 97 | 83 | yes |
| | train | GSE73408 | 35 | 39 | yes |
| | test | GSE42834 | 35 | 71 | yes |
| | test | GSE83456 | 45 | 49 | yes |

- **ATB vs LTB and LTB vs CON: every training cohort contains both classes.**
- **ATB vs OD: GSE19491 now has no OD**, so it contributes only ATB. If kept, it re-creates a cohort-equals-class confound.

## 2. Global shift (`results/global_shift_all_cohorts.csv`)

The per-sample mean over every gene in the cohort is scored against class (AUC; < 0.5 means the first-named class has the lower mean). Each AUC has a bootstrap CI (1,000 resamples, seed 42).

| Cohort | Contrast | Role | AUC (95% CI) | Shared-gene AUC |
|---|---|---|---|---|
| GSE19491 | ATB vs LTB | training | 0.377 (0.27–0.48) | 0.389 |
| GSE37250 | ATB vs LTB | training | 0.370 (0.29–0.46) | 0.435 |
| GSE73408 | ATB vs LTB | training | 0.402 (0.27–0.53) | 0.345 |
| GSE101705 | ATB vs LTB | test | 0.603 (0.42–0.78) | **0.929** |
| GSE107994 | ATB vs LTB | test | **0.193 (0.10–0.30)** | 0.257 |
| GSE19439 | ATB vs LTB | – | 0.186 (0.05–0.35) | 0.439 |
| GSE19444 | ATB vs LTB | – | 0.270 (0.12–0.42) | 0.333 |
| GSE28623 | ATB vs LTB | – | 0.384 (0.25–0.53) | 0.393 |
| GSE19439 | LTB vs CON | training | 0.289 (0.10–0.50) | 0.466 |
| GSE19444 | LTB vs CON | training | 0.504 (0.26–0.74) | 0.563 |
| GSE28623 | LTB vs CON | training | 0.625 (0.46–0.77) | 0.589 |
| GSE107994 | LTB vs CON | test | 0.513 (0.39–0.64) | 0.511 |
| GSE107994 | ATB vs CON | – | 0.174 (0.09–0.28) | 0.227 |
| GSE19439 | ATB vs CON | – | 0.083 (0.00–0.21) | 0.404 |

"Shared-gene AUC" uses the 11,477 genes present in all ten cohorts. ATB vs OD rows and the GEO-processed variants are in the CSV.

**Findings:**
- **ATB has a lower all-gene mean than LTB in all three training cohorts** (0.37–0.40) **and, most strongly, in GSE107994** (0.19). It is the opposite way in GSE101705 (0.60; 0.93 on shared genes). So the shift exists in both training and test cohorts, but its sign is not consistent.
- **The earlier 0.07 for GSE107994 came from the old z-file** (different source, 8 progressors labelled LTB). On GEO's own data it is 0.19.
- **The number depends on gene set and normalisation for RNA-seq.** In GSE101705 it is 0.60 on all genes but 0.93 on shared genes.
- **Two cohorts are near-constant by construction.** GSE19491 and GSE42834 are quantile-normalised, so their all-gene distributions are forced to be almost identical.
- **LTB vs CON shows no consistent shift.**

**Rank scoring is immune to the intensity shift, but not to a transcriptome-wide difference** (`results/rank_composition_check.csv`, `results/pergene_separation.csv`).
- Unsigned random 20-gene rank scores centre on AUC 0.5 in every cohort.
- In GSE107994 their spread is very wide (2.5–97.5%: 0.10–0.91). There, **54% of all genes individually separate ATB from LTB with |AUC − 0.5| > 0.2**, against 13% in GSE19491 and GSE37250.
- So once a gene's direction is learned in training, almost any gene set classifies GSE107994. That is pervasive differential expression, not a global intensity offset.

## 3. Rank-based scoring

- **singscore:** Foroutan 2018 `simpleScore`, centred, with up and down sets. It uses within-sample ranks over all genes of the cohort and no cohort statistics.
- **DirAvg:** the mean of direction-signed per-cohort z-scores.
- In both, each gene's direction is the sign of the positive-minus-negative mean in the training samples of that fit (inside the fold for internal estimates).

## 4. Feature-selection placement

The original code is in `Work\Code\Gene Compression and processing` (`MI + mrmr.py`, `20 from Gene Universe`, `Fianl 20 gene generator.py`).

**Selection ran once, on all training cohorts, outside any LOCO fold.** Internal LOCO estimates for Native20 were therefore optimistic.

Further bugs in the cross-task step (`Fianl 20 gene generator.py`):
- **Training files swapped:** the LTB vs CON ranking was computed on GSE19491, GSE37250 and GSE73408, and the ATB vs LTB ranking on GSE19439, GSE19444 and GSE28623.
- **No subsetting to the task's classes:** negatives were every other class in the file. For example, LTB vs CON scored LTB against ATB + OD + CON.
- **The positional-merge bug:** `concat(axis=1)` followed by `columns.duplicated()` misaligned genes across cohorts.

**The inside-fold version is implemented** (`scripts/02_native_select.py`). It follows the original algorithm with corrected cohorts and classes:
- per cohort: MI top 2,000, then mRMR top 20;
- union, then intersection across the fold's cohorts;
- seed 5 genes, forward selection to 20 on mean cross-cohort AUC, then swap refinement.

## 5. Native panels re-derived per task (`panels/Native_ATB_vs_LTB.csv`, `panels/Native_LTB_vs_CON.csv`, `results/native_panels_by_fold.csv`)

| Task | Final panel (all training cohorts) | Retained from original Native20 |
|---|---|---|
| ATB vs LTB | VAMP5, LHFPL2, APOL3, MEF2A, TMEM70, IGFBP4, SMARCD3, NFXL1, MLC1, ROGDI, ARHGEF17, GFRA2, ZNF229, FRMD1, EGFL6, EPHA4, C1QC, FBXO47, CXCR5, TXLNGY | **VAMP5** only (19 new) |
| LTB vs CON | MBTD1, ANKRD9, GFRA2, NF2, HLA-DQB1, SLC7A3, LYPD1, NTN3, KRT15, DKFZP434K191, UNC5D, DDX23, LCT, ICAM4, CRNKL1, NELFA, GOLM1, CD1E, DSP, CYP4F22 | **CRNKL1** only (19 new) |

- **Very unstable:** no gene appears in all three fold panels for either task, and each fold panel shares only 7–10 genes with the final panel.
- **The LTB vs CON panel looks like noise:** keratins, olfactory receptors, LCT. Per-cohort selection on 29–62 samples is noise-dominated.
- **Sex confound.** Y-linked genes were selected: TXLNGY in the final ATB vs LTB panel, RPS4Y1 in fold panels, and KDM5D in the original Native20.
  - In all three ATB vs LTB training cohorts, ATB are more often male: 0.63–0.71 vs 0.42–0.51 (`results/sex_balance.csv`).
  - In the test cohorts the classes are balanced (0.59 vs 0.57).

## 6. Model settings (`results/model_diagnostics.csv`)

- **MLP `max_iter` = 5000: all 432 MLP fits converged** (median 650–880 iterations, max 3,411). Under the old cap of 500, most would not have.
- **Over-fitting, mean across folds:**

  | Task | MLP training AUC | MLP LOCO AUC | Tuned MLP training AUC | Tuned MLP LOCO AUC |
  |---|---|---|---|---|
  | ATB vs LTB | 0.989 | 0.934 | 0.969 | 0.948 |
  | LTB vs CON | 0.996 | 0.570 | 0.952 | 0.601 |

- **What I did:** added `MLP_alpha`, with alpha chosen from 1e-4 to 10 by inner leave-one-cohort-out within the training cohorts.
  - It chose strong penalties: α = 10 in 18 fits, α = 1 in 11.
  - The gain is small; in LTB vs CON the problem is absence of signal, not capacity.
  - I did not use early stopping.

## 7. Benchmark (`results/curated_auc_with_ci.csv`, `results/external_vs_random.csv`, `results/random_draw_auc.csv`, `logs/04_summarise.log`)

**External AUC (95% CI)**

| ATB vs LTB | Cohort | LR | RF | MLP | MLP_alpha | singscore | DirAvg |
|---|---|---|---|---|---|---|---|
| Kaforou | GSE101705 | 0.891 (0.77–0.98) | 0.949 (0.88–1.00) | 0.884 (0.76–0.98) | 0.922 (0.82–0.99) | 0.922 (0.83–0.99) | 0.935 (0.85–0.99) |
| Native | GSE101705 | 0.908 (0.82–0.99) | 0.945 (0.87–1.00) | 0.879 (0.77–0.97) | 0.924 (0.84–0.99) | 0.926 (0.84–0.99) | 0.942 (0.87–1.00) |
| Sweeney3 | GSE101705 | 0.893 (0.79–0.98) | 0.886 (0.77–0.97) | 0.902 (0.80–0.98) | 0.900 (0.80–0.98) | 0.920 (0.83–0.99) | 0.837 (0.72–0.95) |
| Zak16 | GSE101705 | 0.962 (0.90–1.00) | 0.923 (0.83–0.99) | 0.929 (0.84–0.99) | 0.935 (0.85–0.99) | 0.933 (0.85–1.00) | 0.922 (0.83–1.00) |
| all four panels | GSE107994 | 0.995–1.000 | 0.983–0.999 | 0.992–0.998 | 0.993–0.999 | 0.986–1.000 | 0.984–0.999 |

| LTB vs CON | Cohort | LR | RF | MLP | MLP_alpha | singscore | DirAvg |
|---|---|---|---|---|---|---|---|
| Berry86 | GSE107994 | 0.587 (0.46–0.70) | 0.647 (0.53–0.75) | 0.601 (0.49–0.72) | 0.605 (0.49–0.72) | 0.637 (0.53–0.75) | **0.688 (0.58–0.80)** |
| Native | GSE107994 | 0.537 | 0.545 | 0.550 | 0.544 | 0.554 | 0.552 |
| Sweeney3 | GSE107994 | 0.549 | 0.517 | 0.529 | 0.527 | 0.582 | 0.552 |
| Zak16 | GSE107994 | 0.478 | 0.554 | 0.562 | 0.500 | **0.381 (0.27–0.50)** | 0.515 |

**Size-matched random panels** (50 draws per size; median and 2.5–97.5%):
- **ATB vs LTB, GSE107994:** random medians are 0.91–0.98 for 16–23 genes and 0.79–0.87 for 3 genes. Curated panels sit at the top of that distribution, but the margin is tiny.
- **ATB vs LTB, GSE101705:** random medians are 0.73–0.91. Only some curated combinations beat ≥ 95% of draws: Zak16 LR (p = 0.02), Zak16 MLP (0.02), Native RF (0.04), Sweeney3 RF (0.04).
- **LTB vs CON:** random medians are about 0.51–0.57. Berry86 DirAvg (0.688) and singscore (0.637) exceed all 50 size-matched draws. Everything else sits inside the random distribution.

**Internal LOCO, ATB vs LTB:**
- Kaforou 0.94–0.99, Sweeney3 0.91–0.97, Zak16 0.89–0.95.
- **Native, re-derived inside folds: 0.79–0.96**, lower than the curated panels. The curated panels' internal numbers are inflated by derivation overlap (§8).

**Internal LOCO, LTB vs CON:** 0.35–0.87, about chance on average.

## 8. Derivation overlap (`results/panel_derivation_overlap.csv`)

| Panel | Derived / validated on | Overlap with my cohorts |
|---|---|---|
| Berry86 | **Derived on GSE19444** (London test set) against GSE22098 diseases; validated on GSE19439 and GSE19442 | **LTB vs CON training: GSE19444 (derivation), GSE19439 (validation)** |
| Kaforou (ATB v LTB / ATB v OD) | **Derived on GSE37250** (80% training split); validated on **GSE19491** | **ATB vs LTB training: GSE37250 (derivation), GSE19491 (validation)** |
| Zak16 | Derived on ACS RNA-seq (not used here). The Illumina version was **parameterised on GSE19439** and tested on GSE19444, GSE19442, **GSE37250** and **GSE42834** | ATB vs LTB training: GSE19491, GSE37250. LTB vs CON training: GSE19439, GSE19444. **ATB vs OD test: GSE42834** |
| Sweeney3 | **Discovery: GSE19491, GSE37250, GSE42834**; validation included **GSE28623** | ATB vs LTB training: GSE19491, GSE37250. LTB vs CON training: GSE19439/19444 (inside GSE19491) and GSE28623. **ATB vs OD test: GSE42834 (discovery)** |

- **External test cohorts for ATB vs LTB and LTB vs CON** (GSE101705, GSE107994) were used by none of these panels.
- **Internal LOCO for the curated panels is optimistic** wherever the held-out cohort was a derivation or validation cohort. That is almost every internal cell.
- **ATB vs OD external results on GSE42834 would be optimistic** for Sweeney3 (discovery) and Zak16 (tested).

## 9. HIV status (`results/hiv_status.csv`)

- **GSE37250** is the only cohort with per-sample HIV status in GEO. It is filtered to HIV-negative (274 HIV-positive removed).
- **HIV-negative by study design:** GSE19491, GSE19439/19444, GSE73408, GSE28623 (The Gambia), GSE42834, GSE83456 and GSE107994.
- **GSE101705: HIV status unknown.** GEO and the series text are silent and the paper isn't open access.
- In effect the filter is uniform (HIV-negative everywhere) except possibly GSE101705, an external test cohort.

## Flags

1. The ATB vs OD design breaks: GSE19491 has no OD left.
2. GSE107994 ATB vs LTB is near-trivially separable: random panels reach about 0.95. Its AUCs say little about panel quality.
3. The shift's sign is opposite in GSE101705, and its size there depends on the gene set.
4. Native panels are unstable across folds and pick up sex-linked genes, because ATB are male-enriched in the training cohorts.
5. LTB vs CON shows no transferable signal apart from Berry86 with DirAvg or singscore. Berry86 was derived and validated on two of the three LTB vs CON training cohorts.
6. Zak16 singscore is inverted for LTB vs CON (0.381).
7. Native LTB vs CON loses 8 of 20 genes in GSE107994 (absent from the edgeR table; zero-filled).
8. Zak16 uses 15 genes in LTB vs CON (FCGR1A isn't on GSE28623's platform).
