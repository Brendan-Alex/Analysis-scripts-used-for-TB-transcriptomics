# Stage5: rotating LOCO with the Stage4 decisions applied

## How this was run

- **Computation:** run natively on your Windows machine with `scripts/run_stage5.py`.
  - 10 worker processes.
  - Python 3.14.7, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1.
  - 34 minutes; log in `logs/run.log`.
- **Summaries:** built in the Cowork VM from the saved records, using `scripts/summarise_stage5.py` and `scripts/tables_stage5.py`.
- **Existing files:** none modified.
- **Full tables:** all per-fold tables (every model × panel × method, with CI and random distribution) are in `results/summary_tables.md`.
- **Reporting:** results are per fold only and never averaged.

## Decisions applied

1. **Overlap flags.** The three extra derivation-overlap relations are accepted, in a separate column (`overlap_added_accepted`):
   - Berry86 and Zak16 on GSE19491;
   - Sweeney3 on GSE19439 and GSE19444.

   Your original list is in `overlap_user_list`. `optimistic` is set if either applies.
2. **Selection universe** = genes measured in **every** cohort of the task. This uses the held-out cohort's gene list only.

   | Task | Measured in all cohorts | − blank/LOC/C*ORF/PHAGE | − X/Y genes | **Universe** |
   |---|---|---|---|---|
   | ATB vs LTB | 11,634 | 80 | 416 | **11,138** |
   | ATB vs OD | 19,527 | 466 | 848 | **18,213** |
   | LTB vs CON | 11,620 | 81 | 414 | **11,125** |

   - **No Native gene is zero-filled in any of the 13 folds.**
   - Random panels have no zero-filled genes either.
   - Curated panels were not altered, so a single gene is still zero-filled in a few folds:

     | Panel | Held-out cohort | Genes zero-filled |
     |---|---|---|
     | Kaforou_ATB_v_LTB | GSE107994 | 1 |
     | Berry86 (both versions) | GSE73408 (both ATB tasks) | 1 |
     | Kaforou_ATB_v_OD | GSE73408 | 1 |
     | Zak16 | GSE28623 | 1 |

     These are recorded in `zero_filled_in_heldout`.
3. **Berry86** is reported both as published (`Berry86`, includes TLR7) and as `Berry86_noTLR7`.
4. **X/Y genes** stay excluded from selection. The Native panels contain no X/Y gene.

## Leakage checks

**Native selection: all 13 folds.** I reran the full selection with the held-out cohort replaced by noise and random labels. The panel, cross-AUC and candidate set were identical in every fold (`qc/canary_native.csv`).

**Benchmark: all 13 folds.** I reran with the held-out data replaced by noise. In every fold these were bit-identical:
- features;
- directions;
- the chosen alpha and inner AUCs;
- all training-cohort predictions.

Results are in `qc/canary_bench.jsonl`.

## 1–2. What the changes did, relative to Stage4

Full detail: `results/change_vs_stage4.csv` and `results/native_change_vs_stage4.csv`.

### Platform check

- **Panels that did not change** (Zak16, Kaforou, Sweeney3, Berry86) reproduce Stage4 exactly in all 288 cells. That covers every method, including the MLP and the chosen alpha.
- **So every difference below comes from the decisions, not the move to Windows or newer libraries.**
- **Their random-panel medians shift** by −0.06 to +0.08 (median 0.00), because the random draws now come from the full-coverage universe.

### Native panels were re-selected

Each fold's panel shares only **5–12 of its 20 genes** with Stage4.

**Zero-fill fix**, in the two folds where Stage4 had zero-filled Native genes:

| Task | Held-out cohort | Native genes zero-filled in Stage4 | LR AUC, Stage4 → Stage5 | Random panels ≥ Native |
|---|---|---|---|---|
| LTB vs CON | GSE107994 | 10 / 20 | 0.592 → 0.629 | 0.20 → 0.10 |
| ATB vs LTB | GSE107994 | 6 / 20 | 0.999 → 0.993 | — |

**Folds that had no zero-fill in Stage4 moved by as much or more:**

| Task | Held-out cohort | Method | Change |
|---|---|---|---|
| ATB vs OD | GSE73408 | singscore | **+0.30** (0.47 → 0.78) |
| ATB vs OD | GSE73408 | DirAvg | +0.12 |
| ATB vs OD | GSE83456 | DirAvg | +0.11 |
| LTB vs CON | GSE19444 | DirAvg | +0.19 |
| LTB vs CON | GSE28623 | RF | −0.14 |

These shifts come from changing the candidate gene set, not from zero-fill. **Native performance therefore depends heavily on how the candidate universe is defined.** Read that as a weakness of the selection method, not an improvement.

**Training fit went up, transfer did not.** Training cross-AUC rose, most clearly in LTB vs CON (0.62–0.78 → 0.73–0.84), with no matching gain on the held-out cohorts.

### TLR7 removal

`results/berry86_tlr7_effect.csv`

- **No material effect.** LR, MLP_alpha, singscore and DirAvg change by at most 0.024 in any fold.
- **RF and fixed-alpha MLP move further** (−0.08 to +0.11, all in LTB vs CON). This is model instability when any single feature is dropped (RF's random feature subsets shift), and the CIs overlap almost completely.
  - Example: LTB vs CON, GSE19444 held out, RF: 0.671 [0.48–0.85] → 0.780 [0.62–0.92].
- **Alpha flips:** removing TLR7 changes the chosen alpha in two folds:
  - ATB vs OD, GSE37250: 1e-4 → 10;
  - LTB vs CON, GSE19444: 1e-3 → 1e-4.

  See item 5.

### Headline per-fold results, Stage5 (`per_fold_auc.csv`)

**ATB vs LTB**
- Every panel and method scores 0.84–1.00.
- Native scores 0.85–1.00.

**ATB vs OD**
- Kaforou, Zak16, Sweeney3 and Berry86 clear random panels in most folds.
- **Native: 0.40–0.78.** In 21 of 24 cells, more than 5% of random panels match or beat it.

**LTB vs CON**
- **GSE19439 held out is the one consistent exception.** Native scores 0.72–0.78 across all six methods, with ≤ 8% of random panels ≥ it. It was 0.76 in Stage4 too, despite a different panel. The CIs are wide (n = 29; e.g. DirAvg 0.775 [0.58–0.94]), and GSE19444, a sister cohort from the same lab and platform, is in its training set.
- **Other folds** score 0.43–0.65.
- **GSE28623 held out:** Native is worse than random (0.43–0.51; 50–92% of random panels ≥ it).
- **Berry86** (either version) is in the random range in every fold.

## 3. Native panel stability (final)

Tables: `results/Table_native_stability_summary.csv` (one row per task) and `results/Table_native_stability_genes.csv` (gene × fold, with HGNC locus type and flags). Both are reproduced in `summary_tables.md`.

| | ATB vs LTB | ATB vs OD | LTB vs CON |
|---|---|---|---|
| Folds × genes | 5 × 20 | 4 × 20 | 4 × 20 |
| Distinct genes | 53 | 54 | 46 |
| **In every fold** | **0** | **0** | **0** |
| In ≥ half the folds | 14 | 21 | 25 |
| In one fold only | 24 | 33 | 21 |
| Pairwise overlap (genes) | 2–11 | 1–8 | 4–8 |
| Pairwise Jaccard | 0.05–0.38 | 0.03–0.25 | 0.11–0.25 |
| Per-cohort mRMR top-20 pools, pairwise overlap | 0–3 | **0** | **0** |
| Olfactory receptor / microRNA / pseudogene | 0 / 0 / 1 | **3 / 1 / 2** | 0 / 0 / 0 |

### Flagged genes

**ATB vs OD**
- Olfactory receptors: OR6C4 (2 folds), OR5D13 and OR6T1 (1 fold each).
- microRNA: MIR302B (2 folds).
- Pseudogenes: ANKRD30BP2 and LPAL2 (3 of 4 folds each).

**ATB vs LTB**
- FCGR1BP (pseudogene, 2 folds). It is the FCGR1B pseudogene, and FCGR1A, a known TB gene, is also selected. The probe likely measures FCGR1 family transcript rather than being noise.

### Unexpected (flag)

**In ATB vs OD and LTB vs CON, the per-cohort mRMR top-20 pools share no genes at all between any two cohorts.** The candidate set in each fold is therefore exactly 3 × 20 = 60 genes. This means:
- the union/intersection step of the original algorithm does no consolidation;
- the panel is decided entirely by forward selection on cross-cohort AUC.

That explains both the instability and the implausible members.

## 4. ATB vs LTB task difficulty

`results/task_difficulty.csv`, all tasks for contrast.

| Cohort | Genes measured | % genes AUC > 0.8 | % genes AUC < 0.2 | Random LR median: 3 / 16 / 20 / 72 genes | Curated LR: Zak16 (16 genes) | Native DirAvg |
|---|---|---|---|---|---|---|
| GSE19491 | 24,903 | 1.46 | 2.10 | 0.66 / 0.85 / 0.85 / 0.88 | 0.92 | 0.90 |
| GSE37250 | 30,744 | 1.41 | 2.25 | 0.69 / 0.84 / 0.86 / 0.87 | 0.94 | 0.93 |
| GSE73408 | 20,208 | 5.66 | 8.17 | 0.81 / 0.91 / 0.93 / 0.90 | 0.92 | 0.97 |
| GSE101705 | 18,657 | 9.92 | 10.17 | 0.80 / 0.90 / 0.91 / 0.91 | 0.95 | 0.92 |
| GSE107994 | 14,328 | 13.48 | 18.39 | 0.85 / 0.97 / 0.97 / 0.98 | 1.00 | 1.00 |

Notes on the table:
- The random size-72 column is the Berry86_noTLR7-matched draw; in GSE73408 it uses 73 genes.
- Random medians are cross-cohort models trained on the other cohorts of that fold.
- Two-sided figures (|AUC − 0.5| > 0.3) are in the CSV.

**What it shows**
- **At 16 genes or more, random panels sit within about 0.01–0.10 of the curated panels in every ATB vs LTB cohort.**
- The gap is smallest in GSE107994 (random 0.97 vs 1.00) and GSE73408 (0.91–0.93 vs 0.92).
- Even GSE19491 and GSE37250, where only about 1.4% of genes individually exceed AUC 0.8, give random medians of 0.84–0.88. The signal is spread across many moderately separating genes, not concentrated in a few.
- **Only very small panels show content dependence:** random 3-gene sets reach 0.66–0.85, against Sweeney3's 0.90–1.00.

**Contrast with the other tasks**
- ATB vs OD: ≤ 0.54% of genes exceed 0.8, and random medians are 0.49–0.66.
- LTB vs CON: ≤ 0.3% of genes exceed 0.8, and random medians are about 0.5.

## 5. Alpha tuning

`results/alpha_tuning.csv` (per fold × panel) and `results/alpha_grid_long.csv` (every alpha, with inner per-cohort and held-out AUC).

### Correction to Stage4

I said Stage4 picked "extremes (1e-4 or 10)". **That was wrong.** I had mixed in the fixed-alpha MLP rows, which are always 1e-4.

In both stages the inner LOCO picks **alpha = 10, the top edge of the grid**, nearly every time:

| Chosen alpha | Stage4 (of 61) | Stage5 (of 74) |
|---|---|---|
| 10 | 45 | 59 (80%) |
| 1 | 10 | 9 |
| 1e-2, 1e-3, 1e-4 | 2 each | 2 each |

### How much the grid matters

Medians across fold × panel:

| | ATB vs LTB | ATB vs OD | LTB vs CON |
|---|---|---|---|
| Inner AUC range across the grid | 0.014 | 0.044 | 0.057 |
| Margin, best vs second-best alpha (inner) | 0.007 | 0.028 | 0.024 |
| Held-out AUC range across the grid | 0.022 (max 0.064) | 0.039 (max 0.134) | 0.057 (max 0.211) |
| Held-out: chosen alpha minus 1e-4 | +0.015 | +0.022 | +0.006 |
| Held-out: chosen alpha minus best on grid | 0.000 | 0.000 | −0.024 (worst −0.211) |
| Spearman, inner vs held-out AUC across the grid | 0.59 | 0.46 | −0.02 |

In 24% of cases the inner winner leads the second-best alpha by less than 0.005.

### Reading

**ATB tasks: the choice is not arbitrary.**
- The inner LOCO and the held-out cohort agree that stronger regularisation is better. The chosen alpha is the held-out-best grid value in most folds.
- But the answer is "as much as the grid allows". Because 10 is the boundary, the true optimum may lie above it, so the grid, not the data, sets the value.

**LTB vs CON: the choice is effectively arbitrary.**
- The inner ranking is uncorrelated with held-out performance.
- The selected alpha costs up to 0.21 AUC against the best grid value.

**Knife-edge folds.** In the same fold, panels differing by a single gene can flip between 1e-4 and 10 (Berry86 vs Berry86_noTLR7 above).

**Options (your call):**
- Fix alpha = 10 and report it as a limitation. This reproduces the tuned choice in 80% of cases.
- Extend the grid upward (e.g. 100, 1000) to see whether the optimum is interior.

## Files

**`results/`**
- `per_fold_auc.csv`: one row per task × fold × panel × method. Includes n's, AUC with CI, random distribution, overlap flags, zero-fill and global shift.
- `generalisation_gap.csv`
- `global_shift_heldout.csv`
- `change_vs_stage4.csv`, `native_change_vs_stage4.csv`
- `berry86_tlr7_effect.csv`
- `Table_native_stability_summary.csv`, `Table_native_stability_genes.csv`
- `native_stability_*.csv`
- `task_difficulty.csv`, `gene_level_separability.csv`
- `alpha_tuning.csv`, `alpha_grid_long.csv`
- `summary_tables.md`
- `native/`: per-cohort pools, fold panels and canary panels.
- `bench/`: raw scores.

**`qc/`**
- `folds.csv`
- `selection_universe.csv`
- `canary_native.csv`, `canary_bench.jsonl`
