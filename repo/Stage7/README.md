# Stage7: GSE83456 rebuilt, MLP fixed at alpha 10, consolidated results

## How this was run

- **Main run:** `scripts/run_stage7.py`, natively on your machine.
  - 10 workers; Python 3.14.7, scikit-learn 1.9.1.
  - 17 minutes; log in `logs/run.log`.
- **Rebuild, rankings, relatedness and summaries:** run in the Cowork VM.
  - Scripts `01`–`04` in `scripts/`.
  - Where the VM needed a data cache, it built its own with the same code; each such step cross-checks against your run.
- **Existing files:** none modified.
- **Results:** reported per fold only, never averaged.

**Start with:**
- `results/MASTER_results_table.csv`: the table the paper's tables come from.
- `RESULTS_INVENTORY.txt`: every supported finding, with its numbers and source file.

## 1. GSE83456 rebuilt (`scripts/01_rebuild_gse83456.py`, `qc/gse83456_rebuild_checks.csv`)

**Method**
- Source: `Clean_rebuild/geo_raw/GSE83456_matrix_EPTB_unnormalized.txt.gz`.
- Processing: log2(max(x, 1)), then quantile normalisation across the 155 samples. This is the GSE42834 rule.
- Same probe set and chosen probes as the clean rebuild, whose probe choice already used these raw values.
- Columns are linked to GSMs by the clean rebuild's expression-identity map. All 202 raw columns are accepted, and all 155 included GSMs are mapped.

**Checks**
- **Sample set and labels are unchanged:** 155 GSMs in the same order, 45 ATB / 61 CON / 49 OD, identical to Clean_rebuild and Stage3.
- **Genes:** all 30,652 genes are present.
- **Values:**

  | | Old file | Rebuilt |
  |---|---|---|
  | Range | −8.79 to 6.98 | 0 to 14.44 |
  | Median gene mean | 0.005 | 2.61 |
  | Median within-cohort sample correlation | 0.079 | 0.878 |

**Unexpected: the rebuild changes per-gene z-scores as well as levels**
- Per-gene agreement between the old and new z-scores depends on expression level (`qc/gse83456_old_vs_new_z_agreement_by_expression.csv`):

  | Genes | Median correlation, old vs new z |
  |---|---|
  | Mean log2 5–9 (expressed) | 0.99–1.00 |
  | Curated panel genes | 0.95–0.98 |
  | 16,402 genes near the floor (mean log2 ≤ 3) | 0.17 |

- So methods that z-score within the cohort are **not** unchanged after the rebuild (section 2).

## 2. What the rebuild changed

Files: `results/before_after_vs_stage5_stage6.csv`, `results/gse83456_singscore_before_after.csv`, `results/gse83456_global_shift_before_after.csv`, `results/atb_vs_od_native_before_after.csv`.

### singscore, GSE83456 held out

| Panel | Before (gene-centred input) | After (rebuilt) | Change | Random panels ≥, before → after |
|---|---|---|---|---|
| Sweeney3 | 0.656 [0.54–0.77] | 0.794 [0.69–0.88] | **+0.138** | 0.06 → 0.00 |
| Kaforou_ATB_v_OD | 0.643 [0.52–0.75] | 0.707 [0.59–0.81] | +0.064 | 0.02 → 0.00 |
| Zak16 | 0.738 [0.63–0.84] | 0.767 [0.66–0.86] | +0.029 | 0.00 → 0.00 |
| Berry86 | 0.557 | 0.533 | −0.024 | 0.10 → 0.30 |
| Berry86_noTLR7 | 0.552 | 0.528 | −0.024 | 0.08 → 0.22 |
| Native | 0.566 | 0.547 | −0.019 | 0.28 → 0.32 |

### Global-shift diagnostic, GSE83456

- All-gene mean AUC: 0.532 [0.41–0.65] → 0.465 [0.34–0.59].
- Genes individually AUC > 0.8: 0.45% → 0.35%.

### Other methods, GSE83456 held out (curated panels)

**Not identical,** because the z-scores changed:

| Method | Largest change |
|---|---|
| LR | 0.085 |
| MLP at alpha 10 | 0.056 |
| DirAvg | 0.053 |
| RF | 0.044 |

### Native selection

- GSE83456's own top-20 shortlist keeps only **3 of 20** genes.
- **The three folds where GSE83456 trains get new panels** (8–11 of 20 genes shared with Stage5).
  - Their Native AUCs fall by up to −0.19. Example: GSE73408 held out, MLP at alpha 10, 0.601 → 0.407.
  - Curated panels in those folds move by ≤ 0.030.
- The GSE83456-held-out fold panel is identical to Stage5, as it must be, since selection never sees the held-out cohort.

### Verified unaffected

ATB vs LTB and LTB vs CON contain no GSE83456. They were rerun and diffed against Stage5/Stage6 (`results/identity_check_unaffected_tasks.csv`):
- **LR, RF, singscore, DirAvg:** 0 of 200 cells differ.
- **MLP at alpha 10:** 0 of 50 differ.
- **Tuned MLP:** 0 of 50 differ.

### Leakage canaries (ATB vs OD)

- Native selection: 4/4 identical (`qc/canary_native_ATB_vs_OD.csv`).
- Benchmark: 4/4 pass (`qc/canary_bench.jsonl`).

### Downstream results recomputed for ATB vs OD

- Consensus failure and stability: `FINAL_Table_consensus_failure.csv`, `FINAL_Table_native_stability_*.csv`.
- Task difficulty: `FINAL_Table_task_difficulty.csv`.
- Cohort relatedness: `FINAL_cohort_relatedness.csv`.

What changed:
- **Consensus:** the top-20 shortlists still never overlap. The wider rankings now agree beyond chance in all 4 folds (p 0.003–0.037; Stage6 with the old file: 2 of 4).
- **Relatedness:** GSE83456 pairs now show normal same-platform similarity (mean-profile Spearman 0.97–0.98 with GSE37250 and GSE42834).

## 3. MLP at fixed alpha 10 (`results/FINAL_Table_MLP_fixed_vs_tuned_alpha.csv`)

- **Fixed alpha 10 is the primary MLP result** in every table. The tuned alpha (grid 1e-4…1000, inner LOCO) sits in its own columns with a 95% CI.
- **Tuned minus fixed: median 0.000 in every task.** Ranges:

  | Task | Tuned minus fixed |
  |---|---|
  | ATB vs LTB | −0.018 to +0.005 |
  | ATB vs OD | −0.098 to +0.084 |
  | LTB vs CON | −0.174 to +0.076 |

- **Near-constant predictions** (held-out SD < 1e-3) occur at the tuned alpha in **10 cells:**

  | Task | Cells |
  |---|---|
  | ATB vs LTB | 5 |
  | ATB vs OD | 1 |
  | LTB vs CON | 4 |

  Stage6 had 11: with the rebuilt GSE83456, one ATB vs OD cell no longer collapses. Both flags are columns in the master table.
- **Fixed alpha 10:** all MLP fits converged, and none collapsed.

## 4. Master table (`results/MASTER_results_table.csv`)

**Structure.** 370 rows = 74 task × fold × panel combinations × 5 methods: LR, RF, MLP (alpha 10), singscore, DirAvg. The column names are self-explanatory.

**What each row contains**
- **Design:** task and classes, held-out cohort, training cohorts and n (with class counts), test n and class counts.
- **Panel:** panel name, genes used, genes zero-filled.
- **Performance:** AUC with bootstrap 95% CI (1,000 resamples, seed 42).
- **Random panels:** 50 size-matched draws; median, 2.5–97.5 percentiles, and the proportion ≥ the panel.
- **Flags:**
  - derivation overlap: your list, plus the accepted sub-series containment;
  - the derivation relationship and an "optimistic" flag;
  - the GSE19439 sibling flag;
  - the held-out global shift;
  - a quantile-normalised marker;
  - a GSE83456-rebuilt marker.
- **MLP rows only:** the tuned-alpha result, its collapse flag and convergence.

**Reference for random MLP.** The random-panel comparison for MLP uses random MLPs at alpha 10. Tuned-alpha cells have no random comparison.

## 5. `RESULTS_INVENTORY.txt`

Plain-text list of every supported finding, each with its numbers and source file. It also covers what should *not* be claimed.

## 6. Follow-up: ATB vs OD secondary results on the rebuilt GSE83456

- **Generalisation gap, all tasks:** `results/FINAL_generalisation_gap.csv`, computed from the Stage7 records with no refitting.
  - ATB vs OD, RF: 0.097–0.572.
  - ATB vs OD, DirAvg: median 0.072.
- **ATB vs OD alpha profile:** `results/atb_vs_od_alpha_profile_rebuilt.csv`.
  - Median held-out AUC peaks at alpha 10 (0.786).
  - Tuned minus fixed-10: median 0.000.
- **With-sex ATB vs OD Native panels:** `scripts/run_stage7_withsex.py`, run on your machine in 4.5 min; outputs in `withsex/`.
  - **X/Y genes enter all 4 fold panels** (TTTY7 in 3, SPANXA2-OT1 in 3, MAGEB5, CDX4). Stage4, on the older universe and the old GSE83456, found 1/4.
  - With-sex panels share only 3–7 of 20 genes with the primary panels.
  - Sex balance (`results/sex_balance_ATB_vs_OD.csv`): ATB are more often male in GSE37250 (0.63 vs 0.36) and GSE73408 (0.71 vs 0.59); GSE42834 and GSE83456 are roughly balanced.
- **With-sex panels for all tasks, final universe** (`--task ATB_vs_LTB` / `LTB_vs_CON` runs; `results/FINAL_sex_gene_exclusion.csv`):
  - X/Y genes enter **9 of 13 fold panels**:
    - ATB vs LTB 5/5 (RPS4Y1, TXLNGY, BCORP1);
    - ATB vs OD 4/4;
    - LTB vs CON 0/4.
  - This supersedes Stage4's 7/13, which used the older universe and had SLC7A3 in one LTB vs CON fold.
