# Stage4: rotating leave-one-cohort-out (all tasks)

No existing file was modified. Everything ran on your machine, in this folder.

Every cohort of each task is held out in turn and all the others train. That gives 13 folds:
- ATB vs LTB: 5 folds;
- ATB vs OD: 4 folds;
- LTB vs CON: 4 folds.

Stage 3 exclusions apply. Results are reported **per fold only**; nothing is averaged across folds.

Full per-fold tables (every model × panel × scoring method, with CIs and random-panel comparisons) are in `results/summary_tables.md`. The underlying CSVs are listed at the end.

## Design decisions (yours)

- **Normalisation: label-free per-cohort z-scores.** Each cohort, the held-out one included, is z-scored over all of its included samples regardless of class. This is **transductive but label-free**: the held-out cohort's own mean and SD are used, never its labels.
- **Kaforou:**
  - omitted from LTB vs CON;
  - `Kaforou_ATB_v_LTB` is used for ATB vs LTB;
  - `Kaforou_ATB_v_OD` is used for ATB vs OD.
- **Berry86** is used in all three tasks.

## Methods

**Features and fitting**
- Features are the panel genes present in every training cohort of the fold.
- Genes missing from the held-out cohort are zero-filled, which is the cohort mean on the z scale.
- Directions for singscore up/down sets and DirAvg signs are estimated on the pooled training samples.

**Models**
- LR: max_iter 1000.
- RF: 500 trees, random_state 42.
- MLP: one hidden layer of 50 units, max_iter 5000, alpha 1e-4.
- MLP_alpha: alpha chosen from {1e-4, …, 10} by an inner LOCO over the training cohorts.
- All MLP fits converged (n_iter < 5000).

**Random panels**
- 50 draws per curated panel per fold.
- Size-matched to the genes that panel actually uses in the fold.
- Drawn from the training-cohort gene intersection, minus X/Y genes and minus the names Native excludes (LOC*, C*ORF*, PHAGE).
- Seed: `default_rng([42, task, heldout, size, draw])`.

**CIs**
- 1,000 bootstrap resamples of the held-out cohort, seed 42, percentile method.

## 1. Leakage: how it was verified

**Structural checks**
- `01_native.py fold` asserts the held-out cohort is not among the training cohorts.
- It reads only the training cohorts' data and their per-cohort mRMR pools. Each pool is computed from one cohort alone.
- `02_bench.py` asserts that no held-out GSM appears among the training samples.

**Benchmark canary (all 13 folds, Native panel, all six methods)**
- The held-out cohort's Z, log2 values, ranks and labels were replaced with noise, and the fold was refitted.
- In every fold, all of these were bit-for-bit identical:
  - features;
  - gene directions;
  - the chosen alpha and all inner-LOCO AUCs;
  - the predictions on every training cohort.
- Held-out predictions changed, as they should.
- Results: `qc/canary_bench.jsonl`.

**Native-selection canary (LTB vs CON, GSE19439 held out)**
- The full selection was rerun with the held-out cohort replaced by noise and random labels.
- The panel, cross-AUC and allowed-gene set were identical.
- Results: `qc/canary_native.jsonl`.
- It was run on one fold only, because ATB folds need several calls each. The structural assertions cover the rest.

**Two caveats (not leakage of held-out data)**
- The inner alpha tuning for the Native panel uses a panel selected on all of the fold's training cohorts. The inner-held-out cohort therefore helped choose the panel, so the alpha choice, and only that, is mildly optimistic.
- Normalisation is transductive, as above.

## 2. X/Y-linked genes

- 1,305 genes are annotated X, Y or XY on GPL6947, GPL10558 or GPL11532: `data/sex_chromosome_genes.csv`.
- Removed from each cohort's selection universe:

  | Cohorts | Genes removed per cohort |
  |---|---|
  | GSE37250, GSE42834, GSE83456 | 946 |
  | GSE73408 | 870 |
  | GSE19491, GSE19439, GSE19444 | 865 |
  | GSE28623 | 781 |
  | GSE101705 | 595 |
  | GSE107994 | 438 |

- **The panels change.** When X/Y genes are allowed, they enter **7 of 13** fold panels:

  | Task | Sex genes entering | Folds |
  |---|---|---|
  | ATB vs LTB | RPS4Y1, TXLNGY, BCORP1, EGFL6 | all 5 |
  | ATB vs OD | TTTY7, CLCN5 | fold GSE83456 |
  | LTB vs CON | SLC7A3 | fold GSE107994 |

  RPS4Y1 and TXLNGY are Y-linked, consistent with the sex/class imbalance found in Stage 3.
- With them excluded, none enter.
- Panels with and without sex genes share only part of their genes:

  | Task | Genes shared (of 20) |
  |---|---|
  | ATB vs LTB | 9–14 |
  | ATB vs OD | 2–7 |
  | LTB vs CON | 6–14 |

  So a handful of sex genes reshuffles most of the panel. This is also a sign of how unstable the selection is (see 8).
- **Curated panels were not altered.** Berry86 contains one X-linked gene (TLR7). Say if you want it dropped.
- Details: `results/sex_gene_exclusion.csv`.

## 3–4. Per-fold performance and random panels (headlines)

### ATB vs LTB

- All panels and methods reach AUC 0.84–1.00 in every fold. Random panels of the same size also do very well:
  - size 16–73: random medians 0.72–0.99;
  - Sweeney3 (3 genes): random medians 0.61–0.82.
- The curated panels mostly beat random (proportion ≥ curated usually 0–0.2). The margin is small for the larger panels, and Sweeney3 stands out most against its random baseline.
- **GSE107994 held out:** nearly everything scores 0.98–1.00, and random medians are 0.93–0.98. As in Stage 3, this cohort is near-trivially separable, and it gives no evidence about panels.

### ATB vs OD

The task where the panels differ.

- Kaforou_ATB_v_OD, Zak16 and Sweeney3 reach 0.58–0.93 (83% of cells ≥ 0.70). They clear random panels in most cells: random medians are 0.47–0.66, and in 79% of cells ≤ 5% of random panels reach the curated AUC.
- In the clean folds (no derivation overlap), Kaforou reaches 0.64–0.87.
- Berry86 reaches 0.60–0.89 with the models, but is weak with singscore (0.56–0.64).
- **The Native panel fails here:** 0.43–0.79, often inside the random range.
  - Fold GSE83456: LR 0.50 (proportion of random panels ≥ it: 0.72).
  - Fold GSE73408: singscore 0.47.
- Native's training cross-AUC was 0.82–0.83 in every fold, so the selection overfits the training cohorts (see 5 and 8).

### LTB vs CON

**No transferable signal in any fold.**

- Most AUCs are 0.45–0.65 and inside the random distribution.
- **Berry86's DirAvg/singscore signal from Stage 3 does not reproduce under rotation:**
  - DirAvg: 0.51–0.65;
  - singscore: 0.52–0.63;
  - even in the two Berry-overlap folds.
- Isolated higher values:
  - Native, GSE19439 held out: LR 0.755 [0.57–0.92], none of 50 random panels ≥ it.
  - Sweeney3, GSE19444 held out: LR/singscore 0.72–0.73.
  - Sweeney3, GSE28623: RF 0.70.
- These are single folds with n = 29–62, and they don't recur in other folds. GSE19439 and GSE19444 are sister cohorts (same lab and platform), and each is in the other's training set.

## 5. Generalisation gap

Pooled training resubstitution AUC minus held-out AUC, per fold: `results/generalisation_gap.csv`.

**ATB vs LTB: small gaps.**
- About −0.08 to +0.09.
- Negative gaps in the GSE107994 fold mean the held-out cohort is easier than the training data.

**ATB vs OD: large gaps for the trained models.**
- RF: 0.10–0.47.
- LR: up to 0.37.
- Native gaps are the largest (LR 0.17–0.37), consistent with selection overfitting.
- DirAvg gaps are small for the curated panels (about −0.14 to +0.13).

**LTB vs CON: very large gaps.**
- RF 0.30–0.62; Berry86 LR 0.36–0.47. The models fit the training cohorts and transfer nothing.
- MLP_alpha gaps go up to 0.54.
- DirAvg gaps are mostly ≤ 0.2.

## 6. Global shift (held-out cohort)

Per-sample all-gene mean vs class: `results/global_shift_heldout.csv`, also repeated in the tables.

- The sign is inconsistent across cohorts (AUC 0.19–0.66).
- **GSE107994 (ATB vs LTB):** AUC 0.19, and 54% of genes separate on their own.
- **GSE101705 (ATB vs LTB):** AUC 0.60, with 42% of genes separating.
- **ATB vs OD GSE37250:** AUC 0.66, but only 1.6% of genes separating.
- **GSE19491 and GSE42834** are quantile-normalised, so their all-gene mean is near-constant by construction and the diagnostic is uninformative there.
- The shift doesn't explain the ranking above. LTB vs CON fails both in cohorts with a shift (GSE19439, 0.29) and without one (GSE19444, 0.50).

## 7. Derivation-overlap flags

Columns in `per_fold_auc.csv`:
- `optimistic`;
- `overlap_in_user_list` (your list);
- `overlap_added_subseries_containment`.

**I added three relations beyond your list.** Please confirm or reject them:
- **Berry86 and Zak16 on GSE19491 held out (ATB vs LTB).** The Stage 3 GSE19491 ATB vs LTB set consists *entirely* of samples from GSE19439, GSE19442 and GSE19444 (by GSM sub-series in the manifest). That makes it Berry's derivation and validation material and Zak16's Illumina parameterisation and test material.
- **Sweeney3 on GSE19439 and GSE19444 (LTB vs CON).** These are sub-series of Sweeney's discovery dataset GSE19491.

Flagged folds:
- **ATB vs LTB:**
  - GSE19491: Kaforou, Sweeney3, plus Berry86‡ and Zak16‡;
  - GSE37250: Kaforou, Sweeney3, Zak16.
- **ATB vs OD:**
  - GSE37250: Kaforou, Sweeney3, Zak16;
  - GSE42834: Sweeney3, Zak16.
- **LTB vs CON:**
  - GSE19439: Berry86, Zak16, plus Sweeney3‡;
  - GSE19444: Berry86, Zak16, plus Sweeney3‡;
  - GSE28623: Sweeney3.

The flagged ATB vs OD folds are visibly higher for Kaforou and Sweeney3. For example, Kaforou on GSE37250 reaches 0.88–0.93 against 0.64–0.87 elsewhere.

## 8. Native panel stability

`results/native_stability.csv`, `native_pairwise_overlap.csv`

**No gene is shared by all folds, in any task.**

| Task | Distinct genes across folds | Pairwise overlap (of 20) | Genes in ≥ half the folds |
|---|---|---|---|
| ATB vs LTB | 54 | 6–8 | 12 (APOL3, ASXL1, C1QC, DNAJC21, EPHA4, GFRA2, IL18RAP, MILR1, MTURN, NDRG2, PSTPIP2, VAMP5) |
| ATB vs OD | 50 | 5–8 | listed in the CSV |
| LTB vs CON | 44 | 6–11 | listed in the CSV |

- The ATB vs OD panels contain implausible members such as olfactory receptors, MIR genes and TAS1R2. This fits selection chasing noise when the per-cohort pools barely intersect: only 53–58 genes were allowed per fold.
- A single final Native20 derived from these panels would rest on a set that changes almost completely with the training cohorts.

## 9. ATB vs OD: indication-defined vs disease-defined comparator

`results/atb_vs_od_comparator_type.csv`

- **Only one fold has training that is predominantly one type:** GSE37250 held out, where the training OD is 100% disease-defined.
- In the other three folds, GSE37250 is 41–49% of the training OD, so training is mixed rather than "predominantly indication-defined".
- **The reverse comparison (train mostly indication-defined, test disease-defined) is not available in this design.**
- On GSE37250 with no overlap, the non-optimistic panels score:
  - Native: 0.57–0.79;
  - Berry86: 0.64–0.79.
- On the disease-defined held-out cohorts, the same panels score:
  - Native: 0.43–0.67;
  - Berry86: 0.56–0.89.
- There is no systematic drop when moving from disease-defined training to the indication-defined test. If anything, GSE37250 is the easier held-out cohort for Native.
- With one fold of one type, comparator type can't be separated from cohort.

## Other flags

**Native genes missing from GSE107994.** GSE107994 measures fewer genes (14.5k), and the Native universe is the training intersection, which by design excludes any held-out information. As a result, Native genes are absent from GSE107994 and zero-filled:
- 6/20 in the ATB vs LTB fold;
- 10/20 in the LTB vs CON fold.

Random panels in those folds are zero-filled for up to 26–33% of their genes. ATB vs LTB still scores 1.00 there, but LTB vs CON Native on GSE107994 is effectively a 10-gene panel.

An alternative is to restrict the universe to genes present in all of a task's cohorts. That would use the held-out cohort's gene list (label-free), not its values. Your call.

**Two random draws scored NaN.** In two random Sweeney3-size draws (3 genes) on GSE107994, all three genes were absent, so DirAvg was a mean over nothing. Those scores were set to a constant, consistent with zero-fill, so AUC = 0.5. Listed in `qc/nan_scores_set_to_constant.csv`.

**Inner alpha choices go to the extremes.** The inner LOCO picks 1e-4 or 10 most of the time. The choice is noisy with only 2–4 training cohorts.

**GSMs repeat across tasks.** The same GSMs appear in ATB vs LTB (inside GSE19491) and in LTB vs CON (GSE19439/19444). The tasks are separate, so this is not a leak.

## Files

**Scripts**
- `scripts/common4.py`: tasks, panels and overlap table.
- `scripts/00_prepare.py`: caches, sex genes, folds and global shift.
- `scripts/01_native.py` + `run_native_queue.sh`: per-fold Native selection.
- `scripts/02_bench.py` + `run_bench_queue.sh`: benchmark and canary.
- `scripts/04_canary_native.py`: Native-selection canary.
- `scripts/03_summarise.py`: CSVs.
- `scripts/05_tables_md.py`: `summary_tables.md`.

**Results**
- `per_fold_auc.csv`: one row per task × fold × panel × method. Contains n's, AUC, CI, random statistics, overlap flags, global shift and zero-fill counts.
- `generalisation_gap.csv`
- `native_stability.csv`, `native_pairwise_overlap.csv`
- `sex_gene_exclusion.csv`
- `atb_vs_od_comparator_type.csv`
- `global_shift_heldout.csv`
- `bench/*.jsonl`: raw scores for every job.
- `native/*.json`: fold panels.

**QC**
- `folds.csv`
- `canary_bench.jsonl`, `canary_native.jsonl`
- `nan_scores_set_to_constant.csv`
