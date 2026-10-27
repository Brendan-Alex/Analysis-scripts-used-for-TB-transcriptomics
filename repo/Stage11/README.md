# Stage11: how small can a transferable signature be?

## How this was run

- **Computation:** `scripts/run_stage11.py`, run natively on your machine (46 min).
- **Summary:** `scripts/summarise_stage11.py`.
- **Inputs:** Stage7 data, folds, models and seeds (MLP alpha 10).
- **Leakage canaries:** 26/26 identical. Gene ranking, inner curves and chosen k are unchanged when the held-out cohort is noised.
- **Existing files:** none modified.

## Design

- **Rankings (training cohorts only):**
  - consistent (Stage9 rule);
  - mean mutual-information rank.
- **Panel of size k:** the top k of the ranking.
- **Choosing k:** in each fold, an inner leave-one-cohort-out run over the training cohorts picks the smallest k within 0.01 of the best inner AUC, per method. That k is then scored on the held-out cohort, with a bootstrap 95% CI.
- **Random reference:** 50 size-matched random panels at every k from 1 to 20.
- **Final signature per task:** the same procedure on all cohorts of the task. DirAvg is the primary method.
  - Its expected performance is the fold-level estimate below; it hasn't been tested on an independent cohort.

## Results, consistency ranking

Per-fold detail: `results/chosen_k_per_fold.csv`. Curves: `results/ablation_curves_long.csv`, `median_curves_DirAvg.csv`. A cell is one fold × one method.

| Task | Chosen k, median (range) | AUC at chosen k vs at 20 (median) | Cells beating size-matched random: at chosen k / at 20 |
|---|---|---|---|
| ATB vs LTB | 4 (1–13) | 0.934 vs 0.947 | 20/25 vs 17/25 |
| ATB vs OD | 11 (1–18) | 0.768 vs 0.786 | 15/20 vs 13/20 |
| LTB vs CON | 6 (1–20), erratic | 0.626 vs 0.610 | 4/20 vs 1/20 |

### Median DirAvg held-out AUC by k

| Task | Curve | Random panels by k |
|---|---|---|
| ATB vs LTB | 0.909 (k=1), 0.947 (k=4), plateau 0.94–0.95 to k=20 | 0.65 → 0.87 |
| ATB vs OD | 0.646 (k=1), about 0.75 by k=2–3, then a slow rise to 0.772 at k=20 | about 0.50–0.53 throughout |
| LTB vs CON | flat around 0.60–0.65 | about 0.51–0.55 |

### ATB vs LTB

**Plateau from about 3–4 genes.** Small panels lose little AUC and beat size-matched random more often than 20-gene panels do, because random panels improve sharply with size in this task.

### ATB vs OD

- **The untrained scores need few genes** (DirAvg and singscore final k = 5).
- **The trained models need more** (inner rule picks 17–19 for LR, RF and MLP).
- The curve rises steeply to 2–3 genes, then slowly, so there's no sharp plateau.

### LTB vs CON

- Chosen k is unstable across folds and methods.
- Final-signature genes are unstable: 4 of 11 appear in no fold's top 20.
- The 4 cells beating random are almost all in one fold (GSE107994, k = 3, AUC 0.68–0.69), driven by HLA-DPB1, which is in every fold's ranking.
  - HLA-DPB1 is highly polymorphic; its probe signal can track HLA genotype and therefore population, so this is likely a confound.
- **No signature is supported.**

## Final signatures (consistency ranking, DirAvg): `results/final_signatures.csv`

| Task | Genes (final k) | Stability: in fold top 20 / in fold chosen-k panel |
|---|---|---|
| ATB vs LTB | **ANKRD22, FCGR1A, PSTPIP2** (3) | 5/5 each; chosen 3–4/5 |
| ATB vs OD | **APOL2, WDFY1, BATF2, POLB, GBP2** (5) | 3–4/4; chosen 2–4/4 |
| LTB vs CON | none supported (the procedure returns 11 genes that are unstable across folds) | |

**Packaging:** the ATB vs LTB and ATB vs OD subsets share no genes, giving a **combined 8-gene panel**.

**Expected performance (DirAvg at the fold-chosen k, per held-out cohort):**

| Task | AUC range | Folds where ≤ 5% of random panels reach it |
|---|---|---|
| ATB vs LTB | 0.909–1.000 | 4/5 |
| ATB vs OD | 0.684–0.800 | 2/4 (the other two at 0.06 and 0.10) |

**MI ranking (secondary):**
- ATB vs LTB: POLB, PSTPIP2, LHFPL2 (3).
- ATB vs OD: IFIT3, CSGALNACT2, APOL2, SERPING1 (4).
- Slightly weaker in ATB vs OD (15 vs 10 cells beating random at the chosen k).

## Caveats

1. The fold-level AUCs evaluate the *procedure*. The final gene lists were derived on all cohorts and need an independent cohort to validate them as fixed signatures.
2. For trained models in ATB vs OD, 5 genes lose some performance relative to 17–19. The 5-gene signature is a DirAvg/singscore deliverable.
