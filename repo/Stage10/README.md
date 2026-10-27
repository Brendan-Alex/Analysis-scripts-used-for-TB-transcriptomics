# Stage10: original Native pipeline with mRMR removed

## How this was run

- **Computation:** `scripts/run_stage10.py`, run natively on your machine (48 min).
- **Summary:** `scripts/summarise_stage10.py`.
- **Inputs:** Stage7 data, folds, models and seeds (MLP alpha 10); training cohorts only.
- **Leakage canaries:** 26/26 identical (`qc/canary.csv`).
- **Existing files:** none modified.

## The two arms

- **MI20_FS:** each training cohort's top 20 genes by mutual information, with no mRMR step, then the original forward selection and swaps.
- **consistent60_FS:** the fold's 60 most consistent genes, then the original forward selection and swaps.

Full side-by-side with the Stage7 original Native and the Stage9 selectors: `results/all_selectors_comparison.csv`. Per-fold detail with CIs: `results/selector_results_stage10.csv`.

## ATB vs OD

AUC across 20 cells (4 folds × 5 methods); "beats random" = cells where ≤ 5% of Stage7's 20-gene random panels reach the selector's AUC.

| Selector | Shortlist | Forward selection | AUC (median) | Cells beating random | Training fit (median) |
|---|---|---|---|---|---|
| Original Native | mRMR | yes | 0.40–0.68 (0.55) | 0/20 | 0.83 |
| mRMR_scaled | mRMR, scale fixed | yes | 0.48–0.72 (0.61) | 1/20 | 0.83 |
| **MI20_FS** | **MI, no mRMR** | yes | **0.52–0.81 (0.71)** | **7/20** | 0.82 |
| consistent60_FS | consistent | yes | 0.63–0.83 (0.77) | 13/20 | 0.89 |
| MI_meanrank | MI | no | 0.61–0.82 (0.72) | 10/20 | 0.70 |
| consistent | consistent | no | 0.64–0.82 (0.79) | 13/20 | 0.79 |

**Reading**
- **Removing mRMR alone** takes ATB vs OD from 0/20 to 7/20 cells beating random (median AUC +0.16). mRMR is the main cause of the original failure.
- **Forward selection adds training fit but no transfer.**
  - With MI candidates: 7/20 with forward selection vs 10/20 without.
  - With consistent candidates: 13/20 either way, median 0.77 vs 0.79, while training fit rises from 0.79 to 0.89.
- **Candidate quality decides transfer.** The consistency criterion gives the best and most stable result, with or without forward selection.

## LTB vs CON

- Nothing transfers with any selector: 0–2/20 cells beat random.
- The original Native's 5/20 are all in the sibling-flagged GSE19439 fold.

## ATB vs LTB

- All corrected selectors: 19/25 cells beating random, median AUC 0.95–0.96.
- Original Native: 14/25.
