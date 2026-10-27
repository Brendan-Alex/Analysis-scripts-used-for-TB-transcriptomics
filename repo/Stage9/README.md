# Stage9: is the Native failure the data or the selector?

## How this was run

- **Computation:** `scripts/run_stage9.py`, run natively on your machine.
  - 10 workers; 32 minutes; log in `logs/run.log`.
- **Summary:** `scripts/summarise_stage9.py`, run in the Cowork VM.
- **Inputs:** Stage7 data, folds, universe, models and seeds (MLP alpha fixed at 10).
- **Leakage control:** every gene choice uses training cohorts only. Selector canaries came back identical in 39/39 cases (`qc/canary_selectors.csv`).
- **Existing files:** none modified.

## 1. Corrected selectors: `results/selector_results.csv`

**The three selectors (20 genes per fold)**
- **MI_meanrank:** genes ranked by mutual information in each training cohort; take the 20 with the best mean rank.
- **consistent:** same direction of effect in every training cohort, ranked by the weakest separation across them.
- **mRMR_scaled:** the original Native algorithm with relevance rescaled to the same 0–1 scale as the redundancy penalty.

**Random reference:** Stage7's 20-gene random panels for the same fold and method.

| Task | Original Native (Stage7) | MI_meanrank | consistent | mRMR_scaled |
|---|---|---|---|---|
| ATB vs LTB: AUC; cells beating random | 0.85–1.00; 14/25 | 0.91–1.00; 19/25 | 0.92–1.00; 17/25 | 0.88–1.00; 19/25 |
| **ATB vs OD** | **0.40–0.68; 0/20** | **0.61–0.82; 10/20** | **0.64–0.82; 13/20** | 0.48–0.72; 1/20 |
| LTB vs CON | 0.43–0.78; 5/20 (all sibling-flagged) | 0.44–0.70; 1/20 | 0.46–0.70; 1/20 | 0.44–0.71; 0/20 |

### ATB vs OD

**The simple cross-cohort selectors transfer.**
- consistent: +0.19 median AUC over the original Native, and it beats random in all 5 methods on GSE37250 and GSE42834.
- MI_meanrank: +0.16 median AUC over the original Native.

**They pick the known interferon/TB module** (from 2–3 example folds): BATF2, GBP1/2/5/6, STAT1/2, APOL2, SERPING1, IFIT3, ETV7, SP140. They share 1–4 genes with Kaforou.

**Rescaling mRMR alone does not fix it.**
- mRMR_scaled has the highest training fit (median cross-cohort AUC 0.825) and the worst transfer (1/20).
- The forward-selection-and-swap step still overfits the 3 training cohorts.

### LTB vs CON

No selector transfers (≤ 1/20 cells beat random), including the consistency rule. This supports a real absence of shared LTB-vs-CON signal, consistent with the Stage6 rank analysis.

### ATB vs LTB

Every selector works, slightly better than the original Native.

## 2. Ablation: `results/ablation_shape.csv`, `ablation_informed_vs_random.csv`, `ablation_long.csv.gz`

**Design**
- Panels: Native, Zak16, Kaforou; every fold.
- Removal from n genes down to 1, in three orders:
  - backward (least-harmful to training fit first);
  - univariate (weakest in training first);
  - 50 random orders as the null.

**Results**
- **Halving the Native panel moves held-out AUC very little** (median change −0.05 to +0.02), and in inconsistent directions:

  | ATB vs OD, univariate order | Folds where held-out AUC rises |
  |---|---|
  | LR | 3 of 4 |
  | DirAvg | 0 of 4 |

  There is no clean "remove noise and performance rises" pattern.
- **Training-informed removal versus random removal** (percentile of the informed order within the 50 random orders):

  | Task | Native panel percentile |
  |---|---|
  | ATB vs LTB | 0.88–0.94 (training-based gene importance carries over) |
  | ATB vs OD | about 0.75, with a range of 0.02–0.98 (inconsistent) |
  | LTB vs CON | 0.42–0.64 (no better than random) |

**Reading**
- **Why the ablation doesn't show it:** the original Native panel isn't "a few good genes plus noise". Most of its genes are weak (Stage9 check: median |AUC − 0.5| about 0.04), so removing some can't uncover a strong core.
- **Why the selectors decide it:** replacing the panel with consistently separating genes moves ATB vs OD from 0/20 to 13/20 cells beating random.
