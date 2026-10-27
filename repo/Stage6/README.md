# Stage6: extended alpha grid, consensus-selection failure, GSE19439 sibling check

## How this was run

- **Computation:** `scripts/run_stage6.py`, run natively on your machine.
  - 10 workers; Python 3.14.7, scikit-learn 1.9.1.
  - 12 minutes; log in `logs/run.log`.
- **Summaries:** `scripts/summarise_stage6.py` and `scripts/tables_stage6.py`, run in the Cowork VM.
- **Supplementary diagnostic:** `scripts/sibling_gene_origin.py`, run in the VM on a cache rebuilt there with the Stage5 code. It reproduces the corresponding numbers of your run.
- **Stage5:** caches, panels and code were read only, never modified.
- **Tables:** all of them are in `results/summary_tables.md`. Results are per fold only.

**Consistency checks**
- At the six original alphas, inner and held-out AUCs reproduce Stage5 exactly (checked on the LTB vs CON smoke test, 120/120 cells). Because of this, chosen alphas differ from Stage5 only where 100 or 1000 won.
- The full gene rankings reproduce Stage5's 13 per-cohort top-20 shortlists exactly.

## 1. Alpha grid extended to 100 and 1000

Files:
- `alpha_extended.csv`: per fold × panel; held-out AUC at every grid point, CI at the chosen alpha, random-panel comparison.
- `alpha_extended_long.csv`: every grid point, with inner AUC per inner cohort.

**What was chosen**
- Of the 59 cells that chose 10 in Stage5, 34 stay at 10, 18 move to 100 and 7 move to 1000.
- New choices:

  | Task | 1000 | 100 | 10 | 1 | ≤ 1e-2 |
  |---|---|---|---|---|---|
  | ATB vs LTB | 5 | 6 | 15 | 4 | 0 |
  | ATB vs OD | 0 | 11 | 11 | 1 | 1 |
  | LTB vs CON | 2 | 3 | 8 | 3 | 4 |

- **66/74 choices are now interior.**
  - 7 are at the upper boundary: all five Kaforou ATB vs LTB folds, plus Zak16 and Berry86_noTLR7 on GSE19439.
  - 1 is at the lower boundary.

**Unexpected: high alpha makes the predictions collapse**
- At alpha = 1000, the MLP's predicted probabilities collapse to an almost constant value in every cell (held-out SD around 1e-5).
- At alpha = 100 this happens in 55% of LTB vs CON cells.
- AUC still works, because the scores keep their order. The model outputs, however, are uninformative as probabilities: the same class would be called at any normal threshold.
- **11 of the 74 chosen alphas are collapsed points.** These are marked `chosen_prediction_collapsed` and shown with "·" in the tables.

**Peak or plateau?** Median held-out AUC across the grid:

| alpha | 1e-4 | 1e-3 | 1e-2 | 0.1 | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|---|---|---|---|
| ATB vs LTB | 0.939 | 0.939 | 0.939 | 0.941 | 0.948 | **0.958** | 0.955 | 0.952 |
| ATB vs OD | 0.744 | 0.742 | 0.742 | 0.742 | 0.754 | **0.783** | 0.756 | 0.752 |
| LTB vs CON | 0.557 | 0.553 | 0.555 | 0.564 | 0.594 | 0.575 | 0.594 | 0.586 |

- **ATB vs LTB plateaus** from alpha 1 upwards (within about 0.01). In 30% of cells, held-out AUC varies by ≤ 0.01 across 1–1000.
- **ATB vs OD peaks at 10.**
- **LTB vs CON has no structure.** Values sit around 0.55–0.59, and the held-out-best alpha is spread across the whole grid.

**Does inner tuning predict held-out performance? Same answer as Stage5.**

| | ATB vs LTB | ATB vs OD | LTB vs CON |
|---|---|---|---|
| Spearman, inner vs held-out across the grid (median) | 0.55 | 0.57 | **−0.11** |
| Chosen minus held-out-best, median (worst) | −0.002 (−0.031) | −0.001 (−0.122) | **−0.090 (−0.319)** |
| Chosen minus fixed alpha = 10, median (range) | 0.000 (−0.018 to +0.004) | 0.000 (−0.075 to +0.070) | 0.000 (−0.175 to +0.075) |

- **ATB tasks:** tuning tracks the held-out cohort.
- **LTB vs CON:** tuning is uninformative. With the wider grid it now costs up to **0.32** AUC (Zak16 on GSE19439, chosen 1000, collapsed), against 0.21 in Stage5.

**Reading**
- Extending the grid does not improve anything.
- Fixed alpha = 10 matches tuning in every task (median difference 0). It is the ATB vs OD peak, it sits on the ATB vs LTB plateau, and it never collapses.
- Fixing alpha at 10 and reporting it as a limitation is supported by these data. Your decision.

## 2. Consensus-selection failure (headline)

Files:
- `Table_consensus_failure.csv`: one row per task × fold, publication-structured.
- `consensus_panel_provenance.csv`: every panel gene, with its home pool, seed status and MI rank in each training cohort.
- `consensus_pool_pairwise.csv`

**Definitions**
- Depth is measured on each training cohort's mutual-information ranking over the full task universe. The mRMR ranking within each MI top-2000 is reported too.
- "First gene in ≥ 2 lists" is the smallest k at which any gene sits in the top k of two training cohorts. "In all lists" is the same for every training cohort.
- The null comes from independent random rankings of the same number of genes (1,000 simulations), with the same thresholds.

| Task (folds) | Top-20 pool overlap, any pair | MI depth, ≥ 2 lists (null median) | MI depth, all lists (null median [95%]) | Top-2000 genes in all lists (expected if independent) | Panel genes with no MI top-100 support elsewhere, excluding seed genes |
|---|---|---|---|---|---|
| ATB vs LTB (5) | 0–3 | 2–6 (34), p ≤ 0.014 | **15–62** (984 [403–1530]), p < 0.001 | 257–440 (11.6) | 13–17 / 20 |
| ATB vs OD (4) | **0** in every pair | 10–26 (64), p 0.02–0.11 | 82–390 (615 [208–1090]), p 0.003–0.16 | 86–105 (24.1) | 16–19 / 20 |
| LTB vs CON (4) | **0** in every pair | 11–61 (51), p 0.03–0.63 | **209–734** (439 [146–794]), p 0.08–0.95 | **63–76 (64.6)** | 17–18 / 20 |

**Findings**
- **The zero-overlap finding is confirmed.** In ATB vs OD and LTB vs CON, no two training cohorts' top-20 shortlists share a gene, in any fold.
- **ATB vs LTB does overlap,** at 0–3 genes per pair. Its rankings agree far beyond chance: a gene enters every cohort's list by depth 15–62, against a random median of 984.
- **In LTB vs CON the per-cohort rankings agree no more than random rankings would.**
  - The depth to a gene shared by all cohorts sits inside the random-ranking range in every fold.
  - The number of genes in every cohort's top-2000 is 63–76, against 64.6 expected by chance.
  - So no gene-level LTB vs CON signal is shared across cohorts for the consensus step to find.
- **ATB vs OD sits in between.** There are about 4× more genes in every top-2000 than chance (86–105 vs 24), but the shared-by-all depth is within the random range in 2 of 4 folds.
- **Where the final panel genes come from:**
  - Seed genes (top-5 by summed rank): 1–5 per panel.
  - Genes in two or more pools: 0–3 per panel in ATB vs LTB, 0 elsewhere.
  - Share of panel genes ranked within the MI top-100 of another training cohort: 27% in ATB vs LTB, 6% in ATB vs OD, **0%** in LTB vs CON.
  - **Forward selection with no cross-cohort support (top-100) supplied 13–19 of the 20 genes in every fold.**
  - Even with a loose top-2000 threshold, 7–16 genes per panel have no support in any other training cohort.

## 3. The GSE19439 fold (LTB vs CON)

### Relatedness

`cohort_relatedness.csv`, all pairs in every task, on genes measured in every cohort.

| Pair | Platform | Mean-profile Spearman | Cross- minus within-cohort sample r | LTB vs CON signal Spearman | Top-200 signal Jaccard |
|---|---|---|---|---|---|
| **GSE19439–GSE19444** | same (GPL6947) | **0.977** | −0.021 | 0.089 | 0.008 |
| Other LTB vs CON pairs | different | 0.52–0.68 | −0.31 to −0.46 | 0.014–0.174 | 0.003–0.026 |
| GSE37250–GSE42834 (different studies) | same (GPL10558) | 0.976 | −0.022 | — | — |

- **Overall expression.** The siblings are much closer than any other LTB vs CON pair, but no closer than two *unrelated* studies on the same platform (GSE37250–GSE42834). Overall expression similarity here is a platform effect, not a sign of sibling status.
- **LTB vs CON signal.** The siblings are **not** more alike than other pairs (Spearman 0.09, within the 0.01–0.17 range).
- **No duplicate samples.**
  - The highest cross-cohort sample correlation is 0.959, lower than the highest within-cohort correlations.
  - Every pair, siblings included, is perfectly separable by cohort-of-origin classifiers (AUC 1.0). There is a batch difference even between the siblings.
- **GEO metadata.** Both sub-series are London cohorts from the same submitter, deposited a day apart.

**So relatedness alone does not show that the result is sibling transfer.**

### Sibling-exclusion test

`sibling_exclusion.csv`. GSE19439 held out, GSE19444 removed from training, so the models train on GSE28623 and GSE107994 only.

| Panel | LR | MLP_alpha | singscore | DirAvg | Random panels ≥ |
|---|---|---|---|---|---|
| Stage5 Native panel (selected **with** GSE19444), retrained **without** it | 0.78 | 0.78 | 0.77 | 0.74 | 0.00–0.02 |
| Native panel **re-selected without** GSE19444 | 0.71 | 0.65 | 0.66 | 0.63 | 0.08–0.48 |
| Stage5 (GSE19444 in selection and training) | 0.76 | 0.77 | 0.76 | 0.78 | 0.00–0.04 |

- **What the Stage5 panel's 20 genes keep.** Retraining without the sibling barely changes the result, so the models don't need GSE19444 as training data. What the result does depend on is GSE19444 having been part of panel *selection*.
- **What happens without it.** Without GSE19444 in selection, the result drops and becomes compatible with random panels for RF, MLP_alpha, singscore and DirAvg.
- **The CIs are wide (n = 29).** For example, re-selected LR is 0.71 [0.49–0.88] against 0.78 [0.59–0.93], so the drop is suggestive rather than significant.
- **Where the contribution comes from (`sibling_gene_origin.csv`).** It is not simply GSE19444's own genes. The 9 genes from GSE19444's pool and the 11 from the other pools each score 0.68–0.75 on GSE19439 when trained without the sibling. GSE19444's contribution runs through the forward-selection objective, which picked a gene combination that transfers within the London study.
- **The mirror fold sees no comparable effect.** Holding out GSE19444 without GSE19439 gives 0.47–0.52 with a re-selected panel and 0.58–0.65 with the Stage5 panel. That fold was never positive.

**Flag (added to `results/fold_flags.csv`).** Treat the GSE19439 fold's LTB vs CON result as dependent on the sibling sub-series being in panel selection, not as cross-study generalisation.

## Other flag: GSE83456 input values are gene-median-centred

**Unexpected; affects earlier stages.** GSE83456's expression file holds GEO's per-gene median-centred log values:
- gene means are about 0, and values run −8.8 to +7;
- this is noted in the Clean_rebuild README;
- the median within-cohort sample correlation is only 0.05.

**What is affected and what isn't**
- **Unaffected:** methods using per-cohort z-scores (LR, RF, MLP, DirAvg, and Native selection). Per-gene centring makes no difference to them.
- **Affected:**
  - **singscore with GSE83456 held out** (ATB vs OD, Stages 3–6). Within-sample ranks of centred values measure deviation from the series median, not expression level, so singscore there isn't comparable with other cohorts.
  - The Stage5 all-gene global-shift diagnostic for GSE83456.
  - The absolute-level relatedness measures in pairs involving GSE83456. Those rows show near-zero or negative profile correlations for this reason.

**Option, your call:** rebuild GSE83456 from its non-normalised supplementary file, which is already in `Clean_rebuild/geo_raw`, so singscore can be rerun on it.

## Files (`results/`)

- `alpha_extended.csv`, `alpha_extended_long.csv`
- `Table_consensus_failure.csv`, `consensus_panel_provenance.csv`, `consensus_pool_pairwise.csv`, `rank_reproduces_stage5_pools.csv`
- `rank/` (full per-cohort rankings)
- `cohort_relatedness.csv`
- `sibling_exclusion.csv`, `sibling_gene_origin.csv`
- `fold_flags.csv`
- `summary_tables.md`
