# Stage8: working figures

- **Format:** PNG at 150 dpi, built only from Stage7 outputs (no refitting). Not publication-formatted.
- **Code:** `scripts/figs.py` draws everything (`python scripts/figs.py 1 2 3 4 5 6 7 8`).
- **Random-panel distributions:** `scripts/00_extract_random.py` extracts the 50 draws per cell from the Stage7 benchmark records into `data/random_auc_long.csv`. Their medians are checked against the master table.
- **Captions:** every figure has a `*_caption.txt` beside it saying what it shows, the key numbers and the source files.

| File | Content |
|---|---|
| `Fig1a/b/c_*_curated_vs_random.png` | One file per task. Every curated panel × fold × method: AUC with 95% CI over the random-panel distribution. Filled = clean; open diamond = derivation overlap; open square = GSE19439 sibling flag; * = clean and beats random. |
| `Fig2_beats_random_summary.png` (+ `Fig2_data.csv`) | Share of clean cells beating random per panel. Tick labels give clean folds with ≥3/5 methods beating random, and all-cells counts. |
| `Fig3_ATBvsLTB_panel_size.png` | ATB vs LTB: curated vs random median by panel size, and the advantage per panel. |
| `Fig4_consensus_failure.png` (+ `Fig4_pairwise_overlap_data.csv`) | Top-20 shortlist overlap between cohorts, and depth to the first gene shared by all training cohorts vs a random-ranking null. |
| `Fig5_native_instability.png` | Gene × fold presence of Native panel genes, with olfactory receptor / microRNA / pseudogene flags. |
| `Fig6_generalisation_gap.png` | Training (pooled resubstitution) vs held-out AUC by method. |
| `Fig7_sex_genes.png` | X/Y genes entering Native panels when allowed, and sex balance by class per cohort. |
| `Fig8_global_shift.png` | All-gene-mean AUC per cohort, with quantile-normalised cohorts marked. |

**Note on "cells":** a cell is one panel × one held-out fold × one method, i.e. one row of `Stage7/results/MASTER_results_table.csv`. The 5 methods within a fold share data, so Fig2 also gives a fold-level count.
