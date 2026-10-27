# Cross-cohort benchmarking of tuberculosis blood transcriptomic classifiers

Analysis code for Alex, Patel & Tiwari, *Cross-cohort benchmarking and validation reveals
task-dependent generalisability in tuberculosis blood transcriptomic classification*.

Ten public GEO cohorts, three diagnostic tasks (ATB vs LTB, ATB vs OD, LTB vs CON), rotating
leave-one-cohort-out validation over 13 folds, six gene panels scored by five methods, and two
external validation cohorts that take no part in derivation.

---

## 1. Get the data

The scripts do not download the source expression files. They are archived on Zenodo:

**https://doi.org/10.5281/zenodo.18923347**

Download, extract, and point the code at it:

```bash
export TB_DATA_ROOT=/path/to/extracted/zenodo     # Linux / macOS
setx TB_DATA_ROOT "C:\path\to\extracted\zenodo"   # Windows
```

If `TB_DATA_ROOT` is unset the scripts look in `<repo>/inputs`, so extracting the archive to
`inputs/` inside the repository also works.

Then run the one-time setup, which verifies the download against `SHA256SUMS.txt` and unpacks the
single file that ships as a tar (`GSE19491_RAW.tar`, needed by Stage2 and Stage3):

```bash
python setup_inputs.py
```

## 2. Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Everything in the manuscript was produced with **Python 3.14.7, numpy 2.5.4, pandas 3.0.6,
scipy 1.18.1, scikit-learn 1.9.1** on Windows 11 (12 CPU, 10 workers).

A fixed seed of 42 is used throughout, but seeded reproducibility only holds *within* a
scikit-learn version: `mutual_info_classif`, `MLPClassifier` and `RandomForestClassifier` have all
changed numerical behaviour between releases. Running on a different version may shift the selected
genes and the reported AUCs. Pin the versions above if you intend to reproduce the published numbers.

## 3. Run order

Each stage reads the outputs of earlier stages and writes only inside its own folder. Run them in
order from inside the stage directory. Timings are from the original runs with `--workers 10`.

| # | Stage | Command | Time | Produces |
|---|---|---|---|---|
| 1 | Clean_rebuild | `python scripts/01_parse_geo.py` … `08_verify.py` in filename order | ~20 min | per-cohort log2 gene-level expression, `sample_manifest.csv` |
| 2 | Stage2 | `python scripts/01_full_verify.py` then `02`, `02b`, `03`, `04` | ~30 min | value-level verification against GEO (optional but recommended) |
| 3 | Stage3 | `python scripts/00_build_task_data.py` then `01`–`07` | ~25 min | task datasets, global-shift and per-gene separability diagnostics |
| 4 | Stage4 | `python scripts/00_prepare.py` then `01`–`05` | ~40 min | first full LOCO benchmark |
| 5 | Stage5 | `python scripts/run_stage5.py --workers 10` then `summarise_stage5.py`, `tables_stage5.py` | 34 min | benchmark caches, Native panels, curated and random panel results |
| 6 | Stage6 | `python scripts/run_stage6.py --workers 10` then `summarise_stage6.py`, `tables_stage6.py` | ~35 min | per-cohort MI rankings, cohort relatedness |
| 7 | Stage7 | `python scripts/01_rebuild_gse83456.py`, `run_stage7.py --workers 10`, then `02`–`06` | 17 min | **authoritative benchmark** (MLP alpha fixed at 10, GSE83456 rebuilt) |
| 8 | Stage8 | `python scripts/00_extract_random.py` | ~2 min | random-panel AUCs in long form (needed by Stage16) |
| 9 | Stage9 | `python scripts/run_stage9.py --workers 10` then `summarise_stage9.py` | 33 min | is the Native panel noise; corrected selectors |
| 10 | Stage10 | `python scripts/run_stage10.py --workers 10` then `summarise_stage10.py` | 48 min | six-selector comparison (**S5 Table**) |
| 11 | Stage11 | `python scripts/run_stage11.py --workers 10` then `summarise_stage11.py` | 47 min | panel-size ablation, chosen k per fold, final signatures (**S4, S6 Tables**) |
| 12 | Stage12 | `python scripts/run_stage12.py` | ~15 min | downloads and filters GSE144127 and GSE152218. **Needs internet** |
| 13 | Stage13 | `python scripts/run_stage13.py --workers 10` | ~10 min | external validation (**S7, S9, S10 Tables**) |
| 14 | Stage15 | `python scripts/run_stage15.py` | ~5 min | PCA of cohort structure (**Fig 8**) |
| 15 | Stage16 | `python scripts/run_stage16.py` | ~20 min | **all publication figures** and the paired DeLong table (**S8 Table**) |
| 16 | Stage17 | `python scripts/make_supplementary_tables.py` | <1 min | supplementary tables S1–S10 as markdown |
| 17 | Stage18 | `python scripts/run_stage18.py` | ~5 min | validation-cohort independence check |

Total around four hours on 10 cores, dominated by Stages 9–11.

**Stage12 caveat.** NCBI occasionally serves a reCAPTCHA page instead of a file. The script detects
this and prints manual download instructions; fetch the named file by hand into `Stage12/geo/` and
re-run. This is an NCBI rate limit, not a bug.

## 4. What each script does

### Clean_rebuild — build the expression matrices from source
| script | purpose |
|---|---|
| `01_parse_geo.py` | parse the GEO SOFT download into one row per (cohort, GSM) |
| `02_match_raw_to_gsm.py` | link non-normalised supplementary columns (no GSM IDs) to GSMs |
| `02b_cross_cohort_dups.py` | identical arrays shared between GSE42834 sub-series and GSE83456 |
| `02c_gse19491_duplicates.py` | duplicate arrays inside the GSE19491 SuperSeries |
| `03_build_manifest.py` | `sample_manifest.csv`: class, inclusion and exclusion reason for all 2,125 samples |
| `04_build_expression.py` | per-cohort gene-level log2 expression for included samples |
| `05_compare_probe_selection.py` | how many genes change probe against the older selection (diagnostic) |
| `06_panels.py` | HGNC-map the six gene panels and report per-cohort coverage |
| `07_merge_dryrun.py` | data check: exercise the merge on each task's training cohorts |
| `08_verify.py` | independent checks of the rebuild; fails loudly |
| `hgnc_alias.py`, `tb_data.py` | shared HGNC harmonisation and data access |

### Stage2 — verify the rebuild against GEO
| script | purpose |
|---|---|
| `01_full_verify.py` | every value of every sample re-derived from GEO and compared |
| `01b_verify_gse107994.py` | GSE107994 against the GEO edgeR workbook |
| `02_gse19491_raw_rebuild.py` | GSE19491 rebuilt from per-sample raw files, sub-series normalised together |
| `02b_offset_metrics.py` | does the per-sample mean separate sub-series before and after joint normalisation |
| `03_age_audit.py` | per-sample age from GEO metadata against what the files include |
| `04_gse19491_tb_subseries.py` | GSE19491 class composition and batch by sub-series |

### Stage3 / Stage4 — apply the inclusion decisions, first benchmark
| script | purpose |
|---|---|
| `Stage3/00_build_task_data.py` | apply the decisions and build the three task datasets |
| `Stage3/01_global_shift.py` | per-sample all-gene mean scored against class (global shift) |
| `Stage3/02_native_select.py` | Native panel re-derived with the original algorithm |
| `Stage3/03_benchmark.py` – `07_sex_balance.py` | benchmark and diagnostics (rank scoring, per-gene separability, sex balance) |
| `Stage4/00_prepare.py` – `05_tables_md.py` | label-free per-cohort z-scoring, Native per fold, rotating LOCO, canary, tables |

### Stage5 / Stage6 / Stage7 — the benchmark
| script | purpose |
|---|---|
| `Stage5/run_stage5.py` | rotating LOCO benchmark; builds the caches every later stage imports read-only |
| `Stage6/run_stage6.py` | per-cohort MI rankings and cohort relatedness |
| `Stage6/sibling_gene_origin.py` | which genes carry the GSE19439 / GSE19444 sibling relationship |
| `Stage7/01_rebuild_gse83456.py` | rebuild GSE83456 at absolute expression level from the GEO non-normalised file |
| `Stage7/run_stage7.py` | the authoritative benchmark: MLP alpha fixed at 10, rebuilt GSE83456 |
| `Stage7/02`–`06` | summaries, ATB vs OD relatedness, final tables, generalisation gap, sex balance |
| `Stage7/run_stage7_withsex.py` | supplementary: Native panels with X/Y-linked genes allowed |
| `Stage8/00_extract_random.py` | random-panel AUCs in long form, read by Stage16 |

### Stage9 / Stage10 / Stage11 — why the mRMR selector failed, and how many genes are needed
| script | purpose |
|---|---|
| `Stage9/run_stage9.py` | is the Native panel noise, and does a corrected selector transfer |
| `Stage10/run_stage10.py` | the original pipeline with mRMR removed; isolates the cause of the failure |
| `Stage11/run_stage11.py` | ablation from 20 genes to 1, panel size chosen without the held-out cohort, final signatures |

### Stage12 / Stage13 — external validation
| script | purpose |
|---|---|
| `Stage12/run_stage12.py` | download, independence-check, filter and normalise GSE144127 and GSE152218 |
| `Stage13/run_stage13.py` | locked panels applied without refitting; subgroups, direction, leakage canary |
| `Stage13/delong_stage13.py` | **superseded.** The DeLong table used in the paper is produced by Stage16 with six comparators (Berry86 −TLR7 dropped). Kept for provenance |

### Stage15 – Stage18 — figures, tables, checks
| script | purpose |
|---|---|
| `Stage15/run_stage15.py` | PCA of cohort structure per task, before and after z-scoring |
| `Stage16/run_stage16.py` | all publication figures to PLOS ONE spec, plus the paired DeLong table |
| `Stage17/make_supplementary_tables.py` | supplementary tables S1–S10 from saved outputs; no refitting |
| `Stage18/run_stage18.py` | sample-level independence of the validation cohorts from the discovery cohorts |

## 5. Notes

- **One script cannot be re-run.** `Clean_rebuild/05_compare_probe_selection.py` compares probe
  selection against a 1.8 GB superseded copy of the data that is not deposited. It is a diagnostic
  that feeds no result in the paper; its output is archived as `derived/qc/probe_selection_*.csv`.
  Everything else runs from the Zenodo archive alone.
- **Not included.** Stage14 (an earlier figure set superseded by Stage16) and several exploratory
  folders that produced nothing in the manuscript. Every script here contributes to a reported
  number, table or figure, or verifies one that does.
- **Nothing is averaged across folds.** Results are reported per fold throughout. Several scripts
  assert this.
- **Leakage canaries.** Stages 4, 5, 9, 10, 11 and 13 each re-run a step with the held-out or
  validation cohort replaced by noise and require the selection to be unchanged.
- **Stage folders write only inside themselves.** Later stages import earlier ones read-only, so a
  stage can be re-run without invalidating what came before.

## 6. Citation

If you use this code or the archived data, please cite the paper and the Zenodo record.
