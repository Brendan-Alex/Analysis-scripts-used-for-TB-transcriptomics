# Stage2: verification and data audit (19 Oct 2026)

New folder; nothing that already existed was changed. Everything ran locally. No models and no figures.

New GEO downloads are checksum-verified in `geo_raw/SHA256SUMS.txt`:
- the GSE107994 edgeR xlsx (33 MB) and its series matrix;
- series matrices for GSE37250, GSE28623, GSE73408 and GSE83456;
- `GSE19491_RAW.tar` (263 MB, one raw file per GSM; file names start with the GSM).

## 1. Full verification against GEO (`qc/full_verification.csv`, `logs/01b_verify_gse107994.log`)

Every value of every rebuilt gene × sample was compared with GEO. Membership and labels were re-derived from each series matrix's own characteristics, an independent copy of the GEO metadata.

| Cohort | Values compared | Differences > 1e-4 | Max \|diff\| | Membership | Label mismatches | Probe choice re-derived |
|---|---|---|---|---|---|---|
| GSE37250 | 8,085,672 | 0 | 5.4e-6 | identical (263) | 0 | 0 mismatches |
| GSE28623 | 2,126,736 | 0 | 5.9e-6 | identical (108) | 0 | 0 |
| GSE73408 | 2,202,672 | 0 | 5.5e-6 | identical (109) | 0 | 0 |
| GSE83456 | 4,751,060 | 0 | 5.0e-6 | identical (155) | 0 | n/a (ranked on non-normalised data) |
| GSE107994 | 1,819,656 | 0 | 5.5e-6 | identical (127) | 0 | – |

Max differences are float32 / 5-decimal rounding.

GSE107994 details:
- The local CSV equals GEO's xlsx for all 175 samples × 14,518 genes (max diff 5e-9).
- Your 152-sample list matches GEO's `group` for 152/152.
- **The 19 date-corrupted symbols are already in GEO's own xlsx** (Excel datetime cells), so the corruption is not from your processing. Clean_rebuild recovered them from Ensembl IDs.

**No discrepancies in values, sample membership or labels.**

## 2. GSE19491 rebuilt from raw (`expression/GSE19491_log2_genelevel_jointnorm.csv.gz`, 24,903 genes × 213)

**What the raw files are.** One background-subtracted, unnormalised intensity file per GSM. GEO's processed values are these times a per-sample scale factor (r = 1.000). The factors differ by sub-series: 0.93, 1.23, 0.96 and 0.95 in the samples checked.

So GEO's processing was **scaled**, not centred, per sub-series. This differs slightly from your description, but the remedy is the same.

**Method.** The same as the Clean_rebuild GSE42834 rebuild: log2(max(x,1)), then quantile normalisation across the 213 included samples together.
- Probe per gene: highest mean raw log2. This differs from Clean_rebuild for 18 of 24,897 genes.
- Probe ILMN_2038777 was dropped: it appears twice, with different values, in 13 GSE22098 files.
- GSM484599's file carries 16,732 blank padding rows. They were dropped and the file is otherwise complete.

**Offset metric.** The per-sample mean over all genes, with separation measured as AUC; 0.5 = none (`qc/GSE19491_offset_metrics.csv`).

| Comparison | Before (GEO values) | After (joint normalisation) |
|---|---|---|
| Kruskal–Wallis across the 5 sub-series | H = 169.4 (p = 1e-35) | H = 94.5 (p = 1e-19) |
| GSE22098 (all OD) vs GSE19435 / 19439 / 19444 (London) | 1.000 / 1.000 / 1.000 | 0.688 / 0.884 / 0.539 |
| GSE22098 vs GSE19442 (South Africa) | 0.575 | 0.710 |
| GSE22098 vs all other sub-series | 0.810 | 0.530 |
| ATB vs OD, class AUC of the sample mean | 0.858 | 0.582 |
| ATB vs LTB, class AUC of the sample mean | 0.680 | 0.628 |

**Caveat: this metric is partly tautological after quantile normalisation.** Quantile normalisation makes every sample's probe distribution identical (all-probe mean sd 0.013). What's left in the gene-level mean is the composition of which probes are high, not intensity.

**A non-tautological check, with disease held constant.** The 39 GSE22098 adult healthy controls (normalised jointly for this diagnostic only) were compared with the 24 London controls (`qc/GSE19491_batch_check_controls.csv`):
- sample-mean AUC 1.000 → 0.660;
- genes differing at FDR < 0.05: 14,720 → 7,624;
- of those, genes differing more than 2-fold: 8,478 → 2,477.

**The global intensity offset is largely gone, but a large gene-specific sub-series batch remains.** In GSE19491 it is fully confounded with OD, so ATB vs OD in this cohort cannot be separated from batch by normalisation.

## 3. Age-filter audit (`qc/age_audit_by_cohort.csv`, `qc/age_per_sample.csv`)

**There is no sample-by-sample under-16 filter in any of your files.** I searched every script, the R fetch script, Tables.docx and Supplementary.docx; neither the old cohort files nor Clean_rebuild filter on age.

| Cohort | Per-sample age in GEO | GEO samples < 16 | < 16 in your files | Excluded by an age rule |
|---|---|---|---|---|
| GSE19491 | yes (498/498) | 121 | 0 | 0 (excluded by group choice) |
| GSE107994 | yes, age at baseline (min 16) | 0 | 0 | 0 |
| GSE19439 / GSE19444 | yes (min 19 / 18) | 0 | 0 | 0 |
| GSE73408 | yes (min 20) | 0 | 0 | 0 |
| GSE83456 | yes (min 18) | 0 | 0 | 0 |
| GSE37250 | **no** (series text says adults) | – | – | – |
| GSE28623 | **no** (no age wording at all) | – | – | – |
| GSE42834 | **no** (no age wording) | – | – | – |
| GSE101705 | **no** (no age wording) | – | – | – |

All 121 under-16s are in GSE22098 groups that are excluded as whole groups:

| Group | Age range | Under 16 |
|---|---|---|
| paediatric SLE | 7–19 | 40 of 82 |
| Staph | 0–17 | 39 of 40 |
| Strep | 1–16 | 11 of 12 |
| Strep/Staph controls | 4–18 | 21 of 23 |
| pSLE controls | 9–18 | 10 of 19 |

The outcome resembles an age filter, but it isn't one. **A true per-sample filter would differ: 44 GSE22098 disease samples are ≥ 16** (42 paediatric SLE, 1 Staph, 1 Strep), and your files exclude them too.

## 4. Still's and SLE ages (`qc/GSE19491_OD_ages.csv`)

| Group | n | Median (IQR) | Range | Survive the < 16 filter |
|---|---|---|---|---|
| Adult-onset Still's | 31 | 38 (28–54) | 17–83 | **31** |
| Adult SLE | 28 | 36 (28.75–49) | 18–79 | **28** |

**Correction:** Clean_rebuild's README calls the Still's samples paediatric. That's wrong; they are adults. An autoimmune-OD arm would have 59 adults, but every one is in GSE22098, which carries the batch shown in §2.

## 5. OD composition from GEO metadata (`qc/OD_composition.csv`)

| Cohort | OD diseases (n) | Respiratory? |
|---|---|---|
| GSE37250 (83) | "other disease": GEO gives no per-sample diagnosis | **not determinable from GEO** |
| GSE73408 (39) | pneumonia 39 | yes |
| GSE42834 (71) | active sarcoidosis 41, lung cancer 16, pneumonia 14 | per the series ("similar pulmonary diseases"); no per-sample organ site |
| GSE83456 (49) | sarcoidosis 49 | organ site not given |

This does not support "predominantly respiratory and comparable":
- **Training OD:** GSE37250 is unspecified, GSE73408 is all pneumonia.
- **Test OD:** dominated by sarcoidosis (90 of 120), plus cancer and some pneumonia.
- GSE37250's composition would have to come from the paper, not GEO.

## 6. GSE19491 ATB/LTB by sub-series (`qc/GSE19491_class_by_subseries.csv`, `qc/GSE19491_tb_batch_checks.csv`)

| Sub-series | ATB | LTB | CON | Region |
|---|---|---|---|---|
| GSE19435 | 7 | 0 | 0 | London (longitudinal) |
| GSE19439 | 13 | 17 | 12 | London |
| GSE19442 | 20 | 31 | 0 | South Africa |
| GSE19444 | 21 | 21 | 12 | London |

- **54 of 61 ATB and all 69 LTB sit in sub-series that contain both classes.** The exception is GSE19435's 7 ATB, which have no LTB in their sub-series. This is a small class/batch confound for ATB vs LTB.
- **Large batch between sub-series even within the same class**, after joint normalisation, measured as genes at FDR < 0.05:

  | Comparison | Genes |
  |---|---|
  | CON, GSE19439 vs GSE19444 | 4,833 |
  | LTB, GSE19439 vs GSE19444 | 6,303 |
  | LTB, South Africa vs London | 7,954 |
  | ATB vs LTB biology within one sub-series, for scale | 2,247–4,861 |

  For ATB vs LTB this batch is balanced across classes, so it adds noise rather than confounding, apart from the 7 GSE19435 ATB.
- As noted before, GSE19439 and GSE19444 are also the LTB vs CON training cohorts. Their samples sit inside GSE19491.

## Scripts

`01_full_verify.py <GSE…>`, `01b_verify_gse107994.py`, `02_gse19491_raw_rebuild.py`, `02b_offset_metrics.py` (uses `_diag_matrix.py`), `03_age_audit.py`, `04_gse19491_tb_subseries.py`.
