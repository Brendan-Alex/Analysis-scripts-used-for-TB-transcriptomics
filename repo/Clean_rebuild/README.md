# Clean_rebuild: Tier 1 data rebuild (19 Oct 2026)

Nothing outside this folder was modified. Everything ran locally on your computer's Linux workspace (2 cores, 3 GB, so `n_jobs=-1` = 2). No models were trained and no figures were made.

**Core rule.** Every sample is selected and labelled by GSM, using GEO metadata downloaded from GEO (`geo_raw/`, checksums in `SHA256SUMS.txt`). Titles, column positions and row order are never used to decide class or inclusion. Where GEO's expression file has no GSM headers, columns are linked to GSMs as follows:
- **GSE107994, GSE101705:** by GEO's own title → GSM field, asserted 1:1.
- **GSE42834, GSE83456 non-normalised files:** by expression identity against the GSM-labelled GEO matrix (`qc/raw_column_to_GSM.csv`).

## Result

| Cohort | GEO | Included | ATB | LTB | CON | OD |
|---|---|---|---|---|---|---|
| GSE19491 | 498 | 213 | 61 | 69 | 24 | 59 |
| GSE37250 | 537 | 263 | 97 | 83 | – | 83 |
| GSE73408 | 109 | 109 | 35 | 35 | – | 39 |
| GSE19439 | 42 | 42 | 13 | 17 | 12 | – |
| GSE19444 | 54 | 54 | 21 | 21 | 12 | – |
| GSE28623 | 108 | 108 | 46 | 25 | 37 | – |
| GSE101705 | 44 | 44 | 28 | 16 | – | – |
| GSE107994 | 175 | 127 | 34 | 44 | 49 | – |
| GSE42834 | 356 | 219 | 35 | – | 113 | 71 |
| GSE83456 | 202 | 155 | 45 | – | 61 | 49 |

`sample_manifest.csv` lists all 2,125 GEO samples. For each it gives the class, inclusion status, exclusion reason, the GEO fields used (`geo_basis`), any duplicate-array partner, and whether the GSM also appears in another cohort's file.

## Decisions (yours, 19 Oct, unless marked "mine")

- **GSE107994** (127 = 34 ATB / 44 LTB / 49 CON). Every label matches GEO (127/127).
  - Starts from your 152-GSM list, so all 23 progressor samples are out.
  - The 8 progressor baselines that the old z-file had as LTB: GSM2886275, GSM2886412, GSM2886422, GSM2886429, GSM2886432, GSM2886435, GSM2886441, GSM2886447.
  - 17 samples GEO flags `outlier: Yes` are excluded.
  - 8 more non-outlier extrapulmonary ATB are excluded (EPTB rule below).
  - S14, S43, S123, S131 and S176 are all in GEO with `outlier: No` and normal quality, so they are included.
- **EPTB is excluded wherever GEO annotates TB site:** GSE107994 (8 non-outlier) and GSE83456 (47). The other cohorts don't annotate site.
- **Longitudinal sub-series:** pre-treatment samples are kept and on-treatment samples dropped:
  - GSE19435: 7 pre-treatment ATB kept, 14 on-treatment ATB dropped;
  - GSE42831: 10 on-treatment sarcoid dropped;
  - GSE42827: 5 post-antibiotic pneumonia dropped.
- **Duplicate arrays (same array under two GSMs): one copy kept, and the longitudinal-series copy is the one dropped.**
  - GSE19491: the 12 GSE19435 controls are the same arrays as 12 GSE19444 controls (r = 1.000, constant scale 0.966), so the GSE19444 copies are kept.
  - GSE42834: 5 GSE42831 pre-treatment sarcoid samples are byte-identical to GSE42826 samples, and all 5 GSE42827 pre-treatment pneumonia samples are byte-identical to GSE42830 "Baseline" samples. The cross-sectional copies are kept, so the net addition is 2 sarcoid samples.
  - The third-party note on GEO about identical non-normalised arrays refers to these pairs. No duplicates exist between GSE42834 and GSE83456.
- **GSE19491 OD = 59** (Still's 31, adult SLE 28).
  - The 39 GSE22098 healthy controls are removed. They were identified by the GEO characteristic `healthy control: ASLE/Still`. The methods agree: across GSE22098 the metadata field and the `WB-H-*` titles pick out the same 81 controls, and the 39 old-OD controls are exactly the ASLE/Still subset.
  - GSE22098's other 42 controls and its paediatric SLE, Staph and Strep samples were never part of OD. Note that Still's disease here is paediatric (systemic JIA).
- **GSE42834 (219):**
  - Whole blood only: the 75 sorted-cell/PBMC samples are out.
  - Non-active sarcoidosis (22) excluded.
  - The 8 GSE42830 "Baseline" samples are OD (pneumonia): the series design lists 8 pneumonia patients, and 5 of them are byte-identical to GSE42827 pre-treatment pneumonia arrays.
  - **Rebuilt from GEO non-normalised data:** log2(max(x,1)), then quantile normalisation across the 219. GEO's own values were median-centred separately in each sub-series.
  - The 15 GSE42832 whole-blood samples are excluded because their non-normalised columns can't be linked to GSMs.
- **GSE37250:** HIV-negative only (274 HIV+ out).
- **GSE73408:** multi-gene transcript clusters go to the first gene of GPL11532 `gene_assignment`. This recovers GBP1, FCGR1A, TYROBP, SLC7A6, CERKL, GJA9 and PGA5.
- **Scale:** log2 gene-level only, with no z-scoring (z-score at training time on each task's final samples).
  - Illumina and Agilent linear intensities: log2(max(x,1)) (mine; the background-subtracted values include negatives).
  - RNA-seq counts: log2(x+1).
  - GSE107994 edgeR values and GSE73408 RMA values are already log2.
  - GSE83456 keeps GEO's values (log, median-centred once over the whole series, so no batch issue).

## Merge, probes, symbols

- **`scripts/tb_data.py`:** `load_and_merge` joins on a Gene index with an explicit inner join. It prints the shape after each merge and asserts:
  - unique genes, no duplicate columns, no NaN;
  - gene count = set intersection of the cohorts' genes;
  - sample count = manifest included count.

  It refuses to merge cohorts that share GSMs (GSE19491 contains every GSE19439 and GSE19444 sample). Dry run (`logs/07_merge_dryrun.log`):
  - ATB-task training: 18,468 genes × 585 samples;
  - LTB vs CON training: 17,890 genes × 204 samples.
- **Probe collapsing:** highest mean log2 expression on pre-normalisation values, over the cohort's included samples. GSE42834 and GSE83456 use GEO's non-normalised supplementary values for this ranking.
  - Against the old rule (highest mean z-score, i.e. noise; max |mean z| ≈ 1e-14), 53–57% of multi-probe genes change probe. Totals: 3,905 / 3,840 / 4,062 / 4,913 / 610 / 4,446 / 5,053 / 4,925 genes for GSE19439 / 19444 / 19491 / 37250 / 73408 / 28623 / 42834 / 83456 (`qc/probe_selection_vs_old.csv`).
- **HGNC:** the same mapper and resources as Rerun_corrected, applied to cohorts and panels (`qc/panel_gene_mapping.csv`, `panels/`, `qc/panel_coverage.csv`).
- **Date-mangled symbols:**
  - The local GSE107994 table had 19 symbols turned into dates by Excel (SEPT*/MARCH*, plus SEP15 → SELENOF and MARC1 → MTARC1). All 19 were recovered from their Ensembl IDs; SEPTIN4 and SEPTIN1 are back.
  - GEO's own GSE101705 file has 3 date cells plus a duplicated MARCH1, which can't be resolved. All 5 rows are dropped; none is a panel gene (`qc/date_corrupted_symbols.csv`).
- **Panel genes still missing** (genuinely absent from the platform or RNA-seq table):
  - GSE28623: FCGR1A (not on GPL4133). The GPL4133 map was checked identical to GEO's.
  - GSE107994: CCR6 and CALML4 were filtered out of the edgeR table.

## Verification (`qc/verification_report.txt`: PASS)

- **Labels:** 1,207 included non-GSE107994 labels match an independent GEO coding from 18 Oct with 0 mismatches, and the GSE107994 labels match the separately transcribed GEO table, 127/127.
- **Expression files vs manifest:** every expression file's columns equal the manifest's included GSMs, and all values are finite.
- **GEO value spot-check:** for 18 GSMs across 6 cohorts, the local matrix equals GEO's sample tables (identical row counts and sums, `qc/geo_value_spotcheck.csv`).
- **GSE101705:** the local author file is GEO's supplementary file with only the labels renamed.
- **Duplicates:** no duplicate array has both copies included.
- **Patients:** GSE107994 patients are unique.

## Open items for you

1. **GSE19491 has a per-sub-series intensity offset.**
   - Sample totals are 5.4–6.7 M in the London TB sub-series, against 10.2 M in GSE19442 (South Africa) and 9.5 M in GSE22098, which holds all of OD.
   - In log2 that is a near-constant shift that coincides with the OD class. This is likely the "sample-level global shift" found earlier.
   - Options: per-sample normalisation across GSE19491's included samples, or a rebuild from the sub-series' non-normalised files (needs more GEO downloads). Not changed.
2. **The GSE107994 local edgeR CSV is unverified.** It wasn't checked against GEO's `GSE107994_edgeR_normalized_…xlsx` (33 MB, not downloaded). The date corruption shows the CSV went through Excel.
3. **Local-file provenance.** GSE37250, GSE28623, GSE73408 and GSE83456 come from your local GSM-named tables. They were spot-checked (3 GSMs each), not fully re-downloaded.

## Scripts (run in order from `Clean_rebuild/`)
`01_parse_geo` → `02_match_raw_to_gsm` → `02b_cross_cohort_dups` → `02c_gse19491_duplicates` → `03_build_manifest` → `04_build_expression <GSE…>` → `05_compare_probe_selection <GSE…>` → `06_panels` → `07_merge_dryrun` → `08_verify`. `tb_data.py` is the loader for all later work.
