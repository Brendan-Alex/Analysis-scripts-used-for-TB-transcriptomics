"""Data access for the clean rebuild. Import this instead of reading cohort files directly.

load_cohort(gse)            -> (expr genes x GSM, labels Series GSM -> class) for the manifest's included samples
load_and_merge(cohorts)     -> genes x GSM matrix joined on gene symbol (index), explicit inner join
                               with assertions on uniqueness and on the expected gene and sample counts

Design notes
  * Columns are GSM IDs. A GSM may appear in more than one cohort file (GSE19491 is a SuperSeries that
    contains every GSE19439 and GSE19444 sample); merging two such cohorts fails the duplicate-column
    assertion on purpose instead of silently keeping one copy.
  * Expected counts are computed independently: samples from sample_manifest.csv, genes as the plain
    set intersection of the cohorts' gene indices.
"""
from functools import reduce
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_MAN = None

def manifest():
    global _MAN
    if _MAN is None:
        _MAN = pd.read_csv(ROOT / "sample_manifest.csv")
    return _MAN

def load_cohort(gse):
    X = pd.read_csv(ROOT / f"expression/{gse}_log2_genelevel.csv.gz", index_col="Gene")
    m = manifest(); m = m[(m.cohort == gse) & m.included].set_index("GSM")
    assert X.index.is_unique, f"{gse}: duplicate gene symbols"
    assert X.columns.is_unique, f"{gse}: duplicate sample columns"
    assert set(X.columns) == set(m.index), f"{gse}: expression columns != manifest included GSMs"
    return X, m.loc[X.columns, "class_label"]

def load_and_merge(cohorts, expected_genes=None, expected_samples=None, verbose=True):
    frames, labels = [], []
    for g in cohorts:
        X, y = load_cohort(g)
        if verbose: print(f"  loaded {g}: {X.shape[0]} genes x {X.shape[1]} samples")
        frames.append(X); labels.append(y)
    if expected_genes is None:
        expected_genes = len(reduce(set.intersection, (set(f.index) for f in frames)))
    if expected_samples is None:
        m = manifest(); expected_samples = int(m[m.cohort.isin(cohorts) & m.included].shape[0])
    merged = frames[0]
    for g, f in zip(cohorts[1:], frames[1:]):
        dup = merged.columns.intersection(f.columns)
        assert len(dup) == 0, f"sample columns present in more than one cohort: {list(dup)[:5]} ..."
        merged = merged.join(f, how="inner")              # join on the Gene index, explicit inner join
        if verbose: print(f"  after merging {g}: {merged.shape}")
    assert merged.index.is_unique, "gene index not unique after merge"
    assert merged.columns.is_unique, "duplicate sample columns after merge"
    assert merged.shape[0] == expected_genes, f"genes {merged.shape[0]} != expected {expected_genes}"
    assert merged.shape[1] == expected_samples, f"samples {merged.shape[1]} != expected {expected_samples}"
    assert merged.notna().all().all(), "NaN after inner join"
    y = pd.concat(labels); y = y.loc[merged.columns]
    cohort = pd.Series({s: g for g, f in zip(cohorts, frames) for s in f.columns}).loc[merged.columns]
    return merged, y, cohort
