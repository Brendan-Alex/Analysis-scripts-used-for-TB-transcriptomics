"""One-time setup for the Zenodo data archive.

Verifies the download against SHA256SUMS.txt and unpacks the one file that ships as a tar.
Run once after extracting the archive:

    python setup_inputs.py                      # uses TB_DATA_ROOT, or ./inputs
    python setup_inputs.py --root /path/to/data
    python setup_inputs.py --skip-checksums     # faster; skips the 1.7 GB verification pass
"""
import argparse, hashlib, os, sys, tarfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("TB_DATA_ROOT"))
ap.add_argument("--skip-checksums", action="store_true")
a = ap.parse_args()
ROOT = Path(a.root) if a.root else Path(__file__).resolve().parent / "inputs"
if not ROOT.is_dir():
    sys.exit(f"not a directory: {ROOT}\nExtract the Zenodo archive first, then set TB_DATA_ROOT or pass --root.")
print(f"data root: {ROOT}")

TAR = ROOT / "inputs/geo_raw/GSE19491_RAW.tar"
DEST = ROOT / "inputs/geo_raw/GSE19491_RAW"
if not TAR.exists():                      # archive may have been extracted without its top folder
    TAR = ROOT / "geo_raw/GSE19491_RAW.tar"; DEST = ROOT / "geo_raw/GSE19491_RAW"
if DEST.is_dir() and any(DEST.glob("GSM*.txt.gz")):
    print(f"already unpacked: {DEST} ({len(list(DEST.glob('GSM*.txt.gz')))} files)")
elif TAR.exists():
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"unpacking {TAR.name} -> {DEST} ...")
    with tarfile.open(TAR) as t:
        t.extractall(DEST, filter="data") if sys.version_info >= (3, 12) else t.extractall(DEST)
    print(f"  {len(list(DEST.glob('GSM*.txt.gz')))} per-sample files")
else:
    print(f"WARNING: {TAR} not found. Stage2 and Stage3 need the GSE19491 per-sample raw files.")

sums = ROOT / "SHA256SUMS.txt"
if a.skip_checksums:
    print("checksum verification skipped")
elif not sums.exists():
    print(f"WARNING: {sums} not found; cannot verify the download")
else:
    bad = missing = n = 0
    for line in sums.read_text().splitlines():
        if not line.strip(): continue
        want, _, rel = line.partition("  ")
        f = ROOT / rel.strip()
        if not f.exists(): missing += 1; print(f"  MISSING {rel.strip()}"); continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 22), b""): h.update(blk)
        n += 1
        if h.hexdigest() != want: bad += 1; print(f"  CHECKSUM MISMATCH {rel.strip()}")
    print(f"verified {n} files; {bad} mismatched, {missing} missing")
    if bad or missing: sys.exit("download is incomplete or corrupt - re-download before running the pipeline")
print("\nready. Set TB_DATA_ROOT and follow the run order in README.md.")
