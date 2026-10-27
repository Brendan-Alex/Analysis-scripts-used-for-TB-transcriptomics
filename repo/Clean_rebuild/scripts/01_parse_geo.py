"""Step 1: parse the GEO SOFT (brief) download into one row per (cohort, GSM).
Input : geo_raw/GEO_metadata_SOFT_brief_10series.txt (checksums in geo_raw/SHA256SUMS.txt)
Output: qc/geo_samples_long.csv  (every !Sample_ field, characteristics split into key/value columns)
"""

# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
import re, sys, hashlib
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[1]
txt = open(DATA_ROOT / "geo_raw" / "GEO_metadata_SOFT_brief_10series.txt", encoding="utf-8", newline="").read()
sections = re.split(r"^#### GEO_SOFT_BRIEF (\S+) .*? sha256=(\w+)\n", txt, flags=re.M)
rows = []
for i in range(1, len(sections), 3):
    gse, sha, body = sections[i], sections[i + 1], sections[i + 2][:-1]   # drop the one separator newline
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == sha, f"checksum mismatch {gse}"
    for blk in re.split(r"^\^SAMPLE = ", body, flags=re.M)[1:]:
        lines = blk.replace("\r", "").split("\n"); gsm = lines[0].strip(); r = {"cohort": gse, "GSM": gsm}
        multi = {}
        for l in lines[1:]:
            if not l.startswith("!Sample_") or " = " not in l: continue
            k, v = l[8:].split(" = ", 1); v = v.strip()
            if k == "characteristics_ch1":
                if ":" in v:
                    ck, cv = v.split(":", 1); r["ch:" + ck.strip().lower()] = cv.strip()
                else:
                    r["ch:_unkeyed"] = (r.get("ch:_unkeyed", "") + "|" + v).strip("|")
            else:
                multi.setdefault(k, []).append(v)
        for k, v in multi.items(): r[k] = ";".join(v)
        rows.append(r)
G = pd.DataFrame(rows)
assert not G.duplicated(["cohort", "GSM"]).any()
G.to_csv(R / "qc" / "geo_samples_long.csv", index=False)
print(G.groupby("cohort").GSM.count().to_string())
print("columns:", len(G.columns))
