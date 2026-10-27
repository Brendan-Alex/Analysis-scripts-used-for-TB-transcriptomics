"""
HGNC symbol harmonisation used by the corrected rerun.

Resources (bundled in ./resources, provenance written to the mapping CSVs):
  1. HGNChelper hgnc.table (waldronlab/HGNChelper, data/hgnc.table.rda,
     package v0.8.15, map updated 2024-11-16). Symbol -> Approved.Symbol for
     every approved, previous and alias symbol in the HGNC complete set.
     Primary map.
  2. HGNC custom export bundled with INDRA (sorgerlab/indra,
     indra/resources/hgnc_entries.tsv, updated 2024-02-04). Used only to label
     each mapping as "previous symbol" vs "alias symbol" and to attach HGNC IDs.

Rules (applied identically to panel genes and cohort gene names):
  - normalise: strip, remove internal whitespace, upper-case, drop ".suffix"
    (same as the original normalize_gene, plus whitespace removal for "SEPTIN 1")
  - if the normalised name is a current approved symbol (case-insensitive) -> keep it
  - else if it maps to exactly one approved symbol -> map
  - else if it maps to >1 approved symbol: map only if exactly one of them lists it
    as a *previous* symbol (HGNC export); otherwise leave unmapped ("ambiguous")
  - else leave unchanged ("not_in_HGNC")
  Matching keys are upper-case approved symbols.
"""
from pathlib import Path
import re
import warnings

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "resources"   # Clean_rebuild/resources (copied from Rerun_corrected)


def normalize_gene(g):
    g = re.sub(r"\s+", "", str(g)).upper()
    return g.split(".")[0]


class HGNCMapper:
    def __init__(self):
        import rdata
        warnings.filterwarnings("ignore", module="rdata")
        t = rdata.conversion.convert(rdata.parser.parse_file(str(RES / "hgnc.table.rda")))["hgnc.table"]
        t = t.astype(str)
        self.approved = {}          # UPPER -> official case
        for s in t.loc[t["Symbol"] == t["Approved.Symbol"], "Approved.Symbol"]:
            self.approved[s.upper()] = s
        self.sym2app = {}
        for s, a in zip(t["Symbol"], t["Approved.Symbol"]):
            self.sym2app.setdefault(s.upper(), set()).add(a.upper())
        for a in self.approved:                  # every approved symbol maps to itself
            self.sym2app.setdefault(a, set()).add(a)

        h = pd.read_csv(RES / "hgnc_entries.tsv", sep="\t", dtype=str).fillna("")
        h = h[h["Status"] == "Approved"]
        self.hgnc_id, self.prev_of, self.alias_of = {}, {}, {}
        for _, r in h.iterrows():
            a = r["Approved symbol"].upper()
            self.hgnc_id[a] = r["HGNC ID"]
            for p in filter(None, (x.strip().upper() for x in r["Previous symbols"].split(","))):
                self.prev_of.setdefault(p, set()).add(a)
            for p in filter(None, (x.strip().upper() for x in r["Alias symbols"].split(","))):
                self.alias_of.setdefault(p, set()).add(a)
        self._cache = {}

    def map(self, raw):
        """Return dict(normalised, current, status, candidates)."""
        n = normalize_gene(raw)
        if n in self._cache:
            return self._cache[n]
        cands = sorted(self.sym2app.get(n, set()))
        if n in self.approved:
            res = dict(normalised=n, current=n, status="approved", candidates=n)
        elif len(cands) == 1:
            c = cands[0]
            kind = ("previous_symbol" if c in self.prev_of.get(n, set())
                    else "alias_symbol" if c in self.alias_of.get(n, set())
                    else "HGNChelper_map")
            res = dict(normalised=n, current=c, status=kind, candidates=c)
        elif len(cands) > 1:
            prev = [c for c in cands if c in self.prev_of.get(n, set())]
            if len(prev) == 1:
                res = dict(normalised=n, current=prev[0], status="previous_symbol(resolved_ambiguity)",
                           candidates=";".join(cands))
            else:
                res = dict(normalised=n, current=n, status="ambiguous_unmapped", candidates=";".join(cands))
        else:
            res = dict(normalised=n, current=n, status="not_in_HGNC", candidates="")
        res["official_case"] = self.approved.get(res["current"], "")
        res["hgnc_id"] = self.hgnc_id.get(res["current"], "")
        self._cache[n] = res
        return res

    def current(self, raw):
        return self.map(raw)["current"]
