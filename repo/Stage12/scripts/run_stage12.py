"""Stage12: independent validation cohorts GSE144127 (ATB vs OD) and GSE152218 (ATB vs LTB) - download, independence check,
characterisation, filtering, preprocessing. NO signature scoring. Run on your laptop (needs internet to ftp.ncbi.nlm.nih.gov):
    cd "<repo root>\\Stage12"   then   python scripts\\run_stage12.py
Outputs: REPORT.md (read this first), geo/ (downloads + SHA-256), tables/ (per-GSM metadata, filter decisions, class mapping),
expression/ (gene-level log2 matrices), logs/. Nothing outside Stage12 is modified. Every section is wrapped: an error is written into
REPORT.md and the script continues, so one run gives the full picture.
Rules (same as the clean rebuild): selection/labelling by GSM from GEO metadata; adult whole blood; untreated; HIV-negative; no EPTB;
no paediatric samples. Class/filter keyword rules are heuristics - every raw value -> decision is tabulated for review (tables/*mapping*.csv)."""
import os, re, sys, io, gzip, json, hashlib, time, traceback, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
S12 = Path(__file__).resolve().parents[1]; MT = S12.parent; CR = MT / "Clean_rebuild"
for d in ("geo", "tables", "expression", "logs"): (S12 / d).mkdir(exist_ok=True)
sys.path.insert(0, str(CR / "scripts")); from hgnc_alias import HGNCMapper
COH = {"GSE144127": "ATB_vs_OD", "GSE152218": "ATB_vs_LTB"}
EXISTING = ["GSE19491", "GSE19439", "GSE19442", "GSE19444", "GSE19435", "GSE19447", "GSE22098", "GSE37250", "GSE73408", "GSE101705", "GSE107994",
            "GSE42834", "GSE42825", "GSE42826", "GSE42827", "GSE42830", "GSE42831", "GSE42832", "GSE83456", "GSE28623"]
REP = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.append(s)
    with open(S12 / "logs/run.log", "a", encoding="utf-8") as fh: fh.write(time.strftime("%H:%M:%S ") + s + "\n")
def section(t): say(f"\n## {t}\n")
def stub(g): return g[:-3] + "nnn"
def fetch(url, dest, max_bytes=None):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        if open(dest, "rb").read(200).lstrip().lower().startswith((b"<!doctype", b"<html")):
            raise RuntimeError(f"{dest.name} is an HTML/reCAPTCHA page, not data. Download manually in a browser from {url} and save it over {dest}")
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        n = int(r.headers.get("Content-Length") or 0)
        if max_bytes and n > max_bytes: raise RuntimeError(f"{url} is {n/1e6:.0f} MB > limit; not downloaded")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            while True:
                b = r.read(1 << 20)
                if not b: break
                fh.write(b)
    if n and tmp.stat().st_size != n: raise RuntimeError(f"size mismatch for {url}: {tmp.stat().st_size} vs {n}")
    if open(tmp, "rb").read(200).lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise RuntimeError(f"NCBI returned an HTML/reCAPTCHA page for {url}; download it manually in a browser and save as {dest}")
    os.replace(tmp, dest); return dest
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def gz_ok(p):
    if not str(p).endswith(".gz"): return True
    try:
        with gzip.open(p, "rb") as fh:
            while fh.read(1 << 22): pass
        return True
    except Exception: return False
def parse_soft(p):
    series, samples, platforms, cur, kind, ptab, intab = {}, {}, {}, None, None, {}, None
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("^SERIES"): kind, cur = "S", series
            elif line.startswith("^SAMPLE"): kind = "M"; g = line.split("=", 1)[1].strip(); cur = samples.setdefault(g, {"GSM": g, "char": []})
            elif line.startswith("^PLATFORM"): kind = "P"; pid = line.split("=", 1)[1].strip(); cur = platforms.setdefault(pid, {"rows": []})
            elif kind == "P" and line.startswith("!platform_table_begin"): intab = pid
            elif kind == "P" and line.startswith("!platform_table_end"): intab = None
            elif intab: platforms[intab]["rows"].append(line.split("\t"))
            elif line.startswith("^"): kind, cur = "X", {}          # ^DATABASE and other blocks: ignored
            elif line.startswith("!") and " = " in line and cur is not None:
                k, v = line[1:].split(" = ", 1)
                if kind == "M" and k.startswith("Sample_characteristics"): cur["char"].append(v)
                else: cur.setdefault(k, []).append(v)
    return series, samples, platforms
# ---------------------------------------------------------------- per cohort
SUMMARY = {}
for G, task in COH.items():
    section(f"{G} ({task})"); d = S12 / "geo" / G; d.mkdir(exist_ok=True); man = []
    try:   # ---------- download
        files = {"family.soft.gz": f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub(G)}/{G}/soft/{G}_family.soft.gz"}
        try:
            fl = urllib.request.urlopen(urllib.request.Request(f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub(G)}/{G}/suppl/filelist.txt", headers={"User-Agent": "Mozilla/5.0"}), timeout=120).read().decode()
            (d / "suppl_filelist.txt").write_text(fl, encoding="utf-8")
            for row in fl.splitlines()[1:]:
                c = row.split("\t")
                if len(c) >= 2 and c[0].strip().lower() == "file": files[c[1]] = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub(G)}/{G}/suppl/{c[1]}"
        except Exception as e: say(f"- supplementary file list not available ({e})")
        try:
            idx = urllib.request.urlopen(urllib.request.Request(f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub(G)}/{G}/matrix/", headers={"User-Agent": "Mozilla/5.0"}), timeout=120).read().decode()
            for m in sorted(set(re.findall(r'href="([^"]*series_matrix\.txt\.gz)"', idx))): files[m] = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub(G)}/{G}/matrix/{m}"
        except Exception as e: say(f"- series matrix listing failed ({e})")
        files[f"{G}_raw_counts_GRCh38.p13_NCBI.tsv.gz"] = f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={G}&format=file&file={G}_raw_counts_GRCh38.p13_NCBI.tsv.gz"
        files["Human.GRCh38.p13.annot.tsv.gz"] = "https://www.ncbi.nlm.nih.gov/geo/download/?format=file&type=rnaseq_counts&file=Human.GRCh38.p13.annot.tsv.gz"
        for name, url in files.items():
            if name.endswith("_RAW.tar"): say(f"- skipped {name} (archive contents listed in suppl_filelist.txt)"); continue
            try:
                p = fetch(url, d / name, max_bytes=3e9); man.append(dict(file=name, bytes=p.stat().st_size, sha256=sha(p), gzip_intact=gz_ok(p), url=url))
            except Exception as e: say(f"- not available: {name} ({e})")
        M = pd.DataFrame(man); M.to_csv(d / "checksums.csv", index=False)
        say(f"- downloaded {len(M)} files; SHA-256 in geo/{G}/checksums.csv; gzip integrity all OK: {bool(M.gzip_intact.all()) if len(M) else 'n/a'}")
        for r in man: say(f"  - {r['file']}  {r['bytes']/1e6:.1f} MB  sha256 {r['sha256'][:16]}...")
    except Exception: say("- DOWNLOAD SECTION ERROR\n```\n" + traceback.format_exc() + "```"); continue
    try:   # ---------- metadata + independence
        series, samples, platforms = parse_soft(d / "family.soft.gz")
        rel = series.get("Series_relation", []); say(f"- title: {series.get('Series_title', [''])[0]}")
        say(f"- platform(s): {', '.join(series.get('Series_platform_id', []))}; samples in SOFT: {len(samples)}")
        say(f"- contributors: {'; '.join(series.get('Series_contributor', [])[:8])}"); say(f"- relations: {rel if rel else 'none listed'}")
        say(f"- summary: {series.get('Series_summary', [''])[0][:600]}")
        hit_rel = [r for r in rel if any(e in r for e in EXISTING)]
        crm = pd.read_csv(CR / "sample_manifest.csv"); gsm_overlap = sorted(set(samples) & set(crm.GSM))
        parent = [r for r in rel if "SubSeries of" in r]
        say(f"- INDEPENDENCE: relations naming an existing cohort: {hit_rel or 'none'}; parent SuperSeries: {parent or 'none'}; GSMs shared with any existing cohort: {len(gsm_overlap)}")
        SUMMARY[G] = {"independent_by_metadata": not hit_rel and not gsm_overlap, "parent": parent}
        rows = []
        for g, s in samples.items():
            ch = {}
            for c in s["char"]:
                k, v = (c.split(":", 1) + [""])[:2] if ":" in c else ("characteristic", c)
                ch[k.strip().lower()] = v.strip()
            rows.append({"GSM": g, "title": s.get("Sample_title", [""])[0], "source": s.get("Sample_source_name_ch1", [""])[0], "platform": s.get("Sample_platform_id", [""])[0],
                         "molecule": s.get("Sample_molecule_ch1", [""])[0], "description": " | ".join(s.get("Sample_description", [])), **{f"ch:{k}": v for k, v in ch.items()}})
        T = pd.DataFrame(rows); T.to_csv(S12 / f"tables/{G}_samples_long.csv", index=False)
        say(f"- per-GSM metadata: tables/{G}_samples_long.csv ({len(T)} samples). Characteristic fields and values:")
        for c in [c for c in T.columns if c.startswith("ch:")] + ["source", "molecule", "platform"]:
            vc = T[c].fillna("(missing)").value_counts()
            if c.startswith("ch:") and ("age" in c or len(vc) > 25):
                num = pd.to_numeric(T[c].str.extract(r"([\d.]+)")[0], errors="coerce")
                say(f"  - {c}: {len(vc)} distinct; numeric summary: n={num.notna().sum()}, min {num.min()}, median {num.median()}, max {num.max()}, <18: {(num < 18).sum()}")
            else: say(f"  - {c}: " + "; ".join(f"{k} ({v})" for k, v in vc.items()))
    except Exception: say("- METADATA SECTION ERROR\n```\n" + traceback.format_exc() + "```"); continue
    try:   # ---------- class mapping + filters (heuristic, fully tabulated)
        chc = [c for c in T.columns if c.startswith("ch:")]
        txt = (T[["title", "source", "description"] + chc].fillna("").astype(str).agg(" | ".join, axis=1)).str.lower()
        dcols = [c for c in chc if re.search(r"disease|diagnos|group|status|condition|class|phenotype|cohort|illness", c) and not re.search(r"hiv|smok|diabet|treat", c)]
        if "ch:diagnosis" in chc: dcols = ["ch:diagnosis"]
        dx = T[dcols].fillna("").astype(str).agg(" | ".join, axis=1) if dcols else T.title
        def cls(v):
            v = v.lower()
            if re.search(r"latent|ltbi|\bltb\b", v): return "LTB"
            if re.search(r"active ?tb", v): return "ATB"
            if re.search(r"\bptb\b|active|pulmonary tb|pulmonary tuberculosis|\btb\b|tuberculosis|\batb\b", v) and not re.search(r"non[- ]?tb|not tb|no tb|other", v): return "ATB"
            if re.search(r"healthy|control|\bhc\b", v): return "CON"
            return "OD" if v.strip() else "unknown"
        T["raw_class_value"] = dx; T["class_label"] = dx.map(cls)
        if task == "ATB_vs_OD": T["class_label"] = T["class_label"].replace("LTB", "OD")   # symptomatic non-TB patients with LTBI count as OD
        mp = T.groupby(["raw_class_value", "class_label"]).size().reset_index(name="n"); mp.to_csv(S12 / f"tables/{G}_class_mapping_REVIEW.csv", index=False)
        say(f"- class fields used: {dcols or ['title (no disease field found)']}; raw value -> class mapping (REVIEW): tables/{G}_class_mapping_REVIEW.csv")
        for r in mp.itertuples(): say(f"  - '{r.raw_class_value}' -> {r.class_label} ({r.n})")
        age = None
        acol = [c for c in chc if re.search(r"\bage\b|age \(|age_", c)]
        if acol: age = pd.to_numeric(T[acol[0]].str.extract(r"([\d.]+)")[0], errors="coerce")
        hcol = [c for c in chc if "hiv" in c]; tcol = [c for c in chc if re.search(r"treat|time|visit|week|month|day", c)]
        F = pd.DataFrame({"GSM": T.GSM, "class": T.class_label})
        F["not_whole_blood"] = txt.str.contains(r"pbmc|cd4|cd8|cd14|cd15|monocyte|neutrophil|sorted|t cell|b cell|nk cell")
        F["treated_or_followup"] = T[tcol].fillna("").astype(str).agg(" ".join, axis=1).str.lower().str.contains(r"(?:week|month|day)\s*[1-9]|post|on[- ]treatment|after|follow|end of") if tcol else False
        F["hiv_positive"] = T[hcol].fillna("").astype(str).agg(" ".join, axis=1).str.lower().str.contains(r"pos|\+|yes|\btrue\b") if hcol else False
        F["eptb"] = txt.str.contains(r"extra[- ]?pulmonary|\beptb\b|lymph node tb|pleural tb|tb meningitis|spinal tb|non[- ]pulmonary")
        if "ch:eptborptb" in chc:
            ep = T["ch:eptborptb"].fillna("").astype(str).str.strip().str.upper()
            F["eptb"] = ep.isin(["EPTB", "PTB_EPTB"])
            say(f"- EPTB from ch:eptborptb: EPTB {int(ep.eq('EPTB').sum())}, PTB_EPTB {int(ep.eq('PTB_EPTB').sum())} (both excluded, strict), PTB {int(ep.eq('PTB').sum())}")
        F["paediatric"] = (age < 18) if age is not None else False
        F["class_unknown"] = F["class"].eq("unknown")
        crit = {"not_whole_blood": "whole blood", "treated_or_followup": "untreated", "hiv_positive": "HIV-negative", "eptb": "no EPTB", "paediatric": "adult (>=18)", "class_unknown": "class assignable"}
        if "ch:dosanjh" in chc:   # ATB restricted to culture-confirmed (Dosanjh category 1)
            F["atb_not_culture_confirmed"] = F["class"].eq("ATB") & T["ch:dosanjh"].fillna("").astype(str).str.strip().ne("1")
            crit = {**{k: v for k, v in crit.items() if k != "class_unknown"}, "atb_not_culture_confirmed": "ATB culture-confirmed (Dosanjh 1)", "class_unknown": "class assignable"}
        meta = {"untreated": bool(tcol), "HIV-negative": bool(hcol), "adult (>=18)": age is not None}
        keep = pd.Series(True, index=F.index)
        for k, lab in crit.items():
            rm = keep & F[k]; say(f"  - filter '{lab}': removes {int(rm.sum())}" + ("" if meta.get(lab, True) else "  [NO METADATA FIELD - criterion could not be applied]")); keep &= ~F[k]
        pos, neg = task.split("_vs_"); F["included"] = keep & F["class"].isin([pos, neg]); F.to_csv(S12 / f"tables/{G}_filter_decisions.csv", index=False)
        say(f"- final usable set for {task}: {int((F.included & F['class'].eq(pos)).sum())} {pos} vs {int((F.included & F['class'].eq(neg)).sum())} {neg}; other classes after filters: "
            + str(F[keep & ~F['class'].isin([pos, neg])]['class'].value_counts().to_dict()))
        for nm, pat in [("HIV", r"hiv"), ("asthma", r"asthma"), ("diabetes", r"diabet|\bdm\b|t2d")]:
            g = T.GSM[txt.str.contains(pat)].tolist(); say(f"  - comorbidity/subgroup mention '{nm}': {len(g)} GSMs " + (str(g[:15]) + (" ..." if len(g) > 15 else "") if g else ""))
        if task == "ATB_vs_OD": say("  - OD composition by raw diagnosis value: " + str(T.loc[T.class_label.eq('OD'), 'raw_class_value'].value_counts().to_dict()))
        for c in [c for c in chc if re.search(r"bmi|nutri|malnour|weight|tissue|cell type|treat|hiv|smok|diabet|site|ethnic|country|batch", c)]:
            say(f"  - {c} by class: " + str(pd.crosstab(T[c].fillna('(missing)'), T.class_label).to_dict()))
        scol = [c for c in chc if re.search(r"\bsex\b|gender", c)]
        if scol: say("  - sex by class (all samples): " + str(pd.crosstab(T.class_label, T[scol[0]].str.lower()).to_dict()))
        SUMMARY[G].update(included=F.GSM[F.included].tolist(), filter_ok=True)
    except Exception: say("- FILTER SECTION ERROR\n```\n" + traceback.format_exc() + "```"); continue
    try:   # ---------- expression processing
        HG = HGNCMapper(); inc = F.GSM[keep].tolist()           # normalise over all filter-passing samples, whatever their class (as the clean rebuild)
        mats = sorted(d.glob("*series_matrix.txt.gz")); supp = [p for p in d.iterdir() if p.is_file() and p.name not in ("family.soft.gz", "checksums.csv", "suppl_filelist.txt", "Human.GRCh38.p13.annot.tsv.gz") and "series_matrix" not in p.name and not p.name.endswith(".part") and not open(p, "rb").read(200).lstrip().lower().startswith((b"<!doctype", b"<html"))]
        say(f"- expression files: matrix {[p.name for p in mats]}; supplementary {[p.name for p in supp]}")
        mx = pd.concat([pd.read_csv(p, sep="\t", comment="!", index_col=0, low_memory=False) for p in mats], axis=1) if mats else pd.DataFrame()
        mx = mx.apply(pd.to_numeric, errors="coerce"); mx.index = mx.index.astype(str)
        say(f"- series matrix: {mx.shape[0]} rows x {mx.shape[1]} GSM columns; finite values: {int(np.isfinite(mx.to_numpy()).sum()) if mx.size else 0}; max {np.nanmax(mx.to_numpy()) if mx.size else 'n/a'}")
        rnaseq = "rna" in " ".join(T.molecule.astype(str)).lower() and ("hiseq" in json.dumps(platforms.get(T.platform.iloc[0], {}).get("Platform_title", [""])).lower() or mx.dropna(how="all").empty)
        title2gsm = dict(zip(T.title.astype(str).str.strip(), T.GSM)); desc2gsm = {}
        for g, dsc in zip(T.GSM, T.description.astype(str)):
            for tok in re.split(r"[|;,\s]+", dsc):
                if len(tok) > 2: desc2gsm.setdefault(tok.strip(), set()).add(g)
        def map_cols(cols):
            out = {}
            for c in cols:
                cs = str(c).strip(); m = re.search(r"GSM\d+", cs)
                if m and m.group(0) in samples: out[c] = m.group(0)
                elif cs in title2gsm: out[c] = title2gsm[cs]
                elif cs in desc2gsm and len(desc2gsm[cs]) == 1: out[c] = next(iter(desc2gsm[cs]))
            return out
        raw, rawname = None, None
        for p in supp:
            try:
                df = pd.read_csv(p, sep="\t", index_col=0, compression="infer", low_memory=False)
                if df.shape[1] < 2: df = pd.read_csv(p, sep=",", index_col=0, compression="infer", low_memory=False)
            except Exception: continue
            df = df.loc[:, [c for c in df.columns if not re.search(r"p.?val|detection|pval", str(c), re.I)]]
            mpc = map_cols(df.columns); cov = len(set(mpc.values()) & set(inc)) / max(len(inc), 1)
            say(f"  - {p.name}: {df.shape[0]} x {df.shape[1]}; columns mapped to GSM by id/title/description: {len(mpc)}; coverage of filtered samples {cov:.0%}")
            if cov > (0 if raw is None else rawcov):
                raw, rawname, rawcov = df.rename(columns=mpc)[[c for c in mpc.values() if c in inc]].apply(pd.to_numeric, errors="coerce"), p.name, cov
        if raw is not None and rawcov < 0.95 and not mx.empty and not rnaseq:
            say(f"  - supplementary coverage {rawcov:.0%} < 95%: identity matching of the best supplementary file to the series matrix (log2, expressed probes, mutual best r>=0.90, margin>=0.10)")
        if rnaseq:
            X = raw if raw is not None and rawcov >= 0.95 else None
            if X is None: raise RuntimeError("RNA-seq: no supplementary count table mapped to >=95% of filtered GSMs - see columns above; stop and review")
            idx = X.index.astype(str).str.replace(r"\.\d+$", "", regex=True)
            ann = d / "Human.GRCh38.p13.annot.tsv.gz"
            if ann.exists() and idx.str.fullmatch(r"\d+").mean() > 0.9:      # NCBI-generated counts: Entrez GeneID rows -> NCBI symbol -> HGNC
                an = pd.read_csv(ann, sep="\t", dtype=str, usecols=["GeneID", "Symbol"]).set_index("GeneID").Symbol; idx = pd.Index(idx.map(an).fillna(""))
            h = pd.read_csv(CR / "resources/hgnc_entries.tsv", sep="\t", dtype=str); h = h[(h.Status == "Approved") & h["Ensembl gene ID"].notna()]
            ens = h.drop_duplicates("Ensembl gene ID", keep=False).set_index("Ensembl gene ID")["Approved symbol"]
            sym = pd.Series(np.where(idx.str.startswith("ENSG"), idx.map(ens).fillna(""), [HG.map(s)["current"] for s in idx]), index=X.index)
            scale = "counts" if (X.fillna(0) % 1 == 0).all().all() and X.max().max() > 100 else "already-normalised"
            if scale == "counts":   # DESeq2 median-of-ratios size factors, then log2(x+1)
                Xc = X.clip(lower=0).fillna(0); pos_ = (Xc > 0).all(1); lg = np.log(Xc[pos_]); sf = np.exp((lg.sub(lg.mean(1), axis=0)).median(0))
                say(f"  - DESeq2 size factors over {int(pos_.sum())} all-positive genes: min {sf.min():.3f}, median {sf.median():.3f}, max {sf.max():.3f}")
                L = np.log2(Xc.div(sf, axis=1) + 1)
            else: L = X
            L = L[sym.ne("").values]; L.index = sym[sym.ne("")].values; L = L.loc[L.mean(1).sort_values(ascending=False).index]; L = L[~L.index.duplicated()]
            note = f"RNA-seq {scale} from {rawname}: DESeq2 size-factor normalised, log2(x+1); duplicate symbols -> highest mean"
        else:
            pid = T.platform.iloc[0]; pt = platforms[pid]["rows"]; pt = pd.DataFrame(pt[1:], columns=pt[0]).set_index("ID")
            scol = next((c for c in ["Symbol", "ILMN_Gene", "Gene Symbol", "GENE_SYMBOL", "gene_assignment"] if c in pt.columns), None)
            symraw = pt[scol].astype(str)
            if scol == "gene_assignment": symraw = symraw.str.split(" /// ").str[0].str.split(" // ").str[1].fillna("")
            if raw is not None and rawcov >= 0.95: V, src = raw, f"non-normalised {rawname}"
            else:
                V, src = mx[[g for g in inc if g in mx.columns]], "series matrix (GEO-normalised values: NO non-normalised data on GEO -> probe ranking and quantile step use these values; differs from the rebuild rule)"
                say(f"  - FLAG: {src}")
            V.index = V.index.astype(str); s = symraw.reindex(V.index).fillna("").str.strip()
            ok = ~s.str.contains(r"[;,/]| /// ") & ~s.isin(["", "nan", "NA", "---"]) & V.notna().all(1); V, s = V[ok], s[ok]
            lin = np.nanmax(V.to_numpy()) > 100; Lr = np.log2(V.clip(lower=1)) if lin else V
            rk = Lr.mean(1); hg = s.map(lambda x: HG.map(x)["current"])
            ch = pd.DataFrame({"probe": V.index, "sym": hg.values, "rk": rk.values}).sort_values(["sym", "rk", "probe"], ascending=[True, False, True]).drop_duplicates("sym")
            a = Lr; rkk = a.rank(axis=0, method="average").to_numpy(); ms = np.sort(a.to_numpy(), axis=0).mean(1); grid = np.arange(1, len(ms) + 1)
            Q = pd.DataFrame(np.column_stack([np.interp(rkk[:, k], grid, ms) for k in range(a.shape[1])]), index=a.index, columns=a.columns)
            L = Q.loc[ch.probe]; L.index = ch.sym.values
            note = f"microarray {pid}; values from {src}; {'log2(max(x,1))' if lin else 'already log scale'} + quantile normalisation over {L.shape[1]} filter-passing samples; probe = highest mean pre-normalisation log2"
        L = L[[g for g in inc if g in L.columns]]; L.index.name = "Gene"; L.astype("float32").to_csv(S12 / f"expression/{G}_log2_genelevel.csv.gz", float_format="%.5f")
        say(f"- processed: {note}; result {L.shape[0]} genes x {L.shape[1]} samples (filtered set incl. non-task classes); missing filtered GSMs: {sorted(set(inc) - set(L.columns))[:10]}")
        sig = {"ATB_vs_LTB": ["ANKRD22", "FCGR1A", "PSTPIP2"], "ATB_vs_OD": ["APOL2", "WDFY1", "BATF2", "POLB", "GBP2"]}[task]
        say(f"- SIGNATURE GENES measured ({task}): " + "; ".join(f"{g}: {'yes' if g in L.index else 'MISSING'}" + (f" (mean log2 {L.loc[g].mean():.2f})" if g in L.index else "") for g in sig))
        SUMMARY[G]["expression"] = L
    except Exception: say("- EXPRESSION SECTION ERROR\n```\n" + traceback.format_exc() + "```")
# ---------------------------------------------------------------- expression-identity duplicate check vs existing cohorts
section("Expression-identity check against existing cohorts (same technology)")
try:
    ref = {"GSE144127": ["GSE37250", "GSE42834", "GSE19491", "GSE19439", "GSE19444"], "GSE152218": ["GSE101705", "GSE107994"]}
    for G, refs in ref.items():
        if "expression" not in SUMMARY.get(G, {}): say(f"- {G}: no processed expression; check not run"); continue
        A = SUMMARY[G]["expression"]
        for rc in refs + (["GSE83456"] if G == "GSE144127" else []):
            f = MT / "Stage7/data/GSE83456_log2_genelevel_rebuilt.csv.gz" if rc == "GSE83456" else (MT / "Stage3/data/GSE19491_log2_genelevel_stage3.csv.gz" if rc == "GSE19491" else CR / f"expression/{rc}_log2_genelevel.csv.gz")
            B = pd.read_csv(f, index_col=0); g = A.index.intersection(B.index)
            g = A.loc[g].var(1).sort_values(ascending=False).index[:2000]
            ra = A.loc[g].rank().to_numpy(); rb = B.loc[g].rank().to_numpy(); ra = (ra - ra.mean(0)) / ra.std(0); rb = (rb - rb.mean(0)) / rb.std(0)
            r = ra.T @ rb / len(g); i, j = np.unravel_index(np.argmax(r), r.shape)
            say(f"- {G} vs {rc}: max sample-pair Spearman r {r.max():.4f} ({A.columns[i]} ~ {B.columns[j]}); pairs r>0.99: {int((r > 0.99).sum())}; median {np.median(r):.3f}")
except Exception: say("- DUPLICATE CHECK ERROR\n```\n" + traceback.format_exc() + "```")
section("Stop point")
say("No signature has been scored. Review tables/*_class_mapping_REVIEW.csv and tables/*_filter_decisions.csv before any scoring.")
(S12 / "REPORT.md").write_text("# Stage12 report (auto-generated)\n" + "\n".join(REP) + "\n", encoding="utf-8")
print("\nStage12 finished - see REPORT.md")
