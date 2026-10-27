
# --- path configuration (added for public release) -------------------------
# Original source files live outside the repository. Set TB_DATA_ROOT to the
# folder where the Zenodo archive was extracted. Defaults to <repo>/inputs.
import os as _os
from pathlib import Path as _Path
DATA_ROOT = _Path(_os.environ.get("TB_DATA_ROOT", _Path(__file__).resolve().parents[2] / "inputs"))
# ---------------------------------------------------------------------------
# helper for 02b: rebuild the diagnostic joint normalisation (213 + 39 GSE22098 controls) at gene level
import glob as _g, os as _o
_M = pd.read_csv(CR / "sample_manifest.csv"); _m = _M[_M.cohort == "GSE19491"].set_index("GSM")
_inc = _m[_m.included].index.tolist(); _need = _inc + H
_files = {_o.path.basename(f).split("_")[0]: f for f in _g.glob(str(DATA_ROOT / "geo_raw/GSE19491_RAW/GSM*.txt.gz"))}
def _rd(g):
    d = pd.read_csv(_files[g], sep="\t"); d = d[d.iloc[:, 0].notna()]
    v = pd.to_numeric(d.iloc[:, 1], errors="coerce"); v.index = d.iloc[:, 0].astype(str).values
    return v[~v.index.duplicated(keep=False)].rename(g)
_R = pd.concat([_rd(g) for g in _need], axis=1, join="inner"); _R = _R[_R.notna().all(1)]
_L = np.log2(np.clip(_R, 1, None)); _rk = _L.rank(axis=0).to_numpy(); _ms = np.sort(_L.to_numpy(), axis=0).mean(1); _grid = np.arange(1, len(_ms) + 1)
_Q = pd.DataFrame(np.column_stack([np.interp(_rk[:, k], _grid, _ms) for k in range(_L.shape[1])]), index=_L.index, columns=_L.columns)
_sel = pd.read_csv(S / "qc/GSE19491_probe_selection_jointnorm.csv"); _sel = _sel[_sel.selected]
_sel = _sel[_sel.probe.isin(_Q.index)]
D = _Q.loc[_sel.probe, H + Lc]; D.index = _sel.symbol_hgnc.values
print("all-probe per-sample mean after quantile normalisation: sd =", float(_Q.mean().std()))
