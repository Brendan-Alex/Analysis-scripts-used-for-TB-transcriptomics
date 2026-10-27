"""Leakage canary for Native selection: re-run 01_native.fold for one fold with the held-out cohort's cached data replaced by noise
(Z ~ N(0,1), random labels); outputs go to qc/canary_native/. The per-cohort mRMR pools of the TRAINING cohorts are copied in
(each is computed from that cohort alone). The panel must be identical to results/native/<task>__fold_<held>__nosex.json.
Usage: python 04_canary_native.py TASK HELDOUT   (resumable through the checkpoint, like 01_native)"""
import os, sys, json, shutil, pickle, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from common4 import *
spec = importlib.util.spec_from_file_location("nat", os.path.join(os.path.dirname(__file__), "01_native.py")); nat = importlib.util.module_from_spec(spec); spec.loader.exec_module(nat)
task, held = sys.argv[1], sys.argv[2]; out = S4 / "qc/canary_native"; out.mkdir(exist_ok=True)
for c in TASKS[task]["cohorts"]:
    if c != held: shutil.copy(S4 / f"results/native/{task}__{c}__nosex__cohort.json", out)
real = nat.load
def noisy(t):
    d = real(t); h = dict(d[held]); rng = np.random.default_rng(7)
    h["Z"] = pd.DataFrame(rng.normal(size=h["Z"].shape), index=h["Z"].index, columns=h["Z"].columns)
    h["y"] = pd.Series(rng.integers(0, 2, len(h["y"])), index=h["y"].index); d[held] = h; return d
nat.load = noisy; nat.OUT = out
nat.fold(task, held, "nosex")
f = out / f"{task}__fold_{held}__nosex.json"
if f.exists():
    a = json.load(open(f)); b = json.load(open(S4 / f"results/native/{task}__fold_{held}__nosex.json"))
    res = dict(task=task, heldout=held, panel_identical=a["panel"] == b["panel"], cross_auc_identical=a["mean_cross_auc"] == b["mean_cross_auc"], allowed_identical=a["allowed"] == b["allowed"])
    print(res); open(S4 / "qc/canary_native.jsonl", "a").write(json.dumps(res) + "\n")
