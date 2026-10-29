#!/usr/bin/env python3
"""Regenerate ONLY the kept onml candidates end-to-end from raw data (universe(1500), n_jobs=4).
Honours STRAT_END through rlib.DEV_END (loaders, walk-forward cuts and book() all cut there).
Run unattended:  nohup python3 agents/onml/final.py > agents/onml/final.log 2>&1 &
Candidate list KEEP is filled in from the exploration results (see REPORT.md)."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
NJ = int(os.environ.get("ONML_NJOBS", "4"))
F28 = FEATS18 + EXTRA10
SETS = {"f18": FEATS18, "f28": F28, "f28cal": F28 + CAL, "fall": F28 + XR + MR + CAL + ["nnames"]}
# name -> dict(fs, target(s), retrain(s), seeds, k, construction kwargs).  Predictions of all (target, retrain, seed)
# members are per-day rank-normalised and averaged before the book is formed.
KEEP = {}
exec(open(f"{HERE}/agents/onml/keep.py").read())      # defines KEEP (kept separate so exploration can update it)
t0 = time.time()
D = load(int(os.environ.get("ONML_N", "1500"))); log(f"final: {D.tk.nunique()} names, DEV_END={rlib.DEV_END}, n_jobs={NJ}")
cache = {}
def pred_of(fsn, tg, rt, s):
    key = (fsn, tg, rt, s)
    if key not in cache:
        t1 = time.time(); cache[key] = wf(D, SETS[fsn], tg, rt, seed=s, n_jobs=NJ); log(f"  trained {key} {time.time()-t1:.0f}s")
    return cache[key]
for name, c in KEEP.items():
    members = [xrank(D, pred_of(c["fs"], tg, rt, s)) for tg in c["targets"] for rt in c["retrain"] for s in c["seeds"]]
    p = sum(members) / len(members)
    r = book2(D, p, k=c["k"], p=False, **c.get("kw", {}))
    rlib.save_cand("onml", name, r, c.get("note", ""))
    log(report_line(f"FINAL {name}", r))
log(f"final done {time.time()-t0:.0f}s")
