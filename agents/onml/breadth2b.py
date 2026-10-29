"""2000-name overnight book without the closing-print feature (zmoc) and without the last-5-minute return (z5l):
implementability check for a market-on-close fill. Yearly retrain, seed 0, dev dates."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
NJ = int(os.environ.get("ONML_NJOBS", "2")); CAP = 400000.0
FS = FEATS18 + EXTRA10 + CAL
SETS = {"f28cal": FS, "f28cal-zmoc": [f for f in FS if f != "zmoc"], "f28cal-zmoc-z5l": [f for f in FS if f not in ("zmoc", "z5l")], "f28cal-zmoc-z5l-z15l": [f for f in FS if f not in ("zmoc", "z5l", "z15l")]}
D = load(2000); log(f"breadth2b: {D.tk.nunique()} names")
for name, fs in SETS.items():
    t0 = time.time(); p = xrank(D, wf(D, fs, "yr", "Y", seed=0, n_jobs=NJ))
    for k in [10, 20]:
        r = book2(D, p, k=k, p=False); rn = r - 2 * k * 6 / CAP
        log(f"  u2000 {name:22s} k{k}: spread-only foldOOS {rlib.sh(rlib.fold_oos(r, p=False)[0]):.2f} | $0.4M Sh {rlib.sh(rn):.2f} foldOOS {rlib.sh(rlib.fold_oos(rn, p=False)[0]):.2f} | halves {rlib.sh(rn[rn.index < '2021-10-28']):.2f}/{rlib.sh(rn[rn.index >= '2021-10-28']):.2f}  ({time.time()-t0:.0f}s)")
log("BREADTH2B DONE")
