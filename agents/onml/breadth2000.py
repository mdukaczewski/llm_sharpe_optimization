"""Breadth test: the f28cal overnight recipe on universe(1500) vs universe(2000), yearly retrain, seed 0,
books k10/k20 at spread-only and at the $0.4M commission. Dev dates only."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml")
from olib import *
NJ = int(os.environ.get("ONML_NJOBS", "2")); FS = FEATS18 + EXTRA10 + CAL; CAP = 400000.0
for n in [1500, 2000]:
    t0 = time.time(); D = load(n); log(f"breadth {n}: {D.tk.nunique()} names")
    p = xrank(D, wf(D, FS, "yr", "Y", seed=0, n_jobs=NJ)); log(f"  wf {time.time()-t0:.0f}s")
    for k in [10, 20]:
        r = book2(D, p, k=k, p=False); rn = r - 2 * k * 6 / CAP
        log(f"  u{n} k{k}: spread-only Sh {rlib.sh(r):.2f} foldOOS {rlib.sh(rlib.fold_oos(r, p=False)[0]):.2f} | $0.4M Sh {rlib.sh(rn):.2f} foldOOS {rlib.sh(rlib.fold_oos(rn, p=False)[0]):.2f} | halves {rlib.sh(r[r.index < '2021-10-28']):.2f}/{rlib.sh(r[r.index >= '2021-10-28']):.2f}")
    # restrict the 2000-name predictions to the top-1500 names only (same model, liquid book)
    if n == 2000:
        m = D.tk.map(rlib.RANK) <= 1500; r = book2(D[m].reset_index(drop=True), p[m].reset_index(drop=True), k=20, p=False); rn = r - 40 * 6 / CAP
        log(f"  u2000 model, book on ranks<=1500 k20: spread-only foldOOS {rlib.sh(rlib.fold_oos(r, p=False)[0]):.2f} | $0.4M foldOOS {rlib.sh(rlib.fold_oos(rn, p=False)[0]):.2f}")
    del D
log("BREADTH DONE")
