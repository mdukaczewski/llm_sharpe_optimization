#!/usr/bin/env python3
"""Regenerate ONLY the kept onml2 candidates end-to-end from raw data (universe(2000), env ONML2_N).
Honours STRAT_END through rlib.DEV_END (loaders, walk-forward cuts and the book all cut there).
Run unattended:  nohup python3 agents/onml2/final.py > agents/onml2/final.log 2>&1 &     (ONML2_NJOBS, default 2)
Commission convention: $3 per filled order on $0.4M equity -> P positions cost P*6/400000 per day; the saved
candidate series are NET of spread (rlib.cost_bps), borrow AND this commission; the spread-only series and the
daily position count are written next to it in agents/onml2/out/final_<name>.csv."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml2")
from o2lib import *
NJ = int(os.environ.get("ONML2_NJOBS", "2"))
exec(open(f"{HERE}/agents/onml2/keep.py").read())      # defines KEEP: name -> dict(model=..., seeds, retrain, book kwargs, note)
t0 = time.time()
USE_RAW = os.environ.get("ONML2_USE_RAW", "0") == "1"     # reuse out/raw_<model>.npy from sweep1 (same code path, yearly, seed 0) instead of refitting
D = add_cal(load(int(os.environ.get("ONML2_N", "2000")), extra=False)); log(f"final: {D.tk.nunique()} names rows {len(D)} DEV_END={rlib.DEV_END} n_jobs={NJ}")
cache = {}
def raw_of(model, rt, s):
    key = (model, rt, s)
    fp = f"{OUT}/raw_{model}.npy"
    if key not in cache and USE_RAW and rt == "Y" and s == 0 and os.path.exists(fp):
        cache[key] = pd.Series(np.load(fp), index=D.index); log(f"  loaded {fp}")
    if key not in cache:
        t1 = time.time(); mk = dict(MODELS[model]); fs = mk.pop("feats", F26CAL); cache[key] = wf2(D, fs, "yr", rt, seed=s, n_jobs=NJ, **mk); log(f"  trained {key} {time.time()-t1:.0f}s IC {ic(D, cache[key])}")
    return cache[key]
for name, c in KEEP.items():
    raws = [raw_of(m, c["retrain"], s) for m in c["models"] for s in c["seeds"]]
    p = sum(xrank(D, r) for r in raws) / len(raws); raw = sum(raws) / len(raws)
    r, P = book3(D, p, raw=raw, **c["book"]); rc = with_comm(r, P)
    save(f"final_{name}", r, P); rlib.save_cand("onml2", name, rc, c.get("note", ""))
    d = ev(r, P, f"FINAL {name}")
    sl = olib.sleeve(); sp = olib.spy_cc(); j = pd.concat([rc, sl, sp], axis=1).dropna(); o, t = rlib.fold_oos({"x": rc}, p=False)
    log(f"    corr sleeve {j.iloc[:,0].corr(j.iloc[:,1]):.2f} corr SPY {j.iloc[:,0].corr(j.iloc[:,2]):+.2f} | folds min {t.fold_sh.min():.2f} neg {(t.fold_sh<0).sum()}/{len(t)} | yrs "
        + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in rc.groupby(rc.index.year).apply(sh).items()))
C = rlib.load_cands("onml2"); rlib.fold_oos(C, label="onml2 candidate family (comm)")
log(f"final done {time.time()-t0:.0f}s")
