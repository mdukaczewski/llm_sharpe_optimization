#!/usr/bin/env python3
"""dayml FINAL: regenerates ONLY the kept candidates end-to-end from raw minute data (universe(1500)).
Honours STRAT_END through rlib (rlib.DEV_END): the coordinator runs it once with STRAT_END=2026-10-28
for the single holdout evaluation.  Unattended:  nohup python3 agents/dayml/final.py > agents/dayml/final.log 2>&1 &
Per-ticker feature frames are cached (data/cache, data/cache_dayml keyed on DEV_END), the LightGBM
walk-forward (expanding window, label lag 3 days) is refit from scratch.  Expected runtime: see REPORT.md.
Pipeline per kept candidate (all decided at 09:31 of day t after the first minute bar):
  features = yesterday's close-time features (rlib.ticker_frame shifted 1 day within ticker) + today's gap,
             first minute, first-minute volume, realised overnight return, SPY's gap/first-minute/overnight,
             cross-sectional ranks (gap, overnight, first minute), market-relative gap, gap/rv20, day of week
  target   = cross-sectional rank of the return from the 09:32 fill (bin 2) to the exit bar
  book     = long top-k / short bottom-k by prediction, |pred|-weighted, 1 bp/side (1.5 beyond rank 600), borrow 1%/yr"""
import sys, time, os; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml")
import dlib, rlib, numpy as np, pandas as pd
NJ = int(os.environ.get("DAYML_NJOBS", "4")); LANE = "dayml"; N = 1500
# name, target, retrain, k, weight, note
KEPT = [
    ("dayml_x120q_k20p", "y_120", "Q", 20, "pred", "09:31 LightGBM (EXT feats, quarterly) L/S top/bottom 20 |pred|-weighted, fill 09:32 bar, exit 11:30 bar (bin 120)"),
    ("dayml_x120q_k10p", "y_120", "Q", 10, "pred", "same model, k=10"),
    ("dayml_x60q_k20p",  "y_60",  "Q", 20, "pred", "09:31 LightGBM (EXT feats, quarterly) L/S 20 |pred|-weighted, exit 10:30 bar (bin 60)"),
]
t0 = time.time(); print("DEV_END", rlib.DEV_END, "n_jobs", NJ, flush=True)
tks = rlib.universe(N); print("universe", N, "available", len(tks), flush=True)
D = dlib.build(tks); print("rows", len(D), "names", D.tk.nunique(), f"{time.time()-t0:.0f}s", flush=True)
D = D.drop(columns=[c for c in D.columns if c.startswith("t30_") or c.startswith("t330_") or c.startswith("spy_t")])
preds = {}
for name, y, retrain, k, w, note in KEPT:
    key = (y, retrain)
    if key not in preds:
        t = time.time(); Dp, pred = dlib.run(D, dlib.EXT, y, retrain=retrain, n_jobs=NJ, verbose=True); preds[key] = (Dp, pred)
        print(f"--- model {key} {time.time()-t:.0f}s IC {rlib.ic(Dp, pred, y)}", flush=True)
    Dp, pred = preds[key]
    r = dlib.book(Dp, pred, k=k, y=y, weight=w, name=name)
    rlib.save_cand(LANE, name, r, note)
print(f"FINAL DONE {time.time()-t0:.0f}s", flush=True)
