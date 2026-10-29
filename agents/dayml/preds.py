#!/usr/bin/env python3
"""Save the quarterly intraday predictions (delayed-fill targets y_60d / y_120d, full range per STRAT_END) to
agents/dayml/out/pred_{target}_Q.parquet and write k10/k15/k20 |pred|-weighted books as candidates."""
import sys, time, os; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml")
import dlib, rlib, numpy as np, pandas as pd
NJ = int(os.environ.get("DAYML_NJOBS", "2")); t0 = time.time()
tks = rlib.universe(1500); D = dlib.build(tks, verbose=False); print("rows", len(D), "DEV_END", rlib.DEV_END, f"{time.time()-t0:.0f}s", flush=True)
D = D.drop(columns=[c for c in D.columns if c.startswith("t30_") or c.startswith("t330_") or c.startswith("spy_t")])
for y, tag in [("y_60d", "x60d"), ("y_120d", "x120d")]:
    t1 = time.time(); Dp, pred = dlib.run(D, dlib.EXT, y, retrain="Q", n_jobs=NJ, verbose=False)
    pd.DataFrame({"date": Dp.date, "tk": Dp.tk, "p": pred}).dropna().to_parquet(f"/home/user/llm_sharpe_optimization/agents/dayml/out/pred_{y}_Q.parquet")
    print(f"--- {y} {time.time()-t1:.0f}s IC {rlib.ic(Dp, pred, y)}", flush=True)
    for k in [10, 15, 20]:
        r = dlib.book(Dp, pred, k=k, y=y, weight="pred", name=f"dayml_{tag}q_k{k}p", p=True)
        rlib.save_cand("dayml", f"dayml_{tag}q_k{k}p", r, f"09:31 LightGBM quarterly, fill 09:33 bar, exit {'10:31' if y == 'y_60d' else '11:31'} bar, L/S top{k} |pred|-weighted")
print(f"PREDS DONE {time.time()-t0:.0f}s", flush=True)
