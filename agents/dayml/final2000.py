#!/usr/bin/env python3
"""Intraday 10:31-exit book on universe(2000): quarterly LightGBM on dlib.EXT, delayed fill (09:33 bar), predictions saved,
small L/S books (k6/k8/k10) as candidates. Printed stats cut at DEV_CUT so a STRAT_END=2026-10-28 run keeps the holdout unread."""
import sys, time, os; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml")
import dlib, rlib, numpy as np, pandas as pd
NJ = int(os.environ.get("DAYML_NJOBS", "2")); CAP = 400000.0; DEV_CUT = "2026-04-27"; N = 2000; t0 = time.time()
tks = rlib.universe(N); D = dlib.build(tks, verbose=False); print("rows", len(D), "names", D.tk.nunique(), "DEV_END", rlib.DEV_END, f"{time.time()-t0:.0f}s", flush=True)
D = D.drop(columns=[c for c in D.columns if c.startswith("t30_") or c.startswith("t330_") or c.startswith("spy_t")])
y = "y_60d"; Dp, pred = dlib.run(D, dlib.EXT, y, retrain="Q", n_jobs=NJ, verbose=False)
pd.DataFrame({"date": Dp.date, "tk": Dp.tk, "p": pred}).dropna().to_parquet(f"/home/user/llm_sharpe_optimization/agents/dayml/out/pred_{y}_Q_u{N}.parquet")
print(f"--- {y} {time.time()-t0:.0f}s", flush=True)
for k in [6, 8, 10]:
    r = dlib.book(Dp, pred, k=k, y=y, weight="eq", p=False); name = f"dayml_x60dq_u{N}_k{k}"
    rlib.save_cand("dayml", name, r, f"09:31 LightGBM quarterly on universe({N}), fill 09:33 bar, exit 10:31 bar, L/S top{k} eq-weight")
    rd = r[r.index <= DEV_CUT]; rn = rd - 2 * k * 6 / CAP
    print(f"{name:26s} spread-only Sh {rlib.sh(rd):.2f} foldOOS {rlib.sh(rlib.fold_oos(rd, p=False)[0]):.2f} | $0.4M Sh {rlib.sh(rn):.2f} foldOOS {rlib.sh(rlib.fold_oos(rn, p=False)[0]):.2f} | halves {rlib.sh(rn[rn.index < '2021-10-28']):.2f}/{rlib.sh(rn[rn.index >= '2021-10-28']):.2f}", flush=True)
print(f"FINAL2000 DONE {time.time()-t0:.0f}s", flush=True)
