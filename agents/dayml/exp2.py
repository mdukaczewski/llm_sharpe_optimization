#!/usr/bin/env python3
"""exp2: EXT features on universe(1500) (yearly retrain, n_jobs=2) for targets y_120 / y_60 / y_day; saves preds + books."""
import sys, time, os; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml")
import dlib, rlib, numpy as np, pandas as pd
OUT = "/home/user/llm_sharpe_optimization/agents/dayml/out"; n = 1500
targets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["y_120", "y_60", "y_day"]
retrain = sys.argv[2] if len(sys.argv) > 2 else "Y"
if os.path.exists(f"{OUT}/D_{n}.parquet"): D = pd.read_parquet(f"{OUT}/D_{n}.parquet"); print("loaded D", len(D), flush=True)
else:
    tks = rlib.universe(n); print("universe", n, "available", len(tks), flush=True)
    t0 = time.time(); D = dlib.build(tks); print("rows", len(D), "names", D.tk.nunique(), f"{time.time()-t0:.0f}s", flush=True)
    D = D.drop(columns=[c for c in D.columns if c.startswith("t30_") or c.startswith("t330_") or c.startswith("spy_t")])
    D.to_parquet(f"{OUT}/D_{n}.parquet")
print(D.groupby("date").size().describe(), flush=True)
sl = {f: pd.read_csv(f"/home/user/llm_sharpe_optimization/sleeves/{f}.csv", index_col=0, parse_dates=True).iloc[:, 0].loc[:rlib.DEV_END] for f in ["daylgb1200_ls20", "onlgb1200q_ls20"]}
spy = rlib.ticker_frame("SPY")["y_cc"].loc[:rlib.DEV_END]
res = {}
for y in targets:
    tag = f"ext_{y}{retrain}"
    t = time.time(); Dp, pred = dlib.run(D, dlib.EXT, y, retrain=retrain); pred.rename(tag).to_frame().to_parquet(f"{OUT}/pred_{tag}_{n}.parquet")
    print(f"--- {tag} {time.time()-t:.0f}s  IC {rlib.ic(Dp, pred, y)}", flush=True)
    for k, w in [(10, "pred"), (20, "pred"), (30, "pred"), (20, "eq")]:
        r = dlib.book(Dp, pred, k=k, y=y, weight=w, name=f"{tag} k{k} {w}"); res[f"{tag}_k{k}{w[0]}"] = r
        if w == "pred":
            dlib.book(Dp, pred, k=k, y=y, weight=w, cost=1.5, name=f"{tag} k{k} {w} @1.5bp")
            res[f"{tag}_k{k}{w[0]}_m300"] = dlib.book(Dp, pred, k=k, y=y, weight=w, min_names=300, name=f"{tag} k{k} {w} min300")
            print(f"   corr day1200 {r.corr(sl['daylgb1200_ls20']):.3f} on1200q {r.corr(sl['onlgb1200q_ls20']):.3f} spy {r.corr(spy):.3f}", flush=True)
    pd.DataFrame(res).to_parquet(f"{OUT}/res_exp2_{retrain}_{n}.parquet")
    for c in res:
        if c.startswith(tag): rlib.fold_oos(res[c], label=c)
print("DONE exp2", flush=True)
