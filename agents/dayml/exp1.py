#!/usr/bin/env python3
"""exp1: baseline reproduction + feature sets + horizons on the currently available universe (yearly retrain)."""
import sys, time, os; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml")
import dlib, rlib, numpy as np, pandas as pd
OUT = "/home/user/llm_sharpe_optimization/agents/dayml/out"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
tks = rlib.universe(n); print("universe", n, "available", len(tks), flush=True)
t0 = time.time(); D = dlib.build(tks); print("rows", len(D), "names", D.tk.nunique(), f"{time.time()-t0:.0f}s", flush=True)
D.to_parquet(f"{OUT}/D_{n}.parquet")
res = {}
def go(tag, feats, y, **kw):
    t = time.time(); Dp, pred = dlib.run(D, feats, y, **kw); pred.rename(tag).to_frame().to_parquet(f"{OUT}/pred_{tag}_{n}.parquet")
    print(f"--- {tag} ({len(feats)} feats, y={y}) {time.time()-t:.0f}s  IC {dlib.rlib.ic(Dp, pred, y)}", flush=True)
    for k in (20, 30, 40):
        res[f"{tag}_k{k}"] = dlib.book(Dp, pred, k=k, y=y, name=f"{tag} k{k}")
    res[f"{tag}_k20_pw"] = dlib.book(Dp, pred, k=20, y=y, weight="pred", name=f"{tag} k20 predw")
    dlib.book(Dp, pred, k=20, y=y, cost=1.5, name=f"{tag} k20 @1.5bp")
    return Dp, pred
go("base_yday", dlib.BASE, "y_day")
go("ext_yday", dlib.EXT, "y_day")
go("ext_y60", dlib.EXT, "y_60")
go("ext_y120", dlib.EXT, "y_120")
go("f1000", dlib.F1000, "y_1000")
go("f1500", dlib.F1500, "y_1500")
pd.DataFrame(res).to_parquet(f"{OUT}/res_exp1_{n}.parquet")
print("FOLD-OOS per candidate:", flush=True)
for c in res: rlib.fold_oos(res[c], label=c)
rlib.fold_oos({c: res[c] for c in res if c.startswith("ext_yday")}, label="family ext_yday")
rlib.fold_oos({c: res[c] for c in res if c.startswith("base_yday")}, label="family base_yday")
print("DONE exp1", flush=True)
