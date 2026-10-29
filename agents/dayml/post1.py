#!/usr/bin/env python3
"""post-hoc construction analysis on saved predictions (cheap): k, weighting, min_names, cost, fold-OOS, correlations."""
import sys, glob, os; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml")
import dlib, rlib, numpy as np, pandas as pd
OUT = "/home/user/llm_sharpe_optimization/agents/dayml/out"; n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
D = pd.read_parquet(f"{OUT}/D_{n}.parquet")
sl = {f: pd.read_csv(f"/home/user/llm_sharpe_optimization/sleeves/{f}.csv", index_col=0, parse_dates=True).iloc[:, 0].loc[:rlib.DEV_END] for f in ["daylgb1200_ls20", "onlgb1200q_ls20"]}
spy = rlib.ticker_frame("SPY")["y_cc"].loc[:rlib.DEV_END]
res = {}
for p in sorted(glob.glob(f"{OUT}/pred_*_{n}.parquet")):
    tag = os.path.basename(p)[5:-len(f"_{n}.parquet")]; pred = pd.read_parquet(p).iloc[:, 0]
    y = {"base_yday": "y_day", "ext_yday": "y_day", "ext_y60": "y_60", "ext_y120": "y_120", "f1000": "y_1000", "f1500": "y_1500"}[tag]
    print(f"=== {tag} y={y}", flush=True)
    for k, w in [(10, "pred"), (15, "pred"), (20, "pred"), (30, "pred"), (20, "eq")]:
        r = dlib.book(D, pred, k=k, y=y, weight=w, name=f"{tag} k{k} {w}"); res[f"{tag}_k{k}{w[0]}"] = r
        if (k, w) == (20, "pred"):
            dlib.book(D, pred, k=k, y=y, weight=w, cost=1.5, name=f"{tag} k{k} {w} @1.5bp")
            print(f"   corr day1200 {r.corr(sl['daylgb1200_ls20']):.3f} on1200q {r.corr(sl['onlgb1200q_ls20']):.3f} spy {r.corr(spy):.3f}", flush=True)
pd.DataFrame(res).to_parquet(f"{OUT}/res_post1_{n}.parquet")
print("--- fold-OOS per candidate"); 
for c in res: rlib.fold_oos(res[c], label=c)
print("--- fold-OOS families")
for fam in ["base_yday", "ext_yday", "ext_y60", "ext_y120", "f1000", "f1500"]:
    rlib.fold_oos({c: res[c] for c in res if c.startswith(fam)}, label=fam)
rlib.fold_oos(res, label="ALL")
