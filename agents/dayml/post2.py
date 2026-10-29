#!/usr/bin/env python3
"""post2: after exp2 (universe 1500) finishes -> save candidates via rlib.save_cand, write REPORT numbers."""
import sys, os; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml")
import dlib, rlib, numpy as np, pandas as pd
OUT = "/home/user/llm_sharpe_optimization/agents/dayml/out"; n = 1500
D = pd.read_parquet(f"{OUT}/D_{n}.parquet")
sl = {f: pd.read_csv(f"/home/user/llm_sharpe_optimization/sleeves/{f}.csv", index_col=0, parse_dates=True).iloc[:, 0].loc[:rlib.DEV_END] for f in ["daylgb1200_ls20", "onlgb1200q_ls20"]}
spy = rlib.ticker_frame("SPY")["y_cc"].loc[:rlib.DEV_END]
CANDS = [  # name, pred tag, target, k, weight
    ("dayml_x120q_k20p", "ext_y_120Q", "y_120", 20, "pred"), ("dayml_x120q_k10p", "ext_y_120Q", "y_120", 10, "pred"),
    ("dayml_x120q_k30p", "ext_y_120Q", "y_120", 30, "pred"), ("dayml_x60q_k20p", "ext_y_60Q", "y_60", 20, "pred"),
    ("dayml_x60q_k10p", "ext_y_60Q", "y_60", 10, "pred"), ("dayml_xdayy_k20p", "ext_y_dayY", "y_day", 20, "pred")]
lines = ["\n## universe(1500) results (post2.py, written unattended)\n", "| candidate | Sh | halves | mdd | fold-OOS | folds neg | @1.5bp | corr day1200 | corr on1200q | corr SPY |", "|---|---|---|---|---|---|---|---|---|---|"]
res = {}
for name, tag, y, k, w in CANDS:
    p = f"{OUT}/pred_{tag}_{n}.parquet"
    if not os.path.exists(p): print("missing", p); continue
    pred = pd.read_parquet(p).iloc[:, 0]; Dp = D
    r = dlib.book(Dp, pred, k=k, y=y, weight=w, name=name); res[name] = r
    r15 = dlib.book(Dp, pred, k=k, y=y, weight=w, cost=1.5, name=name + " @1.5bp", p=False)
    st = rlib.stats(r, name, p=False); oos, tab = rlib.fold_oos(r, label=name)
    rlib.save_cand("dayml", name, r, f"09:31 LightGBM EXT feats, target {y}, retrain {tag[-1]}, L/S k{k} {w}-weighted, universe(1500)")
    lines.append(f"| {name} | {st['sharpe']:.2f} | {st['h1']:.2f}/{st['h2']:.2f} | {st['mdd']*100:.1f}% | {rlib.sh(oos):.2f} | {(tab.fold_sh<0).sum()}/{len(tab)} | {rlib.sh(r15):.2f} | {r.corr(sl['daylgb1200_ls20']):.2f} | {r.corr(sl['onlgb1200q_ls20']):.2f} | {r.corr(spy):.2f} |")
    yr = r.groupby(r.index.year).apply(rlib.sh); lines.append("    yearly: " + " ".join(f"{a%100:02d}:{b:+.1f}" for a, b in yr.items()))
oos, tab = rlib.fold_oos(res, label="dayml family (all saved)"); lines.append(f"\nfamily fold-OOS over all saved candidates: {rlib.sh(oos):.2f}; picks {tab.pick.value_counts().to_dict()}")
oos, tab = rlib.fold_oos({k: v for k, v in res.items() if "x120" in k}, label="x120 family"); lines.append(f"x120 family fold-OOS: {rlib.sh(oos):.2f}")
open("/home/user/llm_sharpe_optimization/agents/dayml/REPORT.md", "a").write("\n".join(lines) + "\n")
print("POST2 DONE", flush=True)
