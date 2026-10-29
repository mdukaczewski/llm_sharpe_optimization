#!/usr/bin/env python3
"""exp2: EXT features on universe(1500) for the given targets; books, fold-OOS, correlations, save_cand and REPORT
numbers all in-process (no stacked frame is written to disk; predictions kept as small npz for recovery).
usage: exp2.py y_120,y_60 Q   |   exp2.py y_day Y"""
import sys, time, os; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml")
import dlib, rlib, numpy as np, pandas as pd
OUT = "/home/user/llm_sharpe_optimization/agents/dayml/out"; n = 1500
targets = sys.argv[1].split(","); retrain = sys.argv[2]
tks = rlib.universe(n); print("universe", n, "available", len(tks), flush=True)
t0 = time.time(); D = dlib.build(tks); print("rows", len(D), "names", D.tk.nunique(), f"{time.time()-t0:.0f}s", flush=True)
D = D.drop(columns=[c for c in D.columns if c.startswith("t30_") or c.startswith("t330_") or c.startswith("spy_t")])
print(D.groupby("date").size().describe().to_dict(), flush=True)
sl = {f: pd.read_csv(f"/home/user/llm_sharpe_optimization/sleeves/{f}.csv", index_col=0, parse_dates=True).iloc[:, 0].loc[:rlib.DEV_END] for f in ["daylgb1200_ls20", "onlgb1200q_ls20"]}
spy = rlib.ticker_frame("SPY")["y_cc"].loc[:rlib.DEV_END]
SHORT = {"y_120": "x120", "y_60": "x60", "y_day": "xday"}
lines = [f"\n## universe(1500) results: targets {targets}, retrain {retrain} (exp2.py, written unattended)\n",
         "| candidate | Sh | ret | vol | halves | mdd | fold-OOS | folds neg | @1.5bp | corr day1200 | corr on1200q | corr SPY |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
res = {}
for y in targets:
    tag = f"ext_{y}{retrain}"
    t = time.time(); Dp, pred = dlib.run(D, dlib.EXT, y, retrain=retrain); np.savez_compressed(f"{OUT}/pred_{tag}_{n}.npz", idx=pred.index.values, p=pred.values.astype(np.float32))
    print(f"--- {tag} {time.time()-t:.0f}s  IC {rlib.ic(Dp, pred, y)}", flush=True)
    for k, w in [(10, "pred"), (20, "pred"), (30, "pred"), (20, "eq")]:
        name = f"dayml_{SHORT[y]}{retrain.lower()}_k{k}{w[0]}"
        r = dlib.book(Dp, pred, k=k, y=y, weight=w, name=name); res[name] = r
        r15 = dlib.book(Dp, pred, k=k, y=y, weight=w, cost=1.5, name=name + " @1.5bp", p=False)
        st = rlib.stats(r, name, p=False); oos, tab = rlib.fold_oos(r, label=name)
        if w == "pred" and k in (10, 20):
            rlib.save_cand("dayml", name, r, f"09:31 LightGBM EXT feats, target {y}, retrain {retrain}, L/S k{k} {w}-weighted, universe(1500)")
            rm = dlib.book(Dp, pred, k=k, y=y, weight=w, min_names=300, name=name + " min300", p=False); print(f"   min300 variant Sh {rlib.sh(rm):.2f} (days {int((rm!=0).sum())} vs {int((r!=0).sum())})", flush=True)
        print(f"   corr day1200 {r.corr(sl['daylgb1200_ls20']):.3f} on1200q {r.corr(sl['onlgb1200q_ls20']):.3f} spy {r.corr(spy):.3f}", flush=True)
        lines.append(f"| {name} | {st['sharpe']:.2f} | {st['ret']*100:.1f}% | {st['vol']*100:.1f}% | {st['h1']:.2f}/{st['h2']:.2f} | {st['mdd']*100:.1f}% | {rlib.sh(oos):.2f} | {(tab.fold_sh<0).sum()}/{len(tab)} | {rlib.sh(r15):.2f} | {r.corr(sl['daylgb1200_ls20']):.2f} | {r.corr(sl['onlgb1200q_ls20']):.2f} | {r.corr(spy):.2f} |")
        yr = r.groupby(r.index.year).apply(rlib.sh); lines.append("| " + " ".join(f"{a%100:02d}:{b:+.1f}" for a, b in yr.items()) + " | | | | | | | | | | | |")
    oos, tab = rlib.fold_oos({c: v for c, v in res.items() if SHORT[y] in c}, label=f"family {y}"); lines.append(f"\nfamily {y} fold-OOS (k10p/k20p/k30p/k20e): {rlib.sh(oos):.2f}, picks {tab.pick.value_counts().to_dict()}")
if len(targets) > 1: oos, tab = rlib.fold_oos(res, label="all"); lines.append(f"all-candidates family fold-OOS: {rlib.sh(oos):.2f}, picks {tab.pick.value_counts().to_dict()}")
open("/home/user/llm_sharpe_optimization/agents/dayml/REPORT.md", "a").write("\n".join(lines) + "\n")
print("DONE exp2", flush=True)
