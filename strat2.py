#!/usr/bin/env python3
import numpy as np, pandas as pd, sys, time, itertools
from bt import *
from feats import *

P = Panel(); F = build(P)
print("features built", flush=True)

def make_fn(lb=252, skip=21, top=30, min_adv=10e6, score="vs", weight="eq", regime="none",
            stock_trend=False, maxw=0.1):
    def fn(i):
        m = base_mask(F, i, min_adv)
        if stock_trend:
            m &= (F["closef"][i] > F["ma100"][i]) & (F["closef"][i] >= 0.75 * F["hi252"][i])
        s = mom(F, i, lb, skip)
        v = F["vol60"][i] * np.sqrt(252)
        if score == "vs":
            s = s / (v + 1e-9)
        elif score == "raw":
            pass
        elif score == "sh":   # sharpe-like: mean/std over lookback window, excl skip
            s = s / (F["vol120"][i] * np.sqrt(252) + 1e-9)
        s = np.where(m & np.isfinite(s), s, -np.inf)
        if regime == "spy200" and F["spy_c"][i] < F["spy_ma200"][i]:
            return {}
        idx = np.argsort(-s)[:top]
        idx = idx[np.isfinite(s[idx])]
        if len(idx) == 0:
            return {}
        if weight == "eq":
            w = np.full(len(idx), 1.0 / len(idx))
        else:  # inverse vol, capped
            iv = 1.0 / (v[idx] + 0.05)
            w = iv / iv.sum()
            w = np.minimum(w, maxw); w = w / w.sum()
        return {int(c): float(x) for c, x in zip(idx, w)}
    return fn

grid = []
for lb, score, top, min_adv, weight, regime, st in itertools.product(
        [252, 126], ["vs", "raw"], [10, 20, 40], [10e6, 50e6], ["eq", "iv"], ["none", "spy200"], [False, True]):
    grid.append(dict(lb=lb, skip=21, top=top, min_adv=min_adv, score=score, weight=weight, regime=regime, stock_trend=st))
print(len(grid), "configs", flush=True)
res = []
for c in grid:
    fn = make_fn(**c)
    eq, info = run_fast(P, fn, F["rebal_m"], START, END, cost_bps=15)
    s = stats(eq)
    res.append(dict(**c, cagr=s["cagr"], sharpe=s["sharpe"], maxdd=s["maxdd"], vol=s["vol"], turn=info["turnover"]))
    print(f"{c}  CAGR {s['cagr']*100:.1f}% Sh {s['sharpe']:.2f} DD {s['maxdd']*100:.0f}%", flush=True)
df = pd.DataFrame(res).sort_values("cagr", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(df.head(40))
df.to_csv("logs/strat2_grid.csv", index=False)
