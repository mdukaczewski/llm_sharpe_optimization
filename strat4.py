#!/usr/bin/env python3
"""Robustness around the vol-scaled 12-1 momentum core + residual momentum."""
import numpy as np, pandas as pd, sys, time, itertools
from bt import *
from feats import *

P = Panel(); F = build(P)
spy_r = F["spy_ret"]
# rolling 252d beta vs SPY and residual momentum
dret = F["dret"]
W = 252
def rmean(a, w): return pd.DataFrame(a).rolling(w, min_periods=int(w*0.9)).mean().values
mx = pd.Series(spy_r).rolling(W, min_periods=int(W*0.9)).mean().values[:, None]
my = rmean(dret, W)
mxy = rmean(dret * spy_r[:, None], W)
vx = pd.Series(spy_r).rolling(W, min_periods=int(W*0.9)).var().values[:, None]
beta = (mxy - mx * my) / (vx + 1e-12)
beta = np.clip(np.nan_to_num(beta), -1, 4)
resid = dret - beta * spy_r[:, None]
F["resid_sum252"] = rolling_sum(resid, 252)
F["resid_sum21"] = rolling_sum(resid, 21)
F["resid_std252"] = rolling_std(resid, 252, 200)
F["beta"] = beta
print("features built", flush=True)

def make_fn(top=25, min_adv=50e6, score="vs", lb=252, skip=21, volw=60, min_px=5.0, maxbeta=None):
    def fn(i):
        m = base_mask(F, i, min_adv, min_px)
        v = F[f"vol{volw}"][i] * np.sqrt(252)
        if score == "vs":
            s = mom(F, i, lb, skip) / (v + 1e-9)
        elif score == "resid":
            s = (F["resid_sum252"][i] - F["resid_sum21"][i]) / (F["resid_std252"][i] * np.sqrt(252) + 1e-9)
        elif score == "residraw":
            s = (F["resid_sum252"][i] - F["resid_sum21"][i])
        elif score == "mix":
            s = 0.5 * mom(F, i, lb, skip) / (v + 1e-9) + 0.5 * (F["resid_sum252"][i] - F["resid_sum21"][i]) / (F["resid_std252"][i] * np.sqrt(252) + 1e-9)
        if maxbeta is not None:
            m &= F["beta"][i] <= maxbeta
        s = np.where(m & np.isfinite(s), s, -np.inf)
        idx = np.argsort(-s)[:top]; idx = idx[np.isfinite(s[idx])]
        if len(idx) == 0: return {}
        return {int(c): 1.0 / len(idx) for c in idx}
    return fn

base = dict(top=25, min_adv=50e6, score="vs", lb=252, skip=21, volw=60, min_px=5.0)
variants = [("base", {})]
for top in [15, 20, 30, 40]: variants.append((f"top={top}", dict(top=top)))
for adv in [20e6, 100e6, 200e6]: variants.append((f"adv={adv/1e6:.0f}M", dict(min_adv=adv)))
for skip in [5, 10, 42]: variants.append((f"skip={skip}", dict(skip=skip)))
for lb in [189, 210, 300]: variants.append((f"lb={lb}", dict(lb=lb)))
for vw in [20, 120]: variants.append((f"volw={vw}", dict(volw=vw)))
variants.append(("px10", dict(min_px=10.0)))
for sc in ["resid", "residraw", "mix"]: variants.append((f"score={sc}", dict(score=sc)))
for sc in ["resid", "mix"]:
    for top in [15, 40]: variants.append((f"score={sc} top={top}", dict(score=sc, top=top)))
variants.append(("maxbeta=1.5", dict(maxbeta=1.5)))

rows = []
def evaluate(name, cfg, reb, cost=15):
    fn = make_fn(**{**base, **cfg})
    eq, info = run_fast(P, fn, reb, START, END, cost_bps=cost)
    s = stats(eq)
    c1 = stats(eq[:"2021-10-27"])["cagr"]; c2 = stats(eq["2021-10-27":])["cagr"]
    rows.append(dict(name=name, cagr=s["cagr"], sharpe=s["sharpe"], maxdd=s["maxdd"], h1=c1, h2=c2, turn=info["turnover"]))
    print(f"{name:28s} CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:.2f} DD {s['maxdd']*100:4.0f}% h1 {c1*100:4.0f}% h2 {c2*100:4.0f}% turn {info['turnover']:.0f}", flush=True)
    return eq

for name, cfg in variants:
    evaluate(name, cfg, F["rebal_m"])
# rebalance-offset and cost sensitivity for base
for off in [5, 10, 15]:
    reb = F["rebal_m"] + off
    reb = reb[reb < P.T]
    evaluate(f"base offset+{off}d", {}, reb)
evaluate("base cost30", {}, F["rebal_m"], cost=30)
evaluate("base cost50", {}, F["rebal_m"], cost=50)
pd.DataFrame(rows).to_csv("logs/strat4.csv", index=False)
