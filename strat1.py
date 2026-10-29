#!/usr/bin/env python3
"""Strategy experiments round 1: cross-sectional momentum variants."""
import numpy as np, pandas as pd, sys, time
from bt import *

START, END = "2016-10-28", "2026-10-27"
t0 = time.time()
P = Panel()
print(f"panel {P.T}x{P.N} loaded in {time.time()-t0:.0f}s", flush=True)

# ---------- universe: common stocks, liquid, price >= 5 ----------
is_cs = np.isin(P.ttype, ["CS", ""])          # common stock or unknown type
# unknown types: keep only if looks like plain symbol and not 5-letter ending W/R/U/P/Q
tk = P.tickers
sus = np.array([len(t) == 5 and t[-1] in "WRUPQ" for t in tk])
is_cs &= ~sus
print("universe tickers (static):", is_cs.sum(), flush=True)

close = P.close
adv20 = rolling_mean(P.dollar_vol, 20, 15)      # 20d avg dollar volume
adv20 = np.where(np.isnan(adv20), 0, adv20)
px_ok = close >= 5.0
has_hist = ~np.isnan(ffill(close, 5))

def liquid_mask(i, min_adv=5e6, top=None):
    m = is_cs & px_ok[i] & (adv20[i] >= min_adv) & ~np.isnan(close[i])
    return m

# ---------- signals ----------
logc = np.log(ffill(close, 5))
def mom(i, lb, skip):
    """return over [i-lb, i-skip]"""
    a = logc[i - lb]; b = logc[i - skip]
    return b - a

dret = np.where(np.isnan(P.ret), 0.0, P.ret)
vol60 = rolling_std(dret, 60, 40)

spy = P.col("SPY")
spy_c = ffill(close[:, spy:spy+1])[:, 0]
spy_ma200 = pd.Series(spy_c).rolling(200).mean().values

rebal = month_ends(P.dates)

def make_mom_fn(lb=252, skip=21, top=30, min_adv=5e6, regime=False, volscale=False, min_days=252):
    def fn(i):
        m = liquid_mask(i, min_adv)
        # require history
        m &= ~np.isnan(logc[i - lb]) & ~np.isnan(logc[i - lb + 5])
        s = mom(i, lb, skip)
        if volscale:
            s = s / (vol60[i] * np.sqrt(252) + 1e-9)
        s = np.where(m, s, -np.inf)
        if regime and spy_c[i] < spy_ma200[i]:
            return {}
        idx = np.argsort(-s)[:top]
        idx = idx[np.isfinite(s[idx])]
        if len(idx) == 0:
            return {}
        w = 1.0 / len(idx)
        return {int(c): w for c in idx}
    return fn

# benchmark
spy_eq = (1 + pd.Series(np.where(np.isnan(P.ret[:, spy]), 0, P.ret[:, spy]), index=P.dates)).cumprod()
i0 = int(np.searchsorted(P.dates, pd.Timestamp(START))); i1 = int(np.searchsorted(P.dates, pd.Timestamp(END), side="right"))
report(spy_eq.iloc[i0:i1], "SPY total return")
qqq = P.col("QQQ")
qqq_eq = (1 + pd.Series(np.where(np.isnan(P.ret[:, qqq]), 0, P.ret[:, qqq]), index=P.dates)).cumprod()
report(qqq_eq.iloc[i0:i1], "QQQ total return")

configs = [
    dict(lb=252, skip=21, top=30),
    dict(lb=252, skip=21, top=50),
    dict(lb=252, skip=21, top=30, regime=True),
    dict(lb=126, skip=21, top=30),
    dict(lb=126, skip=21, top=30, regime=True),
    dict(lb=252, skip=21, top=30, volscale=True),
    dict(lb=252, skip=21, top=30, volscale=True, regime=True),
    dict(lb=126, skip=0, top=30, volscale=True, regime=True),
    dict(lb=252, skip=21, top=30, min_adv=20e6, regime=True),
    dict(lb=252, skip=21, top=100, min_adv=20e6, regime=True),
]
for c in configs:
    fn = make_mom_fn(**c)
    eq, info = run_fast(P, fn, rebal, START, END, cost_bps=15)
    report(eq, "mom " + " ".join(f"{k}={v}" for k, v in c.items()), extra=dict(turn=info["turnover"], hold=info["avg_hold"]))
