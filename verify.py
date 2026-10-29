#!/usr/bin/env python3
"""Sanity checks: split handling on known names, holdings, P&L concentration."""
import numpy as np, pandas as pd
from bt import *
from feats import *
from scipy.stats import rankdata
P = Panel(); F = build(P)

def tr(t, a, b):
    c = P.col(t); i0 = np.searchsorted(P.dates, pd.Timestamp(a)); i1 = np.searchsorted(P.dates, pd.Timestamp(b), side="right")
    r = np.nan_to_num(P.ret[i0:i1, c])
    return np.prod(1 + r), np.abs(r).max(), P.dates[i0 + np.abs(r).argmax()].date()
for t in ["AAPL", "NVDA", "TSLA", "AMZN", "GOOGL", "MSFT", "SPY", "QQQ", "META", "NFLX"]:
    try:
        g, mx, d = tr(t, START, END)
        print(f"{t:6s} total return x{g:6.2f}  max|daily|={mx*100:.1f}% on {d}")
    except KeyError:
        print(t, "missing")

def ranks(s, m):
    out = np.full_like(s, np.nan); idx = np.where(m & np.isfinite(s))[0]; out[idx] = rankdata(s[idx]) / len(idx); return out
P4 = ((252, 21), (252, 42), (210, 21), (189, 21))
hist = []
def fn(i, top=25):
    m = base_mask(F, i, 50e6, 5.0); v = F["vol60"][i] * np.sqrt(252)
    s = np.zeros(P.N)
    for lb, skip in P4: s += ranks(mom(F, i, lb, skip) / (v + 1e-9), m)
    s = np.where(m & np.isfinite(s), s, -np.inf)
    idx = np.argsort(-s)[:top]; idx = idx[np.isfinite(s[idx])]
    hist.append((P.dates[i].date(), int(m.sum()), [P.tickers[c] for c in idx]))
    return {int(c): 1.0 / len(idx) for c in idx}
eq, info = run_fast(P, fn, F["rebal_m"], START, END, cost_bps=15)
report(eq, "ens4 top25 monthly (offset 0)")
print("universe size over time:", [(d, n) for d, n, _ in hist[::24]])
for d, n, h in hist[::12]:
    print(d, n, " ".join(h))
print("LATEST", hist[-1][0], " ".join(hist[-1][2]))
# P&L attribution: contribution of each ticker = sum over days of w*r (approx, using held list and equal weights)
contrib = {}
hold_days = {}
i0 = np.searchsorted(P.dates, pd.Timestamp(START))
rebal = set(F["rebal_m"].tolist())
cur = []; k = 0; pend = None
for i in range(i0, np.searchsorted(P.dates, pd.Timestamp(END), side="right")):
    for c in cur:
        r = P.ret[i, c]
        if np.isfinite(r):
            contrib[c] = contrib.get(c, 0) + r / len(cur)
            hold_days[c] = hold_days.get(c, 0) + 1
    if pend is not None:
        cur = pend; pend = None
    if i in rebal and k < len(hist):
        pend = [P.col(t) for t in hist[k][2]]; k += 1
s = pd.Series({P.tickers[c]: v for c, v in contrib.items()}).sort_values(ascending=False)
print("top contributors (sum of weighted daily returns):")
print(s.head(25).round(3).to_string())
print("bottom:"); print(s.tail(10).round(3).to_string())
print("n names ever held:", len(s), " top10 share of positive:", s.head(10).sum() / s[s > 0].sum())
# largest single-day weighted gains/losses -> data error hunt
