"""Momentum strategy (strategy.py rules) with a glitch guard: a name with any
|daily return| > 40% in the last 10 days is excluded at selection, and a held
name is written off at the pre-jump price if the jump is > +100% in one day
(bogus unadjusted reverse split)."""
import numpy as np, pandas as pd
from bt import *; from feats import *; from scipy.stats import rankdata
import strategy as S
P = Panel(path="/home/trumpet/strat20/data/panel.npz"); F = build(P)
big = pd.DataFrame(np.abs(F["dret"]) > 0.40).rolling(10, min_periods=1).max().values.astype(bool)
def ranks(s, m):
    out = np.full_like(s, np.nan); idx = np.where(m & np.isfinite(s))[0]; out[idx] = rankdata(s[idx]) / len(idx); return out
def sel(i):
    m = base_mask(F, i, S.MIN_ADV, S.MIN_PX) & ~big[i]
    v = F["vol60"][i] * np.sqrt(252); s = np.zeros(P.N)
    for lb, skip in S.PARAMS: s += ranks(mom(F, i, lb, skip) / (v + 1e-9), m)
    s = np.where(m & np.isfinite(s), s, -np.inf); idx = np.argsort(-s)[:S.TOP]; return idx[np.isfinite(s[idx])]
eqs = []
for off in S.TRANCHE_OFFSETS:
    reb = F["rebal_m"] + off; reb = reb[reb < P.T]
    eq, info = run_fast(P, lambda i: {int(c): 1.0 / len(sel(i)) for c in sel(i)}, reb, START, END, cost_bps=15)
    eqs.append(eq)
eq = sum(eqs) / 2
report(eq, "momentum with glitch guard")
eq.pct_change().fillna(0).to_csv("sleeves_mom_guard.csv")
# count how many held names had a > +100% day while held (bogus-split suspects)
