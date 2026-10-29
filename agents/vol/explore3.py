"""Phase 3: leveraged-ETF pair decay harvesting with rebalancing bands (no daily rebalance)."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/trumpet/strat20/agents/vol")
from vlib import *
D = load(); ret = D["ret"]; c = D["close"]
def pair_bt(a, b, wa, wb, band, cost_bps=2.0, start=START, end=END, filt=None):
    """Short wa of a and wb of b (weights negative). Hold shares; rebalance to target at next close when
    any |drift - target| > band or filter turns position on/off. Returns daily gross/net frame."""
    R = ret[[a, b]].loc[start:end].fillna(0).values
    idx = ret.loc[start:end].index
    br = np.array([BORROW.get(a, 0.01), BORROW.get(b, 0.01)]) / 252
    tgt0 = np.array([wa, wb]); held = np.zeros(2); pend = None
    f = np.ones(len(idx), bool) if filt is None else filt.reindex(idx).fillna(False).values
    g = np.zeros(len(idx)); n = np.zeros(len(idx)); to = np.zeros(len(idx))
    for i in range(len(idx)):
        r = R[i]; pr = float(held @ r); g[i] = pr
        drift = held * (1 + r) / (1 + pr)
        cost = float(np.sum(np.abs(np.minimum(held, 0)) * br))
        if pend is not None:
            t = float(np.abs(pend - drift).sum()); cost += t * cost_bps / 1e4; to[i] = t; held = pend; pend = None
        else: held = drift
        n[i] = pr - cost
        tgt = tgt0 if f[i] else np.zeros(2)
        if np.abs(held - tgt).max() > band or (f[i] != (np.abs(held).sum() > 0)):
            pend = tgt
    return pd.DataFrame({"gross": g, "net": n, "turn": to, "borrow": 0.0}, index=idx)
for a, b in [("TQQQ", "SQQQ"), ("SPXL", "SPXS"), ("UPRO", "SPXU"), ("TNA", "TZA"), ("SOXL", "SOXS"), ("SSO", "SDS"), ("QLD", "QID")]:
    for band in [0.02, 0.05, 0.10, 0.20]:
        stats(pair_bt(a, b, -0.5, -0.5, band), f"short {a}+{b} 0.5/0.5 band {band}", full=False)
print("== UVXY/SVXY neutral-ish pair: short 0.25 UVXY + short 0.75 SVXY (net ~0 futures exposure, post-2018 ratio) ==")
for band in [0.02, 0.05, 0.10, 0.20]:
    stats(pair_bt("UVXY", "SVXY", -0.25, -0.75, band), f"short UVXY/SVXY .25/.75 band {band}", full=False)
    stats(pair_bt("UVXY", "SVXY", -0.25, -0.75, band, start="2018-03-01"), f"  same, from 2018-03", full=False)
print("== plain short UVXY vs short VIXY (rebalance bands) ==")
for band in [0.02, 0.10]:
    stats(pair_bt("UVXY", "VIXY", -0.33, 0.0, band), f"short UVXY .33 band {band}", full=False)
    stats(pair_bt("VIXY", "UVXY", -0.5, 0.0, band), f"short VIXY .5 band {band}", full=False)
