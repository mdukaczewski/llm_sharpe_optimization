import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/trumpet/strat20/agents/pairs")
from common import *
P, F = load(); T, N = P.T, P.N
m = np.zeros((T, N), bool)
for i in range(260, T): m[i] = base_mask(F, i, 50e6, 5.0) & ~F["is_etf"]
d = F["dret"]
# extreme returns inside the mask
big = np.where(m & (np.abs(d) > 0.5))
print("n |ret|>50% in mask:", len(big[0]))
for t, j in list(zip(*big))[:40]:
    print(P.dates[t].date(), P.tickers[j], f"ret {d[t,j]:+.2f} close {P.close[t-1,j]:.2f}->{P.close[t,j]:.2f} raw {P.close_raw[t-1,j]:.2f}->{P.close_raw[t,j]:.2f} div {P.div[t,j]:.3f} oc {F['oc'][t,j]:+.3f}")
# rank weights on a dummy signal (rev1) and daily contributions
from scipy.stats import rankdata
sig = -d
W = np.zeros((T, N))
for i in range(T):
    idx = np.where(m[i])[0]
    if len(idx) < 50: continue
    r = rankdata(sig[i, idx]) / len(idx) - 0.5 - 0.5 / len(idx)
    w = np.zeros(N); pos = r > 0; neg = r < 0
    w[idx[pos]] = r[pos] / r[pos].sum() * 0.5; w[idx[neg]] = -r[neg] / (-r[neg]).sum() * 0.5
    W[i] = w
print("max |w|:", np.abs(W).max(), "gross:", np.abs(W).sum(1).max())
net, g, info = backtest(P, F, W)
report("rev1 rank", net, g, info, full=True)
gs = g.abs().sort_values(ascending=False)
print(gs.head(15))
Wp = np.vstack([np.zeros((1, N)), W[:-1]]); Wpp = np.vstack([np.zeros((1, N)), Wp[:-1]])
for dt in gs.index[:5]:
    t = P.dates.get_loc(dt)
    contrib = Wpp[t] * np.nan_to_num(d[t]) + (Wp[t] - Wpp[t]) * F["oc"][t]
    o = np.argsort(-np.abs(contrib))[:5]
    print(dt.date(), [(P.tickers[j], round(contrib[j], 4), round(d[t, j], 3), round(F["oc"][t, j], 3), round(Wp[t, j], 4), round(Wpp[t, j], 4)) for j in o])
# also: nextclose variant and no-oc variant
net2, g2, info2 = backtest(P, F, W, exec="nextclose"); report("rev1 rank nextclose", net2, g2, info2)
