import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/trumpet/strat20/agents/pairs")
from common import *
import resid_rev as RR
P, F, T, N = RR.P, RR.F, RR.T, RR.N
m = np.zeros((T, N), bool)
for i in range(260, T): m[i] = base_mask(F, i, 50e6, 5.0) & ~F["is_etf"]
res = RR.residuals(252)
rvol = rolling_std(np.nan_to_num(res), 60, 40)
d = F["dret"]; spy = F["spy_ret"]
G = {}
for k in [1, 10]:
    sig = -rolling_sum(np.nan_to_num(res), k) / (rvol * np.sqrt(k) + 1e-9)
    W = RR.rank_weights(sig, m)
    net, g, info = backtest(P, F, W); G[k] = g
    report(f"rev{k} rank", net, g, info)
    Wp = np.vstack([np.zeros((1, N)), W[:-1]]); Wpp = np.vstack([np.zeros((1, N)), Wp[:-1]])
    for dt in ["2020-03-16", "2025-04-09", "2022-11-10"]:
        t = P.dates.get_loc(pd.Timestamp(dt))
        L = Wpp[t] > 0; S = Wpp[t] < 0
        dd = np.nan_to_num(d[t])
        print(dt, f"long n={L.sum()} w={Wpp[t][L].sum():.3f} ret_eq={dd[L].mean():+.3f} ret_w={(Wpp[t][L]*dd[L]).sum()/Wpp[t][L].sum():+.3f} | short n={S.sum()} w={Wpp[t][S].sum():.3f} ret_eq={dd[S].mean():+.3f} ret_w={(Wpp[t][S]*dd[S]).sum()/Wpp[t][S].sum():+.3f} | carried pnl {np.dot(Wpp[t], dd):+.4f} new pnl {np.dot(Wp[t]-Wpp[t], F['oc'][t]):+.4f} g={g.loc[dt]:+.4f} spy={spy[t]:+.3f}")
        # weights at t-2 vs t-1 difference
        print("   turnover that day", np.abs(Wp[t]-Wpp[t]).sum(), " sum Wp", Wp[t].sum(), " sum Wpp", Wpp[t].sum())
print("corr gross rev1 vs rev10:", np.corrcoef(G[1], G[10])[0, 1])
# regress gross on spy
for k in G:
    b = np.polyfit(spy[-len(G[k]):], G[k].values, 1)[0]; print(f"beta of gross rev{k} to SPY: {b:.2f}")
