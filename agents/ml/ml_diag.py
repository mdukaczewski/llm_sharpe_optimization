#!/usr/bin/env python3
"""Audit a saved weight matrix: top contributing (day,name) pairs with raw prices, monthly table,
concentration, holding period, names/day. Usage: ml_diag.py --w tag"""
import numpy as np, pandas as pd, sys, argparse
sys.path.insert(0, "/home/trumpet/strat20"); sys.path.insert(0, "/home/trumpet/strat20/agents/ml")
from mlfeat import *
ap = argparse.ArgumentParser(); ap.add_argument("--w", required=True); ap.add_argument("--min_adv", type=float, default=50e6)
args = ap.parse_args(); OUT = "/home/trumpet/strat20/agents/ml/out"
P = load_sub(args.min_adv); F = build(P); F = extra_feats(P, F)
Wt = np.load(f"{OUT}/W_{args.w}.npy").astype(np.float64)
net, gross, info = backtest_W(P, F, Wt)
i0 = int(np.searchsorted(P.dates, pd.Timestamp(START))); i1 = int(np.searchsorted(P.dates, pd.Timestamp(END), side="right"))
eq = (1 + net).cumprod(); dd = (eq / eq.cummax() - 1)
print(f"net Sharpe {sh(net):.2f} gross {sh(gross):.2f} ann ret {net.mean()*252*100:.1f}% vol {net.std()*np.sqrt(252)*100:.1f}% maxDD {dd.min()*100:.1f}% turn {info['turn_yr']:.1f}x")
print("monthly net returns (%):"); mo = net.groupby([net.index.year, net.index.month]).sum().unstack() * 100
print(mo.round(1).to_string())
# contributions per (day, name) for the nextopen model
cc = F["dret"]; oc = F["oc"]
Wp = np.vstack([np.zeros((1, P.N)), Wt[:-1]]); Wpp = np.vstack([np.zeros((1, P.N)), Wp[:-1]])
C = Wpp * np.nan_to_num(cc) + (Wp - Wpp) * np.nan_to_num(oc)
C[:i0] = 0; C[i1:] = 0
flat = np.argsort(-np.abs(C).ravel())[:25]
print("\ntop |contributions| (day, ticker, weight, cc ret, contribution, raw closes t-2..t+1, adj):")
for f in flat:
    t, j = divmod(f, P.N)
    print(f"{P.dates[t].date()} {P.tickers[j]:6s} w={Wpp[t,j]:+.4f} cc={cc[t,j]:+.3f} contrib={C[t,j]*100:+.2f}%  rawclose={np.round(P.close_raw[t-2:t+2,j],2)} adj={np.round(P.close[t-2:t+2,j]/P.close_raw[t-2:t+2,j],3)} div={P.div[t,j]:.2f}")
tot = C[i0:i1].sum()
pos = C[i0:i1].ravel(); pos = pos[pos != 0]
print(f"\nsum contributions {tot*100:.1f}% total; top 1% of |contribs| account for {np.sort(np.abs(pos))[::-1][:len(pos)//100].sum()/np.abs(pos).sum()*100:.1f}% of abs")
# drop the 20 biggest days
d = gross.copy(); big = d.abs().sort_values(ascending=False).index[:20]; d2 = d.drop(big)
print(f"gross Sharpe without 20 largest |days| {sh(d2):.2f}; those days: " + " ".join(f"{x.date()}:{d[x]*100:+.1f}" for x in big[:10]))
nl = (Wt[i0:i1] > 0).sum(1); ns = (Wt[i0:i1] < 0).sum(1)
print(f"names/day long {nl.mean():.0f} short {ns.mean():.0f}; gross exposure {np.abs(Wt[i0:i1]).sum(1).mean():.2f}; net {Wt[i0:i1].sum(1).mean():+.3f}")
# average holding period = sum|w| / daily turnover
turn = np.abs(Wp - Wpp).sum(1)[i0:i1]
print(f"avg daily one-way turnover {turn.mean()*100:.2f}% of NAV -> avg holding {np.abs(Wt[i0:i1]).sum(1).mean()/turn.mean():.1f} days")
# per-leg and beta
spy = F["spy_ret"][i0:i1]
print(f"beta of net to SPY: {np.cov(net.values, spy)[0,1]/np.var(spy):.3f}; corr {np.corrcoef(net.values, spy)[0,1]:.3f}")
