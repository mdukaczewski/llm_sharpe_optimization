#!/usr/bin/env python3
"""Regime-switch candidates for folds.py: for every xs_{tag}_XSLO{k}.csv that has a hedged
twin xs_{tag}_XSLOH{k}.csv, write xs_{tag}_XSLOR{k}.csv = unhedged book when SPY closed above
its 200-day average on the previous day, hedged book otherwise (signal lagged one day)."""
import glob, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); L = f"{HERE}/agents/intraday/logs"
e = pd.read_csv(f"{HERE}/agents/vol/etf.csv", index_col=0, parse_dates=True)
spy = e["SPY_close"] * e["SPY_adj"]; up = (spy > spy.rolling(200).mean()).shift(1).fillna(True)
n = 0
for p in sorted(glob.glob(f"{L}/xs_*_XSLO[0-9]*.csv")):
    ph = p.replace("_XSLO", "_XSLOH")
    if not os.path.exists(ph): continue
    a = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]; b = pd.read_csv(ph, index_col=0, parse_dates=True).iloc[:, 0].reindex(a.index).fillna(0)
    u = up.reindex(a.index).fillna(True).astype(bool); r = a.where(u, b)
    r.rename("ret").to_csv(p.replace("_XSLO", "_XSLOR"), index_label="date"); n += 1
    sh = lambda x: x.mean() / x.std() * np.sqrt(252)
    print(f"{os.path.basename(p):28s} LO {sh(a):5.2f}  H {sh(b):5.2f}  regime {sh(r):5.2f}  (up-days {u.mean()*100:.0f}%)")
print("written", n)
