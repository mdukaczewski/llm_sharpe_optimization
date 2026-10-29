#!/usr/bin/env python3
"""Combine daily net-return sleeves (CSV files: date,ret) into one portfolio.
  python combine.py sleeves/*.csv
Reports each sleeve's Sharpe, the correlation matrix, inverse-vol and
equal-risk combos, and a walk-forward (trailing 2y) weighted combo."""
import sys, numpy as np, pandas as pd
def sh(r): return r.mean() / (r.std() + 1e-12) * np.sqrt(252)
def load(p):
    s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
    return s[~s.index.duplicated()].sort_index()
sl = {p.split("/")[-1].replace(".csv", ""): load(p) for p in sys.argv[1:]}
df = pd.DataFrame(sl).loc["2016-10-28":"2026-10-27"].fillna(0.0)
print("sleeves:")
for c in df: 
    r = df[c]; yr = r.groupby(r.index.year).apply(sh)
    print(f"  {c:30s} Sharpe {sh(r):5.2f} ret {r.mean()*252*100:6.1f}% vol {r.std()*np.sqrt(252)*100:5.1f}% worst-yr {yr.min():5.2f}")
print("\ncorrelation:\n", df.corr().round(2).to_string())
def rep(name, r):
    yr = r.groupby(r.index.year).apply(sh); eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    print(f"{name:34s} Sharpe {sh(r):5.2f} ret {r.mean()*252*100:6.1f}% vol {r.std()*np.sqrt(252)*100:5.1f}% maxDD {dd*100:5.1f}% worst-yr {yr.min():5.2f}  yrs " + " ".join(f"{y}:{v:.1f}" for y, v in yr.items()))
w = 1 / df.std(); w /= w.sum(); rep("inverse-vol", (df * w).sum(axis=1))
rep("equal-weight", df.mean(axis=1))
# walk-forward: weights = trailing-2y inverse-vol x max(Sharpe,0), refreshed monthly
out = []
months = df.resample("M").last().index
for k in range(24, len(months)):
    hist = df.loc[months[k - 24]:months[k - 1]]
    sh_h = hist.mean() / (hist.std() + 1e-12); w = (sh_h.clip(lower=0) / (hist.std() + 1e-12)); w = w / (w.sum() + 1e-12)
    seg = df.loc[months[k - 1] + pd.Timedelta(days=1):months[k]]
    out.append((seg * w).sum(axis=1))
rep("walk-forward (2y trailing)", pd.concat(out))
