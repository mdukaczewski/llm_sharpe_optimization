#!/usr/bin/env python3
"""
Multi-sleeve portfolio: combines the daily NET return series of the sleeves in
sleeves/ with inverse-volatility weights (full-period vol, no return-based
selection) and reports the result.  Also reports the walk-forward variant
(weights from trailing 2 years only, refreshed monthly) and the conservative
variant that uses only the out-of-sample overnight names.

Sleeves (each file = date,ret of daily NET returns, 2016-10-28..2026-10-27):
  mom_longonly   strategy.py: vol-scaled 12-1 momentum, top 25, 2 tranches (15 bps)
  vol_final_net  agents/vol: short 0.5 VIXY + long 0.5 VIXM when SPY rv10 > 12%,
                 flat Thursdays (2 bps + borrow)
  ml_ls          agents/ml: LightGBM ensemble L/S top25/bottom25, 5 tranches,
                 SPY-hedged (10 bps + 1% borrow)
  xslo10_50      xs_on.py: conditional-overnight signal s = 0.5*z30 - z60 on 50
                 mega/large caps, each close long the 10 highest s (if s>0),
                 equal weight, exit 09:31 (1 bp per side)
  on52_ts        on_book.py: time-series version of the same signal on 52
                 names, equal-vol weights (1 bp per side)
  xslo10_oos40   same as xslo10_50 but only the 40 names never used in discovery
  xslo20s1_150   xs_on.py on 150 names: long the 20 highest s (s > 1), exit 09:31
  onrule_x60_lo20  on_ml.py rule book: long top 20 by s, exit at the 10:31 bar
  onlgb_ls20     on_ml.py: walk-forward LightGBM on 18 intraday features ->
                 overnight rank; long top 20 / short bottom 20, exit 09:31
  daylgb_ls20    on_ml_day.py: same machinery at 09:31 (gap, first minute, prior
                 day) -> hold 09:32 to the close, long/short top/bottom 20
  vol_ens2_net   agents/vol ENS2 (spread + post-spike short VIXY), 1.14
  onlgb300_ls20  on_ml.py over 300 names (data/next150b.txt added): LightGBM
                 overnight L/S top20/bottom20, exit 09:31 (1 bp per side), net 1.93
  onlgb1200q_ls20 on_ml.py retrain=Q over 1200 names (data/next600d.txt added):
                 LightGBM refit every quarter, overnight L/S top20/bottom20 (1 bp), net 3.22
  daylgb1200_ls20 on_ml_day.py over 1200 names, 09:32 -> close L/S top20/bottom20, net 1.99

usage: python3 portfolio.py
"""
import numpy as np, pandas as pd, os
HERE = os.path.dirname(os.path.abspath(__file__))
START, END = "2016-10-28", "2026-10-27"
MAIN = ["mom_longonly", "vol_ens2_net", "ml_ls", "xslo20s1_150", "onrule_x60_lo20", "onlgb1200q_ls20", "daylgb1200_ls20"]
MAIN_V2 = ["mom_longonly", "vol_ens2_net", "ml_ls", "xslo20s1_150", "onrule_x60_lo20", "onlgb300_ls20", "daylgb_ls20"]  # 2026-10-29 09:00 version, Sharpe 3.05
CONSERVATIVE = ["mom_longonly", "vol_final_net", "ml_ls", "xslo10_oos40"]

def sh(r): return r.mean() / (r.std() + 1e-12) * np.sqrt(252)
def load(name):
    s = pd.read_csv(f"{HERE}/sleeves/{name}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return s[~s.index.duplicated()].sort_index().loc[START:END]
def stats(r, name):
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min(); yr = r.groupby(r.index.year).apply(sh)
    print(f"{name:36s} Sharpe {sh(r):5.2f}  ret {r.mean()*252*100:5.1f}%  vol {r.std()*np.sqrt(252)*100:5.1f}%  maxDD {dd*100:6.1f}%  worst-yr {yr.min():5.2f}  CAGR {(eq.iloc[-1])**(252/len(r))*100-100:5.1f}%")
    return eq
def combo(names, label):
    df = pd.DataFrame({n: load(n) for n in names}).fillna(0.0)
    print(f"\n=== {label}: {', '.join(names)}")
    for n in names: stats(df[n], "  " + n)
    print("  correlations:\n" + df.corr().round(2).to_string())
    w = 1.0 / df.std(); w = w / w.sum()
    print("  inverse-vol weights: " + " ".join(f"{n}={x:.2f}" for n, x in w.items()))
    r = (df * w).sum(axis=1)
    eq = stats(r, "  PORTFOLIO inverse-vol")
    # walk-forward weights: trailing 2y inverse vol x max(Sharpe,0), monthly refresh
    out = []; months = df.resample("ME").last().index
    for k in range(24, len(months)):
        hist = df.loc[months[k - 24]:months[k - 1]]
        ww = (hist.mean() / (hist.std() + 1e-12)).clip(lower=0) / (hist.std() + 1e-12); ww = ww / (ww.sum() + 1e-12)
        out.append((df.loc[months[k - 1] + pd.Timedelta(days=1):months[k]] * ww).sum(axis=1))
    stats(pd.concat(out), "  PORTFOLIO walk-forward (from 2018-11)")
    return r, eq
if __name__ == "__main__":
    r, eq = combo(MAIN, "MAIN")
    combo(MAIN_V2, "MAIN_V2 (300-name books, previous version)")
    rc, eqc = combo(CONSERVATIVE, "CONSERVATIVE (OOS overnight names only)")
    os.makedirs(f"{HERE}/out", exist_ok=True)
    pd.DataFrame({"main_ret": r, "main_equity": eq, "conservative_ret": rc, "conservative_equity": eqc}).to_csv(f"{HERE}/out/portfolio_equity.csv")
    print("\nwritten out/portfolio_equity.csv")
