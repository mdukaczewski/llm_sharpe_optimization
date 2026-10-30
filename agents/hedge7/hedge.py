#!/usr/bin/env python3
"""Overnight-beta hedge for the auction-executed rule books: r_h = r - beta_t x SPY overnight (official close -> official open,
dated at the decision day like rlib.book), beta_t = trailing 250-day OLS beta of the book on SPY overnight, lagged one day (causal),
clipped to [0, 1.5].  Hedge cost: 1.5 bps per side on |beta| notional + 2 SPY orders x $3 per day on $0.4M (inside the series),
borrow 1%/yr on the short SPY leg.  Candidates are decision-dated (folds.py re-dates the xsrule family)."""
import sys, os; sys.path.insert(0, "/home/user/llm_sharpe_optimization")
import numpy as np, pandas as pd, rlib, liq
L = lambda p: (lambda s: s[~s.index.duplicated()].sort_index())(pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0])
spy = liq.overnight_targets("SPY")["y_oo"]; spy = spy[spy.index <= rlib.DEV_END]
src = {"xsr3_cAx_u50_moo_lo10": "agents/xsrule3/cands/xsr3_cAx_u50_moo_lo10.csv", "xsr3_cAx_u150_fhs10_moo_lo10": "agents/xsrule3/cands/xsr3_cAx_u150_fhs10_moo_lo10.csv",
       "xsr3_cAx_u150_moo_lo10": "agents/xsrule3/cands/xsr3_cAx_u150_moo_lo10.csv"}
for name, p in src.items():
    r = L(p); d = pd.concat([r, spy], axis=1, keys=["r", "m"]).dropna(); d = d[d.index <= rlib.DEV_END]
    cov = (d.r * d.m).rolling(250, min_periods=120).mean() - d.r.rolling(250, min_periods=120).mean() * d.m.rolling(250, min_periods=120).mean()
    beta = (cov / d.m.rolling(250, min_periods=120).var()).shift(1).clip(0, 1.5).fillna(0.0)
    active = (d.r != 0).astype(float)                      # hedge only on nights the book holds positions
    b = beta * active
    rh = d.r - b * d.m - b * 2 * 1.5e-4 - b * 0.01 / 252 - active * (b > 0) * 2 * 3 / 400000
    out = name.replace("xsr3_", "xsr3h_")
    rlib.save_cand("hedge7", out, rh, f"{name} hedged with SPY MOC->MOO at trailing 250d beta (lagged, clipped 0-1.5); hedge cost 1.5bps/side + borrow + 2 SPY orders x $3 on $0.4M inside the series")
    sh = rlib.sh; print(f"{out:34s} unhedged Sh {sh(d.r):.2f} fold {sh(rlib.fold_oos(d.r, p=False)[0]):.2f} | hedged Sh {sh(rh):.2f} fold {sh(rlib.fold_oos(rh, p=False)[0]):.2f} | mean beta {b[b>0].mean():.2f} | corr with SPY o/n {d.r.corr(d.m):.2f} -> {rh.corr(d.m):.2f}")
