#!/usr/bin/env python3
"""lmt lane library: passive (limit-at-the-touch) execution applied to book positions with the minute-bar fill proxy.
ext(): the step-0 extraction (out/ext.parquet) with next-day columns (prefix n_) per ticker.
Fill proxy (calibrated in study1/study2 on the NBBO panel): a buy limit posted at T sits at
  L = round(mid_hat x (1 - s x hs_pred / 1e4), 2), mid_hat = close of the minute bar before T, hs_pred = liq.py predicted
  half-spread for that time; strict fill within W bars iff the min low of bins b0..b0+W-1 < L (sell: max high > L).
  Fill price = L.  Unfilled orders cost nothing; every filled order pays $3 on a position of CAP / (2k)."""
import os, sys, numpy as np, pandas as pd
R = "/home/user/llm_sharpe_optimization"; sys.path.insert(0, R); sys.path.insert(0, f"{R}/agents/onml4")
import rlib
OUT = f"{R}/agents/lmt/out"; LOG = f"{R}/agents/lmt/log.txt"; CAP = 400000.0; ORDER = 3.0
def log(s):
    print(s, flush=True); open(LOG, "a").write(s + "\n")

_E = None
def ext(cols=None):
    global _E
    if _E is None:
        E = pd.read_parquet(f"{OUT}/ext.parquet"); E = E[E.date <= rlib.DEV_END].sort_values(["tk", "date"]).reset_index(drop=True)
        _E = E
    return _E

def next_day(E, cols):
    """columns of the next trading day of the same ticker, prefixed n_"""
    g = E.groupby("tk", sort=False)
    N = pd.DataFrame({"n_" + c: g[c].shift(-1) for c in cols}); N["n_date"] = g["date"].shift(-1)
    return N

def lim(mid, hs, side, s=1.0):
    """passive limit: buy at mid(1 - s hs), sell at mid(1 + s hs), rounded to the cent"""
    sg = -1.0 if side == "buy" else 1.0
    return np.round(mid * (1 + sg * s * hs / 1e4), 2)

def filled(lo, hi, L, side, rule="strict"):
    if side == "buy": return (lo < L) if rule == "strict" else (lo <= L + 1e-6)
    return (hi > L) if rule == "strict" else (hi >= L - 1e-6)

def mdd(r):
    eq = (1 + r.fillna(0)).cumprod(); return (eq / eq.cummax() - 1).min()

def metrics(r):
    r = r.loc[rlib.START:rlib.DEV_END].fillna(0.0)
    fo = rlib.fold_oos(r, p=False)[0]
    h1 = r[r.index < "2021-10-28"]; h2 = r[r.index >= "2021-10-28"]
    yr = r.groupby(r.index.year).apply(rlib.sh)
    return dict(sh=round(rlib.sh(r), 2), fold=round(rlib.sh(fo), 2), h1=round(rlib.sh(h1), 2), h2=round(rlib.sh(h2), 2), mdd=round(mdd(r) * 100, 1),
                ret=round(r.mean() * 252 * 100, 1), vol=round(r.std() * np.sqrt(252) * 100, 1), yrs=" ".join(f"{v:+.1f}" for v in yr.values))

def daily(L, k, ls=True, dates=None):
    """L: positions with w (signed 1/k), ret (gross in the position's own direction? no: signed price return), cost (bps, both legs),
    fill (bool: the position traded), norders (filled orders).  Returns daily net series incl. commission and borrow."""
    L = L.copy(); f = L.fill.astype(float)
    L["pnl"] = f * (L.w * L.ret - L.w.abs() * L.cost / 1e4 - np.where(L.w < 0, L.w.abs(), 0) * 0.01 / 252) - L.norders * ORDER / CAP
    r = L.groupby("date").pnl.sum()
    if dates is not None: r = r.reindex(dates).fillna(0.0)
    return r.loc[rlib.START:rlib.DEV_END]

def per_trade(L):
    x = L[L.fill]; sg = np.sign(x.w)
    return dict(fill=round(L.fill.mean() * 100, 1), n_tr=len(x), gross=round((sg * x.ret).mean() * 1e4, 2), cost=round(x.cost.mean(), 2),
                net=round((sg * x.ret - x.cost / 1e4).mean() * 1e4, 2))
