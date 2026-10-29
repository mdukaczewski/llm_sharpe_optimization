#!/usr/bin/env python3
"""Cross-sectional versions of the conditional-overnight signal across many
stocks.  s = a*z30 - z60 per name per day (same as on_book.py).
  TS   : long-only time-series book (clip(s,0,cap), equal-vol weights)   [reference]
  XSLS : rank-demeaned long/short, dollar neutral, |long|=|short|=1
  XSTOP: long top-k / short bottom-k equal weight
  XSLO : long-only top-k equal weight (keeps the overnight premium)
  TSH  : TS book hedged with short SPY overnight (beta from trailing 60d)
Overnight return = next day's 09:31 fill / official close - 1; cost per side
(bps) on every unit of |w| (entry + exit), borrow 1%/yr on shorts.
usage: xs_on.py TICKERS cost_bps tag"""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
import ilib; from ilib import *
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import rankdata
tks = sys.argv[1].split(","); cost = float(sys.argv[2]); tag = sys.argv[3]
a, cap, smin = 0.5, 1.5, 0.0
for arg in sys.argv[4:]:
    k_, v_ = arg.split("="); a = float(v_) if k_ == "a" else a; smin = float(v_) if k_ == "smin" else smin
def zsd(x, w=20):
    s = pd.Series(x); return (s / s.rolling(w, min_periods=10).std().shift(1)).values
S, R, V = {}, {}, {}
for tk in tks:
    try: M = Min(tk)
    except Exception as e: print(tk, "skip", e); continue
    z60 = np.nan_to_num(zsd(M.r(329, 389))); z30 = np.nan_to_num(zsd(M.r(-1, 29)))
    s = a * z30 - z60; s[M.half] = np.nan
    ron = np.r_[M.c[1:, 1] / M.close[:-1] - 1, np.nan]        # overnight t -> t+1 09:31
    ron[np.abs(ron) > 0.5] = np.nan                             # unadjusted split in raw minute data
    idx = M.dates
    S[tk] = pd.Series(s, idx); R[tk] = pd.Series(ron, idx)
    V[tk] = pd.Series(ron, idx).rolling(60, min_periods=30).std().shift(1)
S = pd.DataFrame(S).loc[START:END]; R = pd.DataFrame(R).reindex(S.index); V = pd.DataFrame(V).reindex(S.index)
spy = Min("SPY"); spy_on = pd.Series(np.r_[spy.c[1:, 1] / spy.close[:-1] - 1, np.nan], spy.dates).reindex(S.index)
print("names:", S.shape[1], "days:", len(S), flush=True)
def rep(name, w, cost_bps=cost, borrow=0.01):
    """w: DataFrame of weights decided at close t (NaN->0); returns net daily series"""
    w = w.fillna(0.0)
    g = (w * R.fillna(0)).sum(axis=1)
    c = w.abs().sum(axis=1) * 2 * cost_bps / 1e4 + w.clip(upper=0).abs().sum(axis=1) * borrow / 252
    n = g - c
    yr = n.groupby(n.index.year).apply(sh); eq = (1 + n).cumprod(); dd = (eq / eq.cummax() - 1).min()
    h1 = n[n.index < "2021-10-28"]; h2 = n[n.index >= "2021-10-28"]
    print(f"{name:34s} Sh gross {sh(g):5.2f} net {sh(n):5.2f} | ret {n.mean()*252*100:6.2f}% vol {n.std()*np.sqrt(252)*100:5.2f}% mdd {dd*100:6.1f}% | halves {sh(h1):.2f}/{sh(h2):.2f} | gross expo {w.abs().sum(axis=1).mean():.2f} | yrs " + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in yr.items()), flush=True)
    n.rename("ret").rename_axis("date").to_csv(f"/home/trumpet/strat20/agents/intraday/logs/xs_{tag}_{name.split()[0]}.csv", float_format="%.8f")
    return n
# TS long-only (equal-vol weights, scaled to 10%/yr like ilib.combine)
w_ts = S.clip(lower=0, upper=cap) / V
w_ts = w_ts.div(w_ts.abs().sum(axis=1) + 1e-12, axis=0)
ts = rep("TS longonly eqvol", w_ts)
# hedged TS: beta of ts to spy overnight, trailing 60d
beta = (ts.rolling(60).cov(spy_on) / spy_on.rolling(60).var()).shift(1).clip(0, 2).fillna(0)
tsh = ts - beta * spy_on.fillna(0) - beta.diff().abs().fillna(0) * 0.5 / 1e4
print(f"{'TSH hedged (beta~%.2f)' % beta.mean():34s} Sh net {sh(tsh):5.2f} | ret {tsh.mean()*252*100:6.2f}% vol {tsh.std()*np.sqrt(252)*100:5.2f}%", flush=True)
tsh.rename("ret").rename_axis("date").to_csv(f"/home/trumpet/strat20/agents/intraday/logs/xs_{tag}_TSH.csv", float_format="%.8f")
# cross-sectional rank L/S
def rankw(row):
    ok = row.notna(); out = pd.Series(0.0, index=row.index)
    if ok.sum() < 10: return out
    r = rankdata(row[ok]) / ok.sum() - 0.5 - 0.5 / ok.sum(); out[ok] = r
    pos = out > 0; neg = out < 0
    out[pos] /= out[pos].sum(); out[neg] /= -out[neg].sum()
    return out
Wr = S.apply(rankw, axis=1)
rep("XSLS rank", Wr)
rep("XSLS rank volscaled", (Wr / V).div((Wr / V).abs().sum(axis=1) / 2 + 1e-12, axis=0))
for k in [5, 10, 15, 20, 30]:
    def topk(row, k=k):
        ok = row.dropna(); out = pd.Series(0.0, index=row.index)
        if len(ok) < 2 * k + 2: return out
        o = ok.sort_values(); out[o.index[-k:]] = 1.0 / k; out[o.index[:k]] = -1.0 / k; return out
    rep(f"XSTOP{k} top/bottom", S.apply(topk, axis=1))
    def topl(row, k=k):
        ok = row.dropna(); out = pd.Series(0.0, index=row.index)
        if len(ok) < k + 2: return out
        o = ok.sort_values(); sel = o.index[-k:]; out[sel] = np.where(o[sel] > smin, 1.0 / k, 0.0); return out
    nlo = rep(f"XSLO{k} longonly topk (s>0)", S.apply(topl, axis=1))
    bk = (nlo.rolling(60).cov(spy_on) / spy_on.rolling(60).var()).shift(1).clip(0, 2).fillna(0)
    h = nlo - bk * spy_on.fillna(0) - (bk.diff().abs().fillna(0) + 2 * bk) * 0.5 / 1e4
    print(f"{'XSLOH%d hedged (beta~%.2f)' % (k, bk.mean()):34s} Sh net {sh(h):5.2f} | ret {h.mean()*252*100:6.2f}% vol {h.std()*np.sqrt(252)*100:5.2f}%", flush=True)
    h.rename("ret").rename_axis("date").to_csv(f"/home/trumpet/strat20/agents/intraday/logs/xs_{tag}_XSLOH{k}.csv", float_format="%.8f")
# dead-zone L/S: long s>1, short s<-1, equal weight
def dz(row):
    out = pd.Series(0.0, index=row.index); L = row > 1.0; Sh_ = row < -1.0
    if L.sum(): out[L] = 1.0 / L.sum()
    if Sh_.sum(): out[Sh_] = -1.0 / Sh_.sum()
    return out
rep("XSDZ dead-zone 1.0 L/S", S.apply(dz, axis=1))
