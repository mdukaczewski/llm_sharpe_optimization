#!/usr/bin/env python3
"""ETF-only sleeves: daily mean-reversion rules on index ETFs, cross-sectional
ETF reversal/momentum. Uses panel_liquid; extracts ETF series once."""
import numpy as np, pandas as pd, os
from bt import Panel
sh = lambda r: r.mean() / (r.std() + 1e-12) * np.sqrt(252)
START, END = "2016-10-28", "2026-10-27"
cache = "/home/trumpet/strat20/data/etf_daily.pkl"
if os.path.exists(cache):
    D = pd.read_pickle(cache)
else:
    P = Panel(path="/home/trumpet/strat20/data/panel_liquid.npz")
    etfs = [t for t in P.tickers[P.ttype == "ETF"]]
    adv = pd.DataFrame(P.dollar_vol[:, [P.col(t) for t in etfs]]).rolling(20, min_periods=15).mean().values
    keep = [t for t, m in zip(etfs, np.nanmax(np.nan_to_num(adv), axis=0)) if m >= 20e6]
    D = {}
    for f in ["close", "open", "ret", "dollar_vol"]:
        D[f] = pd.DataFrame(getattr(P, f)[:, [P.col(t) for t in keep]], index=P.dates, columns=keep)
    pd.to_pickle(D, cache)
C, O, R, DV = D["close"], D["open"], D["ret"], D["dollar_vol"]
print("ETFs:", C.shape[1], flush=True)
def rep(name, r, cost_yr=0.0):
    r = r.loc[START:END] - cost_yr / 252
    yr = r.groupby(r.index.year).apply(sh); eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    print(f"{name:44s} Sharpe {sh(r):5.2f} ret {r.mean()*252*100:6.1f}% vol {r.std()*np.sqrt(252)*100:5.1f}% maxDD {dd*100:5.1f}% worst-yr {yr.min():5.2f} yrs " + " ".join(f"{v:.1f}" for v in yr), flush=True)
    return r
out = {}
# --- time-series MR on index ETFs: enter at close t+1 when rule true at close t; exit when close > MA5 ---
for t in ["SPY", "QQQ", "IWM", "DIA"]:
    c = C[t].ffill(); r = R[t].fillna(0)
    rsi_dn = c.pct_change().rolling(3).sum()
    for name, cond in [("3d ret<-2%", rsi_dn < -0.02), ("close<5d low", c <= c.rolling(5).min()), ("2 down days", (c.diff() < 0) & (c.diff().shift(1) < 0))]:
        pos = pd.Series(0.0, index=c.index)
        inpos = False
        ma5 = c.rolling(5).mean()
        for i in range(10, len(c)):
            if inpos and c.iloc[i] > ma5.iloc[i]: inpos = False
            elif not inpos and cond.iloc[i] and c.iloc[i] > c.rolling(200).mean().iloc[i] * 0.9: inpos = True
            pos.iloc[i] = 1.0 if inpos else 0.0
        w = pos.shift(2).fillna(0)   # decided t, filled close t+1, earns from t+2
        trades = w.diff().abs().sum() / 2
        rr = w * r
        out[f"mr_{t}_{name}"] = rep(f"MR {t} {name} (exit>MA5)", rr, cost_yr=trades * 2 * 0.5 / 1e4 / 10)
# --- cross-sectional weekly reversal / momentum among liquid ETFs (ex leveraged/vol) ---
lev = [t for t in C.columns if t in ("TQQQ","SQQQ","SPXL","SPXS","UPRO","SPXU","SSO","SDS","QLD","QID","TNA","TZA","SOXL","SOXS","UVXY","SVXY","VIXY","VXX","VIXM","VXZ","LABU","LABD","FAS","FAZ","NUGT","DUST","JNUG","JDST","TMF","TMV","UCO","SCO","BOIL","KOLD","UVIX","SVIX","TSLL","NVDL","YINN","YANG","TECL","TECS","ERX","ERY","DRN","DRV","TBT","UDOW","SDOW","URTY","SRTY","QQQM")]
U = [t for t in C.columns if t not in lev and not t.startswith(("UVXY","SVXY"))]
Ru = R[U]; Cu = C[U].ffill(); DVu = DV[U].rolling(20, min_periods=15).mean()
liq = DVu >= 20e6
from scipy.stats import rankdata
def xs_ls(score, mask, hold, name, cost=2.0):
    W = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for i in range(0, len(score), hold):
        s = score.iloc[i].where(mask.iloc[i]); ok = s.notna()
        if ok.sum() < 20: continue
        rk = pd.Series(rankdata(s[ok]) / ok.sum() - 0.5, index=s[ok].index)
        w = pd.Series(0.0, index=score.columns); w[rk.index] = rk
        w[w > 0] /= w[w > 0].sum(); w[w < 0] /= -w[w < 0].sum()
        W.iloc[i:i + hold] = w.values
    Wl = W.shift(2).fillna(0)
    rr = (Wl * Ru.fillna(0)).sum(axis=1)
    turn = Wl.diff().abs().sum(axis=1)
    net = rr - turn * cost / 1e4 - Wl.clip(upper=0).abs().sum(axis=1) * 0.01 / 252
    out[name] = rep(name, net)
lr = np.log(Cu)
for lb in [5, 10, 21]:
    for hold in [5, 10]:
        xs_ls(-(lr - lr.shift(lb)), liq, hold, f"ETF xs reversal lb{lb} hold{hold}")
for lb in [63, 126, 252]:
    xs_ls((lr.shift(21) - lr.shift(lb)), liq, 21, f"ETF xs momentum lb{lb} hold21")
pd.DataFrame(out).to_csv("logs/etf_mr_sleeves.csv")
