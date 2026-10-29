#!/usr/bin/env python3
"""Daily-data time-series sleeves on ETFs: overnight/intraday, turn-of-month,
multi-asset trend, VIX ETP carry proxies. Reports Sharpe net of costs."""
import numpy as np, pandas as pd
from ls import *
P = Panel(); F = build(P); F = extra_feats(P, F)
i0 = int(np.searchsorted(P.dates, pd.Timestamp(START))); i1 = int(np.searchsorted(P.dates, pd.Timestamp(END), side="right"))
D = P.dates[i0:i1]
def S(t, a): return pd.Series(a[i0:i1, P.col(t)], index=D)
def rep(name, r, cost_per_yr=0.0):
    r = r - cost_per_yr / 252
    yr = r.groupby(r.index.year).apply(sh)
    print(f"{name:40s} Sharpe {sh(r):5.2f}  ret {r.mean()*252*100:5.1f}%  vol {r.std()*np.sqrt(252)*100:5.1f}%  worst-yr {yr.min():5.2f} (min yr {yr.idxmin()})  cum x{np.exp(np.log1p(r).sum()):.2f}", flush=True)
    return r
sleeves = {}
for t in ["SPY", "QQQ", "IWM", "TLT", "GLD"]:
    cc = S(t, F["dret"]); co = S(t, F["co"]); oc = S(t, F["oc"])
    rep(f"{t} buy&hold", cc)
    # overnight: 2 trades/day, ~0.5 bp each for SPY/QQQ, more for others
    bps = 0.5 if t in ("SPY", "QQQ") else 1.5
    sleeves[f"{t}_ovn"] = rep(f"{t} overnight only (cost {2*bps}bp/d)", co, cost_per_yr=2 * bps / 1e4 * 252)
    rep(f"{t} intraday only", oc, cost_per_yr=2 * bps / 1e4 * 252)
# turn of month on SPY: last 3 + first 3 trading days
spy = S("SPY", F["dret"])
dom = pd.Series(np.arange(len(D)), index=D)
me = set(month_ends(D).tolist())
pos = np.zeros(len(D), bool)
for k, i in enumerate(sorted(me)):
    pos[max(0, i - 3):i + 1] = True; pos[i + 1:i + 4] = True
tom = spy * np.roll(pos, 1)    # position decided at close of prior day
sleeves["spy_tom"] = rep("SPY turn-of-month (d-3..d+3)", tom, cost_per_yr=2 * 0.5 / 1e4 * 12)
rep("SPY NOT turn-of-month", spy * (~np.roll(pos, 1)))
# day-of-week
for d in range(5):
    rep(f"SPY weekday {d}", spy * (D.weekday == d).astype(float))
# multi-asset trend: 12 ETFs, sign of 12m return (skip 0), vol-scaled to 10% each, 1-day lag
etfs = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "DBC", "USO", "UUP", "VNQ", "XLE", "XLU", "XLK", "XLF", "XLV", "XLP", "XLI", "XLY", "XLB"]
etfs = [t for t in etfs if t in P.ix]
rets = pd.DataFrame({t: S(t, F["dret"]) for t in etfs})
closef = pd.DataFrame({t: pd.Series(F["closef"][:, P.col(t)], index=P.dates) for t in etfs})
def trend_sleeve(lb, name, longonly=False, vol_target=0.10):
    sig = np.sign(np.log(closef / closef.shift(lb)))
    if longonly: sig = sig.clip(lower=0)
    v = rets.rolling(60).std() * np.sqrt(252)
    w = (sig * vol_target / v).shift(2)        # decided t, traded close t+1
    w = w.loc[D].fillna(0)
    r = (w * rets.loc[D]).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).sum() / (len(D) / 252)
    return rep(name, r, cost_per_yr=turn * 3 / 1e4), w
sleeves["trend252"], _ = trend_sleeve(252, "multi-asset trend 12m L/S")
sleeves["trend126"], _ = trend_sleeve(126, "multi-asset trend 6m L/S")
sleeves["trend63"], _ = trend_sleeve(63, "multi-asset trend 3m L/S")
sleeves["trend252_lo"], _ = trend_sleeve(252, "multi-asset trend 12m long-only")
# ensemble of trend lookbacks
r = (sleeves["trend252"] + sleeves["trend126"] + sleeves["trend63"]) / 3
sleeves["trend_ens"] = rep("trend ensemble", r)
# VIX ETP carry: short VXX / long SVXY when VXX 1m/3m trend is down (contango proxy)
for t in ["SVXY", "VXX", "VIXY", "UVXY"]:
    if t in P.ix:
        c = S(t, F["dret"]); print(t, "bars", int(np.isfinite(c).sum()), "first", c.first_valid_index(), "last", c.last_valid_index())
vixy = pd.Series(F["closef"][:, P.col("VIXY")], index=P.dates)
svxy = S("SVXY", F["dret"]).fillna(0)
vxx_r = S("VIXY", F["dret"]).fillna(0)
for lb in [10, 21, 63]:
    sig = (vixy < vixy.rolling(lb).mean()).astype(float).shift(2).loc[D].fillna(0)
    sleeves[f"shortvix_{lb}"] = rep(f"short VIXY when below {lb}d MA", -vxx_r * sig * 0.5, cost_per_yr=1.0 / 100)
    rep(f"long SVXY when VIXY below {lb}d MA", svxy * sig)
# SPY realized-vol-filtered overnight
rv = spy.rolling(20).std() * np.sqrt(252)
co = S("SPY", F["co"])
sleeves["spy_ovn_lowvol"] = rep("SPY overnight when rv20<20%", co * (rv.shift(1) < 0.20), cost_per_yr=1 / 1e4 * 252)
# momentum long-only (tranche A, monthly) as a sleeve from strategy.py output
eqc = pd.read_csv("out/equity_curve.csv", index_col=0, parse_dates=True)["strategy"]
sleeves["mom_lo"] = rep("stock momentum long-only (strategy.py)", eqc.pct_change().reindex(D).fillna(0))
sleeves["spy"] = spy
df = pd.DataFrame(sleeves).fillna(0)
print("\nCorrelation matrix:\n", df.corr().round(2).to_string())
df.to_csv("logs/phaseA_sleeves.csv")
# naive combination: inverse-vol weights of a chosen set
def combo(names, label):
    sub = df[names]
    w = 1.0 / sub.std(); w = w / w.sum()
    r = (sub * w).sum(axis=1)
    rep(f"COMBO {label}: " + ",".join(names), r)
combo(["spy_ovn", "spy_tom", "trend_ens", "mom_lo"], "A")
combo(["spy_ovn", "trend_ens", "mom_lo", "shortvix_21"], "B")
combo(["spy_ovn", "qqq_ovn" if "qqq_ovn" in df else "QQQ_ovn", "trend_ens", "mom_lo", "shortvix_21", "spy_tom"], "C")
