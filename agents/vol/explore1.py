"""Phase 1: signal zoo for short-VIX carry (short VIXY / long SVXY), daily, next-close fill."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/trumpet/strat20/agents/vol")
from vlib import *
D = load(); ret = D["ret"]; c = D["close"]
vixy, vixm, spy, svxy = c["VIXY"], c["VIXM"], c["SPY"], c["SVXY"]
lv = np.log(vixy); lm = np.log(vixm)
sig = {}
# term structure proxy: slope of log(VIXY/VIXM); contango => negative
for w in [5, 10, 21]:
    sig[f"ts{w}"] = ((lv - lm) - (lv - lm).shift(w))         # negative = contango
sig["vixy_ma21"] = lv - np.log(vixy.rolling(21).mean())
sig["vixy_ma10"] = lv - np.log(vixy.rolling(10).mean())
sig["vixy_ma5"] = lv - np.log(vixy.rolling(5).mean())
sig["vixy_mom5"] = lv - lv.shift(5)
sig["vixy_mom21"] = lv - lv.shift(21)
sig["vixm_ma21"] = lm - np.log(vixm.rolling(21).mean())
sig["spy_dd20"] = spy / spy.rolling(20).max() - 1
sig["spy_dd63"] = spy / spy.rolling(63).max() - 1
sig["spy_ma50"] = np.log(spy / spy.rolling(50).mean())
sig["spy_ma200"] = np.log(spy / spy.rolling(200).mean())
sig["rv10"] = rv(ret["SPY"], 10); sig["rv21"] = rv(ret["SPY"], 21); sig["rv63"] = rv(ret["SPY"], 63)
sig["rv_ratio"] = sig["rv10"] / sig["rv63"]
sig["vrv10"] = rv(ret["VIXY"], 10)
sig["vixy_ret1"] = ret["VIXY"]
S = pd.DataFrame(sig)
base = ret.loc[START:END]
print("== raw legs (constant weight, no filter) ==")
for t, w in [("VIXY", -0.5), ("SVXY", 1.0), ("VIXM", -1.0), ("UVXY", -0.33)]:
    W = pd.DataFrame({t: w}, index=ret.index); stats(backtest(W, ret), f"const {w:+.2f} {t}")
# spread: short VIXY, long VIXM beta-hedged
for k in [1.0, 1.5, 2.0, 2.5]:
    W = pd.DataFrame({"VIXY": -0.5, "VIXM": 0.5 * k}, index=ret.index); stats(backtest(W, ret), f"spread -0.5 VIXY +{0.5*k:.2f} VIXM", full=False)
print("\n== single-signal binary filters on short 0.5 VIXY (quantile thresholds) ==")
# next-day return of short VIXY for conditioning diagnostics: signal at t, filled t+1, earns t+2
fwd = -ret["VIXY"].shift(-2)
rows = []
for name in S.columns:
    s = S[name].loc[START:END]
    f = fwd.loc[START:END]
    for q in [0.2, 0.4, 0.6, 0.8]:
        thr = s.quantile(q)
        lo = f[s <= thr]; hi = f[s > thr]
        rows.append((name, q, thr, sh(lo), sh(hi), lo.mean() * 252, hi.mean() * 252))
r = pd.DataFrame(rows, columns=["sig", "q", "thr", "sh_below", "sh_above", "ret_below", "ret_above"])
print(r.round(3).to_string())
r.to_csv("/home/trumpet/strat20/agents/vol/logs/explore1_cond.csv")
