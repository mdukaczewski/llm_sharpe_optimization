"""Robustness of the final config: spread (short VIXY / long VIXM) when SPY rv10>12%, flat on Thursdays.
Correct open-fill alignment, sub-periods, hedge ratio, rv threshold, with/without Thursday skip."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/trumpet/strat20/agents/vol")
from vlib import *
D = load(); ret = D["ret"]; c = D["close"]; idx = ret.index
rv10 = rv(ret["SPY"], 10); wd = pd.Series(idx.weekday, index=idx)
lv = np.log(c["VIXY"]); mom5 = lv - lv.shift(5)
skip_close = (wd != 3).astype(float).shift(-2).fillna(1.0)   # close fill: signal t earns day t+2
skip_open = (wd != 3).astype(float).shift(-1).fillna(1.0)    # open fill: signal t earns intraday t+1 + overnight into t+2
f = (rv10 > 0.12).astype(float)
def W(h, skip, filt=f): return pd.DataFrame({"VIXY": -0.5 * filt * skip, "VIXM": 0.5 * h * filt * skip}, index=idx).fillna(0.0)
print("== fill convention (hedge 1.0 => +0.5 VIXM per -0.5 VIXY) ==")
stats(backtest(W(1.0, skip_close), ret), "close fill, skip Thu")
stats(backtest(W(1.0, skip_open), ret, fill="open", D=D), "OPEN fill, skip Thu (aligned)")
stats(backtest(W(1.0, 1.0), ret), "close fill, no skip")
stats(backtest(W(1.0, 1.0), ret, fill="open", D=D), "OPEN fill, no skip")
print("== sub-periods (close fill, skip Thu) ==")
for a, b in [("2016-10-28", "2019-12-31"), ("2020-01-01", "2022-12-31"), ("2023-01-01", "2026-10-27"), ("2016-10-28", "2021-10-27"), ("2021-10-28", "2026-10-27")]:
    stats(backtest(W(1.0, skip_close), ret, start=a, end=b), f"  {a}..{b}", full=False)
print("== hedge ratio ==")
for h in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]:
    stats(backtest(W(h, skip_close), ret), f"  hedge {h}", full=False)
print("== rv window / threshold ==")
for w in [5, 10, 21]:
    for thr in [0.10, 0.12, 0.15]:
        ff = (rv(ret["SPY"], w) > thr).astype(float)
        stats(backtest(W(1.0, skip_close, ff), ret), f"  rv{w}>{thr}", full=False)
print("== weekday skipped (which single weekday to be flat on; close fill) ==")
for d, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
    sk = (wd != d).astype(float).shift(-2).fillna(1.0)
    stats(backtest(W(1.0, sk), ret), f"  flat on {nm}", full=False)
print("== ENS2 (spread_rv + short VIXY if mom5>5%, both skip Thu, inverse vol) sub-periods ==")
a = backtest(W(1.0, skip_close), ret); b = backtest(pd.DataFrame({"VIXY": -0.5 * (mom5 > 0.05).astype(float) * skip_close}, index=idx), ret)
wa = 1 / a["net"].std(); wb = 1 / b["net"].std(); e = (a * wa + b * wb) / (wa + wb)
stats(e, "ENS2 full")
for s, t in [("2016-10-28", "2021-10-27"), ("2021-10-28", "2026-10-27")]:
    stats(e.loc[s:t], f"  ENS2 {s}..{t}", full=False)
# tail table: 10 worst days of the final config
n = backtest(W(1.0, skip_close), ret)["net"]
print("10 worst days (final config, 1x):"); print((n.nsmallest(10) * 100).round(2).to_string())
print("10 worst 5-day windows:"); print((n.rolling(5).apply(lambda x: np.exp(np.log1p(x).sum()) - 1).nsmallest(10) * 100).round(2).to_string())
pd.DataFrame({"final_net": n, "ens2_net": e["net"]}).to_csv("/home/trumpet/strat20/agents/vol/logs/final_daily_returns.csv")
