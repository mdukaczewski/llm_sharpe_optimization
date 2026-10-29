"""Vol-sleeve candidate grid for folds.py: spread / short VIXY / equal-weight ENS2,
rv10 threshold in {none,10,12,14}%, with and without the Thursday skip. Net daily returns
(2 bps + borrow) -> logs/vol_grid.csv (wide)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/trumpet/strat20/agents/vol")
from vlib import *
D = load(); ret = D["ret"]; c = D["close"]; idx = ret.index
rv10 = rv(ret["SPY"], 10); lv = np.log(c["VIXY"]); mom5 = lv - lv.shift(5)
wd = pd.Series(idx.weekday, index=idx); skip = (wd != 3).astype(float).shift(-2).fillna(1.0)
F = pd.read_csv("/home/trumpet/strat20/agents/vol/fred_vix.csv", index_col=0, parse_dates=True).reindex(idx).ffill()
ratio = F["VIX"] / F["VIX3M"]
out = {}
for thu in [0, 1]:
    s = skip if thu else 1.0
    sc = backtest(pd.DataFrame({"VIXY": -0.5 * (ratio < 0.83).astype(float) * s}, index=idx), ret)["net"]
    lb = backtest(pd.DataFrame({"VIXY": 0.5 * (ratio > 1.05).astype(float)}, index=idx), ret)["net"]
    m5 = backtest(pd.DataFrame({"VIXY": -0.5 * (mom5 > 0.05).astype(float) * s}, index=idx), ret)["net"]
    out[f"contango_thu{thu}"] = sc; out[f"backwd_long"] = lb
    for thr in [0.12, 0.14]:
        f = (rv10 > thr).astype(float); g = f * s
        spread = backtest(pd.DataFrame({"VIXY": -0.5 * g, "VIXM": 0.5 * g}, index=idx), ret)["net"]
        out[f"ens4eq_thr{int(thr*100)}_thu{thu}"] = (spread + sc + lb + m5) / 4
        out[f"ens3eq_thr{int(thr*100)}_thu{thu}"] = (spread + sc + m5) / 3
for thr in [0, 0.10, 0.12, 0.14]:
    f = (rv10 > thr).astype(float) if thr > 0 else pd.Series(1.0, index=idx)
    for thu in [0, 1]:
        s = skip if thu else 1.0; g = f * s; tag = f"thr{int(thr*100)}_thu{thu}"
        spread = backtest(pd.DataFrame({"VIXY": -0.5 * g, "VIXM": 0.5 * g}, index=idx), ret)["net"]
        shortv = backtest(pd.DataFrame({"VIXY": -0.5 * g}, index=idx), ret)["net"]
        m5 = backtest(pd.DataFrame({"VIXY": -0.5 * (mom5 > 0.05).astype(float) * s}, index=idx), ret)["net"]
        out[f"spread_{tag}"] = spread; out[f"shortvixy_{tag}"] = shortv; out[f"ens2eq_{tag}"] = 0.5 * spread + 0.5 * m5
df = pd.DataFrame(out).loc[START:END]; df.to_csv("/home/trumpet/strat20/logs/vol_grid.csv")
print((df.mean() / df.std() * np.sqrt(252)).round(2).to_string()); print("VOLGRID DONE")
