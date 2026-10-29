"""Final candidates with full stats. Fill: signal at close t, filled at close t+1 (default) -- also next-open fill shown.
Costs: 2 bps/side on turnover; borrow 2%/yr VIXY/VIXM shorts, 5%/yr SVXY/UVXY shorts. Cash earns 0."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/trumpet/strat20/agents/vol")
from vlib import *
D = load(); ret = D["ret"]; c = D["close"]; idx = ret.index
F = pd.read_csv("/home/trumpet/strat20/agents/vol/fred_vix.csv", index_col=0, parse_dates=True).reindex(idx).ffill()
vix, vix3 = F["VIX"], F["VIX3M"]; ratio = vix / vix3
rv10 = rv(ret["SPY"], 10)
lv = np.log(c["VIXY"]); mom5 = lv - lv.shift(5)
wd = pd.Series(idx.weekday, index=idx)
skip_thu = (wd != 3).astype(float).shift(-2).fillna(1.0)     # signal t -> fill t+1 -> earns t+2: be flat on Thursdays
def run(name, w, fill="close", full=True, lev=1.0):
    W = pd.DataFrame(w, index=idx).fillna(0.0) * lev
    return stats(backtest(W, ret, fill=fill, D=D), name, full=full)
print("#### 1. Unconditional short VIXY, with and without Thursday skip ####")
run("short0.5 VIXY always", {"VIXY": -0.5 + 0 * skip_thu})
run("short0.5 VIXY skip Thu", {"VIXY": -0.5 * skip_thu})
run("short0.5 VIXY skip Thu (OPEN fill)", {"VIXY": -0.5 * skip_thu}, fill="open")
print("#### 2. Term-structure spread (short VIXY, long VIXM) ####")
run("spread -0.5 VIXY +0.5 VIXM always", {"VIXY": -0.5, "VIXM": 0.5})
run("spread skip Thu", {"VIXY": -0.5 * skip_thu, "VIXM": 0.5 * skip_thu})
f = (rv10 > 0.12).astype(float)
run("spread if rv10>12%", {"VIXY": -0.5 * f, "VIXM": 0.5 * f})
run("spread if rv10>12% skip Thu", {"VIXY": -0.5 * f * skip_thu, "VIXM": 0.5 * f * skip_thu})
run("spread if rv10>12% skip Thu x2 lev", {"VIXY": -0.5 * f * skip_thu, "VIXM": 0.5 * f * skip_thu}, lev=2.0)
run("spread if rv10>12% skip Thu (OPEN fill)", {"VIXY": -0.5 * f * skip_thu, "VIXM": 0.5 * f * skip_thu}, fill="open")
for thr in [0.10, 0.14]:
    f2 = (rv10 > thr).astype(float)
    run(f"spread if rv10>{int(thr*100)}% skip Thu", {"VIXY": -0.5 * f2 * skip_thu, "VIXM": 0.5 * f2 * skip_thu})
print("#### 3. Ensemble (inverse-vol: spread_rv, short if ratio<0.83, long if ratio>1.05, short if mom5>5%), each with Thu skip ####")
sl = {}
sl["spread_rv"] = backtest(pd.DataFrame({"VIXY": -0.5 * f * skip_thu, "VIXM": 0.5 * f * skip_thu}, index=idx), ret)
sl["short_contango"] = backtest(pd.DataFrame({"VIXY": -0.5 * (ratio < 0.83).astype(float) * skip_thu}, index=idx), ret)
sl["long_backwd"] = backtest(pd.DataFrame({"VIXY": 0.5 * (ratio > 1.05).astype(float)}, index=idx), ret)
sl["short_mom5"] = backtest(pd.DataFrame({"VIXY": -0.5 * (mom5 > 0.05).astype(float) * skip_thu}, index=idx), ret)
for k, v in sl.items(): stats(v, f"  sleeve {k}", full=False)
def ens(names, label, lev=1.0):
    sub = pd.DataFrame({k: sl[k]["net"] for k in names}); w = 1 / sub.std(); w /= w.sum()
    bt = sum(sl[k] * w[k] for k in names) * lev
    print("   weights:", {k: round(float(w[k]), 3) for k in names})
    return stats(bt, label)
ens(["spread_rv", "short_contango", "long_backwd", "short_mom5"], "ENS4 (1x)")
ens(["spread_rv", "short_contango", "long_backwd", "short_mom5"], "ENS4 (2.3x -> ~15% vol)", lev=2.3)
ens(["spread_rv", "short_mom5"], "ENS2 spread_rv + short_mom5 (1x)")
print("#### 4. Reference: SPY buy&hold and short VIXY vs SVXY long (regime change 2018-02-27 only matters for SVXY/UVXY, not used above) ####")
run("SPY", {"SPY": 1.0 + 0 * skip_thu}, full=False)
run("long 1.0 SVXY", {"SVXY": 1.0 + 0 * skip_thu}, full=False)
