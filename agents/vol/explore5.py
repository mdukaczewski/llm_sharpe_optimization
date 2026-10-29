"""Phase 5: VIX-index-based signals (FRED VIX, VIX3M) -- term structure ratio, VRP, regime switch; both fill modes."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/trumpet/strat20/agents/vol")
from vlib import *
D = load(); ret = D["ret"]; c = D["close"]; idx = ret.index
F = pd.read_csv("/home/trumpet/strat20/agents/vol/fred_vix.csv", index_col=0, parse_dates=True).reindex(idx).ffill()
vix, vix3 = F["VIX"], F["VIX3M"]
ratio = vix / vix3
rv10, rv21 = rv(ret["SPY"], 10) * 100, rv(ret["SPY"], 21) * 100
vrp10, vrp21 = vix - rv10, vix - rv21
vixy, vixm = c["VIXY"], c["VIXM"]
print("ratio quantiles:", ratio.loc[START:END].quantile([0.05, 0.2, 0.5, 0.8, 0.95]).round(3).to_dict())
print("vrp21 quantiles:", vrp21.loc[START:END].quantile([0.05, 0.2, 0.5, 0.8, 0.95]).round(2).to_dict())
fwd = -ret["VIXY"].shift(-2)   # next-close fill convention: signal t -> earns day t+2
print("== conditional Sharpe of short VIXY (lag1 convention) by signal quantile bins ==")
for nm, s in [("ratio", ratio), ("vrp21", vrp21), ("vrp10", vrp10), ("vix", vix), ("dvix5", vix - vix.shift(5))]:
    x = s.loc[START:END]; f = fwd.loc[START:END]
    qs = x.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
    bins = np.digitize(x, qs)
    print(f"{nm:6s} " + " | ".join(f"b{b}: n={int((bins==b).sum())} Sh={sh(f[bins==b]):+.2f} r={f[bins==b].mean()*252*100:+.0f}%" for b in range(6)))
def run(name, w, fill="close", lag=1, full=False):
    W = pd.DataFrame(w, index=idx).fillna(0.0)
    return stats(backtest(W, ret, fill=fill, lag=lag, D=D), name, full=full)
print("\n== classic: short VIXY when ratio<a ; long VIXY when ratio>b (lag0/1/2 close fill; open fill) ==")
for a, b in [(0.95, 1.05), (0.95, 1.0), (0.9, 1.0), (1.0, 1.0), (0.95, 9)]:
    s = -0.5 * (ratio < a) + 0.5 * (ratio > b)
    for lag in [0, 1, 2]:
        run(f"ratio<{a} short / >{b} long lag{lag}", {"VIXY": s}, lag=lag)
    run(f"ratio<{a} short / >{b} long OPEN", {"VIXY": s}, fill="open")
print("\n== VRP filters ==")
for v in [0, 3, 5]:
    s = -0.5 * ((ratio < 0.95) & (vrp21 > v))
    run(f"short if ratio<.95 & vrp21>{v} lag1", {"VIXY": s})
    run(f"short if ratio<.95 & vrp21>{v} OPEN", {"VIXY": s}, fill="open")
    s = -0.5 * ((ratio < 0.95) & (vrp10 > v))
    run(f"short if ratio<.95 & vrp10>{v} lag1", {"VIXY": s})
print("\n== continuous: w = -k*(1-ratio) clipped ==")
for k in [5, 10]:
    s = np.clip(-k * (1 - ratio), -0.8, 0.8)
    run(f"w=-{k}(1-ratio) clip.8 lag1", {"VIXY": s}, full=True)
    run(f"w=-{k}(1-ratio) clip.8 OPEN", {"VIXY": s}, fill="open", full=True)
print("\n== spread (short VIXY, long 0.5 VIXM) with ratio filter ==")
for a in [0.95, 1.0]:
    f = (ratio < a).astype(float)
    run(f"spread -.5 VIXY +.5 VIXM if ratio<{a} lag1", {"VIXY": -0.5 * f, "VIXM": 0.5 * f}, full=True)
    f2 = (ratio > 1.0).astype(float)
    run(f"spread if ratio<{a} + reverse spread if ratio>1 lag1", {"VIXY": -0.5 * f + 0.5 * f2, "VIXM": 0.5 * f - 0.5 * f2}, full=True)
