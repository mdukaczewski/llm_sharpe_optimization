"""Phase 4: fix vol-target test; next-open fill; VIXY overnight/intraday; spread with filters; lag sensitivity."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/trumpet/strat20/agents/vol")
from vlib import *
D = load(); ret = D["ret"]; c = D["close"]
vixy, vixm, spy = c["VIXY"], c["VIXM"], c["SPY"]
lv, lm = np.log(vixy), np.log(vixm)
ma = lambda s, w: s - np.log(np.exp(s).rolling(w).mean())
vma10 = ma(lv, 10); mom5 = lv - lv.shift(5)
dd20 = spy / spy.rolling(20).max() - 1
rv10 = rv(ret["SPY"], 10); vrv21 = rv(ret["VIXY"], 21)
idx = ret.index
def run(name, w, fill="close", lag=1, full=False):
    W = pd.DataFrame(w, index=idx).fillna(0.0)
    return stats(backtest(W, ret, fill=fill, lag=lag, D=D), name, full=full)
print("== VIXY/VIXM/UVXY/SVXY overnight vs intraday (unconditional short VIXY 0.5 pieces, gross) ==")
for t in ["VIXY", "VIXM", "SVXY", "UVXY", "SPY"]:
    co, oc, cc = D["co"][t].loc[START:END], D["oc"][t].loc[START:END], ret[t].loc[START:END]
    print(f"{t:5s} overnight: mean {co.mean()*252*100:6.1f}%/yr Sh {sh(co):5.2f} | intraday: mean {oc.mean()*252*100:6.1f}%/yr Sh {sh(oc):5.2f} | cc: {cc.mean()*252*100:6.1f}%/yr Sh {sh(cc):5.2f}")
    for y in range(2017, 2027):
        a, b = co[str(y)], oc[str(y)]
        print(f"      {y}: ovn {a.mean()*252*100:+6.1f}% intra {b.mean()*252*100:+6.1f}%", end="")
    print()
print("\n== B fixed: short VIXY after spike with vol targeting ==")
for a in [0.0, 0.03]:
    for tv in [0.15, 0.3]:
        on = (vma10 > a).astype(float)
        run(f"B short VIXY if ma10>{a} tv={tv}", {"VIXY": -on * np.minimum(tv / vrv21, 0.8)})
print("\n== lag sensitivity (close fill, lag 0 = same-close fill [not tradable], 1, 2) ==")
for nm, s in [("ma10>0.03", vma10 > 0.03), ("mom5>0.05", mom5 > 0.05), ("rv10>12%", rv10 > 0.12), ("dd20<-2%", dd20 < -0.02), ("ts const", pd.Series(True, index=idx))]:
    for lag in [0, 1, 2]:
        run(f"short0.5 VIXY if {nm} lag{lag}", {"VIXY": -0.5 * s.astype(float)}, lag=lag)
    run(f"short0.5 VIXY if {nm} OPEN fill", {"VIXY": -0.5 * s.astype(float)}, fill="open")
print("\n== spread short VIXY + long VIXM with filters (close fill) ==")
for k in [0.75, 1.0]:
    for nm, s in [("always", pd.Series(True, index=idx)), ("ma10>0", vma10 > 0), ("rv10>12%", rv10 > 0.12), ("dd20<-2%", dd20 < -0.02), ("ma10<0", vma10 < 0), ("rv10<12%", rv10 < 0.12)]:
        f = s.astype(float)
        run(f"spread -0.5 VIXY +{0.5*k:.2f} VIXM if {nm}", {"VIXY": -0.5 * f, "VIXM": 0.5 * k * f}, full=(nm == "always"))
