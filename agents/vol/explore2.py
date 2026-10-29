"""Phase 2: composite rules for VIX-ETP carry with vol targeting; regime-switch long/short vol."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/trumpet/strat20/agents/vol")
from vlib import *
D = load(); ret = D["ret"]; c = D["close"]
vixy, vixm, spy = c["VIXY"], c["VIXM"], c["SPY"]
lv, lm = np.log(vixy), np.log(vixm)
ma = lambda s, w: s - np.log(np.exp(s).rolling(w).mean())
vma5, vma10, vma21 = ma(lv, 5), ma(lv, 10), ma(lv, 21)
mom5 = lv - lv.shift(5)
dd20 = spy / spy.rolling(20).max() - 1; dd63 = spy / spy.rolling(63).max() - 1
rv10, rv21 = rv(ret["SPY"], 10), rv(ret["SPY"], 21)
vrv21 = rv(ret["VIXY"], 21); vrv10 = rv(ret["VIXY"], 10)
ts10 = (lv - lm) - (lv - lm).shift(10)
spy_ma200 = np.log(spy / spy.rolling(200).mean())
idx = ret.index
def run(name, w_vixy, w_vixm=None, w_spy=None, full=False):
    W = pd.DataFrame({"VIXY": w_vixy}, index=idx)
    if w_vixm is not None: W["VIXM"] = w_vixm
    if w_spy is not None: W["SPY"] = w_spy
    W = W.fillna(0.0)
    return stats(backtest(W, ret), name, full=full)
print("== A: short VIXY after spike (VIXY above MA by a) ==")
for sigN, s in [("ma5", vma5), ("ma10", vma10), ("ma21", vma21), ("mom5", mom5)]:
    for a in [0.0, 0.02, 0.05, 0.10]:
        run(f"A short0.5 VIXY if {sigN}>{a}", -0.5 * (s > a))
print("== B: + vol target (w = -tv/vrv21, cap 0.8) ==")
for sigN, s in [("ma10", vma10), ("mom5", mom5)]:
    for a in [0.0, 0.03, 0.06]:
        for tv in [0.15, 0.25]:
            run(f"B short VIXY if {sigN}>{a} tv={tv}", -(s > a) * np.minimum(tv / vrv21, 0.8))
print("== C: SPY drawdown / rv filters alone and combined ==")
run("C short0.5 if dd20<-1%", -0.5 * (dd20 < -0.01))
run("C short0.5 if dd20<-2%", -0.5 * (dd20 < -0.02))
run("C short0.5 if dd20<-3%", -0.5 * (dd20 < -0.03))
run("C short0.5 if rv10>12%", -0.5 * (rv10 > 0.12))
run("C short0.5 if rv10>15%", -0.5 * (rv10 > 0.15))
run("C short0.5 if dd20<-1% & ma10>0", -0.5 * ((dd20 < -0.01) & (vma10 > 0)))
run("C short0.5 if dd20<-2% & ma10>0", -0.5 * ((dd20 < -0.02) & (vma10 > 0)))
run("C short0.5 if dd20<-2% | ma10>0.03", -0.5 * ((dd20 < -0.02) | (vma10 > 0.03)))
run("C short0.5 if rv10>12% & ma10>0", -0.5 * ((rv10 > 0.12) & (vma10 > 0)))
print("== D: regime switch: long VIXY in calm (rv10<b & ma10<0), short after spike ==")
for b in [0.07, 0.09, 0.11]:
    for wl in [0.15, 0.3]:
        calm = (rv10 < b) & (vma10 < 0)
        run(f"D long{wl} VIXY if rv10<{b}&ma10<0 (only)", wl * calm)
        run(f"D long{wl} calm + short0.5 if ma10>0.03", wl * calm - 0.5 * (vma10 > 0.03))
        run(f"D long{wl} calm + short0.5 if dd20<-2%&ma10>0", wl * calm - 0.5 * ((dd20 < -0.02) & (vma10 > 0)))
print("== E: hedged versions of best-ish short rule (dd20<-2% & ma10>0) ==")
m = ((dd20 < -0.02) & (vma10 > 0)).astype(float)
for k in [0.5, 1.0, 1.5]:
    run(f"E short0.5 VIXY +{0.5*k} VIXM if m", -0.5 * m, 0.5 * k * m)
for k in [0.5, 1.0, 1.5]:
    run(f"E short0.5 VIXY short{0.5*k} SPY if m", -0.5 * m, None, -0.5 * k * m)
print("== F: contango proxy ts10 (slope log VIXY/VIXM) ==")
for a in [-0.06, -0.03, 0.0]:
    run(f"F short0.5 if ts10<{a}", -0.5 * (ts10 < a))
    run(f"F short0.5 if ts10<{a} & ma10>0", -0.5 * ((ts10 < a) & (vma10 > 0)))
