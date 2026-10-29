"""VIXY overnight carry book: short w(t) VIXY at close t (fill = official close), exit at 09:31 bar of t+1 (y_on),
or at 10:31 (y_on60) or next close (y_cc).  Cost 1.5 bps/side, borrow 2%/yr on the short.  w(t) from info <= close t."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/etfmacro")
from elib import *
D = load_etf(); idx = D["ret"].index; F = load_fred(idx); ratio = (F.VIX / F.VIX3M)
rv10 = vlib.rv(D["ret"]["SPY"], 10); rv20 = vlib.rv(D["ret"]["SPY"], 20)
vix = F.VIX / 100; spread = vix - rv10
d = tf("VIXY")                          # includes half days; y_on from a half day = 13:00 close -> next 09:31 (fine)
d = d.reindex(idx)
wd = pd.Series(idx.weekday, index=idx)
COST, BORROW = 1.5, 0.02
def leg(w, y="y_on", name="", p=True):
    """w: Series of weights (negative = short) decided at close t; returns net daily series indexed at t+1 (the day of the exit)."""
    w = w.reindex(idx).fillna(0.0); r = d[y]
    g = w * r; cost = w.abs() * 2 * COST / 1e4 + w.clip(upper=0).abs() * BORROW / 252
    n = (g - cost).shift(1).fillna(0.0).loc[START:DEV_END]     # booked on the exit day
    n[d[y].shift(1).isna()] = 0.0
    if p: stats(n, name)
    return n
notmon = (wd != 4).astype(float)          # decided Friday close -> exit Monday: skip
res = {}
print("== unconditional and weekday")
res["on_all"] = leg(-0.5 + 0 * notmon, name="short0.5 VIXY overnight all")
res["on_skipfri"] = leg(-0.5 * notmon, name="short0.5 VIXY overnight skip Fri->Mon")
for w_ in range(5): leg(-0.5 * (wd == w_), name=f"  only weekday {w_} decision")
print("== exit variants (skip Fri->Mon)")
leg(-0.5 * notmon, y="y_on60", name="exit 10:31")
leg(-0.5 * notmon, y="y_cc", name="exit next close")
print("== ratio conditioning (lagged = known at close t)")
for thr in [0.9, 0.95, 1.0]:
    res[f"on_r{thr}"] = leg(-0.5 * notmon * (ratio < thr), name=f"short overnight if ratio<{thr}, skip Fri")
    leg(-0.5 * (ratio < thr), name=f"   same incl Fri")
    leg(-0.5 * notmon * (ratio < thr), y="y_cc", name=f"   same, exit next close")
for lo in [0.8, 0.85]:
    res[f"on_r{lo}_95"] = leg(-0.5 * notmon * (ratio >= lo) * (ratio < 0.95), name=f"short overnight if {lo}<=ratio<0.95, skip Fri")
print("== continuous sizing: w = -clip((1.0-ratio)/0.1, 0, 1) * 0.5")
res["on_ramp"] = leg(-0.5 * notmon * ((1.0 - ratio) / 0.1).clip(0, 1), name="ramp 0.9..1.0, skip Fri")
res["on_ramp2"] = leg(-0.5 * notmon * ((1.0 - ratio) / 0.15).clip(0, 1), name="ramp 0.85..1.0, skip Fri")
print("== VIX - RV10 spread")
for lo in [0.0, 0.03]:
    res[f"on_sp{lo}"] = leg(-0.5 * notmon * (spread > lo), name=f"short overnight if VIX-rv10>{lo}, skip Fri")
    leg(-0.5 * notmon * (spread > lo) * (ratio < 0.95), name=f"   and ratio<0.95")
print("== rv10 (the existing sleeve's filter)")
for thr in [0.10, 0.12, 0.14]:
    leg(-0.5 * notmon * (rv10 > thr), name=f"short overnight if rv10>{thr}, skip Fri")
    leg(-0.5 * notmon * (rv10 > thr) * (ratio < 0.95), name=f"   and ratio<0.95")
print("== no Wed / Thu decisions etc (robustness of the weekday pattern)")
leg(-0.5 * notmon * (ratio < 0.95) * (wd != 0), name="ratio<0.95 skip Fri & Mon decisions")
leg(-0.5 * (ratio < 0.95) * (wd == 2), name="ratio<0.95 only Wed decision")
leg(-0.5 * notmon * (ratio < 0.95) * (wd != 2), name="ratio<0.95 skip Fri & Wed decisions")
print("== fold-OOS of the family")
oos, tab = fold_oos(res, label="vixon family"); print(tab.to_string())
for k, v in res.items(): fold_oos(v, label=k)
pd.DataFrame(res).to_csv("/home/user/llm_sharpe_optimization/agents/etfmacro/x4_res.csv")
