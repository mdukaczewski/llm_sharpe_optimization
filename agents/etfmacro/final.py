#!/usr/bin/env python3
"""etfmacro lane: regenerate the kept candidates end-to-end (honours STRAT_END through rlib.DEV_END).
Runtime ~30-60 s (minute npz loads for VIXY/UVXY/SPY; SPY feature frame is cached by rlib.ticker_frame).

Book: VIX-ETP overnight carry.  Decision at bin 389 (fill = official close, bin 390) or at bin 374 (fill = close of the
15:45 bar, "_1545"), exit at the 09:31 bar (bin 1) of the next day.  Short 0.5 VIXY (borrow 2%/yr) or 0.25 UVXY (borrow
5%/yr), cost 1.5 bps per side.  Regime (all known at the previous day's close): VIX/VIX3M (FRED) < 0.95 = contango, and
no position decided on a Friday (VIX calendar-day weekend effect).  "_ls" adds a long 0.5 VIXY overnight leg when the
term structure is NOT in contango and SPY's intraday signal s = 0.5*z30 - z60 < -1 (the project's overnight rule signal).
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elib import *
log(f"final.py start DEV_END={DEV_END}")
D = load_etf(); idx = D["ret"].index; F = load_fred(idx)
ratio_l = (F.VIX / F.VIX3M).shift(1)                                   # 16:15 VIX close of t-1: known before any decision bar of t
wd = pd.Series(idx.weekday, index=idx); notfri = (wd != 4).astype(float)
f95 = (ratio_l < 0.95).astype(float)
spy = tf("SPY").reindex(idx); s_spy = (0.5 * spy.z30 - spy.z60).clip(-5, 5).fillna(0.0)   # known at the close of t (used for the close decision only)
C = {}
for e, tag in [(390, ""), (375, "_1545")]:
    rV = on_ret("VIXY", e); rU = on_ret("UVXY", e)
    C[f"vixon_r95{tag}"] = leg_ret(-0.5 * notfri * f95, rV, 1.5, 0.02)
    C[f"uvxon_r95{tag}"] = leg_ret(-0.25 * notfri * f95, rU, 1.5, 0.05)
rV = on_ret("VIXY", 390)
C["vixon_r95_ls"] = C["vixon_r95"] + leg_ret(0.5 * ((ratio_l >= 0.95) & (s_spy < -1)).astype(float), rV, 1.5, 0.0)
NOTE = {"vixon_r95": "short 0.5 VIXY close->09:31 if VIX/VIX3M(t-1)<0.95, no Friday decisions, 1.5bps/side, borrow 2%",
        "uvxon_r95": "short 0.25 UVXY close->09:31, same regime, borrow 5%",
        "vixon_r95_1545": "as vixon_r95 but entered at the 15:45 bar (decision bin 374)",
        "uvxon_r95_1545": "as uvxon_r95 but entered at the 15:45 bar",
        "vixon_r95_ls": "vixon_r95 + long 0.5 VIXY overnight when ratio(t-1)>=0.95 and SPY s=0.5*z30-z60<-1"}
# fresh NOTES
nf = f"{HERE}/cands/NOTES.txt"
if os.path.exists(nf): os.remove(nf)
for k, v in C.items():
    save_cand(LANE, k, v, NOTE[k]); stats(v, k)
log("saved " + ", ".join(C))
fold_oos(C, label="etfmacro family")
for k, v in C.items(): fold_oos(v, label=k)
log("final.py done")
