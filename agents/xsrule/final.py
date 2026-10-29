#!/usr/bin/env python3
"""xsrule lane: regenerate ONLY the kept candidates end-to-end from raw minute data.
Honours STRAT_END through rlib (rlib.DEV_END); everything is loaded through rlib.universe(1500) /
rlib.minute_frame (cut at DEV_END).  Run unattended:  nohup python3 agents/xsrule/final.py > agents/xsrule/final.log 2>&1 &
Kept books (all decided at the close of t, filled at the official close, exit at the 09:31 bar of t+1):
  signal  cAx = 0.5*z30 - z60 - 0.5*z5l + 0.3*zon_prev - rank(vlast_x)      [all inputs known by the 15:59 bar]
  signal  cBx = cAx but with -0.3*z5l - 0.3*zmoc instead of -0.5*z5l          [zmoc uses the closing print: see REPORT]
  xsr_cAx_lo20_s15_u150 : long top-20 names with cAx > 1.5 on the 150 most liquid names, equal weight
  xsr_cAx_ls20_u1500    : long top-20 / short bottom-20 by cAx on universe(1500)
  xsr_cBx_lo20_s15_u150, xsr_cBx_ls20_u1500 : same with cBx
Expected runtime: see REPORT.md (dominated by rlib.ticker_frame cache builds on first run)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xlib import *
LANE = "xsrule"
t0 = time.time()
tks = rlib.universe(1500); print(f"universe(1500): {len(tks)} names, DEV_END {rlib.DEV_END}", flush=True)
D = rlib.add_market(rlib.minute_frame(tks)); D = add_extras(D); D = add_extra2(D)
D = rlib.clean(D, ["z30", "z60", "z5l", "zmoc", "zon_prev"])
print(f"frame rows {len(D)} {time.time()-t0:.0f}s", flush=True)
b = 0.5 * D.z30 - D.z60
S = {"cAx": b - 0.5 * D.z5l + 0.3 * D.zon_prev - 1.0 * D.r_vlast_x,
     "cBx": b - 0.3 * D.z5l - 0.3 * D.zmoc + 0.3 * D.zon_prev - 1.0 * D.r_vlast_x}
is150 = D.tk.map(lambda t: rlib.RANK.get(t, 9999) <= 150)
for nm, s in S.items():
    # the rank within the 150-name subset must be computed on that subset only
    r = rlib.book(D[is150], s[is150], k=20, mode="lo", smin=1.5, name=f"xsr_{nm}_lo20_s15_u150")
    rlib.save_cand(LANE, f"xsr_{nm}_lo20_s15_u150", r, f"overnight LO top20 {nm}>1.5 on 150 names, exit 09:31, 1bp/side")
    rlib.book(D[is150], s[is150], k=20, mode="lo", smin=1.5, cost=1.5, name=f"   (1.5 bps) xsr_{nm}_lo20_s15_u150")
    r = rlib.book(D[is150], s[is150], k=20, mode="ls", name=f"xsr_{nm}_ls20_u150")
    rlib.save_cand(LANE, f"xsr_{nm}_ls20_u150", r, f"overnight L/S top/bottom 20 by {nm} on 150 names, exit 09:31, 1bp/side")
    rlib.book(D[is150], s[is150], k=20, mode="ls", cost=1.5, name=f"   (1.5 bps) xsr_{nm}_ls20_u150")
    r = rlib.book(D, s, k=20, mode="ls", name=f"xsr_{nm}_ls20_u1500")
    rlib.save_cand(LANE, f"xsr_{nm}_ls20_u1500", r, f"overnight L/S top/bottom 20 by {nm} on universe(1500) ({len(tks)} names), exit 09:31, per-name cost")
    rlib.book(D, s, k=20, mode="ls", cost=1.5, name=f"   (1.5 bps) xsr_{nm}_ls20_u1500")
C = rlib.load_cands(LANE); rlib.fold_oos(C, label="xsrule family")
for k_, v in C.items(): rlib.fold_oos(v, label=k_)
print(f"done {time.time()-t0:.0f}s", flush=True)
