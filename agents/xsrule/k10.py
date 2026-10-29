#!/usr/bin/env python3
"""Smaller rule books for the fixed-commission regime: long-only top-10 (cAx > 1.5 / 2.0) and L/S top/bottom 10 on 150 names."""
import sys, time; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/xsrule"); sys.path.insert(0, "/home/user/llm_sharpe_optimization")
import rlib, pandas as pd, numpy as np
from xlib import *
LANE = "xsrule"; t0 = time.time()
tks = rlib.universe(1500); D = rlib.add_market(rlib.minute_frame(tks, verbose=False)); D = add_extras(D); D = add_extra2(D)
D = rlib.clean(D, ["z30", "z60", "z5l", "zmoc", "zon_prev"]); print(f"frame rows {len(D)} {time.time()-t0:.0f}s", flush=True)
b = 0.5 * D.z30 - D.z60; cAx = b - 0.5 * D.z5l + 0.3 * D.zon_prev - 1.0 * D.r_vlast_x
is150 = D.tk.map(lambda t: rlib.RANK.get(t, 9999) <= 150)
for k, smin, name in [(10, 1.5, "xsr_cAx_lo10_s15_u150"), (10, 2.0, "xsr_cAx_lo10_s2_u150"), (10, None, "xsr_cAx_ls10_u150")]:
    r = rlib.book(D[is150], cAx[is150], k=k, mode="lo" if "lo" in name else "ls", smin=smin, name=name)
    rlib.save_cand(LANE, name, r, f"overnight {'LO' if 'lo' in name else 'L/S'} top{k} cAx{'>'+str(smin) if smin else ''} on 150 names, exit 09:31; all inputs pre-close")
    for cap in [400000]:
        npos = k * (1 if "lo" in name else 2); print(f"   at $0.4M ({npos} positions): Sh {rlib.sh(r - npos*6/cap):.2f} fold-OOS {rlib.sh(rlib.fold_oos(r - npos*6/cap, p=False)[0]):.2f}")
print(f"done {time.time()-t0:.0f}s", flush=True)
