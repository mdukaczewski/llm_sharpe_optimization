#!/usr/bin/env python3
"""Conditional-overnight book (rule fixed by the intraday agent, ON leg only)
applied to an arbitrary ticker list with an explicit per-side cost.
usage: on_book.py TICKERS cost_bps tag [a=0.5] [cap=1.5]"""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
import ilib
from ilib import *
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
tks = sys.argv[1].split(","); cost = float(sys.argv[2]); tag = sys.argv[3]
a = 0.5; cap = 1.5
for arg in sys.argv[4:]:
    k, v = arg.split("="); 
    if k == "a": a = float(v)
    if k == "cap": cap = float(v)
for t in tks: ilib.COST[t] = cost
def zsd(x, w=20):
    s = pd.Series(x); return (s / s.rolling(w, min_periods=10).std().shift(1)).values
L = "/home/trumpet/strat20/agents/intraday/logs"
ds = {}
for tk in tks:
    try:
        M = Min(tk)
    except Exception as e:
        print(tk, "skip:", e); continue
    nd = len(M.dates); pos = np.zeros((nd, 391))
    z60 = np.nan_to_num(zsd(M.r(329, 389))); z30 = np.nan_to_num(zsd(M.r(-1, 29)))
    w = np.clip(a * z30 - z60, 0, cap); w[M.half] = 0
    pos[:, 390] += w; pos[1:, 0] += w[:-1]
    g, n, t = pathbt(M, pos); ms = M.insample
    d = stats(M.dates, g, n, None, tk, ms, p=False); d["trades"] = t[ms].sum() / (ms.sum() / 252)
    ds[tk] = d
    print(f"{tk:6s} Sh gross {d['sh_g']:5.2f} net {d['sh_n']:5.2f} | ret {d['ret']*100:6.2f}% vol {d['vol']*100:5.2f}% mdd {d['mdd']*100:6.1f}% | trades/yr {d['trades']:4.0f} | days {int(ms.sum())} | yrs " + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in d["yr"].items()), flush=True)
    d["n"].rename("ret").rename_axis("date").to_csv(f"{L}/{tag}_{tk}_net.csv", float_format="%.8f")
port = combine([ds[t]["n"] for t in ds], p=False); pg = combine([ds[t]["g"] for t in ds], p=False)
eq = (1 + port).cumprod(); dd = (eq / eq.cummax() - 1).min(); yr = port.groupby(port.index.year).apply(sh)
print(f"COMBO[{len(ds)}] {tag:14s} Sh gross {sh(pg):5.2f} net {sh(port):5.2f} | ret {port.mean()*252*100:6.2f}% vol {port.std()*np.sqrt(252)*100:5.2f}% mdd {dd*100:6.1f}% | yrs " + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in yr.items()), flush=True)
pd.DataFrame({"ret": port}).rename_axis("date").to_csv(f"{L}/{tag}_combo_net.csv", float_format="%.8f")
