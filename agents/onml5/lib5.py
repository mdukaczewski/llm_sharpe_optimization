#!/usr/bin/env python3
"""onml5 lane: model improvements targeted at the liquid tier where the realistic overnight book trades.
Reuses agents/onml4/lib4.py (book, routes, causal calibration, metrics) unchanged; only LOG/OUT point to this lane.
Inputs: agents/onml4/out/G.parquet (targets, costs, p40/p26) and agents/onml3/out/frame_u2000.parquet (features; read by
column, never rewritten).  Everything <= rlib.DEV_END."""
import os, sys, time
import numpy as np, pandas as pd
HERE = "/home/user/llm_sharpe_optimization"; sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/agents/onml4")
import rlib, liq
import lib4
from lib4 import (book, route_book, prep, metrics, mrow, tier_table, corr_with, commission, causal_cal, load_ref, TNAMES, CAP, ORDER_USD)
from rlib import sh
LANE = f"{HERE}/agents/onml5"; OUT = f"{LANE}/out"; os.makedirs(OUT, exist_ok=True); LOG = f"{LANE}/log.txt"
lib4.LOG = LOG
FRAME = f"{HERE}/agents/onml3/out/frame_u2000.parquet"
REF_RTX = "agents/onml4/cands/onml4_pR_lam3_rte_rtx_k6.csv"; REF_MOO = "agents/onml4/cands/onml4_pR_lam3_rte_moo_k6.csv"

def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as fh: fh.write(s + "\n")

def tlog(title, df): log(f"--- {title}\n" + df.to_string(index=False))

def load_G():
    G = pd.read_parquet(f"{HERE}/agents/onml4/out/G.parquet"); return G[G.date <= rlib.DEV_END].reset_index(drop=True)

def prank(s, d):
    return (s.groupby(d).rank(pct=True) - 0.5).astype(np.float32)

def daily_ic(df, p, y, by=None):
    """mean per-day Spearman IC of p vs y (computed within `by` groups if given), t-stat"""
    g = df[[p, y, "date"] + ([by] if by else [])].dropna()
    keys = ["date"] + ([by] if by else [])
    rp = g.groupby(keys)[p].rank(); ry = g.groupby(keys)[y].rank(); g = g.assign(rp=rp, ry=ry)
    def c(x): return x.rp.corr(x.ry) if len(x) > 20 else np.nan
    s = g.groupby(keys)[["rp", "ry"]].apply(c).dropna()
    return s

def books_for(G, pcol, ks=(6,), lam=3.0, sens=True, tiers=False):
    """onml4 pipeline unchanged on prediction column pcol: entry route -> MOO, and entry route -> exit route, c=0.3 f=1.
    returns rows (dict) and series."""
    H = prep(G, pcol); rows = []; ser = {}; tabs = []
    for k in ks:
        for ex, tag in [("moo", "rte_moo"), ("route", "rte_rtx")]:
            r, L = route_book(H, k, lam, "route", ex, 0.3, 1.0, ret_pos=True); nm = f"{pcol}_{tag}_k{k}"; ser[nm] = r
            m = metrics(r, k); row = mrow(nm, m); row["gross_bps"] = round((np.sign(L.w) * L.y).mean() * 1e4, 2)
            row["t1_pct"] = round((L.tier == 0).mean() * 100, 1)
            if sens:
                for c in [0.1, 1.0]:
                    mb = metrics(route_book(H, k, lam, "route", ex, c, 1.0), k); row[f"fold_c{c:g}"] = round(mb["fold"], 2)
            rows.append(row)
            if tiers: t = tier_table(L); t.insert(0, "book", nm); tabs.append(t)
    return rows, ser, tabs
