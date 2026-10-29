#!/usr/bin/env python3
"""Ensemble of lookbacks + staggered tranches."""
import numpy as np, pandas as pd, itertools
from bt import *
from feats import *
from scipy.stats import rankdata

P = Panel(); F = build(P)
print("features built", flush=True)

def ranks(s, m):
    out = np.full_like(s, np.nan)
    idx = np.where(m & np.isfinite(s))[0]
    out[idx] = rankdata(s[idx]) / len(idx)
    return out

def make_fn(top=25, min_adv=50e6, params=((252, 21), (252, 42), (210, 21), (189, 21)), volw=60, min_px=5.0):
    def fn(i):
        m = base_mask(F, i, min_adv, min_px)
        v = F[f"vol{volw}"][i] * np.sqrt(252)
        s = np.zeros(P.N)
        for lb, skip in params:
            s += ranks(mom(F, i, lb, skip) / (v + 1e-9), m)
        s = np.where(m & np.isfinite(s), s, -np.inf)
        idx = np.argsort(-s)[:top]; idx = idx[np.isfinite(s[idx])]
        if len(idx) == 0: return {}
        return {int(c): 1.0 / len(idx) for c in idx}
    return fn

def evaluate(name, cfg, offsets=(0,), cost=15, verbose=False):
    eqs = []
    for off in offsets:
        reb = F["rebal_m"] + off; reb = reb[reb < P.T]
        eq, info = run_fast(P, make_fn(**cfg), reb, START, END, cost_bps=cost)
        eqs.append(eq)
    eq = sum(eqs) / len(eqs)
    s = stats(eq)
    c1 = stats(eq[:"2021-10-27"])["cagr"]; c2 = stats(eq["2021-10-27":])["cagr"]
    print(f"{name:34s} CAGR {s['cagr']*100:5.1f}% Sh {s['sharpe']:.2f} DD {s['maxdd']*100:4.0f}% h1 {c1*100:4.0f}% h2 {c2*100:4.0f}% turn {info['turnover']:.0f}", flush=True)
    if verbose:
        print("   yearly: " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in s["yearly"].items()), flush=True)
    return eq

P4 = ((252, 21), (252, 42), (210, 21), (189, 21))
P2 = ((252, 21), (252, 42))
P3 = ((252, 21), (210, 21), (189, 21))
for top in [20, 25, 30]:
    evaluate(f"ens4 top={top}", dict(top=top, params=P4))
evaluate("ens2 top=25", dict(top=25, params=P2))
evaluate("ens3 top=25", dict(top=25, params=P3))
evaluate("ens4 top=25 adv100", dict(top=25, params=P4, min_adv=100e6))
evaluate("ens4 top=25 stagger 0/10", dict(top=25, params=P4), offsets=(0, 10), verbose=True)
evaluate("ens4 top=25 stagger 0/5/10/15", dict(top=25, params=P4), offsets=(0, 5, 10, 15), verbose=True)
evaluate("ens4 top=20 stagger 0/10", dict(top=20, params=P4), offsets=(0, 10), verbose=True)
evaluate("ens4 top=25 stagger 0/10 cost30", dict(top=25, params=P4), offsets=(0, 10), cost=30)
for off in [3, 7, 13, 17]:
    evaluate(f"ens4 top=25 offset {off}", dict(top=25, params=P4), offsets=(off,))
