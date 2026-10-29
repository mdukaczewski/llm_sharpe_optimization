#!/usr/bin/env python3
"""Momentum candidate grid for the fold validation (folds.py): daily net return
series of strategy.py variants (lookback set x TOP), written to logs/mom_grid/."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "/home/trumpet/strat20")
import strategy as S
from bt import Panel, run_fast
from feats import build
OUT = "/home/trumpet/strat20/logs/mom_grid"; os.makedirs(OUT, exist_ok=True)
P = Panel(); F = build(P)
GRID = {"E4": ((252, 21), (252, 42), (210, 21), (189, 21)), "S252": ((252, 21),), "S126": ((126, 21),), "E2": ((252, 21), (126, 21))}
for pname, params in GRID.items():
    for top in (15, 25, 40):
        S.PARAMS = params; S.TOP = top
        def fn(i):
            idx = S.select(F, P, i); return {int(c): 1.0 / len(idx) for c in idx}
        eqs = []
        for off in S.TRANCHE_OFFSETS:
            reb = F["rebal_m"] + off; reb = reb[reb < P.T]
            eq, info = run_fast(P, fn, reb, S.START, S.END, cost_bps=S.COST_BPS); eqs.append(eq)
        eq = sum(eqs) / len(eqs); r = eq.pct_change().dropna()
        name = f"mom_{pname}_top{top}"
        r.rename("ret").to_csv(f"{OUT}/{name}.csv", header=True, index_label="date")
        print(name, round(r.mean() / r.std() * np.sqrt(252), 2), flush=True)
print("MOMGRID DONE")
