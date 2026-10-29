#!/usr/bin/env python3
"""Conditioned residual reversal: exclude news-like moves (big |resid| or
abnormal volume), use VWAP-based variants, concentrate in extremes, hedge."""
import numpy as np, pandas as pd, sys, time
sys.path.insert(0, "/home/trumpet/strat20/agents/pairs")
from common import *
import resid_rev as RR
P, F, T, N = RR.P, RR.F, RR.T, RR.N
dret = F["dret"]
m = np.zeros((T, N), bool)
for i in range(260, T): m[i] = base_mask(F, i, 50e6, 5.0) & ~F["is_etf"]
jump = rolling_max(np.abs(dret), 10, 1) > 0.4
m &= ~np.nan_to_num(jump).astype(bool)
res = RR.residuals(252)
r0 = np.nan_to_num(res)
rvol = rolling_std(r0, 60, 40) + 1e-9
absvol = P.dollar_vol / (F["adv60"] + 1)            # abnormal volume today
zres = r0 / rvol
rows = []
def run(name, sig, mm, modes=("z",), hs=(1, 2, 3, 5), dead=1.0, full=False):
    for mode in modes:
        W0 = RR.rank_weights(sig, mm) if mode == "rank" else RR.z_weights(sig, mm, dead=dead)
        for h in hs:
            W = RR.hedge(tranche(W0, h))
            net, g, info = backtest(P, F, W)
            rows.append(dict(report(f"{name} {mode} h{h}", net, g, info, full=full), sig=name, mode=mode, h=h))

# baseline 1d residual z
run("z1", -zres, m)
# exclude news-like moves: |z|>3 or abnormal volume > 2.5
calm = (np.abs(zres) < 3.0) & (np.nan_to_num(absvol) < 2.5)
run("z1 calm", -zres, m & calm)
calm2 = (np.abs(zres) < 2.5) & (np.nan_to_num(absvol) < 1.5)
run("z1 calm2", -zres, m & calm2)
# low-volume moves only (liquidity-driven)
run("z1 lowvol", -zres, m & (np.nan_to_num(absvol) < 1.0))
# 2-3 day calm residual
for k in [2, 3]:
    s = -rolling_sum(r0, k) / (rvol * np.sqrt(k))
    calmk = rolling_max(np.abs(zres), k, 1) < 3.0
    calmk &= rolling_max(np.nan_to_num(absvol), k, 1) < 2.5
    run(f"z{k} calm", s, m & np.nan_to_num(calmk).astype(bool))
# VWAP-based: close vs vwap deviation (residualised crudely by subtracting the SPY close/vwap dev)
cv = np.log(P.close / P.vw); cv = np.where(np.isfinite(cv) & (np.abs(cv) < 0.2), cv, 0.0)
cv_res = cv - cv[:, [P.col("SPY")]]
run("vwapdev", -cv_res / (rolling_std(cv_res, 60, 40) + 1e-9), m)
run("vwapdev calm", -cv_res / (rolling_std(cv_res, 60, 40) + 1e-9), m & calm)
# combo: z1 + vwap dev
comb = -zres - cv_res / (rolling_std(cv_res, 60, 40) + 1e-9)
run("combo calm", comb, m & calm, modes=("z", "rank"))
# different dead zones / extremes only
run("z1 calm dead1.5", -zres, m & calm, dead=1.5)
run("z1 calm dead2", -zres, m & calm, dead=2.0)
# conditional on big sector move days: residual reversal only after |factor| moves
fabs = np.abs(RR.Fm).max(axis=1)                      # max |factor ret| today
bigf = (fabs > np.nanquantile(fabs, 0.8))[:, None]
run("z1 calm bigfactor", -zres, m & calm & bigf)
# nextclose execution for the best-looking
df = pd.DataFrame(rows).sort_values("sh_net", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(df.to_string())
df.to_csv("/home/trumpet/strat20/agents/pairs/resid_rev3.csv", index=False)
