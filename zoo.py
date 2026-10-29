#!/usr/bin/env python3
import numpy as np, pandas as pd, sys
from ls import *
P = Panel(); F = build(P); F = extra_feats(P, F)
dret = F["dret"]; logc = F["logc"]
T, N = P.T, P.N
# liquid mask per day
MINADV = float(sys.argv[1]) if len(sys.argv) > 1 else 50e6
m = np.zeros((T, N), bool)
for i in range(260, T):
    m[i] = base_mask(F, i, MINADV, 5.0)
print("mask built; avg names/day:", m[260:].sum(axis=1).mean(), flush=True)

def lagdiff(k, skip=0):
    out = np.full((T, N), np.nan); out[k:] = logc[k - skip:T - skip] - logc[:T - k]; return out
spy = F["spy_ret"]
# residual returns vs SPY (beta over 252d)
W = 252
def rmean(a, w): return pd.DataFrame(a).rolling(w, min_periods=int(w*0.9)).mean().values
mx = pd.Series(spy).rolling(W, min_periods=200).mean().values[:, None]
my = rmean(dret, W); mxy = rmean(dret * spy[:, None], W)
vx = pd.Series(spy).rolling(W, min_periods=200).var().values[:, None]
beta = np.clip(np.nan_to_num((mxy - mx * my) / (vx + 1e-12)), -1, 4)
resid = dret - beta * spy[:, None]
vol60 = F["vol60"]
sig = {}
sig["mom12_1_vs"] = lagdiff(252, 21) / (vol60 + 1e-9)
sig["mom12_1"] = lagdiff(252, 21)
sig["mom6_1"] = lagdiff(126, 21)
sig["rev5"] = -lagdiff(5)
sig["rev10"] = -lagdiff(10)
sig["rev21"] = -lagdiff(21)
sig["rev5_resid"] = -rolling_sum(resid, 5)
sig["rev10_resid"] = -rolling_sum(resid, 10)
sig["rev21_resid"] = -rolling_sum(resid, 21)
sig["rev5_vs"] = -lagdiff(5) / (vol60 + 1e-9)
sig["lowvol"] = -vol60
sig["lowbeta"] = -beta
sig["ivol"] = -rolling_std(resid, 60, 40)
sig["hi52"] = F["closef"] / F["hi252"]
sig["max5"] = -pd.DataFrame(dret).rolling(21).max().values
sig["absvol"] = -(P.dollar_vol / (F["adv60"] + 1))          # low abnormal volume
sig["turn_chg"] = -(F["adv20"] / (F["adv60"] + 1))
sig["ovn21"] = rolling_sum(F["co"], 21)                        # overnight persistence
sig["intra21"] = -rolling_sum(F["oc"], 21)
sig["ovn5"] = rolling_sum(F["co"], 5)
sig["residmom"] = (rolling_sum(resid, 252) - rolling_sum(resid, 21)) / (rolling_std(resid, 252, 200) + 1e-9)
sig["n_trades_chg"] = -(rolling_mean(P.n, 5, 3) / (rolling_mean(P.n, 60, 40) + 1))
sig["amihud"] = -rolling_mean(np.abs(dret) / (P.dollar_vol + 1), 21, 15)
sig["vwap_dev"] = -(P.close / P.vw - 1)           # close above vwap -> short (reversal)
sig["range_pos"] = -((P.close - P.low) / (P.high - P.low + 1e-9))
sig["gap_rev"] = -F["co"]

reb_d = np.arange(T); reb_w = F["rebal_w"]; reb_m = F["rebal_m"]
results = []
for name, s in sig.items():
    s = np.where(m, s, np.nan)
    for rb_name, rb in [("d", reb_d), ("w", reb_w), ("m", reb_m)]:
        if name.startswith("mom") and rb_name == "d": continue
        if name in ("gap_rev", "range_pos", "vwap_dev", "rev5", "rev5_resid", "rev5_vs", "ovn5", "absvol", "n_trades_chg") and rb_name == "m": continue
        Wm = weights_from_score(s, m, rb, mode="rank")
        for ex in ["nextopen", "nextclose"]:
            net, gross, info = backtest_W(P, F, Wm, cost_bps=10, borrow_bps=100, exec=ex)
            summarize(f"{name} [{rb_name},{ex}]", net, gross, info)
            results.append(dict(sig=name, reb=rb_name, exec=ex, sh_net=sh(net), sh_gross=sh(gross), ret=net.mean()*252, vol=net.std()*np.sqrt(252), turn=info["turn_yr"]))
        np.save(f"logs/zoo_net_{name}_{rb_name}.npy", net.values) if rb_name == "w" else None
df = pd.DataFrame(results).sort_values("sh_net", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(df.to_string())
df.to_csv(f"logs/zoo_{int(MINADV/1e6)}.csv", index=False)
