#!/usr/bin/env python3
"""ETF-vs-ETF spreads: residual reversal of liquid equity ETFs vs SPY (+QQQ/IWM/EFA),
hedged with the factor ETFs. Costs 2 bps. Also the 11-sector-ETF-only book."""
import numpy as np, pandas as pd, sys, time
sys.path.insert(0, "/home/trumpet/strat20/agents/pairs")
from common import *
from scipy.stats import rankdata
P, F = load(); T, N = P.T, P.N
dret = F["dret"]; R = np.clip(dret, -0.25, 0.25)
is_etf = F["is_etf"]
SECT = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
rows = []

def residuals(fact, win=252, step=5, ridge=1e-4):
    fcols = [P.col(t) for t in fact]; Fm = R[:, fcols]; K = len(fcols)
    res = np.full((T, N), np.nan); BET = {}; BV = np.full(T, -1)
    for i in range(win, T, step):
        X = Fm[i - win + 1:i + 1]; Y = R[i - win + 1:i + 1]
        Xc = X - X.mean(0); Yc = Y - Y.mean(0)
        B = np.linalg.solve(Xc.T @ Xc + ridge * win * np.eye(K), Xc.T @ Yc); B = np.clip(B, -4, 4)
        j1 = min(i + step, T)
        res[i + 1:j1 + 1] = R[i + 1:j1 + 1] - Fm[i + 1:j1 + 1] @ B
        BET[i] = B.astype(np.float32); BV[i + 1:j1 + 1] = i
    def hedge(W):
        W = W.copy()
        for i in range(T):
            v = BV[i]
            if v < 0 or not np.any(W[i]): continue
            W[i, fcols] += -(BET[v] @ W[i])
        return W
    # beta/corr to first factor for the universe filter
    return res, hedge, BET, BV, fcols

def zw(s, mm, dead=1.0, cap=3.0):
    z = np.where(mm & np.isfinite(s), np.clip(s, -cap, cap), 0.0); z = np.where(np.abs(z) > dead, z, 0.0)
    pos = np.clip(z, 0, None); neg = np.clip(-z, 0, None)
    ps = pos.sum(1, keepdims=True); ns = neg.sum(1, keepdims=True)
    return np.where(ps > 0, pos / (ps + 1e-12), 0) * 0.5 - np.where(ns > 0, neg / (ns + 1e-12), 0) * 0.5
def rw(s, mm):
    W = np.zeros((T, N))
    for i in range(T):
        idx = np.where(mm[i] & np.isfinite(s[i]))[0]
        if len(idx) < 8: continue
        r = rankdata(s[i, idx]) / len(idx) - 0.5 - 0.5 / len(idx)
        w = np.zeros(N); pos = r > 0; neg = r < 0
        w[idx[pos]] = r[pos] / r[pos].sum() * 0.5; w[idx[neg]] = r[neg] / (-r[neg]).sum() * 0.5
        W[i] = w
    return W

# universe: equity-like ETFs (corr to SPY > 0.6 over 252d, beta in [0.4, 1.6]), ADV >= 20M, >1y history
spy = R[:, P.col("SPY")]
c252 = pd.DataFrame(R).rolling(252, min_periods=200).corr(pd.Series(spy)).values
cov = pd.DataFrame(R * spy[:, None]).rolling(252, min_periods=200).mean().values - pd.DataFrame(R).rolling(252, min_periods=200).mean().values * pd.Series(spy).rolling(252, min_periods=200).mean().values[:, None]
beta = cov / (pd.Series(spy).rolling(252, min_periods=200).var().values[:, None] + 1e-12)
mE = is_etf[None, :] & (F["adv20"] >= 20e6) & (F["nbars252"] >= 230) & (np.nan_to_num(c252) > 0.6) & (np.nan_to_num(beta) > 0.4) & (np.nan_to_num(beta) < 1.6)
mE[:, P.col("SPY")] = False
jump = rolling_max(np.abs(dret), 10, 1) > 0.3; mE &= ~np.nan_to_num(jump).astype(bool)
print("ETF universe avg size:", mE[700:].sum(1).mean(), " sample:", list(P.tickers[np.where(mE[-1])[0][:40]]), flush=True)
mS = np.zeros((T, N), bool); mS[:, [P.col(t) for t in SECT]] = True; mS &= (F["nbars252"] >= 230)

for fname, fact in [("spy", ["SPY"]), ("4f", ["SPY", "QQQ", "IWM", "EFA"])]:
    res, hedge, BET, BV, fcols = residuals(fact)
    r0 = np.nan_to_num(res); rvol = rolling_std(r0, 60, 40) + 1e-9
    for uname, mm in [("etf", mE), ("sect", mS)]:
        for k in [1, 2, 3, 5, 10]:
            s = -rolling_sum(r0, k) / (rvol * np.sqrt(k))
            for mode in ["rank", "z"]:
                W0 = rw(s, mm) if mode == "rank" else zw(s, mm, dead=1.0)
                for h in [1, 3, 5, 10]:
                    for hd in [0, 1]:
                        W = tranche(W0, h); W = hedge(W) if hd else W
                        net, g, info = backtest(P, F, W)
                        rows.append(dict(report(f"{fname} {uname} rev{k} {mode} h{h} hedge{hd}", net, g, info), fact=fname, uni=uname, k=k, mode=mode, h=h, hedge=hd))
df = pd.DataFrame(rows).sort_values("sh_net", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(df.head(60).to_string())
df.to_csv("/home/trumpet/strat20/agents/pairs/etf_spread.csv", index=False)
