#!/usr/bin/env python3
"""Stock residual reversal vs SPY + 11 sector ETFs (Avellaneda-Lee style).
Betas: ridge OLS on trailing window, re-estimated weekly, applied strictly
out-of-sample. Signals: k-day residual return, OU s-score of cumulative
residual. Construction: rank / z-deadzone / hysteresis bands, overlapping tranches."""
import numpy as np, pandas as pd, sys, time, itertools
sys.path.insert(0, "/home/trumpet/strat20/agents/pairs")
from common import *
from scipy.stats import rankdata

P, F = load()
T, N = P.T, P.N
dret = F["dret"]; 
FACT = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
fcols = [P.col(t) for t in FACT]
R = np.clip(dret, -0.25, 0.25)
Fm = R[:, fcols]                                      # T x K factor returns
K = len(fcols)

BETAS = {}      # vintage day i -> K x N beta matrix (estimated with data <= i)
BVINT = np.full(T, -1)   # day t -> vintage used for t
def residuals(win=252, step=5, ridge=1e-4):
    """out-of-sample residual returns; betas from [i-win+1, i] applied to (i, i+step]"""
    res = np.full((T, N), np.nan)
    BETAS.clear()
    for i in range(win, T, step):
        X = Fm[i - win + 1:i + 1]; Y = R[i - win + 1:i + 1]
        Xc = X - X.mean(0); Yc = Y - Y.mean(0)
        XtX = Xc.T @ Xc + ridge * win * np.eye(K)
        B = np.linalg.solve(XtX, Xc.T @ Yc)           # K x N
        B = np.clip(B, -3, 3)
        j1 = min(i + step, T)
        res[i + 1:j1 + 1] = R[i + 1:j1 + 1] - Fm[i + 1:j1 + 1] @ B
        BETAS[i] = B.astype(np.float32); BVINT[i + 1:j1 + 1] = i
    return res

def ou_sscore(res, win=60):
    """AR(1) on cumulative residual over trailing win days -> s-score (Avellaneda-Lee)"""
    C = np.nancumsum(np.nan_to_num(res), axis=0)
    x = C[:-1]; y = C[1:]
    def rs(a): return pd.DataFrame(a).rolling(win, min_periods=win).mean().values
    mx, my, mxx, mxy, myy = rs(x), rs(y), rs(x * x), rs(x * y), rs(y * y)
    vx = mxx - mx * mx; cxy = mxy - mx * my
    b = cxy / (vx + 1e-12)
    a = my - b * mx
    vare = (myy - my * my) - b * cxy            # residual variance
    b = np.clip(b, -0.99, 0.999)
    m = a / (1 - b)
    sig_eq = np.sqrt(np.clip(vare, 1e-12, None) / (1 - b * b))
    s = np.full((T, N), np.nan)
    s[1:] = (y - m) / (sig_eq + 1e-9)
    kappa = np.full((T, N), np.nan); kappa[1:] = -np.log(np.clip(b, 1e-6, None)) * 252
    return s, kappa

def band_weights(s, m, s_in=1.25, s_out=0.5, maxhold=30, kcap=200, kappa=None, kmin=None):
    """hysteresis: open +-1 unit at |s|>s_in (short when s>0), close at |s|<s_out
    or after maxhold days; unit weight = 1/kcap, gross capped at 1 (scaled)."""
    W = np.zeros((T, N)); pos = np.zeros(N); age = np.zeros(N)
    for i in range(T):
        si = s[i]; ok = m[i] & np.isfinite(si)
        if kappa is not None and kmin is not None:
            ok &= (kappa[i] >= kmin)
        # close
        close = (pos != 0) & (~ok | (np.abs(si) < s_out) | (age >= maxhold) | (np.sign(si) == pos))
        pos[close] = 0; age[close] = 0
        # open
        op = (pos == 0) & ok & (np.abs(si) > s_in)
        pos[op] = -np.sign(si[op]); age[op] = 0
        age[pos != 0] += 1
        w = pos / kcap
        g = np.abs(w).sum()
        if g > 1: w = w / g
        W[i] = w
    return W

def hedge(W):
    """add ETF legs so that the stock book is factor-neutral: w_etf = -B w_stock (B from latest vintage <= t-?)
    W[t] is decided at close t; BVINT[t] vintage was estimated with data <= t, so it is known at t."""
    W = W.copy()
    for i in range(T):
        v = BVINT[i]
        if v < 0 or not np.any(W[i]): continue
        h = -(BETAS[v] @ W[i])
        W[i, fcols] += h
    return W

def rank_weights(s, m):
    W = np.zeros((T, N))
    for i in range(T):
        ok = m[i] & np.isfinite(s[i]); idx = np.where(ok)[0]
        if len(idx) < 50: continue
        r = rankdata(s[i, idx]) / len(idx) - 0.5 - 0.5 / len(idx)
        w = np.zeros(N); pos = r > 0; neg = r < 0
        w[idx[pos]] = r[pos] / r[pos].sum() * 0.5; w[idx[neg]] = r[neg] / (-r[neg]).sum() * 0.5
        W[i] = w
    return W

def z_weights(s, m, dead=1.0, cap=3.0):
    z = np.where(m & np.isfinite(s), np.clip(s, -cap, cap), 0.0)
    z = np.where(np.abs(z) > dead, z, 0.0)
    pos = np.clip(z, 0, None); neg = np.clip(-z, 0, None)
    ps = pos.sum(1, keepdims=True); ns = neg.sum(1, keepdims=True)
    return np.where(ps > 0, pos / (ps + 1e-12), 0) * 0.5 - np.where(ns > 0, neg / (ns + 1e-12), 0) * 0.5

if __name__ == "__main__":
    MINADV = 50e6
    m = np.zeros((T, N), bool)
    for i in range(260, T):
        m[i] = base_mask(F, i, MINADV, 5.0) & ~F["is_etf"]
    jump = rolling_max(np.abs(dret), 10, 1) > 0.4          # unadjusted splits / bad prints / real crashes: stand aside 10d
    m &= ~np.nan_to_num(jump).astype(bool)
    print("avg names:", m[700:].sum(1).mean(), flush=True)
    rows = []
    for win in [252]:
        t0 = time.time(); res = residuals(win=win); print(f"residuals win={win} {time.time()-t0:.0f}s", flush=True)
        rvol = rolling_std(np.nan_to_num(res), 60, 40)
        # (a) k-day residual reversal, vol-scaled
        for k in [1, 2, 3, 5, 10]:
            sig = -rolling_sum(np.nan_to_num(res), k) / (rvol * np.sqrt(k) + 1e-9)
            for mode in ["rank", "z"]:
                W0 = rank_weights(sig, m) if mode == "rank" else z_weights(sig, m, dead=1.0)
                for h in [1, 3, 5, 10]:
                    W = tranche(W0, h)
                    for hd in [0, 1]:
                        Wh = hedge(W) if hd else W
                        net, g, info = backtest(P, F, Wh)
                        rows.append(dict(report(f"win{win} rev{k} {mode} h{h} hedge{hd}", net, g, info, full=(h == 5 and hd == 1)), win=win, k=k, mode=mode, h=h, hedge=hd))
        # (b) OU s-score
        for sw in [40, 60]:
            s, kappa = ou_sscore(res, sw)
            s = np.where(np.isfinite(s), np.clip(s, -6, 6), np.nan)
            sig = -s
            for mode in ["rank", "z"]:
                W0 = rank_weights(sig, m) if mode == "rank" else z_weights(sig, m, dead=1.0)
                for h in [1, 5]:
                    for hd in [0, 1]:
                        Wh = tranche(W0, h); Wh = hedge(Wh) if hd else Wh
                        net, g, info = backtest(P, F, Wh)
                        rows.append(dict(report(f"win{win} ou{sw} {mode} h{h} hedge{hd}", net, g, info), win=win, k=sw, mode="ou"+mode, h=h, hedge=hd))
            for s_in, s_out, mh in [(1.25, 0.5, 30), (1.5, 0.75, 30), (2.0, 0.5, 20), (2.0, 1.0, 20), (1.25, 0.25, 60)]:
                for kmin in [None, 8.0]:
                    W = band_weights(s, m, s_in, s_out, mh, kcap=150, kappa=kappa, kmin=kmin)
                    for hd in [0, 1]:
                        Wh = hedge(W) if hd else W
                        net, g, info = backtest(P, F, Wh)
                        rows.append(dict(report(f"win{win} ou{sw} band {s_in}/{s_out}/{mh} kmin{kmin} hedge{hd}", net, g, info, full=(hd == 1)), win=win, k=sw, mode="band", h=mh, hedge=hd))
    df = pd.DataFrame(rows).sort_values("sh_net", ascending=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(df.head(40).to_string())
    df.to_csv("/home/trumpet/strat20/agents/pairs/resid_rev2.csv", index=False)
