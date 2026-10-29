#!/usr/bin/env python3
"""Walk-forward pairs trading among ADV>=$50M stocks.
Every `resel` days: candidate pairs = top-`topk` return-correlation partners
per stock (trailing `win` days); keep pairs passing Engle-Granger (own ADF on
the log-price regression residual) and a half-life filter. Trade the spread
z-score (rolling `zwin` mean/std with the fixed hedge ratio): enter |z|>z_in,
exit |z|<z_out or after maxhold days. Dollar legs 1 : beta, unit = 1/kcap."""
import numpy as np, pandas as pd, sys, time
sys.path.insert(0, "/home/trumpet/strat20/agents/pairs")
from common import *

P, F = load()
T, N = P.T, P.N
logc = F["logc"]; dret = F["dret"]

def adf_stat(E, lags=1):
    """ADF t-stat (constant, `lags` lagged diffs) for each column of E (win x npairs)"""
    dE = np.diff(E, axis=0)
    y = dE[lags:]; x1 = E[lags:-1]
    cols = [x1, np.ones_like(x1)] + [dE[lags - j - 1:-j - 1] for j in range(lags)]
    n, p = y.shape[0], len(cols)
    X = np.stack(cols, axis=2)                      # n x npairs x p
    XtX = np.einsum("nkp,nkq->kpq", X, X)
    Xty = np.einsum("nkp,nk->kp", X, y)
    try:
        B = np.linalg.solve(XtX, Xty[..., None])[..., 0]   # npairs x p
    except np.linalg.LinAlgError:
        return np.full(E.shape[1], np.nan), np.full(E.shape[1], np.nan)
    resid = y - np.einsum("nkp,kp->nk", X, B)
    s2 = (resid ** 2).sum(0) / (n - p)
    inv = np.linalg.inv(XtX)
    se = np.sqrt(s2 * inv[:, 0, 0])
    return B[:, 0] / se, B[:, 0]

def select_pairs(i, m, win=252, topk=5, min_corr=0.6, adf_crit=-3.0, hl_rng=(2, 30)):
    idx = np.where(m[i] & (F["nbars252"][i] >= 250))[0]
    Rw = dret[i - win + 1:i + 1][:, idx]
    Rw = np.clip(Rw, -0.2, 0.2)
    Rw = (Rw - Rw.mean(0)) / (Rw.std(0) + 1e-12)
    C = Rw.T @ Rw / win
    np.fill_diagonal(C, -1)
    cand = set()
    part = np.argpartition(-C, topk, axis=1)[:, :topk]
    for a in range(len(idx)):
        for b in part[a]:
            if C[a, b] >= min_corr:
                cand.add((min(a, b), max(a, b)))
    cand = np.array(sorted(cand))
    if len(cand) == 0: return []
    A = logc[i - win + 1:i + 1][:, idx[cand[:, 0]]]; B = logc[i - win + 1:i + 1][:, idx[cand[:, 1]]]
    ok = np.isfinite(A).all(0) & np.isfinite(B).all(0)
    cand = cand[ok]; A = A[:, ok]; B = B[:, ok]
    Am = A - A.mean(0); Bm = B - B.mean(0)
    beta = (Am * Bm).sum(0) / ((Bm ** 2).sum(0) + 1e-12)
    E = Am - beta * Bm
    tstat, rho = adf_stat(E, 1)
    hl = -np.log(2) / np.log(np.clip(1 + rho, 1e-6, 0.9999))
    keep = (tstat < adf_crit) & (hl >= hl_rng[0]) & (hl <= hl_rng[1]) & (beta > 0.3) & (beta < 3)
    out = [(int(idx[cand[j, 0]]), int(idx[cand[j, 1]]), float(beta[j]), float(tstat[j]), float(hl[j])) for j in np.where(keep)[0]]
    return out

def run(resel=20, win=252, topk=5, min_corr=0.6, adf_crit=-3.0, hl_rng=(2, 30), zwin=60, z_in=2.0, z_out=0.5,
        maxhold=40, kcap=100, max_pairs_per_sel=400, stop_z=4.0, verbose=True):
    m = np.zeros((T, N), bool)
    for i in range(260, T):
        m[i] = base_mask(F, i, 50e6, 5.0) & ~F["is_etf"]
    W = np.zeros((T, N))
    active = {}      # (a,b) -> dict(beta, pos, age, born)
    i_start = int(np.searchsorted(P.dates, pd.Timestamp(START))) - 5
    nsel = []
    for i in range(i_start, T):
        if (i - i_start) % resel == 0:
            sel = select_pairs(i, m, win, topk, min_corr, adf_crit, hl_rng)
            sel.sort(key=lambda x: x[3])
            sel = sel[:max_pairs_per_sel]
            nsel.append(len(sel))
            newkeys = set((a, b) for a, b, *_ in sel)
            # drop flat pairs not in new list; keep open ones until exit
            for k in list(active):
                if k not in newkeys and active[k]["pos"] == 0:
                    del active[k]
            for a, b, beta, ts, hl in sel:
                if (a, b) not in active:
                    active[(a, b)] = dict(beta=beta, pos=0, age=0)
                elif active[(a, b)]["pos"] == 0:
                    active[(a, b)]["beta"] = beta
        if not active: continue
        keys = list(active)
        a = np.array([k[0] for k in keys]); b = np.array([k[1] for k in keys]); beta = np.array([active[k]["beta"] for k in keys])
        S = logc[i - zwin + 1:i + 1, a] - beta * logc[i - zwin + 1:i + 1, b]
        mu = S.mean(0); sd = S.std(0) + 1e-9
        z = (S[-1] - mu) / sd
        w = np.zeros(N)
        for j, k in enumerate(keys):
            st = active[k]
            zj = z[j]
            if not np.isfinite(zj) or not (m[i, k[0]] and m[i, k[1]]):
                st["pos"] = 0; continue
            if st["pos"] != 0:
                st["age"] += 1
                if abs(zj) < z_out or st["age"] >= maxhold or abs(zj) > stop_z or np.sign(zj) == st["pos"]:
                    st["pos"] = 0; st["age"] = 0
            if st["pos"] == 0 and abs(zj) > z_in and abs(zj) < stop_z:
                st["pos"] = -int(np.sign(zj)); st["age"] = 0     # z>0: spread rich -> short a, long b
            if st["pos"] != 0:
                u = 1.0 / kcap
                w[k[0]] += st["pos"] * u
                w[k[1]] -= st["pos"] * u * beta[j]
        g = np.abs(w).sum()
        if g > 1: w /= g
        W[i] = w
    if verbose: print(f"avg pairs selected per date: {np.mean(nsel):.0f}", flush=True)
    return W

if __name__ == "__main__":
    rows = []
    for cfg in [dict(), dict(z_in=1.5, z_out=0.5), dict(z_in=2.5, z_out=1.0), dict(zwin=40), dict(zwin=120), dict(adf_crit=-2.5, topk=8), dict(adf_crit=-3.5),
                dict(hl_rng=(2, 15)), dict(resel=10), dict(win=126), dict(maxhold=20), dict(kcap=60), dict(min_corr=0.75)]:
        t0 = time.time()
        W = run(**cfg)
        for ex in ["nextopen", "nextclose"]:
            net, g, info = backtest(P, F, W, exec=ex)
            rows.append(dict(report(f"pairs {cfg} {ex}", net, g, info, full=(ex == "nextopen")), cfg=str(cfg), exec=ex))
        print(f"   ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values("sh_net", ascending=False)
    pd.set_option("display.width", 250)
    print(df.to_string())
    df.to_csv("/home/trumpet/strat20/agents/pairs/pairs.csv", index=False)
