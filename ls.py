"""Vectorized long/short cross-sectional backtester + signal zoo.

Weights W[t] decided at close t from data <= t.  Execution models:
  'nextopen'  : trades filled at open of t+1, so day t+1 return =
                sum(W[t-1]*cc[t+1]) + sum((W[t]-W[t-1])*oc[t+1])
  'nextclose' : trades filled at close of t+1 (position earns cc from t+2)
Costs: cost_bps per unit of one-way turnover, borrow_bps/yr on the short book.
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata
from bt import *
from feats import *

def xs_rank(a, m):
    """cross-sectional rank in [-0.5, 0.5] within mask m, NaN elsewhere (row-wise)"""
    out = np.full(a.shape, np.nan)
    for i in range(a.shape[0]):
        idx = np.where(m[i] & np.isfinite(a[i]))[0]
        if len(idx) > 10:
            r = rankdata(a[i, idx]) / len(idx)
            out[i, idx] = r - 0.5 - 0.5 / len(idx)
    return out

def xs_demean(a, m):
    out = np.where(m & np.isfinite(a), a, np.nan)
    mu = np.nanmean(out, axis=1, keepdims=True)
    sd = np.nanstd(out, axis=1, keepdims=True)
    return (out - mu) / (sd + 1e-12)

def weights_from_score(score, m, rebal_days, top=None, mode="rank", lag_hold=None):
    """score: T x N (already masked/NaN). Weights: long +1 total, short -1 total.
    mode 'rank': linear in rank; 'topq': top/bottom `top` names equal weight.
    Weights are set on rebal days and held (ffilled) until the next."""
    T, N = score.shape
    W = np.zeros((T, N))
    last = None
    rset = set(int(i) for i in rebal_days)
    for i in range(T):
        if i in rset:
            s = score[i]
            ok = m[i] & np.isfinite(s)
            idx = np.where(ok)[0]
            w = np.zeros(N)
            if len(idx) > 20:
                if mode == "rank":
                    r = rankdata(s[idx]) / len(idx) - 0.5 - 0.5 / len(idx)
                    pos = r > 0; neg = r < 0
                    w[idx[pos]] = r[pos] / r[pos].sum()
                    w[idx[neg]] = -r[neg] / r[neg].sum()
                else:
                    o = idx[np.argsort(-s[idx])]
                    k = min(top, len(o) // 2)
                    w[o[:k]] = 1.0 / k; w[o[-k:]] = -1.0 / k
            last = w
        if last is not None:
            W[i] = last
    return W

def backtest_W(P, F, W, start=START, end=END, cost_bps=10.0, borrow_bps=100.0, exec="nextopen"):
    i0 = int(np.searchsorted(P.dates, pd.Timestamp(start))); i1 = int(np.searchsorted(P.dates, pd.Timestamp(end), side="right"))
    cc = F["dret"]
    oc = F["oc"]; 
    Wp = np.vstack([np.zeros((1, P.N)), W[:-1]])      # W[t-1]
    if exec == "nextopen":
        # day t: held from W[t-2]?? no: positions entered at open t are W[t-1] (decided close t-1)
        # positions carried into day t were W[t-2]; the new piece W[t-1]-W[t-2] earns oc[t]
        Wpp = np.vstack([np.zeros((1, P.N)), Wp[:-1]])
        pr = np.nansum(Wpp * cc, axis=1) + np.nansum((Wp - Wpp) * oc, axis=1)
        turn = np.abs(Wp - Wpp).sum(axis=1)
        shortbook = np.clip(-Wp, 0, None).sum(axis=1)
    else:  # nextclose: W[t-1] traded at close t-1?? decided t-2, filled close t-1, earns cc[t]
        Wpp = np.vstack([np.zeros((1, P.N)), Wp[:-1]])
        pr = np.nansum(Wpp * cc, axis=1)
        turn = np.abs(Wpp - np.vstack([np.zeros((1, P.N)), Wpp[:-1]])).sum(axis=1)
        shortbook = np.clip(-Wpp, 0, None).sum(axis=1)
    cost = turn * cost_bps / 1e4 + shortbook * borrow_bps / 1e4 / 252
    net = pr - cost
    s = pd.Series(net[i0:i1], index=P.dates[i0:i1])
    g = pd.Series(pr[i0:i1], index=P.dates[i0:i1])
    return s, g, dict(turn_yr=turn[i0:i1].sum() / ((i1 - i0) / 252))

def sh(r):
    return r.mean() / (r.std() + 1e-12) * np.sqrt(252)

def summarize(name, net, gross, info, extra=""):
    yr = net.groupby(net.index.year).apply(lambda x: x.mean() / (x.std() + 1e-12) * np.sqrt(252))
    print(f"{name:34s} Sharpe net {sh(net):5.2f} gross {sh(gross):5.2f}  ret {net.mean()*252*100:5.1f}%/yr vol {net.std()*np.sqrt(252)*100:4.1f}%  turn {info['turn_yr']:5.1f}x/yr  worst-yr Sh {yr.min():5.2f} {extra}", flush=True)

def extra_feats(P, F):
    openf = np.where(np.isnan(P.open), np.nan, P.open)
    oc = P.close / openf - 1.0
    oc = np.where(np.isfinite(oc) & (np.abs(oc) < 0.5), oc, 0.0)
    F["oc"] = oc
    prevc = np.vstack([np.full((1, P.N), np.nan), P.close[:-1]])
    co = openf / prevc - 1.0
    F["co"] = np.where(np.isfinite(co) & (np.abs(co) < 0.5), co, 0.0)
    return F
