"""Shared helpers for the pairs/stat-arb agent.
Column-subset panel (liquid stocks + liquid ETFs) so everything fits in RAM
and runs fast; per-asset cost backtester (stocks 10/20 bps, ETFs 2 bps,
1%/yr borrow); reporting."""
import numpy as np, pandas as pd, os, sys, time
sys.path.insert(0, "/home/trumpet/strat20")
from bt import Panel, ffill, rolling_mean, rolling_std, rolling_sum, rolling_max, week_ends, month_ends
from feats import build, base_mask, START, END
from ls import extra_feats, sh

SUB = "/home/trumpet/strat20/agents/pairs/panel_sub.npz"
RAW = "/home/trumpet/strat20/data/panel_liquid.npz"

def make_subset(min_adv_stock=50e6, min_adv_etf=20e6):
    z = np.load(RAW, allow_pickle=True)
    close = z["close"]; vol = z["vol"]; ttype = z["ttype"]; tk = z["tickers"]
    dv = close.astype(np.float64) * vol.astype(np.float64)
    adv = pd.DataFrame(dv).rolling(20, min_periods=15).mean().values
    mx = np.nanmax(np.where(np.isnan(adv), 0, adv), axis=0)
    keep = (np.isin(ttype, ["CS", "ADRC"]) & (mx >= min_adv_stock)) | (np.isin(ttype, ["ETF"]) & (mx >= min_adv_etf))
    keep |= np.isin(tk, ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"])
    print("subset: keeping", keep.sum(), "of", len(keep), "stocks", (keep & np.isin(ttype, ["CS", "ADRC"])).sum(), "etfs", (keep & (ttype == "ETF")).sum(), flush=True)
    out = {"dates": z["dates"]}
    for k in ["tickers", "ttype", "adj", "div", "open", "high", "low", "close", "vw", "vol", "n"]:
        a = z[k]; out[k] = a[:, keep] if a.ndim == 2 else a[keep]
    np.savez(SUB, **out)

def load(force=False):
    if force or not os.path.exists(SUB):
        make_subset()
    t0 = time.time()
    P = Panel(path=SUB)
    F = build(P); F = extra_feats(P, F)
    F["is_etf"] = (P.ttype == "ETF")
    print(f"loaded {P.T}x{P.N} in {time.time()-t0:.0f}s", flush=True)
    return P, F

def cost_vector(P, F, i):
    """per-asset one-way cost in bps at day i: ETF 2, stock ADV>=50M 10, else 20"""
    c = np.where(F["is_etf"], 2.0, np.where(F["adv20"][i] >= 50e6, 10.0, 20.0))
    return c

def backtest(P, F, W, start=START, end=END, exec="nextopen", borrow_bps=100.0, cost_mult=1.0, fixed_cost=None):
    """W[t] decided at close t. Returns net, gross daily return Series + info.
    Per-asset costs (cost_vector) applied to one-way turnover per asset."""
    i0 = int(np.searchsorted(P.dates, pd.Timestamp(start))); i1 = int(np.searchsorted(P.dates, pd.Timestamp(end), side="right"))
    cc = F["dret"]; oc = F["oc"]
    Wp = np.vstack([np.zeros((1, P.N)), W[:-1]])        # W[t-1] = filled at open t
    Wpp = np.vstack([np.zeros((1, P.N)), Wp[:-1]])      # W[t-2] = carried into day t
    if exec == "nextopen":
        pr = np.nansum(Wpp * cc, axis=1) + np.nansum((Wp - Wpp) * oc, axis=1)
        dW = np.abs(Wp - Wpp)
        shortbook = np.clip(-Wp, 0, None).sum(axis=1)
    else:
        pr = np.nansum(Wpp * cc, axis=1)
        dW = np.abs(Wpp - np.vstack([np.zeros((1, P.N)), Wpp[:-1]]))
        shortbook = np.clip(-Wpp, 0, None).sum(axis=1)
    turn = dW.sum(axis=1)
    if fixed_cost is not None:
        cost = turn * fixed_cost / 1e4
    else:
        cost = np.zeros(P.T)
        for i in range(i0, i1):
            cost[i] = (dW[i] * cost_vector(P, F, i)).sum() / 1e4 * cost_mult
    cost = cost + shortbook * borrow_bps / 1e4 / 252
    net = pr - cost
    npos = (np.abs(Wp) > 1e-9).sum(axis=1)
    s = pd.Series(net[i0:i1], index=P.dates[i0:i1]); g = pd.Series(pr[i0:i1], index=P.dates[i0:i1])
    gross_exp = np.abs(Wp).sum(axis=1)
    return s, g, dict(turn_yr=turn[i0:i1].sum() / ((i1 - i0) / 252), npos=float(npos[i0:i1].mean()),
                      npos_min=int(npos[i0:i1].min()), gross_exp=float(gross_exp[i0:i1].mean()))

def report(name, net, gross, info, full=False):
    yr = net.groupby(net.index.year).apply(sh)
    eq = (1 + net).cumprod(); dd = (eq / eq.cummax() - 1).min()
    line = (f"{name:38s} Sh net {sh(net):5.2f} gross {sh(gross):5.2f} | ret {net.mean()*252*100:5.1f}% vol {net.std()*np.sqrt(252)*100:4.1f}% "
            f"mdd {dd*100:5.1f}% | turn {info['turn_yr']:6.1f}x | npos {info['npos']:5.0f} (min {info['npos_min']}) gexp {info['gross_exp']:.2f} | worst-yr {yr.min():5.2f}")
    print(line, flush=True)
    if full:
        print("    per-year Sharpe: " + " ".join(f"{y}:{v:+.2f}" for y, v in yr.items()), flush=True)
        yret = net.groupby(net.index.year).apply(lambda x: x.sum() * 100)
        print("    per-year ret%:   " + " ".join(f"{y}:{v:+.1f}" for y, v in yret.items()), flush=True)
    return dict(name=name, sh_net=sh(net), sh_gross=sh(gross), ret=net.mean() * 252, vol=net.std() * np.sqrt(252), mdd=dd, turn=info["turn_yr"], npos=info["npos"], worst=yr.min())

def tranche(W, h):
    """overlapping tranches: average of the last h daily target weight vectors"""
    if h <= 1: return W
    c = np.cumsum(W, axis=0)
    out = c.copy(); out[h:] = c[h:] - c[:-h]
    return out / h

def scale_gross(W, g=1.0):
    s = np.abs(W).sum(axis=1, keepdims=True)
    return np.where(s > 0, W / (s + 1e-12) * g, 0.0)
