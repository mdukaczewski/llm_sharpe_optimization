"""xsrule lane helpers (do not edit rlib.py). All loaders go through rlib (cut at rlib.DEV_END)."""
import os, sys, time
sys.path.insert(0, "/home/user/llm_sharpe_optimization")
import numpy as np, pandas as pd, rlib
from rlib import sh, stats, fold_oos, cost_bps, DEV_END, START
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.environ.get("XS_SCRATCH", "/tmp/claude-0/-home-user-llm-sharpe-optimization/418755a4-2f35-58dd-83cb-2d5774e89f89/scratchpad")

def load(n=400, cache=True, verbose=True):
    """Stacked frame for rlib.universe(n) + market features + extras (all known at the close of t)."""
    tks = rlib.universe(n)
    cp = f"{SCR}/D_{n}_{len(tks)}_{DEV_END}.parquet"
    if cache and os.path.exists(cp):
        D = pd.read_parquet(cp)
        if verbose: print(f"load cached {cp} rows {len(D)} names {D.tk.nunique()}", flush=True)
        return D
    D = rlib.add_market(rlib.minute_frame(tks, verbose=verbose))
    D = add_extras(D)
    if cache: D.to_parquet(cp)
    if verbose: print(f"built rows {len(D)} names {D.tk.nunique()}", flush=True)
    return D

def add_extras(D):
    D = D.sort_values(["tk", "date"]).reset_index(drop=True)
    g = D.groupby("tk", sort=False)
    # intraday leg of day t+1 (09:31 fill of t+1 -> official close of t+1), i.e. the complement of y_on inside y_cc
    D["y_id"] = (1 + D.y_cc) / (1 + D.y_on) - 1
    D["y_cc2"] = (1 + D.y_cc) * (1 + g.y_cc.shift(-1)) - 1          # close t -> close t+2
    D["y_cc5"] = (1 + D.y_cc) * (1 + g.y_cc.shift(-1)) * (1 + g.y_cc.shift(-2)) * (1 + g.y_cc.shift(-3)) * (1 + g.y_cc.shift(-4)) - 1
    for c in ["y_cc2", "y_cc5"]: D.loc[D[c].abs() > 0.5, c] = np.nan
    # lags of close-time features (known at close t)
    D["zcc_l1"] = g.zcc.shift(1); D["z60_l1"] = g.z60.shift(1); D["z30_l1"] = g.z30.shift(1)
    D["zon_l1"] = g.zon_prev.shift(1)
    # sum of the last 5 overnight returns / sd (overnight momentum), computed from zon_prev: approximate with rolling sum of z
    D["zon5"] = g.zon_prev.transform(lambda s: s.rolling(5).sum()) / np.sqrt(5)
    D["dow"] = D.date.dt.dayofweek.astype(np.float32)
    D["mdays_left"] = D.date.map(_mdays_left_map(D.date.unique())).astype(np.float32)
    # cross-sectional ranks (pct) of key signals, demeaned
    for c in ["z30", "z60", "z5l", "z15l", "zmoc", "zvwap", "rngpos", "vlast_share", "vmoc_share", "cc5", "zon_prev", "zcc", "cc21", "rv5_20", "vday_rel"]:
        D[f"r_{c}"] = (D.groupby("date")[c].rank(pct=True) - 0.5).astype(np.float32)
    return D

def _mdays_left_map(dates):
    d = pd.DatetimeIndex(sorted(dates)); s = pd.Series(np.arange(len(d)), d)
    ym = d.to_period("M"); last = s.groupby(ym).transform("max")
    return pd.Series((last.values - s.values).astype(float), d)

def heldbook(D, pred, k=20, y="y_cc", mode="ls", cost=None, smin=None, hold=1, borrow=0.01, name="", p=True, turn=False):
    """Book decided at the close, filled at the close, held `hold` days with overlapping tranches (1/hold each),
    costs charged on the actual change of weights (|dw| x cost per side) + borrow on shorts. y must be the 1-day c->c target."""
    g = D.loc[pred.notna(), ["date", "tk", y]].copy(); g["p"] = pred[g.index].values; g = g[g[y].notna()]
    g["rk"] = g.groupby("date")["p"].rank(method="first"); g["n"] = g.groupby("date")["p"].transform("size"); g = g[g.n >= k + 5]
    top = g[g.rk > g.n - k].copy(); bot = g[g.rk <= k].copy() if mode == "ls" else g.iloc[0:0].copy()
    if smin is not None: top = top[top.p > smin]
    top["w"] = 1.0 / k; bot["w"] = -1.0 / k
    T = pd.concat([top, bot])[["date", "tk", "w"]]
    W0 = T.pivot_table(index="date", columns="tk", values="w", aggfunc="sum").reindex(sorted(D.date.unique())).fillna(0.0)
    if hold > 1: W = sum(W0.shift(i).fillna(0.0) for i in range(hold)) / hold
    else: W = W0
    R = D.pivot_table(index="date", columns="tk", values=y).reindex(index=W.index, columns=W.columns)
    gross = (W * R.fillna(0.0)).sum(1)
    # with daily close-to-close returns, holdings drift; approximate rebalance back to target each day (cost on |dW|)
    dW = (W - W.shift(1).fillna(0.0)).abs()
    cb = pd.Series({t: (cost_bps(t) if cost is None else cost) for t in W.columns})
    c = (dW * cb).sum(1) / 1e4 + W.clip(upper=0).abs().sum(1) * borrow / 252
    n = (gross - c).loc[START:DEV_END]
    to = dW.sum(1).mean() / 2  # one-way turnover per day (fraction of book, book gross = 1 (lo) or 2 (ls))
    if p: stats(n, name or f"{mode} top{k} {y} h{hold}"); 
    if p: print(f"    one-way turnover/day {to:.3f}  gross expo {W.abs().sum(1).mean():.2f}", flush=True)
    return (n, to) if turn else n

def onbook(D, pred, k=20, mode="lo", smin=None, cost=None, name="", p=True, y="y_on"):
    return rlib.book(D, pred, k=k, y=y, mode=mode, cost=cost, smin=smin, name=name, p=p)

def refs():
    out = {}
    for nm, f in [("xslo20s1_150", "sleeves/xslo20s1_150.csv"), ("onlgb1200q", "sleeves/onlgb1200q_ls20.csv")]:
        out[nm] = pd.read_csv(f"/home/user/llm_sharpe_optimization/{f}", index_col=0, parse_dates=True).iloc[:, 0].loc[:DEV_END]
    spy = rlib.ticker_frame("SPY").loc[:DEV_END]
    out["spy_cc"] = spy.y_cc.astype(float)      # index t = SPY close t -> close t+1 (same convention as candidates)
    out["spy_on"] = spy.y_on.astype(float)      # SPY close t -> 09:31 t+1
    return out

def corr_table(r, R=None):
    R = R or refs(); return {k: round(float(r.corr(v.reindex(r.index))), 3) for k, v in R.items()}

def summary(r, name):
    s = stats(r, name, p=False); oos, tab = fold_oos(r, p=False)
    c = corr_table(r)
    print(f"{name:34s} Sh {s['sharpe']:.2f} OOS {sh(oos):.2f} halves {s['h1']:.2f}/{s['h2']:.2f} mdd {s['mdd']*100:.1f}% folds min {tab.fold_sh.min():.2f} neg {(tab.fold_sh<0).sum()} | corr {c}", flush=True)
    return s, oos

# ---- extra per-ticker features built in this lane (own cache; everything known by the 15:59 bar, i.e. before the closing print)
def ticker_extra(tk):
    cd = f"{SCR}/xcache"; os.makedirs(cd, exist_ok=True); cp = f"{cd}/{tk}.parquet"; src = f"{rlib.DATA}/minute_npz/{tk}.npz"
    if os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(src): return pd.read_parquet(cp)
    M = rlib.Min(tk, end="2026-10-28")
    d = pd.DataFrame(index=M.dates); d["tk"] = tk
    vday_x = M.v[:, :390].sum(1)                               # continuous-session volume, auction excluded
    d["vlast_x"] = (M.v[:, 330:390].sum(1) / (vday_x + 1)).astype(np.float32)   # last-hour share ex auction
    d["v15_x"] = (M.v[:, 375:390].sum(1) / (vday_x + 1)).astype(np.float32)     # last-15-min share ex auction
    d["z5l_x"] = rlib.zsd(M.r(384, 389)).astype(np.float32)                      # same as z5l (15:54 -> 15:59), kept for clarity
    d.index.name = "date"; d.to_parquet(cp); return d

def add_extra2(D, verbose=True):
    rows = []; t0 = time.time(); tks = list(D.tk.unique())
    for i, tk in enumerate(tks):
        try: rows.append(ticker_extra(tk))
        except Exception as e: print(tk, "extra skip", e, flush=True)
        if verbose and (i + 1) % 100 == 0: print(f"  extra {i+1}/{len(tks)} {time.time()-t0:.0f}s", flush=True)
    E = pd.concat(rows).reset_index()
    D = D.merge(E, on=["date", "tk"], how="left")
    for c in ["vlast_x", "v15_x"]: D[f"r_{c}"] = (D.groupby("date")[c].rank(pct=True) - 0.5).astype(np.float32)
    return D
