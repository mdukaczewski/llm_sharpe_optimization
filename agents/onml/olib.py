#!/usr/bin/env python3
"""onml lane: learned overnight book helpers (features beyond rlib.FEATS18, targets, runs).
All features are known at the CLOSE of day t; targets are forward.  Never edits rlib."""
import os, sys, time
import numpy as np, pandas as pd
HERE = "/home/user/llm_sharpe_optimization"; sys.path.insert(0, HERE)
import rlib
from rlib import FEATS18, sh
OUT = f"{HERE}/agents/onml/out"; os.makedirs(OUT, exist_ok=True)
LOG = f"{HERE}/agents/onml/log.txt"

def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as fh: fh.write(s + "\n")

EXTRA10 = ["z15l", "z5l", "zmoc", "zvwap", "vfirst_share", "vmoc_share", "ldv", "cc21", "rv5_20", "spy_zgap"]
XR_BASE = ["z30", "z60", "zgap", "zcc", "zoc", "z120", "rngpos", "vday_rel", "cc5", "rv20", "zon_prev", "ldv", "z30l"]
XR = [f + "_xr" for f in XR_BASE]
MR = ["z60_mr", "z30_mr", "zcc_mr", "zoc_mr"]
CAL = ["dow", "dme", "qend", "dsgap"]

def load(n=400, tickers=None, extra=True):
    """Stacked frame for universe(n) (or explicit tickers) with FEATS18 + EXTRA10 + cross-sectional + calendar
    features and targets yr (rank), yv (vol-scaled), ycls (decile class), yr60 (10:31 exit rank)."""
    t0 = time.time()
    tks = tickers if tickers is not None else rlib.universe(n)
    D = rlib.minute_frame(tks, verbose=False); D = rlib.add_market(D)
    log(f"load: {len(tks)} names rows {len(D)} {time.time()-t0:.0f}s")
    D = D.reset_index(drop=True)   # keep universe order (per-ticker blocks, date-sorted) = baseline row order
    if extra:
        g = D.groupby("date")
        for f in XR_BASE: D[f + "_xr"] = (g[f].rank(pct=True) - 0.5).astype(np.float32)
        for f in ["z60", "z30", "zcc", "zoc"]: D[f + "_mr"] = (D[f] - g[f].transform("median")).astype(np.float32)
        D["dow"] = D.date.dt.dayofweek.astype(np.float32)
        D["dme"] = np.clip(((D.date + pd.offsets.BMonthEnd(0)) - D.date).dt.days, 0, 25).astype(np.float32)
        D["qend"] = ((D.date.dt.month % 3 == 0) & (D.dme <= 2)).astype(np.float32)
        big = (D.zgap.abs() > 2.5).astype(int)                      # days since last |gap z| > 2.5 (per name)
        grp = big.groupby(D.tk)
        idx = np.arange(len(D)); last = pd.Series(np.where(big == 1, idx, np.nan)).groupby(D.tk).ffill()
        first = pd.Series(idx).groupby(D.tk).transform("min")
        D["dsgap"] = np.clip((idx - last.fillna(first).values), 0, 60).astype(np.float32)
        D["nnames"] = g["tk"].transform("size").astype(np.float32)
    # targets (y_on known at 09:31 of t+1)
    D["yr"] = (D.groupby("date").y_on.rank(pct=True) - 0.5).astype(np.float32)
    von = D.groupby("tk").y_on.transform(lambda s: s.rolling(60, min_periods=30).std().shift(1))
    D["yv"] = (D.y_on / (von + 1e-9)).clip(-5, 5).astype(np.float32); D.loc[D.y_on.isna(), "yv"] = np.nan
    D["ycls"] = np.where(D.yr > 0.4, 1.0, np.where(D.yr < -0.4, -1.0, 0.0)).astype(np.float32); D.loc[D.yr.isna(), "ycls"] = np.nan
    D["yr60"] = (D.groupby("date").y_on60.rank(pct=True) - 0.5).astype(np.float32)
    D["ivol"] = (1.0 / (D.rv20.replace(0, np.nan).clip(0.05, 2.0))).astype(np.float32)
    feats = [c for c in D.columns if c.endswith("_xr") or c.endswith("_mr")] + EXTRA10 + CAL + FEATS18
    D = rlib.clean(D, [f for f in feats if f in D.columns])
    log(f"load done {len(D)} rows, {time.time()-t0:.0f}s, mem {D.memory_usage().sum()/1e9:.2f}GB")
    return D

def wf(D, feats, target="yr", retrain="Y", seed=0, n_jobs=2, extra_params=None, label_lag_days=3, first="2017-01-01"):
    P = dict(random_state=seed)
    if extra_params: P.update(extra_params)
    return rlib.walk_forward(D, feats, target, retrain=retrain, params=P, n_jobs=n_jobs, verbose=False, label_lag_days=label_lag_days, first=first)

def xrank(D, pred):
    """per-day pct rank of a prediction (for averaging models with different scales)"""
    s = pd.Series(pred, index=D.index); ok = s.notna()
    out = pd.Series(np.nan, index=D.index); out[ok] = s[ok].groupby(D.date[ok]).rank(pct=True) - 0.5
    return out

def book2(D, pred, k=20, y="y_on", weight="eq", qmin=None, volscale=False, cost=None, name="", p=True, half=None, borrow=0.01):
    """Extension of rlib.book: optional qmin (keep only names with |rank-pct| beyond per-day quantile -> fewer names),
    volscale (weights ~ 1/rv20 within side, sum |w| = 1 per side), half ('even'/'odd' rank split check)."""
    g = D.loc[pred.notna(), ["date", "tk", y, "rv20"]].copy(); g["p"] = pred[g.index].values; g = g[g[y].notna()]
    g["c"] = g.tk.map(rlib.cost_bps) if cost is None else cost
    g["rk"] = g.groupby("date")["p"].rank(method="first"); g["n"] = g.groupby("date")["p"].transform("size")
    g = g[g.n >= k + 5]
    top = g[g.rk > g.n - k].copy(); bot = g[g.rk <= k].copy()
    if half is not None:
        top["o"] = top.groupby("date").rk.rank(method="first"); bot["o"] = bot.groupby("date").rk.rank(method="first")
        m = 0 if half == "even" else 1; top = top[top.o % 2 == m]; bot = bot[bot.o % 2 == m]
    if qmin is not None:   # require prediction beyond per-day quantile (in |p| terms) -> variable book size
        pa = g.p.abs(); thr = pa.groupby(g.date).transform(lambda s: s.quantile(qmin))
        top = top[top.p.abs() >= thr[top.index]]; bot = bot[bot.p.abs() >= thr[bot.index]]
    def side(x, sgn):
        if len(x) == 0: return pd.DataFrame(columns=["date", "w", "r", "c"])
        if weight == "pred": w = x.p.abs()
        elif volscale: w = 1.0 / x.rv20.clip(0.05, 2.0)
        else: w = pd.Series(1.0, index=x.index)
        nn = x.groupby("date").p.transform("size")
        w = w / w.groupby(x.date).transform("sum") * (nn / k).clip(upper=1.0)   # sum|w| per side = min(n,k)/k
        return pd.DataFrame({"date": x.date, "w": sgn * w, "r": x[y], "c": x.c})
    L = pd.concat([side(top, 1), side(bot, -1)])
    L["g"] = L.w * L.r; L["cost"] = L.w.abs() * 2 * L.c / 1e4 + L.w.clip(upper=0).abs() * borrow / 252
    out = L.groupby("date").agg(g=("g", "sum"), cost=("cost", "sum")); n = (out.g - out.cost).loc[rlib.START:rlib.DEV_END]
    if p: rlib.stats(n, name or f"ls top{k} {y}")
    return n

def save(name, r):
    r.rename("ret").rename_axis("date").to_csv(f"{OUT}/{name}.csv", float_format="%.8f")

def load_out(pattern="*"):
    import glob
    return {os.path.basename(p)[:-4]: pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0] for p in sorted(glob.glob(f"{OUT}/{pattern}.csv"))}

def sleeve():
    s = pd.read_csv(f"{HERE}/sleeves/onlgb1200q_ls20.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return s.loc[:rlib.DEV_END]

def spy_cc():
    f = rlib.ticker_frame("SPY"); return f.close.pct_change().loc[:rlib.DEV_END]

def report_line(name, r, oos=None):
    s = rlib.stats(r, name, p=False); sl = sleeve(); sp = spy_cc()
    j = pd.concat([r, sl, sp], axis=1).dropna()
    c1 = j.iloc[:, 0].corr(j.iloc[:, 1]); c2 = j.iloc[:, 0].corr(j.iloc[:, 2])
    o = rlib.fold_oos({"x": r}, p=False)[0] if oos is None else oos
    return f"{name:34s} full {s['sharpe']:.2f} foldOOS {sh(o):.2f} halves {s['h1']:.2f}/{s['h2']:.2f} mdd {s['mdd']*100:.1f}% corr_sleeve {c1:.2f} corr_spy {c2:+.2f}"
