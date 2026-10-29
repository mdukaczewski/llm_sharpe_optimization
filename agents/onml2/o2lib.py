#!/usr/bin/env python3
"""onml2 lane: commission-aware overnight LightGBM book.  Builds on agents/onml/olib.py (never edits rlib/olib).
Commission convention (coordinator): $3 per filled order, $0.4M equity per book, every position traded in and
out daily -> a day with P positions costs P * 6 / 400000 of equity on top of the per-name spread cost in rlib."""
import os, sys, time
import numpy as np, pandas as pd
HERE = "/home/user/llm_sharpe_optimization"; sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/agents/onml")
import rlib
from rlib import sh, FEATS18
import olib
from olib import EXTRA10, CAL, load, xrank
OUT = f"{HERE}/agents/onml2/out"; os.makedirs(OUT, exist_ok=True)
LOG = f"{HERE}/agents/onml2/log.txt"; olib.LOG = LOG   # olib.load logs here, not into the onml lane
COMM = 6.0 / 400000.0          # per position per day ($3 in + $3 out on $0.4M)
F28CAL = FEATS18 + EXTRA10 + CAL
MOC = ["zmoc", "vmoc_share"]                 # closing print / closing-auction volume: known only AT the close -> not usable for a MOC order
F26CAL = [f for f in F28CAL if f not in MOC]  # the implementable set used for every kept candidate
F24CAL = [f for f in F26CAL if f not in ("z5l", "z15l")]   # also without the 15:44-15:59 returns (MOC must be entered by 15:50)
BASE = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)

def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as fh: fh.write(s + "\n")

# ---------------------------------------------------------------- walk-forward with window / recency weights
def wf2(D, feats, target="yr", retrain="Y", seed=0, n_jobs=2, params=None, window_years=None, halflife_years=None,
        label_lag_days=3, first="2017-01-01", verbose=False):
    """Like rlib.walk_forward but with optional rolling window (years of training data before the cut) and
    exponential recency sample weights (half-life in years). Returns raw predictions aligned to D.index."""
    import lightgbm as lgb
    P = dict(BASE); P.update(dict(verbose=-1, n_jobs=n_jobs, random_state=seed))
    if params: P.update(params)
    cuts = list(pd.date_range(first, rlib.DEV_END, freq="QS" if retrain == "Q" else "YS")) + [pd.Timestamp("2027-01-01")]
    pred = pd.Series(np.nan, index=D.index); ok = D[target].notna()
    for i, c0 in enumerate(cuts[:-1]):
        c1 = cuts[i + 1]; lab_end = c0 - pd.Timedelta(days=label_lag_days)
        tr = ok & (D.date < lab_end)
        if window_years is not None: tr &= D.date >= lab_end - pd.Timedelta(days=int(365.25 * window_years))
        te = (D.date >= c0) & (D.date < c1)
        if te.sum() == 0 or tr.sum() < 1000: continue
        X = D.loc[tr, feats].values; y = D.loc[tr, target].values; w = None
        if halflife_years is not None:
            age = (lab_end - D.loc[tr, "date"]).dt.days.values / 365.25
            w = np.power(0.5, age / halflife_years).astype(np.float32)
        t1 = time.time(); m = lgb.LGBMRegressor(**P).fit(X, y, sample_weight=w)
        pred[te] = m.predict(D.loc[te, feats].values)
        if verbose: log(f"  wf2 {c0.date()} train {int(tr.sum())} {time.time()-t1:.0f}s")
    return pred

# ---------------------------------------------------------------- commission-aware book
def book3(D, pred, k=20, y="y_on", mode="ls", weight="eq", qmin=None, raw=None, disp=None, disp_win=120, cost=None,
          borrow=0.01, name="", p=False, ks=None, sgross=1.0):
    """ks: number of shorts (default k); sgross: gross of the short side (1.0 = dollar neutral)."""
    ks = k if ks is None else ks
    """Top-k book from per-(date,tk) predictions.  Returns (net spread-only series, positions per day series).
    mode 'ls' (top-k long / bottom-k short) or 'lo' (top-k long).  weight 'eq' or 'pred' (|rank| weights).
    qmin: keep only names whose |raw| (raw model output, default pred) is >= the per-day qmin-quantile of |raw|
          -> variable book size (sum|w| per side scaled by n/k as in olib.book2).
    disp: trade only on days whose cross-sectional std of `raw` is above its trailing `disp_win`-day median
          (shifted one day: causal), disp = quantile of the trailing window (0.5 = median)."""
    g = D.loc[pred.notna(), ["date", "tk", y]].copy(); g["p"] = pred[g.index].values
    g["raw"] = (raw[g.index].values if raw is not None else g.p.values); g = g[g[y].notna()]
    g["c"] = g.tk.map(rlib.cost_bps) if cost is None else cost
    g["rk"] = g.groupby("date")["p"].rank(method="first"); g["n"] = g.groupby("date")["p"].transform("size")
    g = g[g.n >= k + 5]
    top = g[g.rk > g.n - k].copy(); bot = g[g.rk <= ks].copy() if mode == "ls" else g.iloc[0:0].copy()
    if qmin is not None:
        ra = g.raw.abs(); thr = ra.groupby(g.date).transform(lambda s: s.quantile(qmin))
        top = top[top.raw.abs() >= thr[top.index]]; bot = bot[bot.raw.abs() >= thr[bot.index]]
    if disp is not None:
        sd = g.groupby("date").raw.std(); thr = sd.rolling(disp_win, min_periods=40).quantile(disp).shift(1)
        good = sd.index[(sd > thr) | thr.isna()]
        top = top[top.date.isin(good)]; bot = bot[bot.date.isin(good)]
    def side(x, sgn, kk=k, gross=1.0):
        if len(x) == 0: return pd.DataFrame(columns=["date", "w", "r", "c"])
        if weight == "pred": w = x.p.abs()
        elif weight == "raw": w = x.raw.abs()                                   # |raw model output|
        elif weight == "lin": w = x.groupby("date").p.rank(ascending=(sgn > 0))  # 1..n, best name gets n
        else: w = pd.Series(1.0, index=x.index)
        nn = x.groupby("date").p.transform("size")
        w = w / w.groupby(x.date).transform("sum") * (nn / kk).clip(upper=1.0) * gross
        return pd.DataFrame({"date": x.date, "w": sgn * w, "r": x[y], "c": x.c, "raw": x.raw})
    L = pd.concat([side(top, 1), side(bot, -1, ks, sgross)])
    L["g"] = L.w * L.r; L["cost"] = L.w.abs() * 2 * L.c / 1e4 + L.w.clip(upper=0).abs() * borrow / 252
    out = L.groupby("date").agg(g=("g", "sum"), cost=("cost", "sum"), P=("w", "size"))
    alld = pd.Index(sorted(g.date.unique()))
    out = out.reindex(alld).fillna(0.0)
    n = (out.g - out.cost).loc[rlib.START:rlib.DEV_END]; P = out.P.loc[rlib.START:rlib.DEV_END]
    if p: rlib.stats(n, name)
    return n, P

def with_comm(r, P, comm=COMM):
    return r - P.reindex(r.index).fillna(0.0) * comm

def ev(r, P, name="", p=True):
    """full-sample and fold-OOS Sharpe at spread-only and at the $0.4M commission"""
    rc = with_comm(r, P)
    f0 = sh(rlib.fold_oos({"x": r}, p=False)[0]); f1 = sh(rlib.fold_oos({"x": rc}, p=False)[0])
    s0 = rlib.stats(r, p=False); s1 = rlib.stats(rc, p=False)
    d = dict(full0=s0["sharpe"], fold0=f0, full1=s1["sharpe"], fold1=f1, h1=s1["h1"], h2=s1["h2"], mdd=s1["mdd"], ret=s1["ret"], vol=s1["vol"], P=float(P.mean()))
    if p: log(f"{name:38s} spread-only {d['full0']:.2f}/{d['fold0']:.2f} | comm {d['full1']:.2f}/{d['fold1']:.2f} halves {d['h1']:.2f}/{d['h2']:.2f} mdd {d['mdd']*100:.1f}% ret {d['ret']*100:.0f}% vol {d['vol']*100:.1f}% P {d['P']:.1f}")
    return d

def save(name, r, P=None):
    df = pd.DataFrame({"ret": r.values, "P": (P.reindex(r.index).values if P is not None else np.nan)}, index=r.index)
    df.rename_axis("date").to_csv(f"{OUT}/{name}.csv", float_format="%.8f")

def load_out(name):
    df = pd.read_csv(f"{OUT}/{name}.csv", index_col=0, parse_dates=True); return df.ret, df.P

def hold_through(D, pred, k=20):
    """fraction of names in the top-k (bottom-k) on day t that are in it again on the next trading day"""
    g = D.loc[pred.notna(), ["date", "tk"]].copy(); g["p"] = pred[g.index].values
    g["rk"] = g.groupby("date")["p"].rank(method="first"); g["n"] = g.groupby("date")["p"].transform("size")
    g = g[g.n >= k + 5]; dates = sorted(g.date.unique()); nxt = dict(zip(dates[:-1], dates[1:]))
    top = g[g.rk > g.n - k]; bot = g[g.rk <= k]
    def frac(x):
        s = set(zip(x.date, x.tk)); hits = sum((nxt.get(d), t) in s for d, t in s if d in nxt); return hits / max(1, sum(d in nxt for d, _ in s))
    return frac(top), frac(bot)

def ic(D, pred, y="y_on"):
    return rlib.ic(D, pred, y)
