#!/usr/bin/env python3
"""dayml lane: learned INTRADAY books decided at 09:31 (and 10:00 / 15:00) -- feature builder.
All rows are (date, tk); every feature of a row is known at the DECISION time of that row's book:
  * close-time features of rlib.ticker_frame are shifted by ONE day within each ticker (yesterday's close)
  * o_* columns (gap, first minute, first-minute volume) are today's, known at 09:31
  * extra columns from ilib.Min: raw gap / realised overnight return, first-30-minute and 15:00 features,
    and explicit targets for alternative horizons (all |y|>0.5 -> NaN).
Cut at rlib.DEV_END everywhere (Min is opened with end=DEV_END)."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/llm_sharpe_optimization")
import rlib
from rlib import DEV_END, zsd, Min
HERE = os.path.dirname(os.path.abspath(__file__))
CD = f"{rlib.DATA}/cache_dayml2"; os.makedirs(CD, exist_ok=True)      # v2: delayed-fill targets (decide 09:31, fill at the 09:33 bar = bin 3)

LAG = ["z30", "z60", "z30l", "z120", "z15l", "z5l", "zmoc", "zoc", "zo15", "zgap", "rngpos", "rng", "vlast_share", "vfirst_share",
       "vmoc_share", "vday_rel", "zvwap", "zcc", "cc5", "cc21", "rv20", "rv5_20", "zon_prev", "ldv"]

def extra_frame(tk):
    """Min-based extras, cached per ticker (keyed on npz mtime and DEV_END)."""
    cp = f"{CD}/{tk}_{DEV_END}.parquet"; src = f"{rlib.DATA}/minute_npz/{tk}.npz"
    if os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(src):
        try: return pd.read_parquet(cp)
        except Exception: os.remove(cp)
    M = Min(tk, end=DEV_END); r = M.r; c = M.c; v = M.v
    d = pd.DataFrame(index=M.dates)
    d["gap"] = M.gap                                   # raw gap open(t)/close(t-1)-1 (split adj)
    d["ron"] = c[:, 1] / M.prev_close - 1              # realised overnight close(t-1) -> 09:31 bar of t
    d["o_zron"] = zsd(d["ron"].values)
    d["m1"] = r(-1, 1)
    hi30 = np.nanmax(M.h_raw[:, :31], axis=1); lo30 = np.nanmin(M.l_raw[:, :31], axis=1)
    # ---- known at 10:00 (end of bin 30) ----
    d["t30_zo"] = zsd(r(-1, 30)); d["t30_z1"] = zsd(r(1, 30)); d["t30_z5"] = zsd(r(25, 30)); d["t30_z15"] = zsd(r(15, 30))
    d["t30_rngpos"] = (c[:, 30] - lo30) / (hi30 - lo30 + 1e-9)
    d["t30_rng"] = (hi30 - lo30) / c[:, 30]; d["t30_rng"] = d["t30_rng"] / (d["t30_rng"].rolling(20).mean().shift(1) + 1e-9)
    d["t30_zvwap"] = zsd(c[:, 30] / M.cvwap[:, 30] - 1)
    v31 = v[:, :31].sum(1); d["t30_vrel"] = v31 / (pd.Series(v31).rolling(20).mean().shift(1).values + 1)
    # ---- known at 15:00 (end of bin 330) ----
    hi330 = np.nanmax(M.h_raw[:, :331], axis=1); lo330 = np.nanmin(M.l_raw[:, :331], axis=1)
    d["t330_zo"] = zsd(r(-1, 330)); d["t330_z30"] = zsd(r(300, 330)); d["t330_z60"] = zsd(r(270, 330)); d["t330_z5"] = zsd(r(325, 330))
    d["t330_rngpos"] = (c[:, 330] - lo330) / (hi330 - lo330 + 1e-9)
    d["t330_zvwap"] = zsd(c[:, 330] / M.cvwap[:, 330] - 1)
    v331 = v[:, :331].sum(1); d["t330_vrel"] = v331 / (pd.Series(v331).rolling(20).mean().shift(1).values + 1)
    d["t330_vlast30"] = v[:, 300:331].sum(1) / (v331 + 1)
    # ---- targets (decision bin m fills at c[:, m+1]) ----
    d["y_60"] = c[:, 60] / c[:, 2] - 1                 # 09:31 decision, exit at the 10:30 bar
    d["y_120"] = c[:, 120] / c[:, 2] - 1               # exit 11:30
    d["y_1000"] = M.close / c[:, 31] - 1               # 10:00 decision -> close
    d["y_1500"] = M.close / c[:, 331] - 1              # 15:00 decision -> close
    d["y_day2"] = M.close / c[:, 2] - 1                # same as rlib y_day (check)
    # DELAYED-FILL targets: the 09:32 bar is a single print that still carries the 09:31 bounce the model
    # keys on (diag_liq.py: 4.4 -> 2.4 Sharpe on the thin names); fill at the 09:33 bar (bin 3) instead,
    # exit one bar later as well.  These are the targets the kept candidates use.
    d["y_60d"] = c[:, 61] / c[:, 3] - 1; d["y_120d"] = c[:, 121] / c[:, 3] - 1; d["y_dayd"] = M.close / c[:, 3] - 1
    for k in ["y_60", "y_120", "y_1000", "y_1500", "y_day2", "y_60d", "y_120d", "y_dayd"]:
        x = d[k].values.astype(float); x[np.abs(x) > 0.5] = np.nan; d[k] = x
    d = d.astype(np.float32); d.index.name = "date"; d.to_parquet(cp); return d

def ticker_rows(tk, _retry=True):
    try: f = rlib.ticker_frame(tk).copy()
    except Exception as ex:                       # another agent may be writing the shared cache file: retry once
        if not _retry: raise
        time.sleep(3); return ticker_rows(tk, _retry=False)
    for c in LAG: f[c] = f[c].shift(1)                 # yesterday's close-time features
    f["y_on_prev"] = f["y_on"].shift(1)                 # realised overnight into day t (raw)
    e = extra_frame(tk)
    f = f.join(e, how="left")
    return f

def build(tks, verbose=True):
    t0 = time.time(); rows = []
    for i, tk in enumerate(tks):
        try: rows.append(ticker_rows(tk))
        except Exception as ex: print(tk, "skip", ex, flush=True)
        if verbose and (i + 1) % 100 == 0: print(f"  build {i+1}/{len(tks)} {time.time()-t0:.0f}s", flush=True)
    D = pd.concat(rows).reset_index()
    D = D[(D.date >= "2016-06-01") & (D.date <= DEV_END) & (~D.half)].reset_index(drop=True)
    # SPY of the same day: today's gap / first minute / realised overnight, yesterday's close-time
    s = rlib.ticker_frame("SPY"); se = extra_frame("SPY")
    sp = pd.DataFrame({"spy_zgap": s["o_zgap"], "spy_zm1": s["o_zm1"], "spy_z60": s["z60"].shift(1), "spy_z30": s["z30"].shift(1),
                       "spy_oc": s["zoc"].shift(1), "spy_zron": se["o_zron"], "spy_ron": se["ron"], "spy_gap": se["gap"],
                       "spy_t30_zo": se["t30_zo"], "spy_t30_z1": se["t30_z1"], "spy_t330_zo": se["t330_zo"], "spy_t330_z30": se["t330_z30"]})
    D = D.merge(sp, left_on="date", right_index=True, how="left")
    # cross-sectional features of the day
    g = D.groupby("date")
    D["n_cs"] = g["tk"].transform("size")
    D["gap_rank"] = g["gap"].rank(pct=True) - 0.5
    D["ron_rank"] = g["ron"].rank(pct=True) - 0.5
    D["m1_rank"] = g["m1"].rank(pct=True) - 0.5
    D["gap_mkt"] = D["gap"] - g["gap"].transform("median")
    D["ron_mkt"] = D["ron"] - g["ron"].transform("median")
    D["mkt_gap"] = g["gap"].transform("median")
    dvol = D["rv20"] / np.sqrt(252) + 1e-4                   # rv20 is lagged (yesterday's), daily vol
    D["gap_rv"] = (D["gap"] / dvol).clip(-6, 6); D["ron_rv"] = (D["ron"] / dvol).clip(-6, 6); D["gapmkt_rv"] = (D["gap_mkt"] / dvol).clip(-6, 6)
    D["t30_rank"] = g["t30_zo"].rank(pct=True) - 0.5; D["t330_rank"] = g["t330_zo"].rank(pct=True) - 0.5
    D["dow"] = D.date.dt.dayofweek.astype(np.float32)
    D["gap_x_z60"] = D["o_zgap"] * D["z60"]
    D["rank"] = D.tk.map(rlib.RANK).astype(np.float32)
    return D

# feature sets
BASE = ["z30", "z60", "z30l", "z120", "zoc", "zo15", "rngpos", "rng", "vlast_share", "vday_rel", "zcc", "cc5", "zon_prev", "rv20",
        "o_zgap", "o_zm1", "o_vm1_rel", "gap_x_z60", "spy_z60", "spy_z30", "spy_oc", "spy_zgap", "spy_zm1"]      # = on_ml_day.py feature set
EXT = BASE + ["z15l", "z5l", "zmoc", "zvwap", "vfirst_share", "vmoc_share", "cc21", "rv5_20", "o_zron", "gap_rank", "ron_rank", "m1_rank",
              "gap_mkt", "gapmkt_rv", "gap_rv", "ron_rv", "mkt_gap", "spy_zron", "dow", "ldv"]
F1000 = EXT + ["t30_zo", "t30_z1", "t30_z5", "t30_z15", "t30_rngpos", "t30_rng", "t30_zvwap", "t30_vrel", "t30_rank", "spy_t30_zo", "spy_t30_z1"]
F1500 = EXT + ["t330_zo", "t330_z30", "t330_z60", "t330_z5", "t330_rngpos", "t330_zvwap", "t330_vrel", "t330_vlast30", "t330_rank", "spy_t330_zo", "spy_t330_z30"]

def prep(D, feats, y):
    """rank target per day; clean features; returns D (copy) with 'yr'"""
    D = D.copy(); D = rlib.clean(D, feats)
    D["yr"] = D.groupby("date")[y].rank(pct=True) - 0.5
    return D

def run(D, feats, y, retrain="Y", n_jobs=2, params=None, first="2017-01-01", verbose=False):
    D = prep(D, feats, y)
    pred = rlib.walk_forward(D, feats, "yr", retrain=retrain, n_jobs=n_jobs, params=params, first=first, verbose=verbose)
    return D, pred

def book(D, pred, k=20, y="y_day", min_names=0, cost=None, weight="eq", name="", p=True, smin=None):
    m = pred.copy()
    if min_names: m[D.n_cs < min_names] = np.nan
    return rlib.book(D, m, k=k, y=y, mode="ls", cost=cost, weight=weight, name=name, p=p, smin=smin)
