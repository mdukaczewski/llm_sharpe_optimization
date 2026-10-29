#!/usr/bin/env python3
"""dayml2 lane: commission-aware intraday books decided at 09:31 (fill 09:33 bar = bin 3).
Builds on agents/dayml/dlib (features + y_60d / y_120d targets) and adds
  * extra 09:31 features: 09:30 auction volume share, first-minute range, QQQ first minute / gap,
    cross-sectional dispersion of gaps / overnight moves, yesterday's realised-overnight rank,
    the overnight LightGBM prediction made at yesterday's close (agents/onml/out/pred_f28cal_q3.parquet)
  * extra exits: y_30d (bin 31 = 10:01) and y_90d (bin 91 = 11:01), both from the bin-3 fill
  * a lean walk-forward (numpy matrices, optional rolling window / recency weights / LightGBM params)
  * commission-aware evaluation: r_net = r - P x $6 / $400k per day on top of rlib's spread cost
Everything is cut at rlib.DEV_END (Min opened with end=DEV_END)."""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/llm_sharpe_optimization"); sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml")
import rlib, dlib
from rlib import DEV_END, zsd, Min
HERE = os.path.dirname(os.path.abspath(__file__))
CD = f"{rlib.DATA}/cache_dayml2x"; os.makedirs(CD, exist_ok=True)
ONML_PRED = "/home/user/llm_sharpe_optimization/agents/onml/out/pred_f28cal_q3.parquet"
CAP, COMM = 400000.0, 3.0            # $0.4M equity per book, $3 per filled order (2 orders per position per day)
NJ = int(os.environ.get("DAYML2_NJOBS", "2"))

def log(msg, fn=f"{HERE}/log.txt"):
    line = f"{time.strftime('%H:%M:%S')} {msg}"; print(line, flush=True)
    with open(fn, "a") as fh: fh.write(line + "\n")

def extra2(tk):
    """per-ticker extras known at 09:31 + the two new exit targets; cached (keyed on npz mtime and DEV_END)"""
    cp = f"{CD}/{tk}_{DEV_END}.parquet"; src = f"{rlib.DATA}/minute_npz/{tk}.npz"
    if os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(src):
        try: return pd.read_parquet(cp)
        except Exception: os.remove(cp)
    M = Min(tk, end=DEV_END); c = M.c; v = M.v
    d = pd.DataFrame(index=M.dates)
    v0 = v[:, 0]; vday = v[:, :391].sum(1)
    d["auc_share"] = v0 / (pd.Series(vday).rolling(20).mean().shift(1).values + 1)            # 09:30 bar volume / 20d mean daily volume
    d["auc_rel"] = v0 / (pd.Series(v0).rolling(20).mean().shift(1).values + 1)                 # vs its own 20d mean
    hi1 = np.nanmax(M.h_raw[:, :2], axis=1); lo1 = np.nanmin(M.l_raw[:, :2], axis=1)
    rng1 = (hi1 - lo1) / (c[:, 1] + 1e-9); d["m1_rng"] = rng1
    d["m1_rng_rel"] = rng1 / (pd.Series(rng1).rolling(20).mean().shift(1).values + 1e-9)
    d["m1_pos"] = (c[:, 1] - lo1) / (hi1 - lo1 + 1e-9)                                           # where the 09:31 close sits in the first-2-bar range
    d["y_30d"] = c[:, 31] / c[:, 3] - 1; d["y_90d"] = c[:, 91] / c[:, 3] - 1                    # exits 10:01 / 11:01, fill 09:33 bar
    for k in ["y_30d", "y_90d"]:
        x = d[k].values.astype(float); x[np.abs(x) > 0.5] = np.nan; d[k] = x
    d = d.astype(np.float32); d.index.name = "date"; d.to_parquet(cp); return d

def build(tks, verbose=True, onml=True):
    """dlib.build + extras.  Returns D sorted by (tk, date)."""
    t0 = time.time()
    D = dlib.build(tks, verbose=verbose)
    D = D.drop(columns=[c for c in D.columns if c.startswith("t30_") or c.startswith("t330_") or c.startswith("spy_t")])
    D = D[D.date <= DEV_END].sort_values(["tk", "date"]).reset_index(drop=True)
    ex = []
    for tk in D.tk.unique():
        try: e = extra2(tk); e["tk"] = tk; ex.append(e.reset_index())
        except Exception as ex_: print(tk, "extra2 skip", ex_, flush=True)
    E = pd.concat(ex); D = D.merge(E, on=["date", "tk"], how="left")
    # index ETFs of the same day (first minute / gap known at 09:31)
    q = rlib.ticker_frame("QQQ"); qe = dlib.extra_frame("QQQ"); se = dlib.extra_frame("SPY")
    sp = pd.DataFrame({"qqq_zm1": q["o_zm1"], "qqq_zgap": q["o_zgap"], "qqq_m1": qe["m1"], "spy_m1": se["m1"]})
    D = D.merge(sp, left_on="date", right_index=True, how="left")
    # cross-sectional dispersion of the day's gaps / overnight moves (and relative to their trailing 20-day mean)
    g = D.groupby("date")
    disp = pd.DataFrame({"gap_disp": g["gap"].apply(lambda s: s.clip(-0.2, 0.2).std()), "ron_disp": g["ron"].apply(lambda s: s.clip(-0.2, 0.2).std()),
                         "m1_disp": g["m1"].apply(lambda s: s.clip(-0.1, 0.1).std())})
    for c in ["gap_disp", "ron_disp", "m1_disp"]: disp[c + "_rel"] = disp[c] / disp[c].rolling(20).mean().shift(1)
    D = D.merge(disp, left_on="date", right_index=True, how="left")
    D["auc_rank"] = g["auc_share"].rank(pct=True) - 0.5
    D["m1rng_rank"] = g["m1_rng_rel"].rank(pct=True) - 0.5
    # yesterday's realised-overnight rank and yesterday's gap rank (shift within ticker; D sorted by tk, date)
    gt = D.groupby("tk")
    D["ron_rank_prev"] = gt["ron_rank"].shift(1); D["gap_rank_prev"] = gt["gap_rank"].shift(1)
    D["ron_rank_prev2"] = gt["ron_rank"].shift(2)
    # overnight model prediction made at the close of t-1 (known at 09:31 of t): shift its date to the next trading day
    D["on_pred"] = np.nan
    if onml and os.path.exists(ONML_PRED):
        P = pd.read_parquet(ONML_PRED); P = P[P.date <= DEV_END]
        cal = np.array(sorted(pd.to_datetime(Min("SPY", end=DEV_END).dates)))          # full trading calendar incl. half days
        nxt = pd.Series(np.r_[cal[1:], pd.NaT], index=cal)
        P["date"] = P["date"].map(nxt); P = P.dropna(subset=["date"]).rename(columns={"p": "on_pred"})
        D = D.drop(columns=["on_pred"]).merge(P[["date", "tk", "on_pred"]], on=["date", "tk"], how="left")
        D["on_pred_rank"] = D.groupby("date")["on_pred"].rank(pct=True) - 0.5
        print("on_pred merged: coverage %.3f" % D.on_pred.notna().mean(), flush=True)
    else:
        D["on_pred_rank"] = np.nan; print("on_pred NOT available", flush=True)
    for c in ["y_30d", "y_90d", "y_60d", "y_120d"]: D[c] = D[c].astype(np.float32)
    D["year"] = D.date.dt.year.astype(np.int16)
    print(f"build2 rows {len(D)} names {D.tk.nunique()} {time.time()-t0:.0f}s", flush=True)
    return D

NEW9 = ["auc_share", "auc_rel", "auc_rank", "m1_rng_rel", "m1rng_rank", "m1_pos", "qqq_zm1", "qqq_m1", "spy_m1"]
DISP = ["gap_disp", "gap_disp_rel", "ron_disp_rel", "m1_disp_rel"]
PREV = ["ron_rank_prev", "gap_rank_prev", "y_on_prev"]
ON = ["on_pred", "on_pred_rank"]
EXT2 = dlib.EXT + NEW9 + DISP + PREV
EXT2ON = EXT2 + ON

def matrix(D, feats):
    X = D[feats].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-5, 5).to_numpy(np.float32); return X

def rank_target(D, y):
    return (D.groupby("date")[y].rank(pct=True) - 0.5).to_numpy(np.float32)

def walk_forward(D, X, yr, retrain="Y", params=None, n_jobs=NJ, first="2017-01-01", label_lag_days=3, window_years=None, halflife_years=None, train_mask=None, verbose=False, seed=0):
    """Expanding (or rolling `window_years`) walk-forward LightGBM on a prebuilt matrix.  Rows with NaN target are
    not trained on.  halflife_years: exponential recency sample weights.  train_mask: extra boolean restriction
    on training rows (e.g. ranks > 600).  Returns a float32 array aligned to D (NaN before `first`)."""
    import lightgbm as lgb
    P = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=n_jobs, random_state=seed)
    if params: P.update(params)
    dates = D.date.to_numpy(); ok = np.isfinite(yr)
    if train_mask is not None: ok = ok & np.asarray(train_mask, bool)
    cuts = list(pd.date_range(first, DEV_END, freq="QS" if retrain == "Q" else "YS")) + [pd.Timestamp("2027-01-01")]
    pred = np.full(len(D), np.nan, np.float32)
    for i, c0 in enumerate(cuts[:-1]):
        c1 = cuts[i + 1]; lim = np.datetime64(c0 - pd.Timedelta(days=label_lag_days))
        tr = ok & (dates < lim)
        if window_years: tr &= dates >= np.datetime64(c0 - pd.Timedelta(days=int(365.25 * window_years)))
        te = (dates >= np.datetime64(c0)) & (dates < np.datetime64(c1))
        if te.sum() == 0 or tr.sum() < 1000: continue
        w = None
        if halflife_years:
            age = (lim - dates[tr]).astype("timedelta64[D]").astype(float) / 365.25; w = 0.5 ** (age / halflife_years)
        t = time.time(); m = lgb.LGBMRegressor(**P).fit(X[tr], yr[tr], sample_weight=w)
        pred[te] = m.predict(X[te])
        if verbose: print(f"  wf {c0.date()} train {int(tr.sum())} {time.time()-t:.0f}s", flush=True)
    return pred

# ---------------------------------------------------------------- books at the commission
def comm(r, npos):
    """subtract the fixed commission: npos positions x 2 orders x $3 / $0.4M per day (npos: int or Series by date)"""
    if np.isscalar(npos): return r - npos * 2 * COMM / CAP
    return r - npos.reindex(r.index).fillna(0) * 2 * COMM / CAP

def book(D, pred, y, k=6, mode="ls", weight="eq", mask=None, skip=None, p=False, name=""):
    """r (spread-only) and r_net (with commission).  mask: boolean per row restricting the tradable names.
    skip: boolean Series by date (True = flat that day)."""
    pr = pd.Series(np.asarray(pred, float), index=D.index)
    if mask is not None: pr = pr.where(np.asarray(mask, bool))
    if skip is not None: pr = pr.where(~D.date.map(skip).fillna(False).to_numpy(bool))
    r = rlib.book(D, pr, k=k, y=y, mode=mode, weight=weight, p=False)
    npos = k * (2 if mode == "ls" else 1)
    traded = (r != 0).astype(float) if skip is not None else None
    rn = comm(r, npos * traded) if traded is not None else comm(r, npos)
    if p: rlib.stats(rn, name)
    return r, rn

def ev(r, rn, label="", p=True):
    """full-sample and fold-OOS Sharpe at spread-only and at the commission + yearly net Sharpe"""
    fo = rlib.fold_oos(rn, p=False)[0]; fos = rlib.fold_oos(r, p=False)[0]
    yr = rn.groupby(rn.index.year).apply(rlib.sh)
    h1 = rn[rn.index < "2021-10-28"]; h2 = rn[rn.index >= "2021-10-28"]
    line = (f"{label:44s} spread Sh {rlib.sh(r):5.2f} fo {rlib.sh(fos):5.2f} | $0.4M Sh {rlib.sh(rn):5.2f} fo {rlib.sh(fo):5.2f} ret {rn.mean()*252*100:5.1f}% halves {rlib.sh(h1):.2f}/{rlib.sh(h2):.2f} | "
            + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in yr.items()))
    if p: log(line)
    return dict(label=label, sh=rlib.sh(r), fo_spread=rlib.sh(fos), sh_net=rlib.sh(rn), fo_net=rlib.sh(fo), ret=rn.mean() * 252, h1=rlib.sh(h1), h2=rlib.sh(h2))

def load_preds(D, y):
    """saved quarterly baseline predictions (agents/dayml/out) aligned to D (NaN where missing), cut at DEV_END"""
    P = pd.read_parquet(f"/home/user/llm_sharpe_optimization/agents/dayml/out/pred_{y}_Q.parquet"); P = P[P.date <= DEV_END]
    M = D[["date", "tk"]].merge(P, on=["date", "tk"], how="left")
    return M["p"].to_numpy(float)
