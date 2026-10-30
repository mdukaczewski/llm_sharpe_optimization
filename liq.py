#!/usr/bin/env python3
"""Per-name daily liquidity / spread proxies from minute bars, and a calibrated half-spread model.

liq_frame(tk) -> DataFrame indexed by date with (all LAGGED = known before the open of day t unless prefixed t_ / c_):
  ldv20      log 20d mean continuous-session dollar volume          lpx       log previous close
  hl1        20d mean (h-l)/c of the 09:31 bar                        hl5       20d mean (h-l)/c over bins 1-5
  hlc        20d mean (h-l)/c over bins 380-389                       roll30    20d mean Roll spread estimate, 1-min closes bins 1-30
  n30        20d mean trades per minute, bins 1-30                    nc30      same, bins 360-389
  dv0        20d mean dollar volume of the 09:30 bar (opening auction proxy)
  dvc        20d mean dollar volume of the 16:00 bar (closing auction proxy)
  t_hl0, t_hl1, t_n0, t_n1, t_dv0   same-day values of bins 0 / 1 (known at the end of the 09:31 bar)
  c_hlc, c_nc30                      same-day values of bins 380-389 / 360-389 (known at 15:59)
  hs0931, hs1559, hs1001  predicted NBBO half-spread in bps at those times (model fit on out/quotes_*.csv, see fit_spread_model)
Cache: data/cache_liq/{tk}.parquet (features), data/spread_model.pkl (LightGBM per time)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib
from rlib import Min
HERE = os.path.dirname(os.path.abspath(__file__)); CD = f"{rlib.DATA}/cache_liq"; os.makedirs(CD, exist_ok=True)
MODEL = f"{rlib.DATA}/spread_model.pkl"
BASE = ["ldv20", "lpx", "hl1", "hl5", "hlc", "roll30", "n30", "nc30", "dv0", "dvc"]
TODAY_OPEN = ["t_hl0", "t_hl1", "t_n0", "t_n1", "t_dv0"]; TODAY_CLOSE = ["c_hlc", "c_nc30"]
FEATS = {"0931": BASE + TODAY_OPEN, "1001": BASE + TODAY_OPEN, "1559": BASE + TODAY_CLOSE}

def _lag_mean(x, w=20):
    return pd.Series(x).rolling(w, min_periods=5).mean().shift(1).to_numpy()

def raw_frame(tk):
    cp = f"{CD}/{tk}.parquet"; src = f"{rlib.DATA}/minute_npz/{tk}.npz"
    if os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(src): return pd.read_parquet(cp)
    M = Min(tk, end="2026-10-28"); c = M.c; h = M.h; l = M.l; v = M.v; n = M.n
    with np.errstate(all="ignore"):
        hl = (h - l) / c
        hl1 = hl[:, 1]; hl5 = np.nanmean(hl[:, 1:6], 1); hlc = np.nanmean(hl[:, 380:390], 1); hl0 = hl[:, 0]
        r = np.diff(np.log(c[:, 0:31]), axis=1)                      # 1-min log returns bins 1..30
        r0, r1 = r[:, :-1], r[:, 1:]; cov = np.nanmean(r0 * r1, 1) - np.nanmean(r0, 1) * np.nanmean(r1, 1)
        roll = 2 * np.sqrt(np.maximum(-cov, 0))
        dv = (v * M.vw)[:, :390]; dvday = np.nansum(dv, 1)
        n30 = np.nanmean(n[:, 1:31], 1); nc30 = np.nanmean(n[:, 360:390], 1)
        dv0 = v[:, 0] * M.vw[:, 0]; dvc = v[:, 390] * M.vw[:, 390]
        d = pd.DataFrame({
            "ldv20": np.log(_lag_mean(dvday) + 1), "lpx": np.log(pd.Series(M.close).shift(1).to_numpy()),
            "hl1": _lag_mean(hl1), "hl5": _lag_mean(hl5), "hlc": _lag_mean(hlc), "roll30": _lag_mean(roll),
            "n30": _lag_mean(n30), "nc30": _lag_mean(nc30), "dv0": np.log(_lag_mean(dv0) + 1), "dvc": np.log(_lag_mean(dvc) + 1),
            "t_hl0": hl0, "t_hl1": hl1, "t_n0": n[:, 0], "t_n1": n[:, 1], "t_dv0": np.log(dv0 + 1),
            "c_hlc": hlc, "c_nc30": nc30}, index=pd.DatetimeIndex(M.dates)).astype(np.float32)
    d.index.name = "date"; d.to_parquet(cp); return d

def predict_spreads(d):
    import pickle
    if not os.path.exists(MODEL): return d
    models = pickle.load(open(MODEL, "rb"))
    for t, (m, factor) in models.items():
        d[f"hs{t}"] = (factor * np.exp(m.predict(d[FEATS[t]].astype(np.float64)))).astype(np.float32)
    return d

def liq_frame(tk):
    return predict_spreads(raw_frame(tk))

def build_all(n=2000, procs=4):
    from multiprocessing import Pool
    tks = rlib.universe(n) + ["SPY", "QQQ", "VIXY", "UVXY", "VIXM"]
    with Pool(procs) as p:
        for i, _ in enumerate(p.imap_unordered(raw_frame, tks, chunksize=8)):
            if i % 200 == 0: print(i, flush=True)
    print("LIQ CACHE DONE", flush=True)

def training_rows():
    """(tk, date, time, half-spread bps) rows from out/quotes_trades.csv (book positions) and out/quotes_panel.csv (unbiased)."""
    rows = []
    p = f"{HERE}/out/quotes_trades.csv"
    if os.path.exists(p):
        T = pd.read_csv(p, parse_dates=["date", "d_in", "d_out"])
        T["hs_in"] = (T.ask_in - T.bid_in) / (T.ask_in + T.bid_in) * 1e4; T["hs_out"] = (T.ask_out - T.bid_out) / (T.ask_out + T.bid_out) * 1e4
        for bk, g in T.groupby("book"):
            tin, tout = ("0933", "1001") if bk.startswith("dayml") else ("1559", "0931")
            rows.append(pd.DataFrame({"tk": g.tk, "date": g.d_in, "t": tin, "hs": g.hs_in}))
            rows.append(pd.DataFrame({"tk": g.tk, "date": g.d_out, "t": tout, "hs": g.hs_out}))
    p = f"{HERE}/out/quotes_panel.csv"
    if not os.path.exists(p) and os.path.exists(p + ".gz"): p = p + ".gz"
    if os.path.exists(p):
        Q = pd.read_csv(p, parse_dates=["date"], dtype={"t": str}).dropna(subset=["bid", "ask"])
        rows.append(pd.DataFrame({"tk": Q.tk, "date": Q.date, "t": Q.t, "hs": (Q.ask - Q.bid) / (Q.ask + Q.bid) * 1e4}))
    R = pd.concat(rows); R = R[(R.hs > 0) & (R.hs < 500)]
    R["t"] = R.t.replace({"0933": "0931"})      # the 09:33 entry spread is modelled with the 09:31 model
    return R.reset_index(drop=True)

def fit_spread_model(verbose=True):
    import lightgbm as lgb, pickle
    R = training_rows(); out = {}; models = {}
    feats = {}
    for tk in R.tk.unique():
        try: feats[tk] = raw_frame(tk)
        except Exception as e: pass
    parts = []
    for tk, g in R.groupby("tk"):
        if tk not in feats: continue
        f = feats[tk].reindex(g.date.values); f["tk"] = tk; f["date"] = g.date.values; f["t"] = g.t.values; f["hs"] = g.hs.values; parts.append(f)
    D = pd.concat(parts).reset_index(drop=True)
    for t in ["0931", "1559", "1001"]:
        S = D[D.t == t].dropna(subset=FEATS[t]); y = np.log(S.hs.values); X = S[FEATS[t]].astype(np.float64)
        if len(S) < 500: continue
        # grouped CV by ticker: hash of name
        grp = pd.util.hash_pandas_object(S.tk, index=False).values % 5; pred = np.zeros(len(S))
        for k in range(5):
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
            m.fit(X[grp != k], y[grp != k]); pred[grp == k] = m.predict(X[grp == k])
        r2 = 1 - np.var(y - pred) / np.var(y); corr = np.corrcoef(np.exp(pred), S.hs)[0, 1]
        factor = S.hs.mean() / np.exp(pred).mean()            # retransformation (smearing) factor so the mean in bps is unbiased
        pred = pred + np.log(factor); mae = np.mean(np.abs(np.exp(pred) - S.hs))
        # by year, mean actual vs predicted
        yr = pd.DataFrame({"y": S.hs.values, "p": np.exp(pred), "yr": S.date.dt.year.values}).groupby("yr").mean().round(1)
        if verbose: print(f"spread model {t}: n {len(S)} names {S.tk.nunique()}  grouped-CV R2(log) {r2:.3f}  corr(bps) {corr:.3f}  MAE {mae:.1f} bps\n{yr.T.to_string()}", flush=True)
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(X, y)
        models[t] = (m, factor); out[t] = (r2, corr, len(S)); print(f"  smearing factor {factor:.3f}", flush=True)
    pickle.dump(models, open(MODEL, "wb")); return out

if __name__ == "__main__":
    if "build" in sys.argv: build_all()
    if "fit" in sys.argv: fit_spread_model()

# ------------------------------------------------------------------ official open target and execution-cost helpers
_panel = None
def official_open(tk, dates):
    global _panel
    if _panel is None:
        z = np.load(f"{rlib.DATA}/panel_liquid.npz", allow_pickle=True)
        _panel = (pd.DatetimeIndex(z["dates"]), {t: i for i, t in enumerate(z["tickers"])}, z["open"].astype(np.float64))
    pdx, ti, op = _panel
    if tk not in ti: return np.full(len(dates), np.nan)
    s = pd.Series(op[:, ti[tk]], index=pdx).reindex(pd.DatetimeIndex(dates)); return s.to_numpy()

def overnight_targets(tk):
    """Series on Min dates: y_oo = official open(t+1) / official close(t) - 1 (split-adjusted, MOC entry + MOO exit),
    y_on = 09:31 bar close(t+1) / official close(t) - 1 (the rlib target, for reference).  |y| > 0.5 -> NaN."""
    M = Min(tk, end="2026-10-28"); fwd = lambda x: np.r_[x[1:], np.nan]
    base = M.close * fwd(M.sf); oo = official_open(tk, M.dates); oo = np.where(np.isfinite(oo), oo, M.open)
    d = pd.DataFrame({"y_oo": fwd(oo) / base - 1, "y_on": fwd(M.c[:, 1]) / base - 1}, index=pd.DatetimeIndex(M.dates)); d.index.name = "date"
    d[d.abs() > 0.5] = np.nan; return d

def add_y_oo(D):
    """merge y_oo into a stacked (date, tk) frame."""
    parts = []
    for tk in D.tk.unique():
        e = overnight_targets(tk)[["y_oo"]].reset_index(); e["tk"] = tk; parts.append(e)
    return D.merge(pd.concat(parts), on=["date", "tk"], how="left")

def auction_cost_bps(order_usd, auction_dv_usd, rv_daily, c=0.3, floor=1.0):
    """Per-side cost of a market-on-open / market-on-close order (bps of notional): floor + c x daily vol (bps) x sqrt(order / auction volume).
    auction_dv_usd: expected auction dollar volume (liq_frame: exp(dv0) for the open, exp(dvc) for the close, lagged 20d means);
    rv_daily: daily return sd (fraction).  c = 0.3 is the central assumption; report 0.1 / 0.3 / 1.0."""
    part = np.asarray(order_usd, float) / np.maximum(np.asarray(auction_dv_usd, float), 1.0)
    return np.maximum(floor, c * np.asarray(rv_daily, float) * 1e4 * np.sqrt(part))

def spread_cost_bps(hs_bps, f=1.0):
    """Per-side cost of a continuous-market order that pays fraction f of the predicted NBBO half-spread (f = 1 crosses, 0.5 ~ midpoint-ish)."""
    return f * np.asarray(hs_bps, float)

def add_liq(D, cols=None):
    """merge liq_frame columns (default: all) into a stacked (date, tk) frame."""
    parts = []
    for tk in D.tk.unique():
        try: e = liq_frame(tk)
        except Exception: continue
        if cols is not None: e = e[cols]
        e = e.reset_index(); e["tk"] = tk; parts.append(e)
    return D.merge(pd.concat(parts), on=["date", "tk"], how="left")
