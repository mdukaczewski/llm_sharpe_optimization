#!/usr/bin/env python3
"""onml6 lane: the overnight book with a FEASIBLE information cutoff.

A market-on-close / limit-on-close order must be entered by 15:50 (NYSE) / 15:55 (Nasdaq).  Minute bin b (0 = 09:30 bar,
389 = 15:59 bar, 390 = official close) closes at 09:31 + b minutes, so
  cut = 378  -> last bar used = the 15:48 bar, closed at 15:49:00: the MOC / LOC decision time (order sent 15:49-15:50)
  cut = 388  -> the 15:58 bar, closed at 15:59:00: the decision time of the 15:59:30 continuous cross
Every feature of the onml3 f26calx / f40_on sets is rebuilt to the cut: returns end at c[cut], "to close" quantities use
c[cut] instead of the official close, volumes / ranges / trade counts use bins 0..cut, lagged quantities are unchanged,
mixed rolling statistics (cc5, cc21, rv20, rv5_20) use full close-to-close returns up to t-1 and the cut return on t.
NOTE: the ORIGINAL rlib.ticker_frame features (zoc, zcc, cc5, cc21, rv20, rv5_20, rngpos, rng, zvwap, ldv, vfirst_share)
use the OFFICIAL CLOSE / 16:00 auction volume of day t, i.e. information after the 15:59 bar as well.
Target unchanged: y_oo = official open(t+1) / official close(t) - 1 (fill at the official close)."""
import os, sys, time, pickle
import numpy as np, pandas as pd
HERE = "/home/user/llm_sharpe_optimization"; sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/agents/onml4")
import rlib, liq
from rlib import Min, zsd, sh
LANE = f"{HERE}/agents/onml6"; OUT = f"{LANE}/out"; os.makedirs(OUT, exist_ok=True); LOG = f"{LANE}/log.txt"
F26X = ['z30', 'z60', 'z30l', 'z120', 'zoc', 'zo15', 'zgap', 'rngpos', 'rng', 'vlast_x', 'vday_rel_x', 'zcc', 'cc5', 'rv20', 'zon_prev',
        'spy_z60', 'spy_z30', 'spy_oc', 'z15l', 'zvwap', 'vfirst_share', 'ldv_x', 'cc21', 'rv5_20', 'spy_zgap', 'dow', 'dme', 'qend', 'dsgap']
LIQF = ["ldv20", "lpx", "hl1", "hl5", "hlc", "roll30", "n30", "nc30", "dv0", "dvc", "c_hlc", "c_nc30", "hs0931", "hs1559"]
F40 = F26X + LIQF
# features recomputed at the cut (everything else in F40 is lagged or known at the open)
CUTF = ["z60", "z30l", "z120", "zoc", "rngpos", "rng", "vlast_x", "vday_rel_x", "zcc", "cc5", "rv20", "z15l", "zvwap", "vfirst_share",
        "ldv_x", "cc21", "rv5_20"]
CUTL = ["c_hlc", "c_nc30", "hs1559"]
SPYF = ["spy_z60", "spy_z30", "spy_oc", "spy_zgap"]

def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as fh: fh.write(s + "\n")

def _mix_std(full, cut, w):
    """std over the window [full[t-w+1..t-1], cut[t]] (ddof 1)"""
    f = pd.Series(full); s1 = f.rolling(w - 1).sum().shift(1).to_numpy() + cut; s2 = (f ** 2).rolling(w - 1).sum().shift(1).to_numpy() + cut ** 2
    return np.sqrt(np.maximum(s2 - s1 ** 2 / w, 0) / (w - 1))

_SM = None
def _spread_model():
    global _SM
    if _SM is None: _SM = pickle.load(open(liq.MODEL, "rb"))
    return _SM

def cut_frame(tk, cut):
    """per-day features of ticker tk with information up to the close of bin `cut` (see module doc).  Dates <= DEV_END."""
    M = Min(tk)                                     # ilib END = STRAT_END = DEV_END: holdout bars never loaded
    b = cut; cx = M.c[:, b]; r = M.r
    with np.errstate(all="ignore"):
        d = pd.DataFrame(index=pd.DatetimeIndex(M.dates))
        d["z30"] = zsd(r(-1, 29)); d["z60"] = zsd(r(b - 60, b)); d["z30l"] = zsd(r(b - 30, b)); d["z120"] = zsd(r(b - 120, b))
        d["z15l"] = zsd(r(b - 15, b)); d["z5l"] = zsd(r(b - 5, b))
        d["zoc"] = zsd(cx / M.open - 1); d["zgap"] = zsd(M.gap)
        hi = np.nanmax(M.h_raw[:, :b + 1], axis=1); lo = np.nanmin(M.l_raw[:, :b + 1], axis=1)
        d["rngpos"] = (cx - lo) / (hi - lo + 1e-9)
        rng = (hi - lo) / cx; d["rng"] = rng / (pd.Series(rng).rolling(20).mean().shift(1).to_numpy() + 1e-9)
        v = M.v; vday = v[:, :b + 1].sum(1)
        d["vlast_x"] = v[:, b - 59:b + 1].sum(1) / (vday + 1); d["vfirst_share"] = v[:, :30].sum(1) / (vday + 1)
        d["vday_rel_x"] = vday / (pd.Series(vday).rolling(20).mean().shift(1).to_numpy() + 1)
        d["ldv_x"] = np.log(vday * cx + 1)
        d["zvwap"] = zsd(cx / M.cvwap[:, b] - 1)
        ccf = M.cc; ccc = cx / M.prev_close - 1                   # full (official close) and cut close-to-close return
        d["zcc"] = zsd(ccc)
        sd20 = pd.Series(ccf).rolling(20).std().shift(1).to_numpy()
        s4 = pd.Series(ccf).rolling(4).sum().shift(1).to_numpy(); s20 = pd.Series(ccf).rolling(20).sum().shift(1).to_numpy()
        d["cc5"] = (s4 + ccc) / (sd20 * np.sqrt(5) + 1e-9); d["cc21"] = (s20 + ccc) / (sd20 * np.sqrt(21) + 1e-9)
        rv20 = _mix_std(ccf, ccc, 20); rv5 = _mix_std(ccf, ccc, 5)
        d["rv20"] = rv20 * np.sqrt(252); d["rv5_20"] = rv5 / (rv20 + 1e-9)
        hl = (M.h - M.l) / M.c
        d["c_hlc"] = np.nanmean(hl[:, b - 9:b + 1], 1); d["c_nc30"] = np.nanmean(M.n[:, b - 29:b + 1], 1)
        d["cb"] = cx; d["c388"] = M.c[:, 388]; d["c389"] = M.c[:, 389]; d["c_off"] = M.close
    # predicted 15:59 half-spread from the cut inputs (same booster as liq.hs1559, today's inputs end at the cut)
    if tk not in ("SPY",):
        try:
            e = liq.raw_frame(tk).reindex(d.index)
            X = pd.concat([e[liq.BASE], d[["c_hlc", "c_nc30"]]], axis=1)[liq.FEATS["1559"]].astype(np.float64)
            m, fac = _spread_model()["1559"]; d["hs1559"] = fac * np.exp(m.predict(X))
        except Exception as ex:
            d["hs1559"] = np.nan
    d = d.astype(np.float32); d.index.name = "date"
    return d.loc[:rlib.DEV_END]

def _one(args):
    tk, cuts = args
    out = {}
    for c in cuts:
        try:
            d = cut_frame(tk, c).reset_index(); d["tk"] = tk; out[c] = d
        except Exception as ex:
            print(tk, "skip", ex, flush=True)
    return out

def build(cuts=(378, 388), procs=3):
    """stacked per-cut frames out/cut{c}.parquet for rlib.universe(2000) (+ SPY features out/spy_cut{c}.parquet)"""
    from multiprocessing import Pool
    t0 = time.time(); tks = rlib.universe(2000); parts = {c: [] for c in cuts}
    with Pool(procs) as p:
        for i, o in enumerate(p.imap_unordered(_one, [(t, cuts) for t in tks], chunksize=4)):
            for c, d in o.items(): parts[c].append(d)
            if i % 200 == 0: print(f"  build {i}/{len(tks)} {time.time()-t0:.0f}s", flush=True)
    for c in cuts:
        X = pd.concat(parts[c], ignore_index=True); X.to_parquet(f"{OUT}/cut{c}.parquet")
        s = cut_frame("SPY", c)[["z60", "z30", "zoc", "zgap"]].rename(columns={"z60": "spy_z60", "z30": "spy_z30", "zoc": "spy_oc", "zgap": "spy_zgap"})
        s.reset_index().to_parquet(f"{OUT}/spy_cut{c}.parquet")
        log(f"build cut {c}: {len(X)} rows {X.tk.nunique()} names {time.time()-t0:.0f}s")

ONML3 = f"{HERE}/agents/onml3/out/frame_u2000.parquet"
KEEP = ["date", "tk", "z30", "zo15", "zgap", "zon_prev", "dow", "dme", "qend", "dsgap", "yr", "y_on", "y_oo", "rank", "tier",
        "ldv20", "lpx", "hl1", "hl5", "hlc", "roll30", "n30", "nc30", "dv0", "dvc", "hs0931"]

def model_frame(cut):
    """onml3 frame (same rows / order, half days dropped) with every post-cut feature replaced by its cut version.
    Features through rlib.clean (fillna 0, clip +-5) as in olib.load, liquidity columns unclipped as in onml3 (fix_liq)."""
    D = pd.read_parquet(ONML3, columns=KEEP)
    X = pd.read_parquet(f"{OUT}/cut{cut}.parquet", columns=["date", "tk"] + CUTF + CUTL + ["z5l"])
    X = X.rename(columns={"rv20": "rv20f"})
    D = D.merge(X, on=["date", "tk"], how="left")
    S = pd.read_parquet(f"{OUT}/spy_cut{cut}.parquet"); D = D.merge(S, on="date", how="left")
    D["rv20_raw"] = D.rv20f                                     # raw (NaN kept) for the cost model
    D["rv20"] = D.rv20f; D = D.drop(columns=["rv20f"])
    feats = [f for f in F26X if f not in ("dow", "dme", "qend", "dsgap")] + ["z5l"]
    D = rlib.clean(D, feats)
    D[["c_hlc", "c_nc30", "hs1559"]] = D[["c_hlc", "c_nc30", "hs1559"]].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return D

def wf(D, feats, target="yr", retrain="Y", seed=0, n_jobs=2, first="2017-01-01"):
    return rlib.walk_forward(D, feats, target, retrain=retrain, params=dict(random_state=seed), n_jobs=n_jobs, verbose=False, label_lag_days=3, first=first)

def xrank(D, pred):
    s = pd.Series(pred, index=D.index); ok = s.notna(); out = pd.Series(np.nan, index=D.index)
    out[ok] = s[ok].groupby(D.date[ok]).rank(pct=True) - 0.5; return out

def save_pred(D, p, name):
    pd.DataFrame({"date": D.date, "tk": D.tk, "p": p.astype(np.float32)}).dropna().to_parquet(f"{OUT}/pred_{name}.parquet")

if __name__ == "__main__":
    if "build" in sys.argv: build()
