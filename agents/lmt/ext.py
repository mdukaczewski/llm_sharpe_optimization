#!/usr/bin/env python3
"""lmt lane, step 0: one pass over the minute bars of rlib.universe(2000), dates <= rlib.DEV_END.
Per (date, tk), raw (unadjusted, same units as the NBBO quotes) prices:
  marks  c{b} = close of bin b (ffilled) for b in MARKS, o_off (official open), c_off (official close), c388
  posts  for each post P = (name, first bin b0, mid-proxy bin, hs column) and window W bars (bins b0 .. b0+W-1):
         lo_{P}_{W} = min raw low, hi_{P}_{W} = max raw high (NaN bars ignored; NaN if no trade)
  hs     predicted half-spreads of day t (liq.liq_frame: hs0931 hs0935 hs1001 hs1559)
Panel rows (out/quotes_panel.csv.gz, dates <= DEV_END) additionally get the raw h / l / c / vw of bins 0..61 + 380..390
saved to out/panel_bars.npz for the exact fill study.  No future information is used here: this is a data extraction;
which columns are used for decisions vs fill determination is decided downstream."""
import os, sys, time, numpy as np, pandas as pd
R = "/home/user/llm_sharpe_optimization"; sys.path.insert(0, R)
import rlib, liq
OUT = f"{R}/agents/lmt/out"; os.makedirs(OUT, exist_ok=True)
MARKS = [0, 1, 2, 3, 4, 5, 9, 14, 29, 30, 31, 34, 39, 44, 60, 388, 389]
POSTS = [("0931", 1), ("0932", 2), ("0935", 5), ("1001", 31)]
WINS = [1, 2, 4, 10, 15, 30, 60]
PB = list(range(0, 62)) + list(range(380, 391))

Q = pd.read_csv(f"{R}/out/quotes_panel.csv.gz"); Q = Q[Q.date <= rlib.DEV_END]
PDATES = set(Q.date.unique())

def one(tk):
    try:
        M = rlib.Min(tk, end=rlib.DEV_END)
    except Exception as e:
        return None
    m = (M.dates >= pd.Timestamp(rlib.START)) & (M.dates <= pd.Timestamp(rlib.DEV_END))
    d = M.dates[m]; c = M.c[m]; h = M.h_raw[m]; l = M.l_raw[m]; vw = M.vw_raw[m]
    out = {"date": d, "tk": tk, "half": M.half[m], "o_off": M.open[m], "c_off": M.close[m]}
    for b in MARKS: out[f"c{b}"] = c[:, b]
    with np.errstate(all="ignore"):
        for nm, b0 in POSTS:
            for W in WINS:
                out[f"lo_{nm}_{W}"] = np.nanmin(l[:, b0:b0 + W], 1); out[f"hi_{nm}_{W}"] = np.nanmax(h[:, b0:b0 + W], 1)
    F = pd.DataFrame(out)
    for k in F.columns:
        if F[k].dtype == np.float64: F[k] = F[k].astype(np.float32)
    try:
        L = liq.liq_frame(tk)[["hs0931", "hs0935", "hs1001", "hs1559"]]
    except Exception:
        L = None
    if L is not None:
        F = F.merge(L.reset_index(), on="date", how="left")
    pm = np.array([str(x.date()) in PDATES for x in d])
    P = None
    if pm.any():
        P = dict(tk=tk, dates=np.array([str(x.date()) for x in d[pm]]), h=h[pm][:, PB].astype(np.float32), l=l[pm][:, PB].astype(np.float32),
                 c=c[pm][:, PB].astype(np.float32), vw=vw[pm][:, PB].astype(np.float32), close=M.close[m][pm], open=M.open[m][pm])
    return F, P

if __name__ == "__main__":
    from multiprocessing import Pool
    tks = rlib.universe(2000); t0 = time.time(); Fs = []; Ps = []
    with Pool(4) as pool:
        for i, res in enumerate(pool.imap_unordered(one, tks, chunksize=4)):
            if res is None: continue
            F, P = res; Fs.append(F)
            if P is not None: Ps.append(P)
            if i % 100 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
    F = pd.concat(Fs, ignore_index=True); F.to_parquet(f"{OUT}/ext.parquet")
    keys = ["tk", "dates", "h", "l", "c", "vw", "close", "open"]
    np.savez_compressed(f"{OUT}/panel_bars.npz", tk=np.concatenate([[p["tk"]] * len(p["dates"]) for p in Ps]),
                        **{k: np.concatenate([p[k] for p in Ps]) for k in keys[1:]}, bins=np.array(PB))
    print("EXT DONE", F.shape, f"{time.time()-t0:.0f}s", flush=True)
