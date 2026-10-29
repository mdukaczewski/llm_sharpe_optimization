#!/usr/bin/env python3
"""Convert Polygon minute JSON (per ticker-year) into one compact npz per ticker:
dates (YYYYMMDD int), and arrays [n_days, 391] for o,h,l,c,v,vw over bins
09:30..16:00 America/New_York (bin k = 09:30 + k minutes; bin 390 = 16:00 bar,
which holds the closing auction print). Missing minutes are NaN (vol 0)."""
import gzip, json, os, sys, glob
import numpy as np, pandas as pd

SRC = "/home/trumpet/strat20/data/minute"
DST = "/home/trumpet/strat20/data/minute_npz"
os.makedirs(DST, exist_ok=True)
NB = 391

def convert(tk):
    out = f"{DST}/{tk}.npz"
    files = sorted(glob.glob(f"{SRC}/{tk}_*.json.gz"))
    if os.path.exists(out) and os.path.getmtime(out) > max(os.path.getmtime(f) for f in files):
        print(tk, "up to date"); return
    rows = []
    for f in files:
        d = json.load(gzip.open(f))
        if not d: continue
        df = pd.DataFrame(d)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    ts = pd.to_datetime(df["t"].values, unit="ms", utc=True).tz_convert("America/New_York")
    mins = ts.hour * 60 + ts.minute - (9 * 60 + 30)
    ok = (mins >= 0) & (mins < NB)
    df = df[ok]; ts = ts[ok]; mins = mins[ok]
    day = ts.year * 10000 + ts.month * 100 + ts.day
    days, dix = np.unique(day, return_inverse=True)
    nd = len(days)
    arrs = {k: np.full((nd, NB), np.nan, np.float32) for k in ["o", "h", "l", "c", "vw"]}
    vol = np.zeros((nd, NB), np.float32)
    n = np.zeros((nd, NB), np.float32)
    for k in arrs:
        arrs[k][dix, mins] = df[k].values
    vol[dix, mins] = df["v"].values
    n[dix, mins] = df["n"].values
    # drop days with very few bars (holidays/half days kept if >= 200 bars)
    nb = np.isfinite(arrs["c"]).sum(1)
    keep = nb >= 100
    np.savez_compressed(out, dates=days[keep], nbars=nb[keep], vol=vol[keep], n=n[keep],
                        **{k: v[keep] for k, v in arrs.items()})
    print(tk, nd, "days, kept", keep.sum(), "bars/day median", np.median(nb[keep]), flush=True)

if __name__ == "__main__":
    tks = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(set(os.path.basename(f).split("_")[0] for f in glob.glob(f"{SRC}/*.json.gz")))
    for tk in tks:
        convert(tk)
