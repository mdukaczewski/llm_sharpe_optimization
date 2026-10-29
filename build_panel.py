#!/usr/bin/env python3
"""Build a dates x tickers panel from Polygon grouped-daily files.

  python build_panel.py --grouped DIR1 DIR2 ... --ref DATA --out DATA/panel.npz

Output arrays (float32, NaN where no bar): open, high, low, close, vw, vol, n.
Also: adj (cumulative split factor so that close*adj is split-adjusted to the
LAST date), div (cash dividend per share on the ex-date, unadjusted), tickers,
dates (YYYY-MM-DD strings), and ttype (ticker type string, '' if unknown).
"""
import argparse, gzip, json, os, re, sys, glob
import numpy as np
from multiprocessing import Pool

SYM_RE = re.compile(r"^[A-Z]{1,5}$")

def read_one(path):
    with gzip.open(path, "rt") as fh:
        d = json.load(fh)
    rows = d.get("results", d) if isinstance(d, dict) else d
    date = os.path.basename(path)[:10]
    out = []
    for r in rows:
        t = r.get("T")
        if not t or not SYM_RE.match(t):
            continue
        out.append((t, r.get("o"), r.get("h"), r.get("l"), r.get("c"), r.get("vw"), r.get("v"), r.get("n")))
    return date, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouped", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2014-06-01")
    a = ap.parse_args()

    files = {}
    for d in a.grouped:           # later dirs win on duplicate dates
        for p in glob.glob(os.path.join(d, "????-??-??.json.gz")):
            date = os.path.basename(p)[:10]
            if date >= a.start:
                files[date] = p
    dates = sorted(files)
    print(f"{len(dates)} files {dates[0]}..{dates[-1]}", flush=True)
    with Pool(14) as pool:
        parsed = pool.map(read_one, [files[d] for d in dates], chunksize=8)
    # drop empty days (holidays fetched as empty)
    parsed = [(d, rows) for d, rows in parsed if len(rows) > 500]
    dates = [d for d, _ in parsed]
    tickers = sorted({r[0] for _, rows in parsed for r in rows})
    tix = {t: i for i, t in enumerate(tickers)}
    T, N = len(dates), len(tickers)
    print(f"{T} days x {N} tickers", flush=True)
    fields = ["open", "high", "low", "close", "vw", "vol", "n"]
    arr = {f: np.full((T, N), np.nan, dtype=np.float32) for f in fields}
    for i, (d, rows) in enumerate(parsed):
        idx = np.fromiter((tix[r[0]] for r in rows), dtype=np.int64, count=len(rows))
        for k, f in enumerate(fields):
            vals = np.array([r[k + 1] if r[k + 1] is not None else np.nan for r in rows], dtype=np.float32)
            arr[f][i, idx] = vals
    del parsed

    # splits -> cumulative adjustment factor to the last date.
    # Polygon: split_from -> split_to on execution_date. Prices BEFORE the
    # execution date are multiplied by split_from/split_to.
    adj = np.ones((T, N), dtype=np.float64)
    dix = {d: i for i, d in enumerate(dates)}
    date_arr = np.array(dates)
    splits = json.load(open(os.path.join(a.ref, "splits.json")))
    n_applied = 0
    for s in splits:
        t = s.get("ticker"); ed = s.get("execution_date")
        if t not in tix or not ed or ed < dates[0] or ed > dates[-1]:
            continue
        sf, st = s.get("split_from"), s.get("split_to")
        if not sf or not st or sf <= 0 or st <= 0:
            continue
        i = int(np.searchsorted(date_arr, ed))    # first date >= ed
        adj[:i, tix[t]] *= sf / st
        n_applied += 1
    print(f"splits applied: {n_applied}", flush=True)

    div = np.zeros((T, N), dtype=np.float32)
    dpath = os.path.join(a.ref, "dividends.json")
    if os.path.exists(dpath):
        divs = json.load(open(dpath))
        nd = 0
        for r in divs:
            t = r.get("ticker"); ed = r.get("ex_dividend_date"); c = r.get("cash_amount")
            if t in tix and ed in dix and c and r.get("currency", "USD") in ("USD", None):
                div[dix[ed], tix[t]] += float(c); nd += 1
        print(f"dividends applied: {nd}", flush=True)

    ttype = np.array([""] * N, dtype=object)
    tpath = os.path.join(a.ref, "tickers.json")
    if os.path.exists(tpath):
        for r in json.load(open(tpath)):
            t = r.get("ticker")
            if t in tix and r.get("type"):
                ttype[tix[t]] = r["type"]
        print("types:", {k: int(v) for k, v in zip(*np.unique(ttype.astype(str), return_counts=True))}, flush=True)

    np.savez(a.out, dates=np.array(dates), tickers=np.array(tickers), ttype=ttype.astype(str),
             adj=adj.astype(np.float32), div=div, **arr)
    print("saved", a.out, flush=True)

if __name__ == "__main__":
    main()
