#!/usr/bin/env python3
"""Memory-lean version of build_panel.py (same output arrays): per-date compact arrays, orjson, small pool.
  python3 build_panel2.py --grouped data/grouped --ref data --out data/panel.npz"""
import argparse, gzip, os, re, glob, json
import numpy as np, orjson
from multiprocessing import Pool
SYM_RE = re.compile(r"^[A-Z]{1,5}$")
FIELDS = ["o", "h", "l", "c", "vw", "v", "n"]
def read_one(path):
    with gzip.open(path, "rb") as fh: d = orjson.loads(fh.read())
    rows = d.get("results", d) if isinstance(d, dict) else d
    tk = []; vals = []
    for r in rows:
        t = r.get("T")
        if not t or not SYM_RE.match(t): continue
        tk.append(t); vals.append([r.get(f) if r.get(f) is not None else np.nan for f in FIELDS])
    return os.path.basename(path)[:10], np.array(tk), np.array(vals, dtype=np.float32).reshape(-1, len(FIELDS))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grouped", nargs="+", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--start", default="2014-06-01"); ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    files = {}
    for d in a.grouped:
        for p in glob.glob(os.path.join(d, "????-??-??.json.gz")):
            date = os.path.basename(p)[:10]
            if date >= a.start: files[date] = p
    dates = sorted(files); print(f"{len(dates)} files {dates[0]}..{dates[-1]}", flush=True)
    with Pool(a.procs) as pool: parsed = pool.map(read_one, [files[d] for d in dates], chunksize=16)
    parsed = [x for x in parsed if len(x[1]) > 500]; dates = [x[0] for x in parsed]
    tickers = np.array(sorted(set(np.concatenate([x[1] for x in parsed]).tolist()))); tix = {t: i for i, t in enumerate(tickers)}
    T, N = len(dates), len(tickers); print(f"{T} days x {N} tickers", flush=True)
    names = ["open", "high", "low", "close", "vw", "vol", "n"]
    arr = {f: np.full((T, N), np.nan, dtype=np.float32) for f in names}
    for i, (d, tk, v) in enumerate(parsed):
        idx = np.fromiter((tix[t] for t in tk), dtype=np.int64, count=len(tk))
        for k, f in enumerate(names): arr[f][i, idx] = v[:, k]
    del parsed
    adj = np.ones((T, N), dtype=np.float64); date_arr = np.array(dates); dix = {d: i for i, d in enumerate(dates)}
    n_applied = 0
    for s in json.load(open(os.path.join(a.ref, "splits.json"))):
        t = s.get("ticker"); ed = s.get("execution_date")
        if t not in tix or not ed or ed < dates[0] or ed > dates[-1]: continue
        sf, st = s.get("split_from"), s.get("split_to")
        if not sf or not st or sf <= 0 or st <= 0: continue
        i = int(np.searchsorted(date_arr, ed)); adj[:i, tix[t]] *= sf / st; n_applied += 1
    print(f"splits applied: {n_applied}", flush=True)
    div = np.zeros((T, N), dtype=np.float32); dpath = os.path.join(a.ref, "dividends.json")
    if os.path.exists(dpath):
        nd = 0
        for r in json.load(open(dpath)):
            t = r.get("ticker"); ed = r.get("ex_dividend_date"); c = r.get("cash_amount")
            if t in tix and ed in dix and c and r.get("currency", "USD") in ("USD", None): div[dix[ed], tix[t]] += float(c); nd += 1
        print(f"dividends applied: {nd}", flush=True)
    ttype = np.array([""] * N, dtype=object)
    for r in json.load(open(os.path.join(a.ref, "tickers.json"))):
        t = r.get("ticker")
        if t in tix and r.get("type"): ttype[tix[t]] = r["type"]
    print("types:", {k: int(v) for k, v in zip(*np.unique(ttype.astype(str), return_counts=True))}, flush=True)
    np.savez(a.out, dates=np.array(dates), tickers=tickers, ttype=ttype.astype(str), adj=adj.astype(np.float32), div=div, **arr)
    print("saved", a.out, flush=True)
if __name__ == "__main__": main()
