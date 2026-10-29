#!/usr/bin/env python3
"""Polygon minute bars -> one compact npz per ticker, CPU-parallel (processes) + network-parallel (threads).
Same output layout as fetch_minute2.py / agents/intraday/convert.py.
  python3 fetch_minute3.py --out data/minute_npz --tickers data/minute_list.txt [--procs 4 --threads 4]"""
import argparse, gzip, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import numpy as np, pandas as pd, orjson
BASE = "https://api.polygon.io"; KEY = os.environ["POLYGON_API_KEY"]; NB = 391

def get(url, retries=8):
    sep = "&" if "?" in url else "?"
    for a in range(retries):
        try:
            req = urllib.request.Request(f"{url}{sep}apiKey={KEY}", headers={"Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
                if (r.headers.get("Content-Encoding") or "") == "gzip": body = gzip.decompress(body)
                return orjson.loads(body)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403): raise
            if e.code == 429: time.sleep(5 * (a + 1)); continue
            if e.code == 404: return {"results": []}
            time.sleep(2 ** a)
        except Exception:
            time.sleep(2 ** a)
    raise RuntimeError(url)

def fetch_year(args):
    t, s, e = args
    url = f"{BASE}/v2/aggs/ticker/{urllib.parse.quote(t)}/range/1/minute/{s}/{e}?adjusted=false&sort=asc&limit=50000"
    cols = {k: [] for k in "tohlcvn"}; cols["vw"] = []
    while url:
        p = get(url); res = p.get("results") or []
        for k in cols:
            kk = k
            cols[k].append(np.array([r.get(kk, np.nan) for r in res], dtype=np.float64))
        url = p.get("next_url")
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}

def to_npz(cols, out):
    t = cols["t"]
    if len(t) == 0:
        np.savez_compressed(out, dates=np.zeros(0, np.int64), nbars=np.zeros(0, np.int64)); return 0
    ts = pd.to_datetime(t.astype(np.int64), unit="ms", utc=True).tz_convert("America/New_York")
    mins = (ts.hour * 60 + ts.minute - (9 * 60 + 30)).values
    ok = (mins >= 0) & (mins < NB)
    mins = mins[ok]; day = (ts.year * 10000 + ts.month * 100 + ts.day).values[ok]
    days, dix = np.unique(day, return_inverse=True); nd = len(days)
    arrs = {k: np.full((nd, NB), np.nan, np.float32) for k in ["o", "h", "l", "c", "vw"]}
    for k in arrs: arrs[k][dix, mins] = cols[k][ok]
    vol = np.zeros((nd, NB), np.float32); vol[dix, mins] = cols["v"][ok]
    n = np.zeros((nd, NB), np.float32); n[dix, mins] = cols["n"][ok]
    nb = np.isfinite(arrs["c"]).sum(1); keep = nb >= 100
    np.savez_compressed(out, dates=days[keep], nbars=nb[keep], vol=vol[keep], n=n[keep], **{k: v[keep] for k, v in arrs.items()})
    return int(keep.sum())

def one(args):
    t, start, end, outdir, threads = args
    out = f"{outdir}/{t}.npz"
    if os.path.exists(out): return t, -1, 0.0
    t0 = time.time()
    try:
        y0, y1 = int(start[:4]), int(end[:4])
        jobs = [(t, max(start, f"{y}-01-01"), min(end, f"{y}-12-31")) for y in range(y0, y1 + 1)]
        with ThreadPoolExecutor(threads) as ex: parts = list(ex.map(fetch_year, jobs))
        cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        nd = to_npz(cols, out + ".tmp.npz"); os.replace(out + ".tmp.npz", out)
        return t, nd, time.time() - t0
    except Exception as e:
        return t, f"ERR {e}", time.time() - t0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tickers", required=True)
    ap.add_argument("--start", default="2016-06-01"); ap.add_argument("--end", default="2026-10-28")
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tks = [l.strip() for l in open(a.tickers) if l.strip() and not l.startswith("#")]
    todo = [t for t in tks if not os.path.exists(f"{a.out}/{t}.npz")]
    print(f"{len(tks)} tickers, {len(todo)} to fetch", flush=True)
    done = 0; t00 = time.time()
    with Pool(a.procs) as pool:
        for t, nd, secs in pool.imap_unordered(one, [(t, a.start, a.end, a.out, a.threads) for t in todo]):
            done += 1; print(f"{t} days={nd} {secs:.0f}s  [{done}/{len(todo)} {(time.time()-t00)/60:.1f}min]", flush=True)
    print("MINUTE DONE", flush=True)
