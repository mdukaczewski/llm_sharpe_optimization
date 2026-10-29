#!/usr/bin/env python3
"""Download Polygon reference data + extra grouped-daily history into DATA dir.

  python fetch_ref.py --data ~/strat20/data --what splits,dividends,tickers,grouped
"""
import argparse, gzip, json, os, sys, time, urllib.request, urllib.parse, datetime as dt
from concurrent.futures import ThreadPoolExecutor

BASE = "https://api.polygon.io"
KEY = os.environ["POLYGON_API_KEY"]

def get(url, retries=6):
    sep = "&" if "?" in url else "?"
    full = f"{url}{sep}apiKey={KEY}"
    for a in range(retries):
        try:
            req = urllib.request.Request(full, headers={"Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                if (r.headers.get("Content-Encoding") or "") == "gzip":
                    body = gzip.decompress(body)
                return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise
            time.sleep(2 ** a)
        except Exception:
            time.sleep(2 ** a)
    raise RuntimeError(f"failed: {url}")

def paginate(endpoint, **params):
    rows = []
    url = f"{BASE}{endpoint}?{urllib.parse.urlencode(params)}"
    n = 0
    while url:
        p = get(url)
        rows.extend(p.get("results") or [])
        url = p.get("next_url")
        n += 1
        if n % 20 == 0:
            print(f"  {endpoint}: {n} pages, {len(rows)} rows", flush=True)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--what", default="splits,dividends,tickers,grouped")
    ap.add_argument("--gstart", default="2014-06-01")
    ap.add_argument("--gend", default="2016-09-18")
    a = ap.parse_args()
    os.makedirs(a.data, exist_ok=True)
    what = a.what.split(",")
    if "splits" in what:
        rows = paginate("/v3/reference/splits", limit=1000, order="asc", sort="execution_date")
        json.dump(rows, open(f"{a.data}/splits.json", "w"))
        print("splits", len(rows), flush=True)
    if "tickers" in what:
        allrows = []
        for active in ("true", "false"):
            rows = paginate("/v3/reference/tickers", market="stocks", active=active, limit=1000)
            for r in rows:
                r["_active"] = active == "true"
            allrows.extend(rows)
            print("tickers active=", active, len(rows), flush=True)
        json.dump(allrows, open(f"{a.data}/tickers.json", "w"))
    if "grouped" in what:
        gd = f"{a.data}/grouped"
        os.makedirs(gd, exist_ok=True)
        d0 = dt.date.fromisoformat(a.gstart); d1 = dt.date.fromisoformat(a.gend)
        dates = []
        d = d0
        while d <= d1:
            if d.weekday() < 5:
                dates.append(d.isoformat())
            d += dt.timedelta(days=1)
        todo = [d for d in dates if not os.path.exists(f"{gd}/{d}.json.gz")]
        print(f"grouped: {len(todo)} to fetch", flush=True)
        def one(d):
            p = get(f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{d}?adjusted=false")
            res = p.get("results") or []
            with gzip.open(f"{gd}/{d}.json.gz", "wt") as fh:
                json.dump(res, fh)
            return d, len(res)
        with ThreadPoolExecutor(8) as ex:
            for i, (d, n) in enumerate(ex.map(one, todo)):
                if i % 50 == 0:
                    print(f"  {d}: {n} rows", flush=True)
                time.sleep(0.1)
        print("grouped done", flush=True)
    if "dividends" in what:
        # month-sharded, parallel: deep pagination of the global list is slow
        months = []
        y, m = 2013, 1
        while (y, m) <= (2027, 12):
            months.append((y, m)); m += 1
            if m == 13: y, m = y + 1, 1
        def one_month(ym):
            y, m = ym
            lo = f"{y:04d}-{m:02d}-01"
            y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
            hi = f"{y2:04d}-{m2:02d}-01"
            return paginate("/v3/reference/dividends", limit=1000, **{"ex_dividend_date.gte": lo, "ex_dividend_date.lt": hi})
        rows = []
        with ThreadPoolExecutor(8) as ex:
            for ym, rr in zip(months, ex.map(one_month, months)):
                rows.extend(rr)
                if ym[1] == 1:
                    print(f"  dividends through {ym[0]}: {len(rows)}", flush=True)
        json.dump(rows, open(f"{a.data}/dividends.json", "w"))
        print("dividends", len(rows), flush=True)

if __name__ == "__main__":
    main()
