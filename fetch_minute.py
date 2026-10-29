#!/usr/bin/env python3
"""Download Polygon minute bars for a few tickers into DATA/minute/{T}_{YEAR}.json.gz"""
import gzip, json, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
BASE = "https://api.polygon.io"; KEY = os.environ["POLYGON_API_KEY"]
def get(url, retries=6):
    sep = "&" if "?" in url else "?"
    for a in range(retries):
        try:
            req = urllib.request.Request(f"{url}{sep}apiKey={KEY}", headers={"Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
                if (r.headers.get("Content-Encoding") or "") == "gzip": body = gzip.decompress(body)
                return json.loads(body)
        except Exception as e:
            time.sleep(2 ** a)
    raise RuntimeError(url)
def fetch(t, year, out):
    path = f"{out}/{t}_{year}.json.gz"
    if os.path.exists(path): return t, year, -1
    rows = []
    url = f"{BASE}/v2/aggs/ticker/{t}/range/1/minute/{year}-01-01/{year}-12-31?adjusted=false&sort=asc&limit=50000"
    while url:
        p = get(url); rows.extend(p.get("results") or []); url = p.get("next_url")
    with gzip.open(path, "wt") as fh: json.dump(rows, fh)
    return t, year, len(rows)
if __name__ == "__main__":
    out = sys.argv[1]; tickers = sys.argv[2].split(","); y0, y1 = int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(out, exist_ok=True)
    jobs = [(t, y) for t in tickers for y in range(y0, y1 + 1)]
    with ThreadPoolExecutor(10) as ex:
        for t, y, n in ex.map(lambda j: fetch(j[0], j[1], out), jobs):
            print(t, y, n, flush=True)
