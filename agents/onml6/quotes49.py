#!/usr/bin/env python3
"""last NBBO at or before 15:49:00 NY (10 s window, widened to 60 s / 5 min) for (tk, date) jobs, quotes_panel.py request pattern.
usage: POLYGON_API_KEY=... python3 quotes49.py jobs.csv out.csv [threads]   (jobs.csv: tk,date; already-fetched rows skipped)"""
import os, sys, time, json, gzip, urllib.request, datetime as dt, csv
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, numpy as np
KEY = os.environ["POLYGON_API_KEY"]; NY = ZoneInfo("America/New_York"); BASE = "https://api.polygon.io"
def get(url, retries=6):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                if (r.headers.get("Content-Encoding") or "") == "gzip": body = gzip.decompress(body)
                return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(2 * (i + 1)); continue
            if e.code in (401, 403): raise
            time.sleep(1 + i)
        except Exception: time.sleep(1 + i)
    return {}
def ns(d, hh, mm, ss): return int(dt.datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=NY).timestamp() * 1e9)
def quote_at(tk, t_ns):
    for w in (10, 60, 300):
        d = get(f"{BASE}/v3/quotes/{tk}?timestamp.gte={t_ns - w * 10**9}&timestamp.lte={t_ns}&order=desc&sort=timestamp&limit=1&apiKey={KEY}")
        res = d.get("results") or []
        if res:
            q = res[0]; b, s = q.get("bid_price"), q.get("ask_price")
            if b and s and s >= b > 0: return b, s, (t_ns - q["sip_timestamp"]) / 1e9
    return None
J = pd.read_csv(sys.argv[1]); out = sys.argv[2]; th = int(sys.argv[3]) if len(sys.argv) > 3 else 8
have = set()
if os.path.exists(out): o = pd.read_csv(out); have = set(zip(o.tk, o.date))
jobs = [(r.tk, r.date) for r in J.itertuples() if (r.tk, r.date) not in have]
print(len(jobs), "lookups", flush=True)
f = open(out, "a", newline=""); w = csv.writer(f)
if not have: w.writerow(["tk", "date", "bid", "ask", "age_s"])
def one(j):
    tk, d = j; q = quote_at(tk, ns(pd.Timestamp(d), 15, 49, 0)); return (tk, d) + (q if q else (np.nan, np.nan, np.nan))
t0 = time.time(); n = 0
with ThreadPoolExecutor(th) as ex:
    for row in ex.map(one, jobs):
        w.writerow(row); n += 1
        if n % 500 == 0: f.flush(); print(f"  {n}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
f.close(); print("DONE", flush=True)
