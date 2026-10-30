#!/usr/bin/env python3
"""Unbiased NBBO panel for the spread model: every --step-th trading day, every name of rlib.universe(2000) and the
VIX ETPs, the last quote at or before 09:31:00, 09:35:00, 10:01:00 and 15:59:30 (10 s window, widened to 60 s / 5 min).
Writes out/quotes_panel.csv (tk, date, t, bid, ask, age_s) incrementally.
  POLYGON_API_KEY=... python3 quotes_panel.py --step 40 --threads 12"""
import argparse, os, sys, time, json, gzip, urllib.request, datetime as dt, csv
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/llm_sharpe_optimization"); os.environ.setdefault("STRAT_END", "2026-10-28")
import rlib
KEY = os.environ["POLYGON_API_KEY"]; NY = ZoneInfo("America/New_York"); BASE = "https://api.polygon.io"
ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, default=40); ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--start", default="2017-01-03"); ap.add_argument("--out", default="out/quotes_panel.csv"); a = ap.parse_args()
TIMES = [("0931", 9, 31, 0), ("0935", 9, 35, 0), ("1001", 10, 1, 0), ("1559", 15, 59, 30)]
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
def ns(date, hh, mm, ss): return int(dt.datetime(date.year, date.month, date.day, hh, mm, ss, tzinfo=NY).timestamp() * 1e9)
def quote_at(tk, t_ns):
    for w in (10, 60, 300):
        d = get(f"{BASE}/v3/quotes/{tk}?timestamp.gte={t_ns - w * 10**9}&timestamp.lte={t_ns}&order=desc&sort=timestamp&limit=1&apiKey={KEY}")
        res = d.get("results") or []
        if res:
            q = res[0]; b, s = q.get("bid_price"), q.get("ask_price")
            if b and s and s >= b > 0: return b, s, (t_ns - q["sip_timestamp"]) / 1e9
    return None
spy = rlib.Min("SPY", end="2026-10-28"); dates = pd.DatetimeIndex(spy.dates); dates = dates[(dates >= a.start) & (~spy.half)][::a.step]
tks = rlib.universe(2000) + ["VIXY", "UVXY", "VIXM", "SPY", "QQQ"]
have = set()
if os.path.exists(a.out):
    old = pd.read_csv(a.out); have = set(zip(old.tk, old.date, old.t))
jobs = [(tk, d, t) for d in dates for tk in tks for t in TIMES if (tk, str(d.date()), t[0]) not in have]
print(len(dates), "days", len(tks), "names", len(jobs), "lookups", flush=True)
f = open(a.out, "a", newline=""); w = csv.writer(f)
if not have: w.writerow(["tk", "date", "t", "bid", "ask", "age_s"])
t0 = time.time(); n = 0
def one(j):
    tk, d, t = j; q = quote_at(tk, ns(d, t[1], t[2], t[3])); return (tk, str(d.date()), t[0]) + (q if q else (np.nan, np.nan, np.nan))
with ThreadPoolExecutor(a.threads) as ex:
    for row in ex.map(one, jobs, chunksize=50):
        w.writerow(row); n += 1
        if n % 5000 == 0: f.flush(); print(f"  {n}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
f.close(); print("PANEL DONE", flush=True)
