#!/usr/bin/env python3
"""FMP per-symbol earnings history -> DATA/earnings.json  (python fetch_earnings.py tickers.txt out.json)"""
import json, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
KEY = os.environ["FMP_API_KEY"]
def one(t):
    url = f"https://financialmodelingprep.com/stable/earnings?symbol={urllib.parse.quote(t)}&apikey={KEY}"
    for a in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read()
            d = json.loads(body)
            if isinstance(d, list):
                return t, d
            return t, {"error": str(d)[:200]}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(10 * (a + 1)); continue
            return t, {"error": f"HTTP {e.code}"}
        except Exception as e:
            time.sleep(2 ** a)
    return t, {"error": "failed"}
tickers = [l.strip() for l in open(sys.argv[1]) if l.strip()]
out = {}
if os.path.exists(sys.argv[2]):
    out = json.load(open(sys.argv[2]))
todo = [t for t in tickers if t not in out]
print(len(tickers), "tickers,", len(todo), "to fetch", flush=True)
with ThreadPoolExecutor(4) as ex:
    for k, (t, d) in enumerate(ex.map(one, todo)):
        out[t] = d
        if k % 100 == 0:
            print(k, t, (len(d) if isinstance(d, list) else d), flush=True)
            json.dump(out, open(sys.argv[2], "w"))
json.dump(out, open(sys.argv[2], "w"))
errs = sum(1 for v in out.values() if isinstance(v, dict))
print("done", len(out), "errors", errs, flush=True)
