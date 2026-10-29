#!/usr/bin/env python3
"""Rank CS/ADRC tickers by median daily dollar volume over 2016-10-28..END using the
grouped-daily files present in data/grouped (sampled every --step files), requiring
presence on >= 80% of the sampled days. Writes one ticker per line, best first.
  python3 universe.py --n 1200 --out data/univ1200.txt [--step 5]"""
import argparse, glob, gzip, json, os, re, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1200); ap.add_argument("--out", required=True)
ap.add_argument("--step", type=int, default=5); ap.add_argument("--start", default="2016-10-28"); ap.add_argument("--end", default="2026-10-28")
ap.add_argument("--exclude", default="", help="file of tickers to skip")
a = ap.parse_args()
files = sorted(f for f in glob.glob("data/grouped/????-??-??.json.gz") if a.start <= os.path.basename(f)[:10] <= a.end)[::a.step]
typ = {r["ticker"]: r.get("type") for r in json.load(open("data/tickers.json")) if r.get("ticker")}
SYM = re.compile(r"^[A-Z]{1,5}$")
cols = {}
for f in files:
    d = os.path.basename(f)[:10]
    for r in json.load(gzip.open(f, "rt")):
        t = r.get("T")
        if t and SYM.match(t) and r.get("c") and r.get("v"): cols.setdefault(t, {})[d] = r["c"] * r["v"]
dv = pd.DataFrame(cols)
ok = [t for t in dv.columns if typ.get(t) in ("CS", "ADRC") and dv[t].notna().mean() >= 0.8 and not re.match(r"^Z[A-Z]ZZT$", t)]
med = dv[ok].median().sort_values(ascending=False)
ex = set(l.strip() for l in open(a.exclude)) if a.exclude else set()
top = [t for t in med.index if t not in ex][: a.n]
open(a.out, "w").write("\n".join(top) + "\n")
print(len(files), "sample days;", len(ok), "eligible;", "wrote", len(top), "to", a.out, "| last median $vol %.1fM" % (med[top[-1]] / 1e6))
