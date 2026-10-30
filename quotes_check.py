#!/usr/bin/env python3
"""Realistic-fill check with Polygon NBBO quotes.  On every --step-th trading day, take the positions the three new
books actually hold and look up the last quote at or before each decision time (window 10 s, widened to 60 s, 5 min,
15 min if empty).  Buys fill at the ask, sells at the bid.  Compare with the backtest's bar-based fills:
  overnight book : enter at the official close (MOC) [also: at the 15:59:30 NBBO, the continuous-market alternative],
                   exit at the 09:31 bar close  -> quote at 09:31:00 next trading day
  intraday book  : enter at the 09:33 bar close -> quote at 09:33:00; exit at the 10:01 bar close -> quote at 10:01:00
Outputs out/quotes_trades.csv (one row per trade) and a summary by book / year / liquidity tier.
  POLYGON_API_KEY=... python3 quotes_check.py --step 10 --threads 8"""
import argparse, os, sys, time, json, gzip, urllib.request, datetime as dt
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/llm_sharpe_optimization"); sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/xsrule")
os.environ.setdefault("STRAT_END", "2026-10-28")
import rlib
from rlib import Min
KEY = os.environ["POLYGON_API_KEY"]; NY = ZoneInfo("America/New_York"); BASE = "https://api.polygon.io"
ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, default=10); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--start", default="2017-01-03")
a = ap.parse_args()

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

def ns(date, hh, mm, ss):
    return int(dt.datetime(date.year, date.month, date.day, hh, mm, ss, tzinfo=NY).timestamp() * 1e9)

def quote_at(tk, t_ns):
    """last NBBO at or before t_ns; returns (bid, ask, age_s, window_s) or None"""
    for w in (10, 60, 300, 900):
        d = get(f"{BASE}/v3/quotes/{tk}?timestamp.gte={t_ns - w * 10**9}&timestamp.lte={t_ns}&order=desc&sort=timestamp&limit=1&apiKey={KEY}")
        res = d.get("results") or []
        if res:
            q = res[0]; b, s = q.get("bid_price"), q.get("ask_price")
            if b and s and s >= b > 0: return b, s, (t_ns - q["sip_timestamp"]) / 1e9, w
    return None

# ---------------------------------------------------------------- positions on the sampled days
dates_all = pd.DatetimeIndex(sorted(pd.read_parquet("agents/onml/out/pred_f26calx_u2000_q3.parquet", columns=["date"]).date.unique()))
dates = dates_all[dates_all >= a.start][::a.step]; nxt = {d: dates_all[i + 1] for i, d in enumerate(dates_all[:-1])}
print(len(dates), "sampled days", dates[0].date(), "..", dates[-1].date(), flush=True)
trades = []   # book, date, tk, side(+1 long/-1 short), entry_time, exit_time, entry_day, exit_day
# overnight 10-name book (2000 names, strict features)
P = pd.read_parquet("agents/onml/out/pred_f26calx_u2000_q3.parquet"); P = P[P.date.isin(dates)]
for d, g in P.groupby("date"):
    if d not in nxt: continue
    o = g.sort_values("p"); 
    for tk in o.tk.iloc[-10:]: trades.append(("onml_k10", d, tk, 1, "15:59:30", "09:31:00", d, nxt[d]))
    for tk in o.tk.iloc[:10]: trades.append(("onml_k10", d, tk, -1, "15:59:30", "09:31:00", d, nxt[d]))
# intraday ensemble 8-name book (1500 names): avg per-day pct ranks of the two models
A = pd.read_parquet("agents/dayml/out/pred_y_60d_Q.parquet"); B = pd.read_parquet("agents/dayml/out/pred_y_120d_Q.parquet")
A = A[A.date.isin(dates)]; B = B[B.date.isin(dates)]
M = A.merge(B, on=["date", "tk"], suffixes=("_a", "_b")); M["r"] = (M.groupby("date").p_a.rank(pct=True) + M.groupby("date").p_b.rank(pct=True)) / 2
for d, g in M.groupby("date"):
    o = g.sort_values("r")
    for tk in o.tk.iloc[-8:]: trades.append(("dayml_e30_k8", d, tk, 1, "09:33:00", "10:01:00", d, d))
    for tk in o.tk.iloc[:8]: trades.append(("dayml_e30_k8", d, tk, -1, "09:33:00", "10:01:00", d, d))
# rule book: long top 10 of 150 names with cAx > 1.5
from xlib import add_extras, add_extra2
tks150 = rlib.universe(150); D = rlib.add_market(rlib.minute_frame(tks150, verbose=False)); D = add_extras(D); D = add_extra2(D, verbose=False)
D = rlib.clean(D, ["z30", "z60", "z5l", "zmoc", "zon_prev"]); D["s"] = 0.5 * D.z30 - D.z60 - 0.5 * D.z5l + 0.3 * D.zon_prev - 1.0 * D.r_vlast_x
D = D[D.date.isin(dates) & (~D.half)]
for d, g in D.groupby("date"):
    if d not in nxt: continue
    o = g.sort_values("s"); top = o.iloc[-10:]; top = top[top.s > 1.5]
    for tk in top.tk: trades.append(("xsr_lo10", d, tk, 1, "15:59:30", "09:31:00", d, nxt[d]))
T = pd.DataFrame(trades, columns=["book", "date", "tk", "side", "t_in", "t_out", "d_in", "d_out"])
print(T.groupby("book").size().to_dict(), "trades;", T.tk.nunique(), "names", flush=True)

# ---------------------------------------------------------------- bar-based prices used by the backtest
bars = {}
for tk in T.tk.unique():
    try:
        m = Min(tk, end="2026-10-28"); bars[tk] = pd.DataFrame({"close": m.close, "c1": m.c[:, 1], "c3": m.c[:, 3], "c31": m.c[:, 31]}, index=m.dates)
    except Exception as e: print(tk, "no bars", e)
def bar(tk, d, col):
    b = bars.get(tk); 
    return float(b.loc[d, col]) if b is not None and d in b.index else np.nan
T["bar_in"] = [bar(tk, d, "close" if t == "15:59:30" else "c3") for tk, d, t in zip(T.tk, T.d_in, T.t_in)]
T["bar_out"] = [bar(tk, d, "c1" if t == "09:31:00" else "c31") for tk, d, t in zip(T.tk, T.d_out, T.t_out)]
T["rank"] = T.tk.map(rlib.RANK).fillna(9999).astype(int)
T["spread_assumed_bps"] = T.tk.map(rlib.cost_bps)

# ---------------------------------------------------------------- quotes
jobs = []
for i, r in T.iterrows():
    h, m_, s = map(int, r.t_in.split(":")); jobs.append((i, "in", r.tk, ns(r.d_in, h, m_, s)))
    h, m_, s = map(int, r.t_out.split(":")); jobs.append((i, "out", r.tk, ns(r.d_out, h, m_, s)))
print(len(jobs), "quote lookups", flush=True); t0 = time.time(); out = {}
def one(j):
    i, leg, tk, t = j; q = quote_at(tk, t); return i, leg, q
with ThreadPoolExecutor(a.threads) as ex:
    for n, (i, leg, q) in enumerate(ex.map(one, jobs)):
        out[(i, leg)] = q
        if n % 2000 == 0: print(f"  {n}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
for leg in ["in", "out"]:
    T[f"bid_{leg}"] = [out.get((i, leg), (np.nan,) * 4)[0] if out.get((i, leg)) else np.nan for i in T.index]
    T[f"ask_{leg}"] = [out.get((i, leg), (np.nan,) * 4)[1] if out.get((i, leg)) else np.nan for i in T.index]
    T[f"age_{leg}"] = [out.get((i, leg), (np.nan,) * 4)[2] if out.get((i, leg)) else np.nan for i in T.index]
# fills: long = buy at ask then sell at bid; short = sell at bid then buy at ask
T["fill_in"] = np.where(T.side > 0, T.ask_in, T.bid_in); T["fill_out"] = np.where(T.side > 0, T.bid_out, T.ask_out)
T["mid_in"] = (T.bid_in + T.ask_in) / 2; T["mid_out"] = (T.bid_out + T.ask_out) / 2
T["halfspread_in_bps"] = (T.ask_in - T.bid_in) / 2 / T.mid_in * 1e4; T["halfspread_out_bps"] = (T.ask_out - T.bid_out) / 2 / T.mid_out * 1e4
# returns per unit position: backtest (bar fills, assumed spread) vs quotes; for overnight books also MOC entry (official close) + quote exit
T["ret_bar_gross"] = T.side * (T.bar_out / T.bar_in - 1)
T["ret_bar_net"] = T.ret_bar_gross - 2 * T.spread_assumed_bps / 1e4
T["ret_quote"] = T.side * (T.fill_out / T.fill_in - 1)                         # continuous-market entry + exit at NBBO
T["ret_moc_quote"] = T.side * (T.fill_out / T.bar_in - 1)                       # MOC entry at the official close, NBBO exit
T.to_csv("out/quotes_trades.csv", index=False)
T["year"] = T.date.dt.year; T["tier"] = pd.cut(T["rank"], [0, 300, 600, 1500, 2000, 99999], labels=["1-300", "301-600", "601-1500", "1501-2000", "other"])
pd.set_option("display.width", 220)
def summ(g):
    return pd.Series({"n": len(g), "quoted%": 100 * g.fill_out.notna().mean(), "half_in": g.halfspread_in_bps.median(), "half_out": g.halfspread_out_bps.median(),
                      "age_out_s": g.age_out.median(), "bar_net_bps": g.ret_bar_net.mean() * 1e4, "quote_bps": g.ret_quote.mean() * 1e4, "moc_quote_bps": g.ret_moc_quote.mean() * 1e4,
                      "gap_vs_bar_bps": (g.ret_quote - g.ret_bar_net).mean() * 1e4, "gap_moc_bps": (g.ret_moc_quote - g.ret_bar_net).mean() * 1e4})
print("\n=== by book"); print(T.groupby("book").apply(summ).round(1).to_string())
print("\n=== by book x tier"); print(T.groupby(["book", "tier"], observed=True).apply(summ).round(1).to_string())
print("\n=== by book x year"); print(T.groupby(["book", "year"]).apply(summ).round(1).to_string())
# daily book returns on the sampled days: bar-based net vs quote-based, equal weight per side
for bk, g in T.groupby("book"):
    gg = g.dropna(subset=["ret_quote", "ret_bar_net"]); k = gg.groupby("date").size().median()
    dly = gg.groupby("date").agg(bar=("ret_bar_net", "mean"), q=("ret_quote", "mean"), mq=("ret_moc_quote", "mean"))
    gross = 2 if (gg.side < 0).any() else 1
    print(f"\n{bk}: {len(dly)} days, mean positions/day {k:.0f}: per-day book return (x gross {gross}) bar-net {dly.bar.mean()*gross*1e4:.1f} bps, quote {dly.q.mean()*gross*1e4:.1f} bps, MOC+quote {dly.mq.mean()*gross*1e4:.1f} bps; Sharpe-equivalent (sqrt252 x mean/std): bar {dly.bar.mean()/dly.bar.std()*np.sqrt(252):.2f} quote {dly.q.mean()/dly.q.std()*np.sqrt(252):.2f} MOC+quote {dly.mq.mean()/dly.mq.std()*np.sqrt(252):.2f}")
print("QUOTES DONE", flush=True)
