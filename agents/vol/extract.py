#!/usr/bin/env python3
"""Extract ETF series from panel_liquid.npz into a small CSV (adjusted OHLC, div, raw close, adj, vol)."""
import numpy as np, pandas as pd, json, sys
sys.path.insert(0, "/home/trumpet/strat20")
z = np.load("/home/trumpet/strat20/data/panel_liquid.npz", allow_pickle=True)
dates = pd.to_datetime(z["dates"]); tk = list(z["tickers"])
want = ["SPY","QQQ","IWM","TLT","IEF","GLD","VIXY","VXX","VIXM","VXZ","SVXY","UVXY","TQQQ","SQQQ","SPXL","SPXS","UPRO","SPXU","SSO","SDS","QLD","QID","TNA","TZA","SOXL","SOXS","SVIX","UVIX","VXXB","XIV","ZIV","VIXM","BIL","SHV"]
have = [t for t in want if t in tk]; print("missing:", [t for t in want if t not in tk])
ix = [tk.index(t) for t in have]
out = {}
for k in ["open","high","low","close","vw","vol","adj","div"]:
    a = z[k][:, ix].astype(np.float64)
    for j, t in enumerate(have): out[f"{t}_{k}"] = a[:, j]
df = pd.DataFrame(out, index=dates)
df.to_csv("/home/trumpet/strat20/agents/vol/etf.csv")
sp = json.load(open("/home/trumpet/strat20/data/splits.json"))
rows = [s for s in (sp if isinstance(sp, list) else sp.get("results", [])) if s.get("ticker") in have]
print("splits for our tickers:", len(rows))
for s in sorted(rows, key=lambda s: (s["ticker"], s["execution_date"])): print(s["ticker"], s["execution_date"], s["split_from"], s["split_to"])
# series checks: first/last bar, nan count, max abs daily log return, adj jumps
for t in have:
    c = df[f"{t}_close"] * df[f"{t}_adj"]
    v = c.dropna(); lr = np.log(v).diff()
    gaps = (v.index.to_series().diff().dt.days > 5).sum()
    print(f"{t:5s} first {v.index[0].date()} last {v.index[-1].date()} nbars {len(v)} nan_inside {int(c.loc[v.index[0]:v.index[-1]].isna().sum())} gaps>5d {gaps} maxlr {lr.max():+.3f} on {lr.idxmax().date()} minlr {lr.min():+.3f} on {lr.idxmin().date()}  adj_changes {int((df[f'{t}_adj'].diff().abs()>1e-9).sum())}")
