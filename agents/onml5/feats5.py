#!/usr/bin/env python3
"""Build out/feat5.parquet: new liquid-tier features per (date, tk), all known at 15:59 of day t.
Inputs: onml3 frame (already-z-scored features of day t, read by column) and sector-ETF minute bars (ilib.Min, end=DEV_END).

Sector map (causal, quarterly): at each quarter start c, on the last 250 trading days with date < c - 3 days, regress each
stock's z-scored daily return zcc on SPY's (z-space), take residuals, and correlate them with each sector ETF's SPY-residual;
sector = ETF with the highest residual correlation if >= 0.15, else SPY.  b = plain corr(zcc_stock, zcc_etf) over the
same window (vol-scaled beta: for z-scores r/sd, beta x sd_e / sd_s = corr).  The map is applied to days in [c, next c).
  (a) <f>_sr = f_stock - b x f_etf for f in z60 z30 z15l zoc zgap zon_prev   (+ sec_b = b)
  (b) residual overnight / intraday sums: ron5_r ron20_r = rolling sum of zgap_sr over 5 / 20 days (incl. today's gap)
      / sqrt(n); rid5_r rid20_r the same for zoc_sr (today's open -> 15:59-bar... zoc uses the official close, see note)
  (c) event proxies: gmax5 = max |zgap| over the last 5 days (incl. today), vmax5 = max vday_rel_x over the last 5 days,
      z60_g5 = z60 x gmax5
  (d) drank = per-day rank of ldv20 (lagged 20-day mean $vol; 1 = most liquid), z60_l600 = per-day pct rank of z60 among
      the names with drank <= 600 (NaN beyond).
Note on zoc: the frame's zoc is open -> official close of t (rlib.ticker_frame), the same few-minutes look-ahead the
project already accepts for zoc/zcc in F26X; it is used here only through the same feature."""
from lib5 import *
sys.path.insert(0, f"{HERE}/agents/intraday")
from ilib import Min
ETFS = ["XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SMH", "KRE", "GDX"]
FB = ["z60", "z30", "z15l", "zoc", "zgap", "zon_prev"]
t0 = time.time()
F = pd.read_parquet(FRAME, columns=["date", "tk", "zcc", "ldv20", "vday_rel_x"] + FB)
F = F.sort_values(["tk", "date"]).reset_index(drop=True)

def etf_frame(tk):
    M = Min(tk, end=rlib.DEV_END); r = M.r; zsd = rlib.zsd
    d = pd.DataFrame({"z60": zsd(r(329, 389)), "z30": zsd(r(-1, 29)), "z15l": zsd(r(374, 389)), "zoc": zsd(M.oc), "zgap": zsd(M.gap),
                      "zon_prev": zsd(np.nan_to_num(np.r_[np.nan, M.c[1:, 1] / M.close[:-1] - 1])), "zcc": zsd(M.cc)}, index=M.dates)
    d = d.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-5, 5)          # same treatment as the frame (rlib.clean)
    return d[~M.half & (d.index <= rlib.DEV_END)]
E = {e: etf_frame(e) for e in ETFS + ["SPY"]}
log(f"feats5: frame {len(F)} rows, ETFs loaded {time.time()-t0:.0f}s; last ETF date {max(v.index.max() for v in E.values()).date()}")
dates = pd.DatetimeIndex(sorted(F.date.unique()))
W = F.pivot(index="date", columns="tk", values="zcc").reindex(dates)
EZ = pd.DataFrame({e: E[e].zcc for e in E}).reindex(dates)
cuts = pd.date_range("2016-10-01", rlib.DEV_END, freq="QS")
maps = []
for i, c in enumerate(cuts):
    hi = int(np.searchsorted(dates, c - pd.Timedelta(days=3))); lo = max(0, hi - 250)
    if hi - lo < 40: continue
    w = W.iloc[lo:hi]; ez = EZ.iloc[lo:hi]; m = ez.SPY
    nobs = w.notna().sum()
    def corr_cols(A, s):   # corr of every column of A with series s (pairwise complete)
        ok = A.notna() & s.notna().values[:, None]; a = A.where(ok); ss = pd.DataFrame(np.repeat(s.values[:, None], A.shape[1], 1), index=A.index, columns=A.columns).where(ok)
        am = a - a.mean(); sm = ss - ss.mean(); return (am * sm).sum() / np.sqrt((am ** 2).sum() * (sm ** 2).sum())
    bm = (w.sub(w.mean())).mul(m - m.mean(), axis=0).sum() / ((m - m.mean()) ** 2).sum()            # z-space beta on SPY
    res = w - np.outer(m.values, bm.values)
    rc = {}; pc = {}
    for e in ETFS:
        ee = ez[e] - (np.cov(ez[e], m)[0, 1] / m.var()) * m                                           # ETF residual on SPY
        rc[e] = corr_cols(res, ee); pc[e] = corr_cols(w, ez[e])
    pc["SPY"] = corr_cols(w, m)
    RC = pd.DataFrame(rc); PC = pd.DataFrame(pc)
    RCf = RC.fillna(-9); best = RCf.idxmax(axis=1); bestv = RC.max(axis=1).fillna(-9)
    sec = best.where(bestv >= 0.15, "SPY")
    b = pd.Series([PC.at[t, s] for t, s in sec.items()], index=sec.index)
    ok = nobs >= 60
    nxt = cuts[i + 1] if i + 1 < len(cuts) else pd.Timestamp("2027-01-01")
    maps.append(pd.DataFrame({"tk": sec.index, "sec": sec.where(ok, "SPY").values, "sec_b": b.where(ok, PC.SPY.where(ok, 0.0)).fillna(0).values,
                              "rcorr": bestv.values, "c0": c, "c1": nxt}))
MP = pd.concat(maps); MP.to_csv(f"{OUT}/sector_map.csv", index=False)
last = MP[MP.c0 == MP.c0.max()]
log("feats5 sector map, last quarter: " + last.sec.value_counts().to_string().replace("\n", " | "))
for t in ["JPM", "XOM", "NVDA", "AAPL", "NEM", "DUK", "PG", "UNH", "AMZN", "CAT", "ZION"]:
    x = MP[MP.tk == t]
    if len(x): log(f"  {t}: " + " ".join(f"{r.c0.date()}:{r.sec}({r.sec_b:.2f})" for r in x.iloc[::8].itertuples()))
# assign per row
F["q"] = F.date.dt.to_period("Q").dt.start_time
MP["q"] = MP.c0
F = F.merge(MP[["tk", "q", "sec", "sec_b"]], on=["tk", "q"], how="left")
F["sec"] = F.sec.fillna("SPY"); F["sec_b"] = F.sec_b.fillna(0.0).astype(np.float32)
EL = pd.concat([E[e][FB].assign(sec=e) for e in E]).rename_axis("date").reset_index().rename(columns={f: f + "_e" for f in FB})
F = F.merge(EL, on=["date", "sec"], how="left")
for f in FB:
    F[f + "_sr"] = (F[f] - F.sec_b * F[f + "_e"].fillna(0)).astype(np.float32)
F = F.sort_values(["tk", "date"]).reset_index(drop=True)
g = F.groupby("tk")
for n in [5, 20]:
    F[f"ron{n}_r"] = (g.zgap_sr.transform(lambda s: s.rolling(n, min_periods=2).sum()) / np.sqrt(n)).astype(np.float32)
    F[f"rid{n}_r"] = (g.zoc_sr.transform(lambda s: s.rolling(n, min_periods=2).sum()) / np.sqrt(n)).astype(np.float32)
F["gmax5"] = g.zgap.transform(lambda s: s.abs().rolling(5, min_periods=1).max()).astype(np.float32)
F["vmax5"] = g.vday_rel_x.transform(lambda s: s.rolling(5, min_periods=1).max()).astype(np.float32)
F["z60_g5"] = (F.z60 * F.gmax5).astype(np.float32)
F["drank"] = F.groupby("date").ldv20.rank(ascending=False, method="first").astype(np.float32)
liqm = F.drank <= 600
F["z60_l600"] = np.nan
F.loc[liqm, "z60_l600"] = F[liqm].groupby("date").z60.rank(pct=True) - 0.5
F["z60_l600"] = F.z60_l600.astype(np.float32)
NEW = [f + "_sr" for f in FB] + ["sec_b", "ron5_r", "ron20_r", "rid5_r", "rid20_r", "gmax5", "vmax5", "z60_g5", "drank", "z60_l600"]
F[["date", "tk", "sec"] + NEW].to_parquet(f"{OUT}/feat5.parquet")
log(f"feats5 done {len(F)} rows, {len(NEW)} new features, {time.time()-t0:.0f}s\n" + F[NEW].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
log("feats5 corr of new features with raw: " + " ".join(f"{f}:{F[f].corr(F[f+'_sr']):.2f}" for f in FB))
