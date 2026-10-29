#!/usr/bin/env python3
"""Learned INTRADAY predictor (decide at 09:31 after the first minute, fill at the 09:32 bar, exit MOC) on intraday features, walk-forward, cross-section of
many stocks.  Features per (day, name) known at the close; target = overnight
return to the 09:31 fill (optionally exit later).  Models: ridge, LightGBM,
retrained yearly on an expanding window (first OOS year 2017; 2016-10..12 uses
the 2016 in-sample fit of Jun-Oct 2016 only -> reported separately).
Portfolio: long-only top-k by prediction (equal weight), and rank L/S.
usage: on_ml.py TICKERS cost_bps tag [exit_bin=1] [k=20]"""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
import ilib; from ilib import *
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
tks = sys.argv[1].split(","); cost = float(sys.argv[2]); tag = sys.argv[3]
exit_bin, K = 1, 20
for arg in sys.argv[4:]:
    k_, v_ = arg.split("="); exit_bin = int(v_) if k_ == "exit_bin" else exit_bin; K = int(v_) if k_ == "k" else K
def zsd(x, w=20):
    s = pd.Series(x); return (s / s.rolling(w, min_periods=10).std().shift(1)).values
rows = []
spy = Min("SPY")
spy_z60 = pd.Series(np.nan_to_num(zsd(spy.r(329, 389))), spy.dates); spy_z30 = pd.Series(np.nan_to_num(zsd(spy.r(-1, 29))), spy.dates)
spy_oc = pd.Series(spy.oc, spy.dates)
spy_zgap = pd.Series(np.nan_to_num(zsd(spy.gap)), spy.dates); spy_zm1 = pd.Series(np.nan_to_num(zsd(spy.r(-1, 1))), spy.dates)
for tk in tks:
    try: M = Min(tk)
    except Exception as e: print(tk, "skip", e); continue
    d = pd.DataFrame(index=M.dates)
    d["tk"] = tk
    d["z30"] = zsd(M.r(-1, 29)); d["z60"] = zsd(M.r(329, 389)); d["z30l"] = zsd(M.r(359, 389)); d["z120"] = zsd(M.r(269, 389))
    d["zoc"] = zsd(M.oc); d["zo15"] = zsd(M.r(-1, 329)); d["zgap"] = zsd(M.gap)
    hi = np.nanmax(M.h_raw[:, :390], axis=1); lo = np.nanmin(M.l_raw[:, :390], axis=1)
    d["rngpos"] = (M.close - lo) / (hi - lo + 1e-9); d["rng"] = (hi - lo) / M.close / (pd.Series((hi - lo) / M.close).rolling(20).mean().shift(1).values + 1e-9)
    vol_last = M.v[:, 330:391].sum(1); vol_day = M.v[:, :391].sum(1)
    d["vlast_share"] = vol_last / (vol_day + 1); d["vday_rel"] = vol_day / (pd.Series(vol_day).rolling(20).mean().shift(1).values + 1)
    cc = pd.Series(M.cc, M.dates); d["zcc"] = zsd(M.cc); d["cc5"] = cc.rolling(5).sum().values / (cc.rolling(20).std().shift(1).values * np.sqrt(5) + 1e-9)
    d["rv20"] = cc.rolling(20).std().values * np.sqrt(252)   # lagged below
    ron_prev = np.r_[np.nan, M.c[1:, 1] / M.close[:-1] - 1]      # yesterday's overnight (known at today's close)
    d["zon_prev"] = zsd(np.nan_to_num(ron_prev))
    # ---- features known at 09:31 of day t (today's gap / first minute; everything else is yesterday's) ----
    for c in ["z30", "z60", "z30l", "z120", "zoc", "zo15", "rngpos", "rng", "vlast_share", "vday_rel", "zcc", "cc5", "zon_prev", "rv20"]:   # rv20 includes today's close -> must be lagged
        d[c] = d[c].shift(1)
    d["zgap"] = zsd(M.gap); d["zm1"] = zsd(M.r(-1, 1)); d["v1rel"] = M.v[:, :2].sum(1) / (pd.Series(M.v[:, :2].sum(1)).rolling(20).mean().shift(1).values + 1)
    d["gap_x_z60"] = d["zgap"] * d["z60"]
    d["spy_z60"] = spy_z60.reindex(M.dates).shift(1).values; d["spy_z30"] = spy_z30.reindex(M.dates).shift(1).values; d["spy_oc"] = spy_oc.reindex(M.dates).shift(1).values
    d["spy_zgap"] = spy_zgap.reindex(M.dates).values; d["spy_zm1"] = spy_zm1.reindex(M.dates).values
    d["half"] = M.half
    ron = (M.close if exit_bin <= 2 else M.c[:, exit_bin]) / M.c[:, 2] - 1; ron[np.abs(ron) > 0.5] = np.nan      # 09:32 fill -> official close (or bin exit_bin)
    d["y"] = ron; d["vol_on"] = pd.Series(ron, M.dates).rolling(60, min_periods=30).std().shift(1).values
    rows.append(d)
D = pd.concat(rows).reset_index().rename(columns={"index": "date"})
D = D[(~D.half) & D.y.notna()].drop(columns="half")
D = D[D.date >= "2016-06-01"]
feats = [c for c in D.columns if c not in ("date", "tk", "y", "vol_on")]
D[feats] = D[feats].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-5, 5)
# cross-sectional rank target (per day) and also vol-scaled target
D["yr"] = D.groupby("date")["y"].rank(pct=True) - 0.5
D["yv"] = (D.y / (D.vol_on + 1e-9)).clip(-5, 5).fillna(0)
print("rows", len(D), "names", D.tk.nunique(), "feats", len(feats), flush=True)
D["pred_ridge"] = np.nan; D["pred_lgb"] = np.nan
years = sorted(D.date.dt.year.unique())
from sklearn.linear_model import Ridge
for y in years:
    if y < 2017: continue
    tr = D.date < pd.Timestamp(f"{y}-01-01"); te = (D.date.dt.year == y)
    tr &= D.date < pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=3)   # label ends before cutoff
    X = D.loc[tr, feats].values; Xt = D.loc[te, feats].values
    r = Ridge(alpha=10.0).fit(X, D.loc[tr, "yv"].values); D.loc[te, "pred_ridge"] = r.predict(Xt)
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8)
    m.fit(X, D.loc[tr, "yr"].values); D.loc[te, "pred_lgb"] = m.predict(Xt)
    print(y, "trained on", int(tr.sum()), flush=True)
D["pred_rule"] = -D.zgap
for p in ["pred_ridge", "pred_lgb", "pred_rule"]:
    ic = D.dropna(subset=[p]).groupby("date").apply(lambda g: np.corrcoef(g[p], g.y)[0, 1] if len(g) > 10 else np.nan)
    print(f"{p}: mean daily IC {ic.mean():.4f}  t={ic.mean()/ic.std()*np.sqrt(len(ic)):.1f}", flush=True)
def book(p, k=K, ls=False, smin=None, name=""):
    g = D.dropna(subset=[p])
    out = []
    for dte, grp in g.groupby("date"):
        if len(grp) < k + 5: continue
        o = grp.sort_values(p)
        top = o.iloc[-k:]
        if smin is not None: top = top[top[p] > smin]
        w = pd.Series(1.0 / k, index=top.index)
        if ls:
            bot = o.iloc[:k]; w = pd.concat([w, pd.Series(-1.0 / k, index=bot.index)])
        ret = (w * g.loc[w.index, "y"]).sum(); turn = w.abs().sum() * 2
        out.append((dte, ret, ret - turn * cost / 1e4 - (w.clip(upper=0).abs().sum() * 0.01 / 252)))
    r = pd.DataFrame(out, columns=["date", "g", "n"]).set_index("date")
    r = r.loc[START:END]
    yr = r.n.groupby(r.index.year).apply(sh); eq = (1 + r.n).cumprod(); dd = (eq / eq.cummax() - 1).min()
    h1 = r.n[r.index < "2021-10-28"]; h2 = r.n[r.index >= "2021-10-28"]
    oos = r.n[r.index >= "2017-01-01"]
    print(f"{name:30s} Sh gross {sh(r.g):5.2f} net {sh(r.n):5.2f} (OOS from 2017: {sh(oos):5.2f}) | ret {r.n.mean()*252*100:6.2f}% vol {r.n.std()*np.sqrt(252)*100:5.2f}% mdd {dd*100:6.1f}% | halves {sh(h1):.2f}/{sh(h2):.2f} | yrs " + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in yr.items()), flush=True)
    r.n.rename("ret").rename_axis("date").to_csv(f"/home/trumpet/strat20/agents/intraday/logs/dayml_{tag}_{name.replace(' ', '_')}.csv", float_format="%.8f")
for p in ["pred_rule", "pred_ridge", "pred_lgb"]:
    book(p, name=f"{p} LO top{K}")
    book(p, ls=True, name=f"{p} LS top{K}")
book("pred_lgb", k=10, name="pred_lgb LO top10"); book("pred_lgb", k=30, name="pred_lgb LO top30")
book("pred_ridge", k=10, name="pred_ridge LO top10"); book("pred_ridge", k=30, name="pred_ridge LO top30")
imp = pd.Series(m.feature_importances_, feats).sort_values(ascending=False); print("lgb importance (last model):", imp.head(10).to_dict())
