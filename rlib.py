#!/usr/bin/env python3
"""Shared research library for the strategy agents (2026-10-29 session, local data).

Data layout (all under data/, gitignored):
  panel.npz / panel_liquid.npz   daily bars for every US ticker (bt.Panel), split-adj + dividends
  minute_npz/{T}.npz             1-minute bars 09:30..16:00 (bins 0..390), see agents/intraday/ilib.py
  univ1500.txt                   CS/ADRC tickers ranked by median dollar volume 2016-10..2026-10
  cache/                         per-ticker feature frames built by minute_frame()

PROTOCOL (read before doing anything):
  * DEV_END (env STRAT_END, default 2026-04-27) is the last date any research may use.
    Every loader here cuts at DEV_END.  The holdout 2026-04-28..2026-10-28 is evaluated
    ONCE by the coordinator at the end.  Do not change STRAT_END.
  * A candidate = a CAUSAL daily NET return series (date,ret), decided with data <= t
    (signal at the close of t or at an intraday bar of t), filled at the next bar after
    the decision, with costs: 1 bp per side per unit of |weight| on liquid US stocks
    (1.5 bps for names ranked > 600 by dollar volume), borrow 1%/yr on shorts.
  * The metric that matters is FOLD-OOS Sharpe (fold_oos): half-year test windows from
    2018-07-01, variant choice made only from dates before each window.  Full-sample
    in-sample Sharpe is for orientation only.  Many candidates = more selection noise.
  * Write candidates with save_cand(lane, name, series).  Keep names informative.
"""
import os, sys, glob, time, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = f"{HERE}/data"
sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/agents/intraday")
DEV_END = os.environ.get("STRAT_END", "2026-04-27")
os.environ["STRAT_END"] = DEV_END
START = "2016-10-28"
import ilib
from ilib import Min
RANK = {t: i + 1 for i, t in enumerate(l.strip() for l in open(f"{DATA}/univ1500.txt") if l.strip())}
if os.path.exists(f"{DATA}/univ2000x.txt"):        # phase 4: 500 more names (ranks 1501-2000) appended, original ranks unchanged
    for t in (l.strip() for l in open(f"{DATA}/univ2000x.txt") if l.strip()):
        if t not in RANK: RANK[t] = len(RANK) + 1

def sh(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    return r.mean() / (r.std() + 1e-12) * np.sqrt(252)

def stats(r, name="", p=True):
    r = r.loc[START:DEV_END].fillna(0.0)
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min(); yr = r.groupby(r.index.year).apply(sh)
    h1 = r[r.index < "2021-10-28"]; h2 = r[r.index >= "2021-10-28"]
    line = (f"{name:40s} Sh {sh(r):5.2f} | ret {r.mean()*252*100:6.1f}% vol {r.std()*np.sqrt(252)*100:5.1f}% mdd {dd*100:6.1f}% | halves {sh(h1):.2f}/{sh(h2):.2f} | yrs "
            + " ".join(f"{y%100:02d}:{v:+.1f}" for y, v in yr.items()))
    if p: print(line, flush=True)
    return dict(sharpe=sh(r), ret=r.mean() * 252, vol=r.std() * np.sqrt(252), mdd=dd, h1=sh(h1), h2=sh(h2))

def universe(n=1500, available=True):   # n up to 2000 once data/minute_npz holds the extension
    """top-n tickers by dollar volume; if available, only those whose minute npz exists"""
    tks = [t for t in RANK if RANK[t] <= n]
    if available: tks = [t for t in tks if os.path.exists(f"{DATA}/minute_npz/{t}.npz")]
    return tks

def cost_bps(tk):
    r = RANK.get(tk, 9999)
    return 1.0 if r <= 600 else (1.5 if r <= 1500 else 2.5)     # ranks 1501-2000: median $vol 3-11M/day

def save_cand(lane, name, r, note=""):
    """r: daily net return Series. Written to agents/<lane>/cands/<name>.csv (dates <= DEV_END)."""
    d = f"{HERE}/agents/{lane}/cands"; os.makedirs(d, exist_ok=True)
    r = r.loc[START:DEV_END].rename("ret").rename_axis("date")
    r.to_csv(f"{d}/{name}.csv", float_format="%.8f")
    with open(f"{d}/NOTES.txt", "a") as fh: fh.write(f"{name}\tSh={sh(r):.2f}\t{note}\n")

def load_cands(lane):
    return {os.path.basename(p)[:-4]: pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0] for p in sorted(glob.glob(f"{HERE}/agents/{lane}/cands/*.csv"))}

def fold_oos(cands, first="2018-07-01", gap=5, minhist=200, sel="all", p=True, label=""):
    """Stitched out-of-sample series of a FAMILY of candidates: in each half-year window
    from `first`, hold the candidate with the best Sharpe over prior dates (minus gap).
    cands: dict name -> daily return Series (or a single Series). Returns (oos, table)."""
    if isinstance(cands, pd.Series): cands = {"only": cands}
    df = pd.DataFrame(cands).loc[START:DEV_END].fillna(0.0); dates = df.index
    starts = pd.date_range(first, DEV_END, freq="6MS"); starts = starts[starts.month.isin([1, 7])]
    parts, rows = [], []
    for k, ts in enumerate(starts):
        te = (starts[k + 1] - pd.Timedelta(days=1)) if k + 1 < len(starts) else pd.Timestamp(DEV_END)
        pos = int(np.searchsorted(dates, ts)); sel_end = pos - gap; sel_start = 0 if sel == "all" else max(0, sel_end - 252)
        s = df.iloc[sel_start:sel_end]; t = df.loc[ts:te]
        if len(t) == 0: continue
        scored = sorted(((sh(s[c][s[c] != 0]), c) for c in df.columns if (s[c] != 0).sum() >= minhist), reverse=True)
        if not scored: continue
        best = scored[0][1]; parts.append(t[best]); rows.append((f"{ts.date()}..{te.date()}", best, round(scored[0][0], 2), round(sh(t[best]), 2)))
    oos = pd.concat(parts) if parts else pd.Series(dtype=float)
    tab = pd.DataFrame(rows, columns=["window", "pick", "prior_sh", "fold_sh"])
    if p and len(oos):
        print(f"FOLD-OOS {label:30s} Sh {sh(oos):5.2f} | folds mean {tab.fold_sh.mean():.2f} min {tab.fold_sh.min():.2f} neg {(tab.fold_sh<0).sum()}/{len(tab)} | picks: " + ", ".join(f"{p_}" for p_ in tab.pick.value_counts().head(3).index), flush=True)
    return oos, tab

# ---------------------------------------------------------------- minute features
def zsd(x, w=20):
    s = pd.Series(np.asarray(x, float)); return (s / s.rolling(w, min_periods=10).std().shift(1)).values

def ticker_frame(tk):
    """Per-day feature frame for one ticker (all features known at the CLOSE of day t unless
    prefixed 'o_' = known at 09:31 of day t). Targets (forward, NaN at the end):
      y_on    : close t -> 09:31 fill of t+1 (exit_bin=1)        [overnight book]
      y_on60  : close t -> 10:31 fill of t+1
      y_day   : 09:32 fill of t+1 -> official close of t+1     [intraday book, decided at 09:31 of t+1]
      y_cc    : close t -> close t+1 (total, no dividends)
    Cached in data/cache/{tk}.parquet (keyed on npz mtime)."""
    cd = f"{DATA}/cache"; os.makedirs(cd, exist_ok=True); cp = f"{cd}/{tk}.v2.parquet"      # v2: split-adjusted forward targets
    src = f"{DATA}/minute_npz/{tk}.npz"
    if os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(src):
        return pd.read_parquet(cp)
    M = Min(tk, end="2026-10-28")           # build on everything; cut at DEV_END when used
    d = pd.DataFrame(index=M.dates); d["tk"] = tk
    r = M.r
    d["z30"] = zsd(r(-1, 29)); d["z60"] = zsd(r(329, 389)); d["z30l"] = zsd(r(359, 389)); d["z120"] = zsd(r(269, 389))
    d["z15l"] = zsd(r(374, 389)); d["z5l"] = zsd(r(384, 389)); d["zmoc"] = zsd(r(389, 390))      # last 15 / 5 min, closing print vs 15:59
    d["zoc"] = zsd(M.oc); d["zo15"] = zsd(r(-1, 329)); d["zgap"] = zsd(M.gap)
    hi = np.nanmax(M.h_raw[:, :390], axis=1); lo = np.nanmin(M.l_raw[:, :390], axis=1)
    d["rngpos"] = (M.close - lo) / (hi - lo + 1e-9)
    rng = (hi - lo) / M.close; d["rng"] = rng / (pd.Series(rng).rolling(20).mean().shift(1).values + 1e-9)
    vol_last = M.v[:, 330:391].sum(1); vol_day = M.v[:, :391].sum(1); vol_first = M.v[:, :30].sum(1)
    d["vlast_share"] = vol_last / (vol_day + 1); d["vfirst_share"] = vol_first / (vol_day + 1); d["vmoc_share"] = M.v[:, 390] / (vol_day + 1)
    d["vday_rel"] = vol_day / (pd.Series(vol_day).rolling(20).mean().shift(1).values + 1)
    d["ldv"] = np.log(vol_day * M.close + 1)
    d["zvwap"] = zsd(M.close / M.cvwap[:, 389] - 1)                                   # close vs day vwap
    cc = pd.Series(M.cc, M.dates); d["zcc"] = zsd(M.cc); d["cc5"] = cc.rolling(5).sum().values / (cc.rolling(20).std().shift(1).values * np.sqrt(5) + 1e-9)
    d["cc21"] = cc.rolling(21).sum().values / (cc.rolling(20).std().shift(1).values * np.sqrt(21) + 1e-9)
    d["rv20"] = cc.rolling(20).std().values * np.sqrt(252); d["rv5_20"] = cc.rolling(5).std().values / (cc.rolling(20).std().values + 1e-9)
    ron_prev = np.r_[np.nan, M.c[1:, 1] / M.close[:-1] - 1]
    d["zon_prev"] = zsd(np.nan_to_num(ron_prev))
    d["half"] = M.half; d["nbars"] = M.nbars
    d["close"] = M.close
    # features known at 09:31 of day t (for intraday books): gap and first minute of THIS day
    d["o_zgap"] = zsd(M.gap); d["o_zm1"] = zsd(r(-1, 1)); d["o_vm1_rel"] = M.v[:, :2].sum(1) / (pd.Series(M.v[:, :2].sum(1)).rolling(20).mean().shift(1).values + 1)
    # targets
    def fwd(x): return np.r_[x[1:], np.nan]
    # forward targets cross a day boundary: prev close must be multiplied by the split factor of day d+1
    # (M.sf[d+1] = split_from/split_to on an execution date, else 1), as ilib.Min does for gap/cc.
    base = M.close * fwd(M.sf)
    y_on = fwd(M.c[:, 1]) / base - 1; y_on60 = fwd(M.c[:, 61]) / base - 1
    y_cc = fwd(M.close) / base - 1
    y_day = M.close / M.c[:, 2] - 1                   # same-day: 09:32 fill -> close (decided at 09:31 of that day)
    for k, v in [("y_on", y_on), ("y_on60", y_on60), ("y_cc", y_cc), ("y_day", y_day)]:
        v = np.asarray(v, float); v[np.abs(v) > 0.5] = np.nan; d[k] = v
    d["o_y_day"] = d["y_day"]                          # alias: target of a decision taken at 09:31 of day t
    d = d.astype({c: np.float32 for c in d.columns if c not in ("tk", "half")})
    d.index.name = "date"
    d.to_parquet(cp); return d

def minute_frame(tickers, drop_half=True, verbose=True):
    """Stacked (date, tk) frame of ticker_frame() for all tickers, cut at DEV_END."""
    t0 = time.time(); rows = []
    for i, tk in enumerate(tickers):
        try: rows.append(ticker_frame(tk))
        except Exception as e: print(tk, "skip", e, flush=True)
        if verbose and (i + 1) % 100 == 0: print(f"  minute_frame {i+1}/{len(tickers)} {time.time()-t0:.0f}s", flush=True)
    D = pd.concat(rows).reset_index()
    D = D[(D.date >= "2016-06-01") & (D.date <= DEV_END)]
    if drop_half: D = D[~D.half]
    return D.reset_index(drop=True)

FEATS18 = ["z30", "z60", "z30l", "z120", "zoc", "zo15", "zgap", "rngpos", "rng", "vlast_share", "vday_rel", "zcc", "cc5", "rv20", "zon_prev", "spy_z60", "spy_z30", "spy_oc"]

def add_market(D):
    """SPY features of the same day, broadcast to every name (used by on_ml.py)."""
    s = ticker_frame("SPY").loc[:, ["z60", "z30", "zoc", "zgap"]].rename(columns={"z60": "spy_z60", "z30": "spy_z30", "zoc": "spy_oc", "zgap": "spy_zgap"})
    return D.merge(s, left_on="date", right_index=True, how="left")

def clean(D, feats):
    D[feats] = D[feats].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-5, 5); return D

def walk_forward(D, feats, target, retrain="Q", model="lgb", first="2017-01-01", params=None, n_jobs=2, verbose=True, label_lag_days=3):
    """Expanding-window walk-forward predictions. Train rows: date < cut - label_lag; test rows: [cut, next cut).
    target: column name; for rank targets build it first (e.g. D['yr'] = D.groupby('date').y_on.rank(pct=True)-0.5).
    Returns a Series aligned to D.index (NaN before `first`)."""
    import lightgbm as lgb
    P = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=n_jobs)
    if params: P.update(params)
    cuts = list(pd.date_range(first, DEV_END, freq="QS" if retrain == "Q" else "YS")) + [pd.Timestamp("2027-01-01")]
    pred = pd.Series(np.nan, index=D.index); ok = D[target].notna()
    for i, c0 in enumerate(cuts[:-1]):
        c1 = cuts[i + 1]; tr = ok & (D.date < c0 - pd.Timedelta(days=label_lag_days)); te = (D.date >= c0) & (D.date < c1)
        if te.sum() == 0 or tr.sum() < 1000: continue
        X = D.loc[tr, feats].values; y = D.loc[tr, target].values
        if model == "lgb": m = lgb.LGBMRegressor(**P).fit(X, y)
        else:
            from sklearn.linear_model import Ridge; m = Ridge(alpha=10.0).fit(X, y)
        pred[te] = m.predict(D.loc[te, feats].values)
        if verbose: print(f"  wf {c0.date()} train {int(tr.sum())}", flush=True)
    return pred

COMMISSION_USD = float(os.environ.get("STRAT_COMMISSION_USD", "0"))   # fixed commission per filled order (user: $3)
CLIP_USD = float(os.environ.get("STRAT_CLIP_USD", "10000"))           # dollars per position -> commission in bps = USD/CLIP*1e4

def commission_bps(commission=None, clip=None):
    """per-side cost in bps of a fixed per-order commission on a position of `clip` dollars"""
    c = COMMISSION_USD if commission is None else commission; k = CLIP_USD if clip is None else clip
    return c / k * 1e4

def book(D, pred, k=20, y="y_on", mode="ls", cost=None, smin=None, weight="eq", borrow=0.01, name="", p=True, commission=None, clip=None):
    """Daily net return of a top-k book from per-(date,tk) predictions.
    mode: 'ls' long top-k / short bottom-k, 'lo' long top-k.  weight: 'eq' or 'pred' (|pred| scaled, sum|w| per side = 1).
    cost: bps per side (spread/impact); default per-name cost_bps(tk).  On top of it a fixed per-order commission
    (commission USD per filled order, clip USD per position; defaults from env STRAT_COMMISSION_USD / STRAT_CLIP_USD,
    i.e. $3 on a $10k clip = 3 bps per side).  Costs: 2 sides x |w| x (cost + commission_bps) + borrow on shorts."""
    g = D.loc[pred.notna(), ["date", "tk", y]].copy(); g["p"] = pred[g.index].values; g = g[g[y].notna()]
    g["c"] = (g.tk.map(cost_bps) if cost is None else cost) + commission_bps(commission, clip)
    g["rk"] = g.groupby("date")["p"].rank(method="first"); g["n"] = g.groupby("date")["p"].transform("size")
    g = g[g.n >= k + 5]
    top = g[g.rk > g.n - k]; bot = g[g.rk <= k] if mode == "ls" else g.iloc[0:0]
    if smin is not None: top = top[top.p > smin]
    def side(x, sgn):
        if weight == "pred":
            w = x.p.abs() if sgn > 0 else x.p.abs(); w = w / x.groupby("date").p.transform(lambda s: s.abs().sum()).replace(0, np.nan)
        else: w = 1.0 / k
        return pd.DataFrame({"date": x.date, "w": sgn * w, "r": x[y], "c": x.c})
    L = pd.concat([side(top, 1), side(bot, -1)])
    L["g"] = L.w * L.r; L["cost"] = L.w.abs() * 2 * L.c / 1e4 + L.w.clip(upper=0).abs() * borrow / 252
    out = L.groupby("date").agg(g=("g", "sum"), cost=("cost", "sum")); n = (out.g - out.cost).loc[START:DEV_END]
    if p: stats(n, name or f"{mode} top{k} {y}")
    return n

def recost(r, extra_bps_per_side, gross=2.0, sides=2):
    """Re-cost a daily net series of a book that is fully turned over every day: subtract
    sides x gross x extra_bps per day (L/S top-k books: gross 2, two sides; LO books: gross 1)."""
    return r - sides * gross * extra_bps_per_side / 1e4

def ic(D, pred, y="y_on"):
    g = D.loc[pred.notna() & D[y].notna(), ["date", y]].copy(); g["p"] = pred[g.index].values
    s = g.groupby("date").apply(lambda x: x.p.corr(x[y]) if len(x) > 10 else np.nan).dropna()
    return s.mean(), s.mean() / s.std() * np.sqrt(len(s))
