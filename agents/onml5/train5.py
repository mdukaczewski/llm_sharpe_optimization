#!/usr/bin/env python3
"""Walk-forward LightGBM (same params / protocol as rlib.walk_forward: expanding window, label lag 3 days, retrain Y or Q from
2017-01-01) with optional sample weights and tier-specific targets.  Saves out/pred_<name>.parquet (date, tk, p = per-day
pct rank - 0.5; for multi-seed runs the seed ranks are averaged and re-ranked; per-seed files out/pred_<name>_s<seed>).
usage: python3 train5.py name=<n> feats=f40|f40n|f40a|f40c [w=none|sqdv|icost] [tgt=yr|r600] [filt=none|l600] [rt=Y|Q] [seeds=0]
  feats  f40  = F26X + 14 liq (= onml3 f40_on)      f40n = f40 + all 16 onml5 features
         f40a = f40 + sector-relative / residual (a,b)   f40c = f40 + event / liquid-tier context (c,d)
  w      sqdv  = sqrt(exp(ldv20)), normalised to mean 1 per day;  icost = 1 / predicted round-trip auction cost (MOC + MOO,
         k6 order $33k, c = 0.3, floor 2 bps), normalised per day;  rv20 = 0 (frame fill) -> day median weight
  tgt    yr    = per-day rank of y_on (onml3 f40_on target);   r600 = per-day rank of y_on computed separately inside the
         dynamic top-600 (drank <= 600, lagged ldv20) and inside the rest, plus the flag in600 as a feature
  filt   l600  = train and predict only on drank <= 600 (rank target recomputed inside); for splicing (see eval5.splice)"""
from lib5 import *
sys.path.insert(0, f"{HERE}/agents/onml")
import olib
from olib import EXTRA10, CAL
import lightgbm as lgb
kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
NAME = kw["name"]; FS = kw.get("feats", "f40"); WT = kw.get("w", "none"); TGT = kw.get("tgt", "yr"); FILT = kw.get("filt", "none")
RT = kw.get("rt", "Y"); SEEDS = [int(s) for s in kw.get("seeds", "0").split(",")]; NJ = int(kw.get("nj", "2"))
FS26 = rlib.FEATS18 + EXTRA10 + CAL
F26 = [f for f in FS26 if f not in ("zmoc", "vmoc_share", "z5l")]
XMAP = {"vlast_share": "vlast_x", "vday_rel": "vday_rel_x", "ldv": "ldv_x"}
F26X = [XMAP.get(f, f) for f in F26]
LIQF = ["ldv20", "lpx", "hl1", "hl5", "hlc", "roll30", "n30", "nc30", "dv0", "dvc", "c_hlc", "c_nc30", "hs0931", "hs1559"]
NA = ["z60_sr", "z30_sr", "z15l_sr", "zoc_sr", "zgap_sr", "zon_prev_sr", "sec_b", "ron5_r", "ron20_r", "rid5_r", "rid20_r"]
NC = ["gmax5", "vmax5", "z60_g5", "drank", "z60_l600"]
FEATS = {"f40": F26X + LIQF, "f40n": F26X + LIQF + NA + NC, "f40a": F26X + LIQF + NA, "f40c": F26X + LIQF + NC}[FS]
t0 = time.time()
cols = list(dict.fromkeys(["date", "tk", "y_on", "rv20", "dv0", "dvc"] + [f for f in FEATS if f not in NA + NC]))
D = pd.read_parquet(FRAME, columns=cols)
X5 = pd.read_parquet(f"{OUT}/feat5.parquet", columns=["date", "tk"] + NA + NC)
D = D.merge(X5, on=["date", "tk"], how="left"); del X5
D = D[D.date <= rlib.DEV_END].sort_values(["date", "tk"]).reset_index(drop=True)
in600 = D.drank <= 600
if FILT == "l600": D = D[in600].reset_index(drop=True); in600 = D.drank <= 600
if TGT == "yr" or FILT == "l600":
    D["t"] = (D.groupby("date").y_on.rank(pct=True) - 0.5).astype(np.float32)
elif TGT == "r600":
    D["in600"] = in600.astype(np.float32); FEATS = FEATS + ["in600"]
    D["t"] = (D.groupby([D.date, D.in600]).y_on.rank(pct=True) - 0.5).astype(np.float32)
D.loc[D.y_on.isna(), "t"] = np.nan
if WT == "none": wcol = None
else:
    if WT == "sqdv": w = np.sqrt(np.exp(D.ldv20.astype(float)))
    elif WT == "icost":
        rv = D.rv20.astype(float).where(D.rv20 > 0)
        G_ = pd.DataFrame({"rv20": rv, "dv0": D.dv0, "dvc": D.dvc, "ldv20": D.ldv20})
        cst = lib4.auction(G_, CAP / 12, 0.3, "close") + lib4.auction(G_, CAP / 12, 0.3, "open")
        w = 1.0 / cst.clip(lower=2.0)
    w = pd.Series(w, index=D.index); w = w.fillna(w.groupby(D.date).transform("median")).fillna(w.median())
    D["w"] = (w / w.groupby(D.date).transform("mean")).astype(np.float32); wcol = "w"
    log(f"train5 {NAME}: weight {WT} quantiles " + " ".join(f"{q}:{D.w.quantile(q):.3f}" for q in [0.01, 0.1, 0.5, 0.9, 0.99])
        + " | mean weight by dynamic rank tier: " + " ".join(f"{a}-{b} {D.w[(D.drank >= a) & (D.drank <= b)].mean():.2f}" for a, b in [(1, 300), (301, 600), (601, 1500), (1501, 3000)]))
log(f"train5 {NAME}: feats {FS} ({len(FEATS)}) w {WT} tgt {TGT} filt {FILT} rt {RT} seeds {SEEDS} rows {len(D)} load {time.time()-t0:.0f}s")

def wf(seed):
    P = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
             verbose=-1, n_jobs=NJ, random_state=seed)
    cuts = list(pd.date_range("2017-01-01", rlib.DEV_END, freq="QS" if RT == "Q" else "YS")) + [pd.Timestamp("2027-01-01")]
    pred = pd.Series(np.nan, index=D.index); ok = D.t.notna(); X = D[FEATS].to_numpy(np.float32); imp = 0
    for i, c0 in enumerate(cuts[:-1]):
        c1 = cuts[i + 1]; tr = (ok & (D.date < c0 - pd.Timedelta(days=3))).values; te = ((D.date >= c0) & (D.date < c1)).values
        if te.sum() == 0 or tr.sum() < 1000: continue
        m = lgb.LGBMRegressor(**P).fit(X[tr], D.t.values[tr], sample_weight=(D[wcol].values[tr] if wcol else None))
        pred[te] = m.predict(X[te]); imp = imp + m.booster_.feature_importance("gain")
    return pred, pd.Series(imp, index=FEATS)
members = []
for s in SEEDS:
    t1 = time.time(); p, imp = wf(s); p = prank(p, D.date); members.append(p)
    pd.DataFrame({"date": D.date, "tk": D.tk, "p": p}).dropna().to_parquet(f"{OUT}/pred_{NAME}_s{s}.parquet")
    log(f"train5 {NAME} seed {s} {time.time()-t1:.0f}s; top gain: " + " ".join(f"{k}:{v/imp.sum()*100:.1f}%" for k, v in imp.sort_values(ascending=False).head(15).items())
        + " | new feats gain share " + f"{imp[[f for f in FEATS if f in NA + NC + ['in600']]].sum()/imp.sum()*100:.1f}%")
p = prank(sum(members) / len(members), D.date)
pd.DataFrame({"date": D.date, "tk": D.tk, "p": p}).dropna().to_parquet(f"{OUT}/pred_{NAME}.parquet")
log(f"train5 {NAME} DONE {time.time()-t0:.0f}s")
