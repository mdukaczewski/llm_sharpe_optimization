#!/usr/bin/env python3
"""step 3b: TWO-SIDED net targets.  A single rank(y - c) target (step 3 B/C) makes the costliest names rank lowest, where the
book shorts them.  Fix: a long model on rank(y_60d - 0.5 c) (= B1 / C1) and a SHORT model on rank(-y_60d - 0.5 c) (S1 / S1_600);
the book is long the top-k of the long model and short the top-k of the short model.  Also the gross-model (A1) version with
the cost subtracted at selection time: long score E[y|rank] - 0.5c, short score -E[y|rank] - 0.5c (bucket calibration in-sample)."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
from d3lib import *
NJ = int(os.environ.get("D3_NJOBS", "1")); RETRAIN = os.environ.get("D3_RETRAIN", "Y")
log(f"==== step3b: two-sided net targets, retrain {RETRAIN} n_jobs {NJ} ====")
D = frame(1500); D["ys_net05"] = -D["y_60d"] - 0.5 * D["hs_rt"] / 1e4
u600 = (D["rank"] <= 600).to_numpy(); u300 = (D["rank"] <= 300).to_numpy()
def load(cfg):
    P = pd.read_parquet(f"{OUT}/pred_step3_{cfg}_{RETRAIN}.parquet"); return D[["date", "tk"]].merge(P, on=["date", "tk"], how="left")["p"].to_numpy(float)
def fit(cfg, feats, y, m):
    fp = f"{OUT}/pred_step3_{cfg}_{RETRAIN}.parquet"
    if os.path.exists(fp): return load(cfg)
    t0 = time.time(); X = d2lib.matrix(D, feats); yr = d2lib.rank_target(D if m is None else D[m], y)
    if m is not None: yf = np.full(len(D), np.nan, np.float32); yf[m] = yr; yr = yf
    pred = d2lib.walk_forward(D, X, yr, retrain=RETRAIN, n_jobs=NJ, seed=0, train_mask=m); del X
    if m is not None: pred[~m] = np.nan
    P = D.loc[np.isfinite(pred), ["date", "tk"]].copy(); P["p"] = pred[np.isfinite(pred)]; P.to_parquet(fp)
    log(f"step3b {cfg}: fit {time.time()-t0:.0f}s"); return pred
rows = []
def books2(tag, ls, ss, unis):
    for uni, m in unis:
        for k in [4, 6, 8]:
            L = positions(D, None, k=k, y="y_30d", mask=m, long_score=ls, short_score=ss); dd = daily(L, k)
            for f in [1.0, 0.5, 0.25]:
                s = stats(net(dd, f)); s.update(cfg=tag, uni=uni, k=k, f=f); rows.append(s)
            s = stats(net(dd, old=True)); s.update(cfg=tag, uni=uni, k=k, f="old"); rows.append(s)
            t = tier_table(L, 0.5); sh_ = " ".join(f"{r.tier}:{r.share:.2f}/{r.net_bps:+.1f}" for r in t.itertuples())
            log(f"{tag:9s} {uni:5s} k{k} | f1 Sh {rows[-4]['sh']:5.2f} fo {rows[-4]['fo']:5.2f} | f0.5 Sh {rows[-3]['sh']:5.2f} fo {rows[-3]['fo']:5.2f} h {rows[-3]['h1']:4.1f}/{rows[-3]['h2']:4.1f} mdd {rows[-3]['mdd']*100:5.0f}% | f0.25 fo {rows[-2]['fo']:5.2f} | old Sh {rows[-1]['sh']:5.2f} fo {rows[-1]['fo']:5.2f} | tiers {sh_}")
            dd.to_parquet(f"{OUT}/step3b_{tag}_{uni}_k{k}_{RETRAIN}.parquet")
# short models
S1 = fit("S1", EXTL, "ys_net05", None)
B1 = load("B1"); books2("B1+S1", B1, S1, [("u1500", None), ("u600", u600), ("u300", u300)])
pd.DataFrame(rows).to_csv(f"{OUT}/step3b_{RETRAIN}.csv", index=False)
S6 = fit("S1_600", EXTL, "ys_net05", u600)
while not os.path.exists(f"{OUT}/pred_step3_C1_{RETRAIN}.parquet"): time.sleep(30)
C1 = load("C1"); books2("C1+S1_600", C1, S6, [("u600", u600), ("u300", u300)])
pd.DataFrame(rows).to_csv(f"{OUT}/step3b_{RETRAIN}.csv", index=False)
# gross model A1 with the cost at selection (bucket-calibrated E[y_30d | per-day pct rank of A1], in-sample)
A1 = load("A1"); r_ = xr(D, A1); ok = np.isfinite(r_) & D.y_30d.notna().to_numpy()
edges = np.r_[np.arange(-0.5, -0.45, 0.01), np.arange(-0.45, 0.45, 0.05), np.arange(0.45, 0.5001, 0.01)]
b = pd.cut(pd.Series(r_), edges, include_lowest=True); cal = D.y_30d[ok].groupby(b[ok], observed=True).mean(); E = b.map(cal).astype(float).to_numpy()
log("A1 bucket calibration (bps): " + " ".join(f"{i.right:.2f}:{v*1e4:+.1f}" for i, v in cal.items() if abs(i.right) > 0.44))
c = 0.5 * D.hs_rt.to_numpy() / 1e4
books2("A1sel05", E - c, -E - c, [("u1500", None), ("u600", u600)])
pd.DataFrame(rows).to_csv(f"{OUT}/step3b_{RETRAIN}.csv", index=False); log("step3b done")
