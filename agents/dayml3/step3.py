#!/usr/bin/env python3
"""step 3: refits (yearly retrain, seed 0, n_jobs=1) with / without the liq features, on three targets:
  (i)   rank of y_60d                                                   (the dayml target)
  (ii)  rank of y_60d NET of the predicted round-trip spread cost at f = 0.5: y_60d - 0.5 x (hs0931 + hs1001) / 1e4
  (iii) target (ii) on the universe restricted to ranks <= 600 (train and predict there only)
Predictions saved to out/pred_step3_<cfg>.parquet (date, tk, p); books L/S k in {4, 6, 8}, fill bin 3, exit bin 31 (y_30d)."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
from d3lib import *
NJ = int(os.environ.get("D3_NJOBS", "1")); RETRAIN = os.environ.get("D3_RETRAIN", "Y")
cfgs = sys.argv[1:] or ["A1", "B1", "C1", "A0", "B0", "C0"]
log(f"==== step3: refits retrain {RETRAIN} n_jobs {NJ} cfgs {cfgs} ====")
D = frame(1500); log(f"frame rows {len(D)} cols {len(D.columns)}")
D["y_60d_net05"] = D["y_60d"] - 0.5 * D["hs_rt"] / 1e4
D["y_120d_net05"] = D["y_120d"] - 0.5 * D["hs_rt"] / 1e4
u600 = (D["rank"] <= 600).to_numpy(); u300 = (D["rank"] <= 300).to_numpy()
CFG = {  # name: (features, target, universe mask for train+predict)
    "A0": (dlib.EXT, "y_60d", None), "A1": (EXTL, "y_60d", None),
    "B0": (dlib.EXT, "y_60d_net05", None), "B1": (EXTL, "y_60d_net05", None),
    "C0": (dlib.EXT, "y_60d_net05", u600), "C1": (EXTL, "y_60d_net05", u600),
    "B1x120": (EXTL, "y_120d_net05", None), "C1x120": (EXTL, "y_120d_net05", u600),
    "A1x120": (EXTL, "y_120d", None),
}
def books(D, pred, tag):
    rows = []
    for uni, m in [("u1500", None), ("u600", u600), ("u300", u300)]:
        for k in [4, 6, 8]:
            L = positions(D, pred, k=k, y="y_30d", mask=m); dd = daily(L, k)
            if len(dd) < 500: continue
            for f in [1.0, 0.5, 0.25]:
                s = stats(net(dd, f)); s.update(cfg=tag, uni=uni, k=k, f=f); rows.append(s)
            s = stats(net(dd, old=True)); s.update(cfg=tag, uni=uni, k=k, f="old"); rows.append(s)
            t = tier_table(L, 0.5); sh_ = " ".join(f"{r.tier}:{r.share:.2f}/{r.net_bps:+.1f}" for r in t.itertuples())
            log(f"{tag:7s} {uni:5s} k{k} | f1 Sh {rows[-4]['sh']:5.2f} fo {rows[-4]['fo']:5.2f} | f0.5 Sh {rows[-3]['sh']:5.2f} fo {rows[-3]['fo']:5.2f} h {rows[-3]['h1']:4.1f}/{rows[-3]['h2']:4.1f} mdd {rows[-3]['mdd']*100:5.0f}% | f0.25 fo {rows[-2]['fo']:5.2f} | old Sh {rows[-1]['sh']:5.2f} fo {rows[-1]['fo']:5.2f} | tiers(share/net bps@f0.5) {sh_}")
    return rows
allrows = []
for cfg in cfgs:
    feats, y, m = CFG[cfg]; t0 = time.time()
    X = d2lib.matrix(D, feats); yr = d2lib.rank_target(D if m is None else D[m], y)
    if m is not None:
        yr_full = np.full(len(D), np.nan, np.float32); yr_full[m] = yr; yr = yr_full
    pred = d2lib.walk_forward(D, X, yr, retrain=RETRAIN, n_jobs=NJ, seed=0, verbose=False, train_mask=m)
    if m is not None: pred[~m] = np.nan
    del X
    P = D.loc[np.isfinite(pred), ["date", "tk"]].copy(); P["p"] = pred[np.isfinite(pred)]; P.to_parquet(f"{OUT}/pred_step3_{cfg}_{RETRAIN}.parquet")
    icm, ict = rlib.ic(D, pd.Series(pred, index=D.index), "y_30d")
    log(f"step3 {cfg}: feats {len(feats)} target {y} uni {'all' if m is None else 'u600'} fit {time.time()-t0:.0f}s | IC(y_30d) {icm:.4f} t {ict:.1f}")
    allrows += books(D, pred, cfg)
    pd.DataFrame(allrows).to_csv(f"{OUT}/step3_{RETRAIN}.csv", index=False)
log("step3 done")
