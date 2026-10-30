#!/usr/bin/env python3
"""quarterly-retrain fit (n_jobs=2, seed 0, expanding window, label lag 3 days) of one step-3 configuration; predictions to
out/pred_final_<cfg>_Q.parquet (date, tk, p).   python3 agents/dayml3/fitq.py A1"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
from d3lib import *
NJ = int(os.environ.get("D3_NJOBS", "2")); cfg = sys.argv[1]
D = frame(1500); u600 = (D["rank"] <= 600).to_numpy()
D["y_60d_net05"] = D["y_60d"] - 0.5 * D["hs_rt"] / 1e4; D["ys_net05"] = -D["y_60d"] - 0.5 * D["hs_rt"] / 1e4
D["yl120"] = D["y_120d"] - 0.5 * D["hs_rt"] / 1e4; D["ys120"] = -D["y_120d"] - 0.5 * D["hs_rt"] / 1e4
CFG = {"A0": (dlib.EXT, "y_60d", None), "A1": (EXTL, "y_60d", None), "A1x120": (EXTL, "y_120d", None), "B1": (EXTL, "y_60d_net05", None), "S1": (EXTL, "ys_net05", None),
       "C1": (EXTL, "y_60d_net05", u600), "S1_600": (EXTL, "ys_net05", u600), "B1x120": (EXTL, "yl120", None), "S1x120": (EXTL, "ys120", None)}
feats, y, m = CFG[cfg]; fp = f"{OUT}/pred_final_{cfg}_Q.parquet"; t0 = time.time()
log(f"fitq {cfg}: quarterly, n_jobs {NJ}, feats {len(feats)}, target {y}, uni {'all' if m is None else 'u600'}")
X = d2lib.matrix(D, feats); yr = d2lib.rank_target(D if m is None else D[m], y)
if m is not None: yf = np.full(len(D), np.nan, np.float32); yf[m] = yr; yr = yf
pred = d2lib.walk_forward(D, X, yr, retrain="Q", n_jobs=NJ, seed=0, verbose=True, train_mask=m); del X
if m is not None: pred[~m] = np.nan
P = D.loc[np.isfinite(pred), ["date", "tk"]].copy(); P["p"] = pred[np.isfinite(pred)]; P.to_parquet(fp)
icm, ict = rlib.ic(D, pd.Series(pred, index=D.index), "y_30d"); log(f"fitq {cfg}: done {time.time()-t0:.0f}s IC(y_30d) {icm:.4f} t {ict:.1f} -> {fp}")
