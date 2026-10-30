"""Step 1: rebuild the agents/ml feature set on the dev-cut panel and train walk-forward LightGBM models.
tags: b{h} = ml_run.py --h h --nomk (target close t -> close t+h), s{h} = target close t+1 -> close t+1+h (skip one day,
for the MOC-at-t+1 execution).  Retrain every 126 trading days, expanding window, n_jobs=2."""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dml2")
from dlib import *
tags = sys.argv[1].split(",")
retrain = sys.argv[2] if len(sys.argv) > 2 else "126"
retrain = "Y" if retrain == "Y" else int(retrain)
P, F, m = setup()
rt, ri = np.nonzero(m); rows = (rt, ri)
xc = f"{OUT}/X16.npy"
if os.path.exists(xc):
    X = np.load(xc).astype(np.float32); names = [l.strip() for l in open(f"{OUT}/X_names.txt")]
else:
    t0 = time.time(); names, X = build_features(P, F, m, rows)
    np.save(xc, X.astype(np.float16)); open(f"{OUT}/X_names.txt", "w").write("\n".join(names))
    log(f"features {X.shape} {time.time()-t0:.0f}s")
i0 = int(np.searchsorted(P.dates, pd.Timestamp(START)))
for tg in tags:
    h = int(tg[1:]); skip = 1 if tg[0] == "s" else 0
    y = fwd_target(P, F, m, h, skip)
    t0 = time.time(); pred, imp = walk_forward(P, m, rows, X, names, y, h + skip, retrain=retrain)
    save_pred(pred, f"{tg}_r{retrain}")
    ic = rank_ic(pred, y, m, i0, P.T)
    ic.index = P.dates[i0:]
    log(f"MODEL {tg} retrain {retrain}: OOS rank IC {ic.mean():.4f} IR {ic.mean()/ic.std():.3f} | {time.time()-t0:.0f}s | by year " +
        " ".join(f"{y_%100}:{v:.3f}" for y_, v in ic.groupby(ic.index.year).mean().items()))
    log("   top feats " + ", ".join(f"{k}:{v*100:.0f}%" for k, v in imp.head(10).items()))
