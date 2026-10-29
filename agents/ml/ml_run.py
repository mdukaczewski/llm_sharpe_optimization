#!/usr/bin/env python3
"""Walk-forward cross-sectional ML. Usage:
  python agents/ml/ml_run.py --tag lgb5 --h 5 --model lgb [--retrain 126] [--min_adv 50e6] [--minyears 2]
Saves predictions (T x N float32, NaN outside mask) to agents/ml/out/pred_<tag>.npy
and prints OOS rank-IC stats. Train rows: days t with t+h < cutoff (label fully known)."""
import numpy as np, pandas as pd, sys, os, time, gc, argparse
sys.path.insert(0, "/home/trumpet/strat20"); sys.path.insert(0, "/home/trumpet/strat20/agents/ml")
from mlfeat import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--h", type=int, default=5)
ap.add_argument("--model", default="lgb"); ap.add_argument("--retrain", type=int, default=126)
ap.add_argument("--min_adv", type=float, default=50e6); ap.add_argument("--minyears", type=float, default=1.0)
ap.add_argument("--window", type=int, default=0, help="rolling window in days (0=expanding)")
ap.add_argument("--target", default="cc"); ap.add_argument("--step", type=int, default=1, help="use every k-th day in training")
ap.add_argument("--rounds", type=int, default=400); ap.add_argument("--leaves", type=int, default=31)
ap.add_argument("--lr", type=float, default=0.03); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--start", default=START); ap.add_argument("--end", default=END)
ap.add_argument("--feats", default="", help="comma list of features to use (default all)")
ap.add_argument("--nomk", action="store_true", help="drop market-level features")
args = ap.parse_args()
OUT = "/home/trumpet/strat20/agents/ml/out"; os.makedirs(OUT, exist_ok=True)
CACHE = f"/home/trumpet/strat20/agents/ml/cache"; os.makedirs(CACHE, exist_ok=True)

t0 = time.time()
P = load_sub(args.min_adv)
F = build(P); F = extra_feats(P, F)
m = build_mask(F, P, args.min_adv)
print("mask avg names/day", m[300:].sum(1).mean(), flush=True)
cf = f"{CACHE}/feats_{int(args.min_adv/1e6)}.npz"
if os.path.exists(cf):
    z = np.load(cf); names = list(z["names"]); X3 = z["X"]
    print("features loaded from cache", X3.shape, flush=True)
else:
    feats = build_features(P, F, m)
    names = list(feats.keys())
    X3 = np.stack([feats[k] for k in names], axis=-1)   # T x N x K float32
    del feats; gc.collect()
    np.savez(cf, names=np.array(names), X=X3)
    print("features built", X3.shape, f"{time.time()-t0:.0f}s", flush=True)
if args.feats:
    sel = [names.index(f) for f in args.feats.split(",")]
    X3 = X3[:, :, sel]; names = [names[i] for i in sel]
if args.nomk:
    sel = [i for i, n in enumerate(names) if not n.startswith("m_")]
    X3 = X3[:, :, sel]; names = [names[i] for i in sel]
K = X3.shape[2]
y, fwdraw = fwd_target(P, F, m, args.h, args.target)
T, N = P.T, P.N
i0 = int(np.searchsorted(P.dates, pd.Timestamp(args.start))); i1 = int(np.searchsorted(P.dates, pd.Timestamp(args.end), side="right"))
pred = np.full((T, N), np.nan, np.float32)

def fit_predict(tr_days, te_days):
    rows = []
    for d in tr_days:
        idx = np.where(m[d] & np.isfinite(y[d]))[0]
        if len(idx): rows.append((d, idx))
    Xtr = np.concatenate([X3[d, idx] for d, idx in rows]); ytr = np.concatenate([y[d, idx] for d, idx in rows])
    if args.model in ("lgb", "rank"):
        import lightgbm as lgb
        params = dict(objective="regression", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=1000,
                      feature_fraction=0.6, bagging_fraction=0.6, bagging_freq=1, lambda_l2=10.0, verbose=-1,
                      num_threads=args.threads, max_bin=63)
        if args.model == "rank":
            nb = 10
            lab = np.clip(np.floor((ytr + 0.5) * nb), 0, nb - 1).astype(int)
            grp = [len(idx) for d, idx in rows]
            ds = lgb.Dataset(Xtr, lab, group=grp, feature_name=names, free_raw_data=True)
            params.update(objective="lambdarank", label_gain=list(range(nb)), lambdarank_truncation_level=60, bagging_fraction=1.0, bagging_freq=0)
        else:
            ds = lgb.Dataset(Xtr, ytr, feature_name=names, free_raw_data=True)
        mdl = lgb.train(params, ds, num_boost_round=args.rounds)
        imp = pd.Series(mdl.feature_importance("gain"), index=names).sort_values(ascending=False)
        print("   top feats:", ", ".join(f"{k}:{v/imp.sum()*100:.0f}%" for k, v in imp.head(8).items()), flush=True)
        f = lambda X: mdl.predict(X, num_threads=args.threads)
    elif args.model == "ridge":
        from sklearn.linear_model import Ridge
        mdl = Ridge(alpha=float(len(ytr)) * 1e-3).fit(Xtr, ytr)
        co = pd.Series(mdl.coef_, index=names); print("   top coefs:", ", ".join(f"{k}:{v:+.3f}" for k, v in co.abs().sort_values(ascending=False).head(8).index.map(lambda k: (k, co[k]))), flush=True)
        f = lambda X: mdl.predict(X)
    del Xtr, ytr; gc.collect()
    for d in te_days:
        idx = np.where(m[d])[0]
        if len(idx): pred[d, idx] = f(X3[d, idx])
    return len(rows)

first_train = 260 + 1
cut = i0
while cut < i1:
    te = np.arange(cut, min(cut + args.retrain, i1))
    last_tr = cut - args.h - 1                       # label of day t ends at t+h <= cut-1
    lo = first_train if args.window == 0 else max(first_train, last_tr - args.window)
    tr = np.arange(lo, last_tr + 1, args.step)
    if (last_tr - lo) < args.minyears * 252:
        print("not enough training history at", P.dates[cut].date()); cut += args.retrain; continue
    t1 = time.time(); nd = fit_predict(tr, te)
    # quick IC on the test block
    ics = []
    for d in te:
        ok = np.isfinite(pred[d]) & np.isfinite(y[d])
        if ok.sum() > 50: ics.append(pd.Series(pred[d, ok]).corr(pd.Series(y[d, ok]), method="spearman"))
    print(f"train {P.dates[lo].date()}..{P.dates[last_tr].date()} ({nd} days) -> test {P.dates[te[0]].date()}..{P.dates[te[-1]].date()}  IC {np.mean(ics):.4f} (t={np.mean(ics)/np.std(ics)*np.sqrt(len(ics)):.1f})  {time.time()-t1:.0f}s", flush=True)
    cut += args.retrain

np.save(f"{OUT}/pred_{args.tag}.npy", pred)
ic = []
for d in range(i0, i1):
    ok = np.isfinite(pred[d]) & np.isfinite(y[d])
    ic.append(pd.Series(pred[d, ok]).corr(pd.Series(y[d, ok]), method="spearman") if ok.sum() > 50 else np.nan)
ic = pd.Series(ic, index=P.dates[i0:i1])
print("OOS rank IC mean %.4f  std %.4f  IR(daily) %.2f  by year:" % (ic.mean(), ic.std(), ic.mean() / ic.std()))
print(ic.groupby(ic.index.year).mean().round(4).to_string())
print("saved", f"{OUT}/pred_{args.tag}.npy", f"total {time.time()-t0:.0f}s")
