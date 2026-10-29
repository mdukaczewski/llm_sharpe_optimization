"""Refit sweeps (YEARLY retrain, universe(1500), target y_60d unless stated): capacity, rolling window, recency
weights, thin-name training, new 09:31 features (incl. the overnight model's prediction), 30/90-bin exits.
usage: python3 lab2.py cfg1 cfg2 ...   (results appended to lab2.csv, predictions to out/pred_<cfg>.npy)"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml2")
from d2lib import *
OUT = f"{HERE}/out"; os.makedirs(OUT, exist_ok=True)
CAP31 = dict(num_leaves=31, n_estimators=600, learning_rate=0.02, min_child_samples=500, colsample_bytree=0.5)
CAP63 = dict(num_leaves=63, n_estimators=1000, learning_rate=0.02, min_child_samples=1000, colsample_bytree=0.5)
CFG = {  # name: (feats, target, kwargs for walk_forward)
    "base":   (dlib.EXT, "y_60d", {}),
    "t600":   (dlib.EXT, "y_60d", dict(params=dict(n_estimators=600, learning_rate=0.02))),
    "cap31":  (dlib.EXT, "y_60d", dict(params=CAP31)),
    "cap63":  (dlib.EXT, "y_60d", dict(params=CAP63)),
    "roll3":  (dlib.EXT, "y_60d", dict(window_years=3)),
    "roll4":  (dlib.EXT, "y_60d", dict(window_years=4)),
    "hl2":    (dlib.EXT, "y_60d", dict(halflife_years=2)),
    "hl1":    (dlib.EXT, "y_60d", dict(halflife_years=1)),
    "thin":   (dlib.EXT, "y_60d", dict(train_mask="rank>600")),
    "ext2":   (EXT2, "y_60d", {}),
    "ext2on": (EXT2ON, "y_60d", {}),
    "new9":   (dlib.EXT + NEW9, "y_60d", {}),
    "y30":    (dlib.EXT, "y_30d", {}),
    "y90":    (dlib.EXT, "y_90d", {}),
    "y120":   (dlib.EXT, "y_120d", {}),
    "ext2_y90": (EXT2, "y_90d", {}),
    "ext2_y120": (EXT2, "y_120d", {}),
    "ext2on_y120": (EXT2ON, "y_120d", {}),
    "ext2_cap31": (EXT2, "y_60d", dict(params=CAP31)),
    "ext2on_cap31": (EXT2ON, "y_60d", dict(params=CAP31)),
    "ext2_hl2": (EXT2, "y_60d", dict(halflife_years=2)),
}
cfgs = sys.argv[1:]
t0 = time.time(); tks = rlib.universe(1500); D = build(tks, verbose=False); log(f"lab2 {cfgs}: rows {len(D)} {time.time()-t0:.0f}s")
rk = D["rank"].to_numpy(); rows = []
for cfg in cfgs:
    feats, y, kw = CFG[cfg]; kw = dict(kw)
    if kw.get("train_mask") == "rank>600": kw["train_mask"] = rk > 600
    if any(f not in D.columns or D[f].notna().mean() < 0.5 for f in feats):
        log(f"{cfg}: SKIPPED (features missing: {[f for f in feats if f not in D.columns or D[f].notna().mean() < 0.5]})"); continue
    t1 = time.time(); X = matrix(D, feats); yr = rank_target(D, y)
    pred = walk_forward(D, X, yr, retrain="Y", **kw); del X
    np.save(f"{OUT}/pred_{cfg}.npy", pred)
    icm, ict = rlib.ic(D, pd.Series(pred, index=D.index), y)
    log(f"=== {cfg} ({y}, {len(feats)} feats) fit {time.time()-t1:.0f}s IC {icm:.4f} t {ict:.1f}")
    for k, w, mode in [(6, "eq", "ls"), (4, "eq", "ls"), (8, "eq", "ls"), (6, "pred", "ls"), (6, "eq", "lo")]:
        r, rn = book(D, pred, y, k=k, weight=w, mode=mode); d = ev(r, rn, f"{cfg} {y} {mode} k{k} {w}"); d["cfg"] = cfg; rows.append(d)
    if cfg == "thin":
        r, rn = book(D, pred, y, k=6, weight="eq", mask=rk > 600); d = ev(r, rn, f"{cfg} {y} ls k6 eq ranks>600 only"); d["cfg"] = cfg; rows.append(d)
    pd.DataFrame(rows).to_csv(f"{OUT}/lab2_{'_'.join(cfgs)[:60]}.csv", index=False)
log(f"LAB2 DONE {time.time()-t0:.0f}s")
