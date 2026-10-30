#!/usr/bin/env python3
"""Walk-forward LightGBM on the cut frame (onml3 protocol: target yr = per-day rank of y_on, 300 trees / 15 leaves / lr 0.03 /
min_child 200, expanding window, label lag 3 days, first test 2017-01-01).
usage: python3 train6.py <cut> <set f40|f26> <Y|Q> <seed> [n_jobs]   -> out/pred_<set>_c<cut>_<Y|Q>_s<seed>.parquet"""
from lib6 import *
cut = int(sys.argv[1]); fset = sys.argv[2]; rt = sys.argv[3]; seed = int(sys.argv[4]); nj = int(sys.argv[5]) if len(sys.argv) > 5 else 2
feats = {"f40": F40, "f26": F26X}[fset]
t0 = time.time(); D = model_frame(cut)
log(f"train6 cut {cut} {fset} ({len(feats)} feats) retrain {rt} seed {seed} n_jobs {nj}: {len(D)} rows, frame {time.time()-t0:.0f}s; NaN share hs1559 {D.hs1559.isna().mean():.3f} z60 {D.z60.isna().mean():.3f}")
p = xrank(D, wf(D, feats, "yr", rt, seed=seed, n_jobs=nj)); name = f"{fset}_c{cut}_{rt}_s{seed}"; save_pred(D, p, name)
ic_oo = rlib.ic(D, p, "y_oo"); log(f"train6 done {name} {time.time()-t0:.0f}s  IC y_oo {ic_oo[0]:.4f} (t {ic_oo[1]:.1f})")
