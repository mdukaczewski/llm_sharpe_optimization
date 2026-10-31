import sys, pandas as pd, numpy as np, glob, os
old, new = sys.argv[1], sys.argv[2]; cut = "2026-04-27"
for f in sorted(glob.glob(f"{old}/*.csv")):
    n = f"{new}/{os.path.basename(f)}"
    a = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]; b = pd.read_csv(n, index_col=0, parse_dates=True).iloc[:, 0]
    bd = b[b.index <= cut]; j = pd.concat([a, bd], axis=1)
    d = (j.iloc[:, 0] - j.iloc[:, 1]).abs()
    print(f"{os.path.basename(f)[:-4]:42s} last {b.index[-1].date()} | dev old {len(a)} new {len(bd)} only_new {sorted(set(bd.index)-set(a.index))[:3]} only_old {len(set(a.index)-set(bd.index))} maxabs {d.max():.2e} n>1e-6 {(d>1e-6).sum()} | holdout dates {(b.index>cut).sum()} NaN {int(b.isna().sum())}")
