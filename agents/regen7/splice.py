"""helpers: continue a saved quarterly walk-forward prediction into the holdout.
The expanding-window model of quarter cut c only sees rows with date < c - 3 days, so the predictions of every quarter that
starts on or before DEV_END-of-the-dev-run are unchanged; the models of cuts >= FIRST (2026-04-01, the last dev quarter, refit as a
reproducibility check, plus 2026-07-01 and 2026-10-01) are fit with rlib.walk_forward(first=FIRST)."""
import numpy as np, pandas as pd
FIRST = "2026-04-01"; OLD_END = "2026-04-27"
def splice(D, new_p, old_path, name, logf, cast=None):
    """D: frame (rows of the saved prediction), new_p: Series aligned to D (NaN before FIRST).  Returns the spliced Series aligned to D
    and writes overlap stats.  old prediction used for dates < FIRST; new for dates >= FIRST."""
    old = pd.read_parquet(old_path, columns=["date", "tk", "p"])
    o = D[["date", "tk"]].merge(old, on=["date", "tk"], how="left").p.values
    n = new_p.values
    ov = (D.date >= FIRST) & (D.date <= OLD_END)
    a, b = o[ov.values], n[ov.values]
    if cast is not None: a = a.astype(cast).astype(float); b = b.astype(cast).astype(float)
    both = np.isfinite(a) & np.isfinite(b)
    msg = (f"{name}: overlap {FIRST}..{OLD_END} rows old {np.isfinite(a).sum()} new {np.isfinite(b).sum()} both {both.sum()} "
           f"maxabs {np.abs(a[both]-b[both]).max() if both.any() else np.nan:.3e} corr {np.corrcoef(a[both], b[both])[0,1]:.8f}; "
           f"old rows {len(old)} old rows matched in D {np.isfinite(o).sum()}")
    out = np.where(D.date.values < np.datetime64(FIRST), o, n)
    s = pd.Series(out, index=D.index)
    if cast is not None: s = s.astype(cast)
    with open(logf, "a") as fh: fh.write(msg + "\n")
    print(msg, flush=True)
    return s
