"""run table shared by step3 / step4: key -> (desc, features, target[, universe filter])
filter 'pc10': train AND predict only on rows whose predicted mean per-side auction cost at c=0.3, k=10 is <= 10 bps
(~800 names/day, known at 15:59 of day t); the rank target is recomputed within that sub-universe."""
from lib3 import F26X, LIQF
RUNS = {"R1": ("f26x_oo", F26X, "yr_oo"), "R2": ("f40_oo", F26X + LIQF, "yr_oo"), "R3": ("f40_oon", F26X + LIQF, "yr_oon"),
        "R4": ("f40_on", F26X + LIQF, "yr"), "R5": ("f26x_on", F26X, "yr"),
        "R6": ("f40_oo_liq", F26X + LIQF, "yr_oo", "pc10"), "R7": ("f26x_oo_liq", F26X, "yr_oo", "pc10")}
def apply_filter(D, filt):
    import numpy as np
    if filt is None: return D
    if filt == "pc10":
        D = D[((D.ce + D.cx) / 2 <= 10) & D.ce.notna()].reset_index(drop=True)
        for y, t in [("y_oo", "yr_oo"), ("y_on", "yr"), ("y_oon", "yr_oon")]: D[t] = (D.groupby("date")[y].rank(pct=True) - 0.5).astype(np.float32)
        return D
    raise ValueError(filt)
