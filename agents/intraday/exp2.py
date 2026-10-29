"""Exp 2: systematic predictor x target grid. Cell = gross Sharpe of sign(pred)*target-leg
(fill at next bar; costs ignored here, assessed afterwards). Prints per-ticker matrices and
the cross-ticker mean / min to find patterns that hold in SPY,QQQ,IWM,DIA alike."""
import sys; sys.path.insert(0, "/home/trumpet/strat20/agents/intraday")
from ilib import *
import warnings; warnings.filterwarnings("ignore")

TARGETS = [("09:31-10:00", 0, 29), ("10:00-11:00", 29, 89), ("11:00-12:00", 89, 149), ("12:00-14:00", 149, 269),
           ("14:00-15:00", 269, 329), ("15:00-15:30", 329, 359), ("15:30-close", 359, 389), ("09:31-close", 0, 389), ("ON close->09:31", None, None)]

def preds(M, a):
    """predictors known at decision bin a (a = start bin of target; a=None for overnight target decided at 389)"""
    D = len(M.dates); c = M.c
    P = {}
    lag = lambda x: np.r_[np.nan, x[:-1]]
    P["prev_cc"] = lag(M.cc); P["prev_oc"] = lag(M.oc); P["prev_gap"] = lag(M.gap)
    P["cc_5d"] = pd.Series(M.cc).rolling(5).sum().shift(1).values
    P["prev_last30"] = lag(M.r(359, 390)); P["prev_last60"] = lag(M.r(329, 390)); P["prev_first30"] = lag(M.r(-1, 29))
    if a is None:  # overnight target: decided at 389
        P["gap"] = M.gap; P["cum_today"] = M.r(-1, 389); P["last30"] = M.r(359, 389); P["last60"] = M.r(329, 389); P["first30"] = M.r(-1, 29)
        P["vwapdev"] = c[:, 389] / M.cvwap[:, 389] - 1
    else:
        P["gap"] = M.gap
        if a > 0:
            P["cum_today"] = M.r(-1, a); P["last30"] = M.r(max(a - 30, -1), a); P["last60"] = M.r(max(a - 60, -1), a)
            P["first30"] = M.r(-1, min(29, a)); P["vwapdev"] = c[:, a] / M.cvwap[:, a] - 1
            P["prev_same"] = lag(M.r(a, min(a + 30, 390)))
    P["range_pos"] = None
    if a is not None and a > 0:
        hi = np.nanmax(M.h_raw[:, :a + 1], 1); lo = np.nanmin(M.l_raw[:, :a + 1], 1)
        P["range_pos"] = (c[:, a] - (hi + lo) / 2) / np.maximum(hi - lo, 1e-9)   # +: near high
    else:
        P.pop("range_pos")
    return P

def grid(tk):
    M = Min(tk); ms = M.insample & ~M.half
    rows = {}
    for lab, a, b in TARGETS:
        if a is None:
            tgt = np.r_[M.c[1:, 0] / (M.c[:-1, 390] * M.sf[1:]) - 1, np.nan]   # close(d) -> 09:31 (d+1), split adj
            P = preds(M, None)
        else:
            tgt = fill(M, b) / fill(M, a) - 1
            P = preds(M, a)
        for k, v in P.items():
            s = np.sign(np.nan_to_num(v)) * tgt
            rows.setdefault(k, {})[lab] = sh(s[ms])
    df = pd.DataFrame(rows).T[[t[0] for t in TARGETS]]
    return df

if __name__ == "__main__":
    tks = sys.argv[1].split(",") if len(sys.argv) > 1 else ETF4
    out = {}
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.float_format", lambda x: f"{x:5.2f}")
    for tk in tks:
        out[tk] = grid(tk); print(f"\n=== {tk} gross Sharpe of sign(pred) x target"); print(out[tk])
    A = pd.concat(out.values(), keys=out.keys())
    print("\n=== MEAN across", tks); print(A.groupby(level=1).mean().loc[out[tks[0]].index])
    print("\n=== MIN across", tks); print(A.groupby(level=1).min().loc[out[tks[0]].index])
