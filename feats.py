"""Shared feature computation + universe for the strategy experiments."""
import numpy as np, pandas as pd, re, time
from bt import *

START, END = "2016-10-28", "2026-10-27"

def build(P):
    F = {}
    tk = P.tickers
    test_sym = np.array([bool(re.match(r"^Z[A-Z]ZZT$", t)) for t in tk])
    is_cs = np.isin(P.ttype, ["CS", "ADRC"]) & ~test_sym
    F["is_cs"] = is_cs
    close = P.close
    F["closef"] = ffill(close, 5)
    F["close_raw"] = ffill(P.close_raw, 5)     # UNADJUSTED close for the price screen
    F["logc"] = np.log(F["closef"])
    adv20 = rolling_mean(P.dollar_vol, 20, 15)
    F["adv20"] = np.nan_to_num(adv20)
    F["adv60"] = np.nan_to_num(rolling_mean(P.dollar_vol, 60, 40))
    dret = np.where(np.isnan(P.ret), 0.0, P.ret)
    F["dret"] = dret
    F["vol20"] = rolling_std(dret, 20, 15)
    F["vol60"] = rolling_std(dret, 60, 40)
    F["vol120"] = rolling_std(dret, 120, 80)
    F["hi252"] = rolling_max(F["closef"], 252, 200)
    F["ma50"] = rolling_mean(F["closef"], 50, 40)
    F["ma100"] = rolling_mean(F["closef"], 100, 80)
    F["ma200"] = rolling_mean(F["closef"], 200, 160)
    # history length in days with a bar
    has = (~np.isnan(close)).astype(np.float64)
    F["nbars252"] = np.nan_to_num(rolling_sum(has, 252))
    F["nbars60"] = np.nan_to_num(rolling_sum(has, 60))
    spy = P.col("SPY")
    spy_c = F["closef"][:, spy]
    F["spy_c"] = spy_c
    F["spy_ma200"] = pd.Series(spy_c).rolling(200).mean().values
    F["spy_ma50"] = pd.Series(spy_c).rolling(50).mean().values
    F["spy_ret"] = dret[:, spy]
    F["rebal_m"] = month_ends(P.dates)
    F["rebal_w"] = week_ends(P.dates)
    return F

def mom(F, i, lb, skip):
    return F["logc"][i - skip] - F["logc"][i - lb]

def base_mask(F, i, min_adv=10e6, min_px=5.0, min_hist=252):
    # price screen on the UNADJUSTED close: a split-adjusted close leaks future
    # reverse splits (a $0.50 stock with a later 1:30 split reads $15)
    return (F["is_cs"] & (F["close_raw"][i] >= min_px) & (F["adv20"][i] >= min_adv)
            & (F["nbars252"][i] >= min_hist * 0.9) & (F["nbars60"][i] >= 55))
