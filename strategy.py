#!/usr/bin/env python3
"""
Vol-scaled momentum ensemble, long-only US large/mid-cap stocks.
==================================================================

Universe (evaluated at each rebalance, point-in-time):
  * US common stocks and ADRs (Polygon type CS / ADRC), Nasdaq test symbols
    excluded
  * close >= $5, 20-day average dollar volume >= $50M
  * at least ~1 year of trading history (>= 227 bars in last 252 days)

Signal: for each (lookback, skip) in {(252,21), (252,42), (210,21), (189,21)}
  score = log(close[t-skip] / close[t-lookback]) / annualised 60-day vol
  rank each score across the universe, sum the four ranks.

Portfolio: top 25 by summed rank, equal weight, 100% invested, no leverage,
  no shorting. Two equal tranches, each rebalanced monthly: tranche A on the
  last trading day of the month, tranche B 10 trading days later. Orders are
  executed at the close of the trading day AFTER the signal day.
  Transaction cost: 15 bps per unit of turnover (buy + sell each 15 bps).

Data: Polygon grouped-daily bars (unadjusted) + Polygon splits + dividends;
  total-return close-to-close. Delisted names kept (no survivorship bias).
  A holding with no bar for 15 consecutive days is written off at its last
  price.

Usage (on the box with ~/strat20/data/panel.npz):
  python strategy.py                 # full report + CSV outputs in out/
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from bt import Panel, run_fast, stats, report, ffill, rolling_mean, rolling_std, rolling_sum, month_ends
from feats import build, base_mask, mom, START, END

PARAMS = ((252, 21), (252, 42), (210, 21), (189, 21))
TOP = 25
MIN_ADV = 50e6
MIN_PX = 5.0
VOLW = 60
COST_BPS = 15
TRANCHE_OFFSETS = (0, 10)

def ranks(s, m):
    out = np.full_like(s, np.nan)
    idx = np.where(m & np.isfinite(s))[0]
    out[idx] = rankdata(s[idx]) / len(idx)
    return out

def score(F, P, i):
    m = base_mask(F, i, MIN_ADV, MIN_PX)
    v = F[f"vol{VOLW}"][i] * np.sqrt(252)
    s = np.zeros(P.N)
    for lb, skip in PARAMS:
        s += ranks(mom(F, i, lb, skip) / (v + 1e-9), m)
    return np.where(m & np.isfinite(s), s, -np.inf)

def select(F, P, i):
    s = score(F, P, i)
    idx = np.argsort(-s)[:TOP]
    idx = idx[np.isfinite(s[idx])]
    return idx

def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)
    P = Panel(); F = build(P)
    holdings_log = []
    def make_fn(tag):
        def fn(i):
            idx = select(F, P, i)
            holdings_log.append(dict(tranche=tag, signal_date=str(P.dates[i].date()), tickers=[str(P.tickers[c]) for c in idx]))
            return {int(c): 1.0 / len(idx) for c in idx}
        return fn
    eqs = []
    for k, off in enumerate(TRANCHE_OFFSETS):
        reb = F["rebal_m"] + off; reb = reb[reb < P.T]
        eq, info = run_fast(P, make_fn("AB"[k]), reb, START, END, cost_bps=COST_BPS)
        report(eq, f"tranche {'AB'[k]} (offset {off}d)", extra=dict(turnover=info["turnover"]))
        eqs.append(eq)
    eq = sum(eqs) / len(eqs)
    s = report(eq, "COMBINED strategy")
    spy = P.col("SPY")
    i0 = np.searchsorted(P.dates, pd.Timestamp(START)); i1 = np.searchsorted(P.dates, pd.Timestamp(END), side="right")
    spy_eq = pd.Series(np.cumprod(1 + np.nan_to_num(P.ret[i0:i1, spy])), index=P.dates[i0:i1])
    sb = report(spy_eq, "SPY (total return)")
    # rolling 3y CAGR & calendar stats
    r = eq.pct_change().dropna()
    yrs = pd.DataFrame({"strategy": s["yearly"], "SPY": sb["yearly"]})
    print("\nCalendar-year returns:\n" + (yrs * 100).round(1).to_string())
    dd = (eq / eq.cummax() - 1)
    print(f"\nWorst drawdown {dd.min()*100:.1f}% ending {dd.idxmin().date()}")
    pd.DataFrame({"strategy": eq, "spy": spy_eq}).to_csv(os.path.join(out, "equity_curve.csv"))
    json.dump(holdings_log, open(os.path.join(out, "holdings_log.json"), "w"), indent=0)
    # current portfolio: latest signal of each tranche
    latest = {}
    for h in holdings_log:
        latest[h["tranche"]] = h
    print("\nCurrent portfolio (two tranches, 50% each, equal weight within):")
    for t, h in sorted(latest.items()):
        print(f"  tranche {t} signal {h['signal_date']}: " + " ".join(h["tickers"]))
    json.dump(latest, open(os.path.join(out, "current_portfolio.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
