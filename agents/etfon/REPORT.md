# etfon lane: ETF overnight sleeves at realistic cost (2026-10-30) — nothing kept

Written by the coordinator from the agent's hand-back (the agent's own write was blocked).  Code: `elib2.py`, `x1_ts.py`
(time-series rule), `x2_ml.py` (pooled LightGBM), `x3_split.py` (regime split), `x4_checks.py`, `x5_short.py`, `save.py`;
log `log.txt` (correlations above its `##### NOTE` marker are off by one day; the rerun below it is correct).

Conventions: EQ = SPY QQQ IWM DIA EEM EFA KRE SMH XLE XLF XLI XLK XLP XLU XLV XLY (16), ALL = EQ + GDX GLD HYG IEF SLV TLT
USO.  MOC entry, official-open exit (09:31 bar compared).  Sizing 0.05 / rv20(t-1) per position, cap 0.5, gross cap 2.
Costs inside every series: 1.5 bps per side (3 bps also reported), 1% borrow on shorts, commission 2 x positions x $3 /
$400k per day.  Targets include the declared dividend (minute and official prices are not dividend-adjusted).  Series
are decision-dated.

## 1. Time-series overnight rule, s = 0.5*z30 - z60 > threshold
Gross mean overnight return (bps/night), s > 1: SPY 13.8 (t 3.8), QQQ 15.2, IWM 12.8, SMH 29.9, EEM 22.0; mean of 16 EQ
11.9 (all nights 4.5); mean of 7 non-equity 6.2 (second half 1.5).  An equity effect; about 4.5 bps of it is the plain
overnight premium.

| book | Sh | fold-OOS | halves | maxDD | positions/day | at 3 bps (fold) | corr xsr3 / onml3 / vixon / mom / SPY |
|---|---|---|---|---|---|---|---|
| EQ every night (ref) | -0.07 | -0.16 | 0.28/-0.54 | -54% | 16.0 | -0.79 | .72/.31/.06/.43/.61 |
| EQ s>0.5 long | 1.02 | 0.95 | 1.53/0.42 | -20% | 5.1 | 0.36 (0.35) | .79/.35/.45/.34/.45 |
| **EQ s>1 long (`etfon_ts_s1_moo`)** | 0.90 | 0.89 | 1.13/0.60 | -20.9% | 2.9 | 0.41 (0.43) | .77/.34/.45/.34/.43 |
| EQ s>1.5 long | 0.71 | 0.68 | 0.78/0.63 | -19% | 1.6 | 0.37 | .75/.37/.42/.35/.43 |
| EQ s>1 L/S | 0.64 | 0.75 | 0.98/0.18 | -32% | 5.5 | 0.03 | .45/.11/.41/-.06/-.11 |
| **EQ s>1.5 L/S (`etfon_ts_s15_both_moo`)** | 0.69 | 0.77 | 0.96/0.32 | -20.4% | 3.0 | 0.28 (0.39) | .48/.14/.47/-.09/-.13 |
| ALL s>1 long | 0.84 | 0.83 | 1.29/0.27 | -20% | 4.0 | 0.18 | .67/.29/.33/.26/.30 |

* Official open vs 09:31-bar exit: same book (0.90 vs 0.94).  Gross Sharpe 1.49 on ~9 bps per unit traded; the round trip
  plus commission is 3.4 bps at 1.5 bps/side and 6.4 at 3 bps, hence the halving.
* Auction impact `liq.auction_cost_bps` on EQ s>1 long: c = 0.1 -> 0.73 fold-OOS, c = 0.3 -> -0.32, c = 1.0 -> -3.9;
  SPY/QQQ/IWM stay at 1.0-1.6 bps per side at c = 0.3, KRE/SMH 17-21, other sectors 3-8.
* Signal frozen at 15:50: 0.71 fold-OOS.  Short side alone: best 0.49 fold-OOS (mostly 2020), dead.
* Overlap with the stock rule book: beta 0.78 on xsr3, residual Sharpe -0.64.  Adding 20% of it to an equal-risk
  xsr3 + onml3 + vixon_ls mix lowers that mix from 2.46 to 2.24-2.33.

## 2. Pooled LightGBM across the 16 EQ ETFs — dead
Target overnight return / rv20(t-1); 200 trees, 7 leaves, min_child 300, 3 seeds, yearly expanding retrain, label lag 3
days; features FEATS18 subset, SPY-relative versions, VIX/VIX3M(t-1), day of week.  Out-of-sample prediction-target
correlation -0.018 (5 of 9 years negative); the model fits one market series.  Best book (top-3 long) 0.63 fold-OOS,
everything <= 0 at 3 bps.

## 3. Overnight / intraday split of SPY QQQ IWM DIA by regime — dead
Fixed rule (overnight long if VIX/VIX3M(t-1) < 1 and SPY 21-day trend > 0; intraday long at the turn of the month,
short when the ratio >= 1): 0.48 fold-OOS, -0.24 at 3 bps.  The index-ETF overnight premium has shrunk (7.1 -> 2.7
bps/day from the first to the second half).

## Verdict
Nothing for the portfolio: the ETF book is the same overnight-reversal premium the stock rule book already harvests, at
a higher cost per unit of edge, with negative marginal value; dead at c = 0.3.  The two candidates are kept for the
record only (NOTES.txt says NOT recommended); their series already include commission (folds.py treats the family as
commission-free).
