# Multi-sleeve US-equity portfolio under realistic, feasible execution: fold-OOS Sharpe 2.1, holdout 1.9

Daily long/short and long-only books on US stocks and ETFs built from Polygon (Massive) daily, minute
and NBBO quote data, combined with inverse-vol weights.  Every variant choice and every weight is made
on dates before each half-year test window; the last half year (2026-04-28..2026-10-28) is a holdout
never used for a decision.  Costs are the user's $3 per filled order on $0.4M of equity per book, plus
a modelled execution cost per fill: auction impact for market-on-close / limit-on-close / market-on-open
orders, a calibrated NBBO half-spread for continuous-market orders (section "Cost model").  Every order
uses only information available when it must be sent: MOC and LOC orders are decided on data to
15:49:00 (exchange cut-offs 15:50 NYSE / 15:55 Nasdaq).

| portfolio (picks and weights from prior dates only, $3 per order, $0.4M per book, inverse-vol) | dev folds 2018-07..2026-04 | holdout 2026-04-28..10-28 |
|-----------------------------------------------------------------------------------------------|----------------------------|---------------------------|
| **A: feasible timing — overnight MOC + LOC books (15:49 cut), daily ML, momentum, VIX (6 families)** | **2.14** (ret 18.2%, vol 8.5%, maxDD -9.1%) | **1.91** (ret 21.0%, vol 11.0%, maxDD -3.6%) |
| B: A + passive-limit intraday book (fragile, see `lmtday`)                                     | 2.26                       | 1.91                      |
| reference: previous headline, 15:59 signal (and the official close) into the MOC — not executable | 2.30                   | not comparable (*)        |

Dev fold Sharpes (A): mean 2.19, median 2.29, min -2.06 (2026 Q1), two negative folds of 16; by year
2018 -1.2, 2019 2.1, 2020 4.5, 2021 3.0, 2022 0.9, 2023 1.3, 2024 4.3, 2025 2.4.  The holdout is 127
trading days (standard error of a Sharpe ~1.4).  Full output: `out/final_eval_phase8.txt`; reproduce with
`bash final_eval7.sh --show_holdout`.

(*) **Holdout correction.**  Earlier README versions reported a holdout of 2.37 / 2.52.  Those reads were
wrong: the auction-executed overnight and rule books (onml3/onml4/xsrule3) were only ever built to
2026-04-27, and `folds.py` gave those sleeves their weight with zero return in the holdout, so the
figure measured the daily sleeves alone, diluted.  The A line above is the first holdout read with every
sleeve's series built through 2026-10-28 (onml6 and dml2 regenerated with `STRAT_END=2026-10-28`,
bit-identical on all dev dates; scripts in `agents/regen7`).  The reference line's holdout still has the
zero-return defect (its onml4x/xsrule3 series were not regenerated) and is not shown.

**What the timing fix cost.**  The previous overnight books computed their signal from minute bars to
15:59 and entered 96% of positions in the closing auction, which cannot be done (MOC/LOC entry closes
at 15:50 / 15:55).  Worse, several "15:59" features (zoc, zcc, cc5, cc21, rv20, rngpos, rng, zvwap,
spy_oc) silently contained the official close of day t, whose print reverts overnight.  Rebuilt at a
15:49:00 cut (`agents/onml6`), the MOC book falls from fold-OOS 1.80 to 1.21-1.29 (a fifth of the loss
is the close leak, the rest the last ten minutes); a limit-on-close at the estimated 15:49 bid
recovers 1.62-1.70.  The rule overnight book does not survive the cut (1.39 -> 0.42) and is dropped.

**Date alignment.**  The overnight stock books (`rlib.book` convention) are dated on the decision day
t, the daily sleeves on the exit day; `folds.py` re-dates `DECISION_DATED` families to their exit day
(`--no_align` restores the old behaviour), which lowered every portfolio number by 0.3-0.5.

The single most important unresolved input is still the auction-impact coefficient c: at c = 0.1 the
15:49 MOC book is ~2.0-2.3 fold-OOS, at c = 1.0 every MOC book is negative and only the LOC books stay
positive (0.8-1.0).

## Cost model

* **Daily-bar sleeves and ETFs**: 15 bps per side on the monthly momentum book, 2 bps on the VIX
  spread, 1.5 bps on the VIX ETPs (their half-spread is 2-4 bps).  The daily ML sleeve (dml2) is
  costed with the auction model below and its own order count.
* **Borrow** 1%/yr on every short (2% VIXY, 5% UVXY).
* **Commission** $3 per filled order (user's figure).  A long/short top-k minute-bar book holds 2k
  positions and pays 2k x 2 x $3 per day; on $0.4M of equity a 6+6 book costs 1.8 bps/day.
  `folds.py --capital_usd 400000 --order_usd 3` subtracts this from candidates whose name carries the
  count (`_k6`); books whose fills are partial (LOC, passive limits) or whose order count varies (dml2)
  carry their commission inside the series (names `..._kSix` / `..._kFour`, families `dml`, `lmt*`,
  exempt in `npos()`); unfilled orders pay nothing.  Momentum, VIX spread and VIX ETPs trade too few
  orders to matter.
* **Execution model (`liq.py`).**  Auction orders (MOC/LOC entry, MOO exit) pay
  `max(1 bp, c x daily vol in bps x sqrt(order / expected auction dollar volume))` per side, with the
  opening-auction volume proxied by the lagged 20-day mean dollar volume of the 09:30 bar, the closing
  one by max(16:00-bar volume, 5% of daily volume), order = $400k / positions, and c = 0.3 as the central
  assumption (0.1 and 1.0 reported).  Continuous-market orders pay a fraction f of the predicted NBBO
  half-spread at that time of day (f = 1 crosses the spread); the half-spread is predicted per name and
  day by a LightGBM fit on 497k measured quotes (section "NBBO data").  A 15:59:30 cross pays a further
  1.85 bps: on the books' own entries the real NBBO fill was 1.22 bps (t 3.7; longs 1.85) worse than
  modelled (`cross1559_check.py`, `agents/onml4/cross_check_positions.py`).
* **Limit orders.**  A resting limit at the bid (sell: ask) is filled only if a print goes strictly
  through it (queue position unknown); a limit-on-close is filled iff the official close is strictly
  through the limit.  See `lmt` and `onml6`.

## Sleeves (families) and what the chooser picks

`final_eval7.sh` holds six families.  In each fold `folds.py` picks, per family, the candidate with the
best Sharpe over all prior dates (5-day gap, >= 200 non-zero days) and weights the families by 1/vol of
that prior window.  Picks and weights in the last dev fold (2026-01..04), family fold-OOS stitched over
the dev folds at $0.4M:

| family   | pick in the last fold                   | weight | family fold-OOS | by year 2018..2026 |
|----------|-----------------------------------------|--------|-----------------|--------------------|
| onmlloc  | onml6_pForty_lamThree_locbid_moo_kSix   | 0.21   | 1.48 (vol 13%, maxDD -14%) | -1.2 0.8 3.3 1.8 0.7 0.3 3.4 1.7 0.9 |
| onml     | onml6_p40c49_lam3_moc_moo_k6            | 0.17   | 1.10 (vol 18%, maxDD -35%) | -0.0 -0.2 4.8 2.9 -0.3 -0.7 3.0 1.2 -0.4 |
| etfmacro | uvxon_r95                               | 0.26   | 0.97 (maxDD -14%) | -0.4 1.4 1.8 1.3 0.1 2.0 1.8 0.6 -0.9 |
| vol      | spread_thr14_thu1                       | 0.18   | 1.03 (maxDD -19%) | 0.0 0.7 1.5 3.2 0.7 1.6 0.0 1.0 0.4 |
| mom      | mom_E4_top25                            | 0.11   | 1.06 (maxDD -30%) | -1.1 1.5 1.9 1.1 -0.2 0.9 2.0 2.0 0.8 |
| dml      | dml2_b5b10_band_n20_r10_b45_H           | 0.07   | 1.11 (vol 41%, maxDD -66%) | -1.5 1.7 0.3 0.2 1.9 0.5 2.7 1.3 1.8 |
| xsrule   | (dropped: 0.42 fold-OOS with a feasible 15:49 signal) | | | |
| dayml / lmtday | (headline: dropped; passive-limit version 0.54-0.67, second half <= 0) | | | |
| ml       | (replaced by dml: the old sleeve re-costed at MOC and $3/order is 0.47) | | | |

(The 2026 column of the stitched family series includes the holdout half year.)

### onml: learned overnight book, 15:49 cut (`agents/onml6`, built on `agents/onml4`, `agents/onml3`)
* Universe `rlib.universe(2000)` (`data/univ1500.txt` + `data/univ2000x.txt`: CS/ADRC names ranked by
  median dollar volume 2016-10..2026-10, survivorship).
* Features: the onml3 sets `f26calx` (26) and `f40_on` (+ 14 liquidity features from `liq.liq_frame`),
  every intraday quantity rebuilt to the 15:48 bar (closing 15:49:00) in `agents/onml6/lib6.py
  cut_frame`: returns end at c[378], "to close" quantities use c[378], volume / range / trade counts
  cover bins 0..378, cc5 / cc21 / rv20 / rv5_20 combine full returns to t-1 with the cut return of t,
  hs1559 is the spread booster fed with the cut inputs.  Target `y_oo` = official open(t+1) /
  official close(t) - 1.  LightGBM (300 trees, 15 leaves, lr 0.03, min_child_samples 200), expanding
  window, quarterly retrain, label lag 3 days, 3 seeds rank-averaged.  Daily rank IC: f40 0.036 at the
  old (leaky) 15:59 cut, 0.033 at 15:59:00, 0.029 at 15:49:00.
* **Cost-aware selection** (`agents/onml4/lib4.py`): the prediction rank is mapped to an expected
  gross return with an expanding, causal calibration; long = E[gross] - 3 x (entry + exit cost), short =
  -E[gross] - 3 x cost - borrow; long the top 6, short the bottom 6, equal weight; ~85% of positions in
  ranks 1-300.  Exit MOO (c = 0.3).
* **MOC books** `onml6_p40c49_lam3_moc_moo_k6` (1.29 fold-OOS, halves 2.30 / 0.53, maxDD -35%),
  `onml6_pR49_lam3_moc_moo_k6` (p40 + p26 rank average, 1.21), `onml6_pR49x59_lam3_two_moo_k6` (MOC at
  15:49, empty slots crossed at 15:59:30 on 15:59 data: 1.22; at hs + 1.85 bps only 3% of slots cross).
* **LOC books (limit-on-close at the estimated 15:49 bid):** same positions, buy limit = round(c378 x
  (1 - hs49 / 1e4), 2) (sell: +), hs49 = predicted half-spread on inputs to 15:49:00; filled iff the
  official close is strictly through the limit, at the close + MOC impact; unfilled = no trade.  Fill
  ~45%; filled names make 12.9 bps gross overnight vs 3.8 unfilled (the 15:49 -> close move reverts).
  `onml6_pForty_lamThree_locbid_moo_kSix` 1.70 fold-OOS (halves 1.82 / 1.60, maxDD -13%),
  `onml6_pRcut_lamThree_locbid_moo_kSix` 1.62; 0.8-1.0 at c = 1.0.  Checked on 3,346 real 15:49:00 NBBO
  quotes: proxy bid - real bid median 0.00 bps, fill agreement 95-97%, the proxy slightly conservative.
* Sensitivity (fold-OOS at c = 0.1 / 0.3 / 1.0): MOC p40 2.26 / 1.29 / -0.93; LOC p40 2.30 / 1.70 / 0.80.
* Dead / superseded: onml3 / onml4 / onml4x / onml5 (15:59 signal into the MOC; onml5's liquid-name
  features gave +0.14 on one book and nothing at the portfolio level); routed exits at 09:35-10:31 and
  passive exits at the open (`lmt`: the fills are the losers); a 15:54 Nasdaq cut (no causal listing
  exchange in the data).  Cost bug fixed in onml4: `rlib.clean` filled NaN `rv20` with 0, giving
  short-history names the 1 bp auction floor.

### lmt: passive limit orders at the touch (`agents/lmt`)
Posting at the bid (sell: ask), filled only on a print strictly through the limit, unfilled = no trade,
$3 per filled order.  Exact fills on the 59 NBBO-panel days, a calibrated minute-bar proxy on the full
history.  In the continuous session the fills are adversely selected: a 09:31 buy at the bid fills 45%
of the time, and the filled names move -31 bps vs +29.5 unfilled.  On the intraday book this lifts the
fold-OOS from -3.8 (crossing) to 0.54 (user's rule: bid entry, sell at the bid) / 0.67 (passive both
legs), with a non-positive second half (`lmtday` family, line B).  Into the closing auction it is the
other way round (closes printing below the bid revert overnight), which is what the onml6 LOC books use.

### dayml: learned intraday book (`agents/dayml3`, built on `agents/dayml`, `agents/dayml2`): not tradable at the spread
Decide at the 09:31 bar, fill at the 09:33 bar, exit at the 10:01 bar, L/S top/bottom k, quarterly
LightGBM on 43 features of the previous close and the first minute plus 17 liquidity features.  The
gross edge is 10-16 bps per trade; the predicted round-trip spread cost is 20 / 34 / 53 bps in ranks
1-300 / 301-600 / 601-1500.  Result: nothing is positive when crossing the spread on any universe
(best -0.5 fold-OOS; the old `dayml2_ens_e30_k8` re-costed is -7.6); at half the half-spread only a
cost-aware 4-name book on ranks <= 600 survives (`dayml3_liqsel_u600_x05_k4`: 1.03 / 0.77, flat since
2021).  The liquidity features raise the mid-quote book from 2.4 to 3.0 (old convention) but push it
into thin names.  The family is dropped from the headline portfolio; its half-spread variants are in
`agents/dayml3/cands` and the second line of the headline table.

### lgb_on: previous learned overnight book (`agents/intraday`, `on_ml.py`) — reference only
LightGBM on FEATS18, 1200 names, quarterly, L/S top 20, close -> 09:31 bar.  No auction-executed
version was built; it is in the old-convention reference run only.

### xsrule: rule overnight book (`agents/xsrule3`, built on `agents/xsrule`) — dropped: needs the 15:59 signal
At a 15:49 cut (`agents/onml6/xs6.py`) the 50-name book falls from 1.39 to 0.42 fold-OOS (150 names 1.08 -> 0.12).  The 15:59 version, for reference:
Signal with fixed, untuned weights, all inputs known by the 15:59 bar:

    cAx = 0.5*z30 - z60 - 0.5*z5l + 0.3*zon_prev - rank(vlast_x)

z30 = first-30-minute return, z60 = last-60-minute return, z5l = last-5-minute return (each over its
lagged 20-day sd), zon_prev = previous night's z-score, vlast_x = cross-sectional rank of the last
hour's share of continuous-session volume.  Long the top 10 with cAx > 1.5, equal weight, MOC entry,
MOO exit.  Kept: `xsr3_cAx_u50_moo_lo10` (50 most liquid names: 1.62 / 1.39 fold-OOS, the only one
positive at c = 1.0), `xsr3_cAx_u150_fhs10_moo_lo10` (150 names, skip predicted hs0931 > 10 bps:
1.54 / 1.28), `_u150_moo_lo10` (1.29 / 1.08), `_lo20`, a half-spread 09:31-exit variant and one L/S
(0.64).  The edge to the official open equals the edge to the 09:31 bar (15 bps per pick); breadth
beyond 150 names fails with the auction cost (300: 0.46, 600: negative); a liquidity tilt inside the
signal adds nothing.  Family fold-OOS 1.06-1.12; 2026 H1 negative for every realistic variant.

### etfmacro: VIX-ETP overnight book (`agents/etfmacro`)
VIX-ETP decay is an overnight phenomenon (VIXY close->09:31 -11 bps/night, 09:32->close -1 bps).
`vixon_r95_ls`: short 0.5 VIXY from the close to the 09:31 bar when VIX/VIX3M of the previous
close (FRED) < 0.95, no Friday decisions (the weekend loses); plus long 0.5 VIXY overnight when
the ratio >= 0.95 and SPY's intraday signal 0.5*z30 - z60 < -1.  Core variants without the long
leg: `vixon_r95`, `uvxon_r95` (0.25 UVXY, 5% borrow), and 15:45-bar entries.  Fold-OOS 1.15,
correlation ~0 with the stock books.  VIXY's half-spread is 2-4 bps, so the 09:31-bar exit at 1.5
bps/side is kept; exiting at the official open loses a third of the overnight decay (1.06).

### mom: vol-scaled momentum, long-only (`strategy.py`, grid `mom_grid.py`)
Universe at each rebalance: CS/ADRC, close >= $5, 20-day average dollar volume >= $50M, >= 1 year
of history.  Score = sum over (lookback, skip) in {(252,21),(252,42),(210,21),(189,21)} of the
cross-sectional rank of log(close[t-skip]/close[t-lookback]) / (60-day vol x sqrt 252).  Hold the
top 25 equal weight; two 50% tranches rebalanced monthly (month end, and 10 trading days later),
traded at the next close at 15 bps.  Grid: lookback sets E4/E2/S252/S126 x top 15/25/40.

### vol: VIX futures-ETP spread (`agents/vol`, grid `agents/vol/grid.py`)
Short 0.5 VIXY / long 0.5 VIXM, close-to-close, only when SPY's 10-day realised vol > 14%
(variants: no threshold, 10%, 12%), skipping Thursday-decided holds; 2 bps per side plus borrow.

### dml: daily ML sleeve at MOC cost (`agents/dml2`, replaces `agents/ml`)
The h5 + h10 LightGBM models (126-day retrain, 59 daily features), score smoothed 5 days, traded at the
close of t on information to the close of t-1, auction impact c = 0.3, $3 per order sent and borrow inside
the series.  Rank-hysteresis "band" books (enter in the top/bottom k, leave only past k + buffer,
rebalance every r days, SPY hedge capped at 0.5 beta) send ~4 orders a day instead of 64:
`dml2_b5b10_band_n20_r10_b45_H` 0.92, `_n15_r10_b20_H` 0.95 fold-OOS; vol 37-47%, maxDD -62% to -72%.
The old sleeve below, re-costed the same way, is 0.47 (commission was 78% of its cost).

### ml: daily-bar cross-sectional LightGBM (`agents/ml`) — superseded by dml
Ensemble of three LightGBM models (1-, 5- and 10-day close-to-close targets) on daily-bar
features over names with 20-day ADV >= $50M, retrained every 126 days; long top 25 / short bottom
25, 5 daily tranches (5-day hold), score smoothed over 5 days, SPY-hedged, 10 bps per side.

## Validation protocol (`folds.py`)

* Test windows: half years from 2018-07-01; 16 dev folds end 2026-04-27, the holdout fold
  2026-04-28..2026-10-28 is evaluated only with `--show_holdout`.
* Candidate series are causal on their own (expanding-window retrains, lagged features); the
  script removes the choice-of-variant and choice-of-weight look-ahead by selecting on prior
  dates only (`--sel all`, 5-day gap).
* `final_eval7.sh` is the headline run (`--lanes`, old minute-bar families, xsrule and ml dropped,
  etfmacro `_ls` omitted because its long leg uses a 15:59 SPY signal); `final_eval.sh` is the
  old-convention reference.  Both fix the candidate grid: `--omit` drops variants that are superseded, use
  close-only information (`cBx`, `onml_f28cal_q_k20`), or are redundant 15:45 entries; `--drop
  intraday,rule_e1,rule_late` removes the old 09:32-fill intraday family and the two old rule
  families superseded by xsrule.
* Dates: overnight stock books are re-dated to their exit day so all sleeves are summed on the day
  their PnL is realised (`DECISION_DATED` in `folds.py`).
* Dead at the portfolio level (dev folds, aligned): averaging the top-2/3 variants per family,
  Sharpe-weighted, risk-parity, min-variance and max-Sharpe weights, trailing-year selection, a
  minimum prior Sharpe, hedging the rule book's overnight beta with SPY (fold-OOS 1.76 -> 0.57 alone),
  hedged momentum, and an ETF overnight sleeve (a costlier copy of the stock rule book, residual
  Sharpe -0.64 against it).
* Everything the agents did ran with `STRAT_END=2026-04-27` (`rlib.DEV_END`), so the loaders
  could not read holdout dates; the final regeneration runs with `STRAT_END=2026-10-28`, and the
  dev-date series are identical.

What remains in-sample: the sleeve families, feature sets, LightGBM parameters, exits and k values
were chosen on the dev span (never on the holdout); the 1500/2000-name universe is ranked by
2016-2026 median dollar volume (survivorship; names ranked 1501-2000 trade $3-11M a day); the
overnight books now decide at 15:49:00 (fixed, see onml); the LOC limit placement, lam, k and p40 over
pR were chosen on the dev span; both learned books have weaker second halves (overnight 4.8 -> 3.3,
intraday 4.2 -> 1.4); the auction-impact coefficient and the spread model are calibrations, not
out-of-sample quantities, and the cost-aware selection's calibration map is causal but its lambda = 3
was chosen on the dev span.

## NBBO data: the fill check, the quote panel and the spread model

**At the NBBO, the three minute-bar books do not survive.**  `quotes_check.py` takes the actual
positions of the three new books on every 10th trading day 2017-01-03..2026-10-16 (245 days, 10,806
trades, 21,612 quote lookups) and looks up the last Polygon NBBO at or before the decision time
(10-second window, widened to 60 s / 5 min / 15 min if empty; 99.9% found, median quote age 1-3 s).
Buys fill at the ask, sells at the bid.  Rows in `out/quotes_trades.csv`; summary at the end of
`logs/quotes_check.log`.

| book, entry -> exit (quote times)                  | median half-spread entry / exit | bps per trade: backtest gross / mid-to-mid / MOC entry + mid exit / MOC entry + NBBO exit / NBBO both | Sharpe-equivalent on the sampled days: backtest / MOC+mid / MOC+NBBO / NBBO both |
|----------------------------------------------------|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| onml_k10 (2000 names), 15:59:30 -> 09:31:00        | 3.8 / **21.4** bps              | 22.2 / 14.5 / 24.3 / **-7.9** / -25.6                                                                 | 2.47 / 3.32 / **-1.16** / -3.65                                                   |
| dayml_e30_k8 (1500 names), 09:33:00 -> 10:01:00    | **20.0** / 7.8 bps              | 16.5 / 15.0 / n.a. / 6.4 (bar entry, NBBO exit) / **-25.7**                                           | 3.44 / 3.68 (mid both) / 1.60 / **-6.10**                                         |
| xsr_lo10 (150 names), 15:59:30 -> 09:31:00         | 1.0 / 6.2 bps                   | 10.3 / 7.6 / 9.9 / **-0.0** / -3.7                                                                    | 1.32 / 1.55 / **-0.03** / -0.61                                                   |

Reading:
* **The signals are real at the mid.**  Mid-to-mid returns match the bar-based backtest (overnight
  book: market-on-close entry + 09:31:00 mid exit 24.3 bps/trade vs 22.2 in the backtest; intraday
  15.0 vs 16.5; rule book 9.9 vs 10.3), so the minute-bar fills are not biased.  The whole gap is
  the spread.
* **The spread at 09:31 is 10x the assumed cost.**  The names the overnight model picks have a
  median NBBO half-spread of 21 bps at 09:31:00 (mean 32; 7.5 bps in ranks 1-300, 16 in 301-600,
  26-27 beyond 600) against the 1-2.5 bps per side in the series.  The intraday book's names have a
  20 bps median half-spread at 09:33:00 (36 bps mean beyond rank 600) and 8 bps at 10:01.  Spreads
  have widened: overnight exit half-spread mean 31 bps in 2017-24, 38 bps in 2025-26; intraday
  entry median 14 bps in 2017, 32 bps in 2026.
* **Breakeven spread capture.**  With a market-on-close entry (no spread) the overnight book earns
  24 bps/trade against a 32 bps mean exit half-spread: it is profitable only if the exit pays less
  than 76% of the half-spread (paying 50%: 15 bps/day, Sharpe-equivalent 1.1).  The rule book on
  150 names breaks even at 99% of its 10 bps exit half-spread (50%: Sharpe 0.8).  The intraday book
  pays two spreads (41 bps of half-spreads for a 15 bps edge) and breaks even at 37%; paying 50% of
  both half-spreads it is negative.
* By tier, only the most liquid names stay positive when crossing the spread: overnight ranks
  1-300 +24 bps/trade at MOC entry + NBBO exit (+19 crossing both), intraday ranks 1-300 +17 bps
  with a bar entry and NBBO exit (+6 crossing both); everything beyond rank 600 is negative.

**Unbiased quote panel (`quotes_panel.py`, `out/quotes_panel.csv.gz`).**  Every 40th trading day
2017-01-03..2026-10-16 (62 days), every name of the 2000-name universe plus the ETFs, the last NBBO at
or before 09:31:00, 09:35:00, 10:01:00 and 15:59:30: 497k lookups, 467k valid quotes.  Median
half-spread in bps:

| rank tier | 09:31 | 09:35 | 10:01 | 15:59 | 09:31 in 2017 / 2024 / 2026 |
|-----------|-------|-------|-------|-------|-----------------------------|
| 1-300     | 9.7   | 6.0   | 3.2   | 1.1   | 7.7 / 10.5 / 19.5           |
| 301-600   | 19.3  | 11.8  | 5.7   | 1.7   | 15.1 / 21.9 / 37.9          |
| 601-1500  | 38.0  | 23.8  | 10.8  | 2.9   | 30.2 / 39.7 / 62.0          |
| 1501-2000 | 52.4  | 36.9  | 17.3  | 4.7   | 55.9 / 46.3 / 65.5          |
| ETFs      | 2.7   | 2.4   | 2.5   | 2.2   |                             |

The closing-auction entry is cheap everywhere; the 09:31 exit is the problem, and it has roughly
doubled in the liquid tiers since 2024 while the 15:59 spread is unchanged.

**Spread model (`liq.fit_spread_model`, `data/spread_model.pkl`).**  LightGBM on log half-spread per
time of day from lagged minute-bar proxies (log dollar volume, price, 20-day means of the 09:31-bar
and last-10-minute ranges, a Roll estimate on the first 30 one-minute returns, trades per minute at
the open and close, opening/closing-bar dollar volume) plus same-day bins 0-1 (09:31/10:01) or bins
380-389 (15:59); trained on the panel and the fill-check trades (120k rows per time), with a
retransformation factor so the mean in bps is unbiased.  Ticker-grouped 5-fold CV R² of the log
spread 0.78 / 0.78 / 0.77 / 0.78 at 09:31 / 09:35 / 10:01 / 15:59 (the 09:35 model, added later, via `liq.py add0935`), MAE 17 / 12.9 / 5.6 / 1.3 bps; it under-predicts the
2025-26 widening by about 20% (the proxies do not see it).  It is a cost calibration fitted on the
whole span, so the predicted spreads are in-sample as costs; they are used both as features and as
the per-name, per-day cost of continuous-market fills.

**Auction impact** is not observable from quotes.  The sqrt model with c = 0.3 is an assumption in
the range of published closing-auction estimates; the paper that would anchor it (Goyal, Jegadeesh,
Wu, JFQA 2025, "Price Impact in Closing Auctions, Opening Auctions, and Continuous Markets") is on a
host outside this environment's network policy.  Measuring the actual auction prints against the
pre-auction mid for the books' names is the most valuable next step.

## Data pipeline (Polygon / Massive, key in `POLYGON_API_KEY`; `data/` is gitignored)

1. `fetch_ref.py`: splits, dividends, tickers, grouped daily bars 2014-06..2026-10-28.
2. `build_panel2.py` -> `data/panel_liquid.npz` (dates x tickers, split-adjusted, dividends).
3. `universe.py --n 1500 --out data/univ1500.txt`: CS/ADRC tickers ranked by median daily dollar
   volume 2016-10-28..2026-10-28, present on >= 80% of sampled days; `data/univ2000x.txt` extends
   it to 2000.
4. `fetch_minute3.py --out data/minute_npz --tickers data/minute_list.txt`: minute bars
   2016-06..2026-10-28, one npz per ticker (o h l c vw vol n over 391 bins per day); 2029 files
   (2000 names + ETFs), 20 GB.
5. `rlib.ticker_frame` caches per-ticker close-time features and split-adjusted targets in
   `data/cache/{tk}.v2.parquet`; `agents/onml/final2000.py` adds `data/cache_x` (ex-auction
   volume features); `agents/dayml/dlib.py` keeps its 09:31 features in `data/cache_dayml2*`.
6. `liq.py build` -> `data/cache_liq/{tk}.parquet` (liquidity proxies), `liq.py fit` ->
   `data/spread_model.pkl`; `quotes_panel.py` -> `out/quotes_panel.csv(.gz)`.
7. VIX / VIX3M daily closes from FRED (`agents/vol/fred_vix.csv`); the Polygon index feed is not
   in the subscription.

## Regenerating the candidate series

```
export STRAT_END=2026-10-28
python3 agents/onml/final2000.py            # ~50 min, 3 quarterly walk-forwards on 2000 names
python3 agents/onml2/final.py
python3 agents/onml/final.py                 # 1500-name f28cal candidates (~50 min)
python3 agents/dayml/final.py && python3 agents/dayml/preds.py && python3 agents/dayml/small.py
python3 agents/dayml/final2000.py && python3 agents/dayml/u2000_e30.py
python3 agents/dayml2/final.py --from-saved
python3 agents/xsrule/final.py && python3 agents/xsrule/k10.py
python3 agents/etfmacro/final.py
python3 liq.py build && python3 liq.py fit            # liquidity cache (data/cache_liq) and spread model
python3 agents/onml3/step4.py                          # cost-aware overnight books (needs the onml predictions; ~2 h)
python3 agents/onml4/step4.py                          # routed pR books (saved predictions, no refit)
python3 agents/xsrule3/s4_save.py                      # auction-executed rule books
python3 agents/dayml3/final.py                         # spread-costed intraday books
python3 agents/onml6/train6.py ... && python3 agents/onml6/final6.py   # 15:49-cut frames, predictions, MOC/LOC books (see agents/onml6/log.txt)
python3 agents/dml2/s1_train.py && python3 agents/dml2/s5_final.py      # daily ML sleeve at MOC cost
#   holdout extension of onml6 / dml2 (walk-forward continued, dev dates bit-identical): agents/regen7/*.py
bash final_eval7.sh --show_holdout > out/final_eval_phase8.txt          # headline
bash final_eval.sh --show_holdout > out/final_eval_phase4.txt   # old-convention reference
```
Candidate series live in `agents/<lane>/cands/*.csv` (date, daily net return at the spread cost)
with a `NOTES.txt` per lane; each lane has a `REPORT.md` with the full exploration record and the
dead ends.  `rlib.py` is the shared library (universe, costs, feature cache, walk-forward
LightGBM, `book()`, `fold_oos()`); `agents/BRIEF.md` is the protocol the agents worked under.

## Layout

* `rlib.py`, `liq.py`, `folds.py`, `final_eval7.sh` (headline; `final_eval6.sh` previous) / `final_eval.sh`: library, liquidity and
  execution-cost module, fold validation, final evaluation (realistic / old convention).
* `agents/onml4`, `agents/onml3`, `agents/xsrule3`, `agents/dayml3`: the realistic-execution lanes (code,
  reports, candidates); `agents/BRIEF6.md` their brief.  `agents/etfon` (ETF overnight books) and
  `agents/hedge7` (SPY-hedged rule books) are dead ends kept for the record.
* `agents/onml`, `agents/onml2`, `agents/dayml`, `agents/dayml2`, `agents/xsrule`,
  `agents/etfmacro`: the active lanes (code, reports, candidates).
* `agents/intraday` (`ilib.py`, `logs/`): the previous minute-bar framework; its `onml_a1200q_*`
  series are still a family.  `agents/vol`, `agents/ml`, `logs/mom_grid`, `logs/vol_grid.csv`:
  the vol, daily-ML and momentum families.
* `strategy.py`, `bt.py`, `feats.py`, `portfolio.py`, `sleeves/`: momentum engine and the older
  fixed-weight portfolio (reference only; its in-sample picks score 3.68 on the dev span).
* `quotes_check.py`, `out/quotes_trades.csv`, `quotes_panel.py`, `out/quotes_panel.csv.gz`: NBBO fill
  check and quote panel.
* Dead lanes kept for the record: `agents/pairs`, `etf_mr.py`, `zoo.py`, `on_ml_pm.py`,
  `regime_var.py` (pairs / residual reversal, ETF spreads, factor zoo, last-hour books, regime
  filters: all net <= 0.5 or negative).
