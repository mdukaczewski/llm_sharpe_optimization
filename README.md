# Multi-sleeve US-equity portfolio under realistic execution: fold-OOS Sharpe 2.5, holdout 2.3

Daily long/short and long-only books on US stocks and ETFs built from Polygon (Massive) daily, minute
and NBBO quote data, combined with inverse-vol weights.  Every variant choice and every weight is made
on dates before each half-year test window; the last half year (2026-04-28..2026-10-28) is a holdout
never used for a decision (read three times in the project's life, the last time for the table below).
Costs are the user's $3 per filled order on $0.4M of equity per book, plus a modelled execution cost
per fill: auction impact for market-on-close / market-on-open orders, a calibrated NBBO half-spread
for continuous-market orders (section "Execution model").

| portfolio (picks and weights from prior dates only, $3 per order, $0.4M per book, inverse-vol) | dev folds 2018-07..2026-04 | holdout 2026-04-28..10-28 |
|-----------------------------------------------------------------------------------------------|----------------------------|---------------------------|
| **auction-executed overnight books + daily sleeves (6 families)**                             | **2.52** (ret 19.9%, vol 7.9%, maxDD -11.5%) | **2.25** (ret 13.1%, vol 5.8%, maxDD -2.3%) |
| same + the intraday book assuming fills at half the half-spread                               | 2.69                       | 2.25                      |
| same as the first line, $2M per book                                                          | 2.68                       | 2.25                      |
| reference: the same signals at the old 1-2.5 bps/side fill assumption (8 families)            | 4.32                       | 3.51                      |

Dev fold Sharpes (first line): mean 2.50, median 2.45, min -1.76 (2026 Q1), one negative fold of 16;
by year 2018 0.1, 2019 2.3, 2020 5.6, 2021 3.1, 2022 0.5, 2023 1.6, 2024 3.8, 2025 3.3.  The holdout is
126 trading days (standard error ~1.4).  Full output: `out/final_eval_phase6.txt` (four variants);
reproduce with `bash final_eval6.sh --show_holdout`.  The old-convention run is `out/final_eval_phase4.txt`
(`bash final_eval.sh --show_holdout`): it measures signal quality at the mid-quote, not achievable
performance, because the NBBO data show a 10-50 bps half-spread at 09:31 in the names those books trade.

The gap between 4.3 and 2.5 is entirely execution: the learned overnight book drops from 4.0 to 1.6-1.8
once it must trade through the auctions with an impact charge and is pushed into liquid names, the
intraday book is negative when it pays the spread, and the rule book loses a third.  The single most
important unresolved input is the auction-impact coefficient c (below): at c = 0.1 the overnight sleeve
is ~2.2 and the portfolio ~2.8; at c = 1.0 the overnight books are negative.

## Cost model

* **Daily-bar sleeves and ETFs**: 15 bps per side on the monthly momentum book, 10 bps on the
  daily-bar ML book, 2 bps on the VIX spread, 1.5 bps on the VIX ETPs (their half-spread is 2-4
  bps).  The old stock-book convention (`rlib.cost_bps`: 1 / 1.5 / 2.5 bps per side by rank tier)
  survives only in the reference run.
* **Borrow** 1%/yr on every short (2% VIXY, 5% UVXY).
* **Commission** $3 per filled order (user's figure).  Every minute-bar book is fully turned over
  every day, so a long/short top-k book holds 2k positions and pays 2k x 2 x $3 per day; on $0.4M of
  equity a 10-name L/S book costs 3.0 bps/day, a 20-name one 6.0 bps/day, an 8-name one 2.4
  bps/day, a 10-name long-only one 1.5 bps/day.  `folds.py --capital_usd 400000 --order_usd 3`
  subtracts this from every minute-bar candidate (position count parsed from the candidate name);
  momentum, VIX spread, daily ML and the VIX-ETP books trade too few orders to matter.
* **Execution model (phase 6, `liq.py`).**  Auction orders (MOC entry, MOO exit) pay
  `max(1 bp, c x daily vol in bps x sqrt(order / expected auction dollar volume))` per side, with the
  opening-auction volume proxied by the lagged 20-day mean dollar volume of the 09:30 bar, the closing
  one by max(16:00-bar volume, 5% of daily volume), order = $400k / positions, and c = 0.3 as the central
  assumption (0.1 and 1.0 reported).  Continuous-market orders pay a fraction f of the predicted NBBO
  half-spread at that time of day (f = 1 crosses the spread, 0.5 is the passive case); the half-spread
  is predicted per name and day by a LightGBM fit on 497k measured quotes (section "NBBO data").  The
  old convention (1 / 1.5 / 2.5 bps per side by rank) is kept only for the reference line.

## Sleeves (families) and what the chooser picks

`folds.py --lanes onml=onml3,xsrule=xsrule3,etfmacro=etfmacro` (as in `final_eval6.sh`) holds six
families.  In each fold it picks, per family, the candidate with the best Sharpe over all prior dates
(5-day gap, >= 200 non-zero days) and weights the families by 1/vol of that prior window.  Picks and
weights in the last dev fold (2026-01..04), family fold-OOS at $0.4M with the realistic costs:

| family   | pick in the last fold                 | weight | family fold-OOS (realistic) | old convention | by year 2018..2026 (realistic) |
|----------|----------------------------------------|--------|------------------------------|----------------|--------------------------------|
| onml     | onml3_f26calx_q3_lam3_moo_k6           | 0.18   | 1.61 (ret 25%, vol 15%, maxDD -28%) | 3.98 | 2.0 0.6 5.9 3.1 -0.1 -0.3 1.5 2.0 -0.5 |
| etfmacro | vixon_r95_ls                           | 0.15   | 1.15 (maxDD -27%)            | 1.15           | -0.4 1.4 2.6 1.2 0.4 2.0 1.7 0.9 -1.7 |
| mom      | mom_E4_top25                           | 0.11   | 1.06 (maxDD -30%)            | 1.06           | -1.1 1.5 1.9 1.1 -0.2 0.9 2.0 2.0 0.8 |
| xsrule   | xsr3_cAx_u50_moo_lo10                  | 0.28   | 1.06 (maxDD -19%)            | 1.78           | -0.4 1.5 3.0 1.1 -0.5 0.6 2.4 1.2 -1.0 |
| vol      | spread_thr14_thu1                      | 0.19   | 1.03 (maxDD -19%)            | 1.03           | 0.0 0.7 1.5 3.2 0.7 1.6 0.0 1.0 0.4 |
| ml       | ml_k25_h5_d_eq_buf0_H_sm5              | 0.08   | 0.98 (maxDD -42%)            | 0.98           | 0.0 0.0 0.4 -0.7 0.9 0.3 2.3 1.7 2.2 |
| dayml    | (dropped: negative when paying the spread; 0.75 at half-spread, flat since 2021) | | | 2.35 | |
| lgb_on   | (dropped: the 1200-name 09:31-exit book has no auction-executed version)   | | | 1.95 | |

### onml: learned overnight book, auction-executed (`agents/onml3`, built on `agents/onml`)
* Universe `rlib.universe(2000)` (`data/univ1500.txt` + `data/univ2000x.txt`: CS/ADRC names ranked by
  median dollar volume 2016-10..2026-10, survivorship).
* Features known by 15:59 of day t (`rlib.ticker_frame`, `agents/onml/olib.py`): FEATS18 = z30 z60
  z30l z120 zoc zo15 zgap rngpos rng vlast_share vday_rel zcc cc5 rv20 zon_prev spy_z60 spy_z30 spy_oc
  (z-scores = window return / lagged 20-day sd), plus z15l zvwap vfirst_share ldv cc21 rv5_20 spy_zgap,
  plus calendar dow / business days to month end / quarter-end dummy / days since the last |gap| z > 2.5.
  Volume shares are ex-auction (`vlast_x`, `vday_rel_x`, `ldv_x`); the closing print, the auction
  volume share and the last-5-minute return are not used (a MOC order cannot know them).  The
  `f40_on` variant adds 14 liquidity features from `liq.liq_frame` (lagged `ldv20 lpx hl1 hl5 hlc
  roll30 n30 nc30 dv0 dvc`, same-day `c_hlc c_nc30`, predicted half-spreads `hs0931 hs1559`).
* Target: per-day percentile rank of the overnight return close -> 09:31 bar (the official-open
  target `y_oo` gives the same book once liquidity features are in).  LightGBM, 300 trees, 15 leaves,
  lr 0.03, min_child_samples 200, expanding window, quarterly retrain from 2017, label lag 3 days,
  3 seeds rank-averaged per day.
* **Cost-aware selection** (`agents/onml3/lib3.py`): each day the prediction rank is mapped to an
  expected gross return with an expanding, causal calibration (fitted on prior years), and names are
  scored long = E[gross] - 3 x (entry + exit auction cost), short = -E[gross] - 3 x cost - borrow; long
  the top 6, short the bottom 6, equal weight, MOC entry, MOO exit.  This moves 81-84% of positions
  into ranks 1-300 (5 bps net per trade on 8 gross).  Without it the broad book is -5 at c = 0.3:
  75% of its picks are beyond rank 600, where a $20k auction order costs 2-3x the 25-37 bps gross.
* Kept candidates (series net of auction cost at c = 0.3 and borrow, before commission; numbers at
  the $0.4M commission, full dev / fold-OOS): `onml3_f26calx_q3_lam3_moo_k6` 1.85 / 1.95 (halves
  2.5 / 1.3), `_k8` 1.73 / 1.86; `onml3_f40_on_lam3_moo_k6` 1.82 / 1.70 (3.0 / 0.8), `_k8`, `_k10`;
  `onml3_f40_on_lam3_x05_k10` 1.79 / 1.82 (MOC entry, 09:31 exit paying half the half-spread);
  `_x1_k10` 0.83 / 0.86 (crossing).  Sensitivity: c = 0.1 -> 2.0-2.3, c = 1.0 -> negative for every
  MOO book.  Correlation with the old book 0.3-0.4, with SPY ~0.
* Liquidity features as model inputs: +1.4 Sharpe at the mid (old convention 5.3 vs 3.8 for the same
  book), +0.15-0.3 under cost-aware selection, +0.4 for the half-spread exit.  Training on the net
  target (gross minus cost) is dead (it pushes expensive names into the shorts); restricting training
  to liquid names loses data and is worse.  Predictions: `agents/onml3/out/pred_f40_on.parquet`,
  `agents/onml/out/pred_f26calx_u2000_q3.parquet`.

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

### xsrule: rule overnight book, auction-executed (`agents/xsrule3`, built on `agents/xsrule`)
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

### ml: daily-bar cross-sectional LightGBM (`agents/ml`)
Ensemble of three LightGBM models (1-, 5- and 10-day close-to-close targets) on daily-bar
features over names with 20-day ADV >= $50M, retrained every 126 days; long top 25 / short bottom
25, 5 daily tranches (5-day hold), score smoothed over 5 days, SPY-hedged, 10 bps per side.

## Validation protocol (`folds.py`)

* Test windows: half years from 2018-07-01; 16 dev folds end 2026-04-27, the holdout fold
  2026-04-28..2026-10-28 is evaluated only with `--show_holdout`.
* Candidate series are causal on their own (expanding-window retrains, lagged features); the
  script removes the choice-of-variant and choice-of-weight look-ahead by selecting on prior
  dates only (`--sel all`, 5-day gap).
* `final_eval6.sh` is the headline run (phase-6 lanes only, `--lanes`, old minute-bar families
  dropped); `final_eval.sh` is the old-convention reference.  Both fix the candidate grid: `--omit` drops variants that are superseded, use
  close-only information (`cBx`, `onml_f28cal_q_k20`), or are redundant 15:45 entries; `--drop
  intraday,rule_e1,rule_late` removes the old 09:32-fill intraday family and the two old rule
  families superseded by xsrule.
* Everything the agents did ran with `STRAT_END=2026-04-27` (`rlib.DEV_END`), so the loaders
  could not read holdout dates; the final regeneration runs with `STRAT_END=2026-10-28`, and the
  dev-date series are identical.

What remains in-sample: the sleeve families, feature sets, LightGBM parameters, exits and k values
were chosen on the dev span (never on the holdout); the 1500/2000-name universe is ranked by
2016-2026 median dollar volume (survivorship; names ranked 1501-2000 trade $3-11M a day); the
overnight books trade a 15:59 signal at the official close, which needs a market-on-close order
entered by 15:50 (NYSE) / 15:55 (Nasdaq), i.e. the last 5-10 minutes of z60 and z15l are not
available in practice; both learned books have weaker second halves (overnight 4.8 -> 3.3,
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
spread 0.78 / 0.77 / 0.78 at 09:31 / 10:01 / 15:59, MAE 17 / 5.6 / 1.3 bps; it under-predicts the
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
python3 agents/xsrule3/s4_save.py                      # auction-executed rule books
python3 agents/dayml3/final.py                         # spread-costed intraday books
bash final_eval6.sh --show_holdout > out/final_eval_phase6.txt
bash final_eval.sh --show_holdout > out/final_eval_phase4.txt   # old-convention reference
```
Candidate series live in `agents/<lane>/cands/*.csv` (date, daily net return at the spread cost)
with a `NOTES.txt` per lane; each lane has a `REPORT.md` with the full exploration record and the
dead ends.  `rlib.py` is the shared library (universe, costs, feature cache, walk-forward
LightGBM, `book()`, `fold_oos()`); `agents/BRIEF.md` is the protocol the agents worked under.

## Layout

* `rlib.py`, `liq.py`, `folds.py`, `final_eval6.sh` / `final_eval.sh`: library, liquidity and
  execution-cost module, fold validation, final evaluation (realistic / old convention).
* `agents/onml3`, `agents/xsrule3`, `agents/dayml3`: the realistic-execution lanes (code, reports,
  candidates); `agents/BRIEF6.md` their brief.
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
