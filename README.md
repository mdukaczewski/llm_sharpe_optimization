# Multi-sleeve US-equity portfolio: fold-OOS Sharpe 4.3 at $3 per order on $0.4M per book

Daily long/short and long-only books on US stocks and ETFs built from Polygon (Massive) daily and
minute bars, combined with inverse-vol weights.  Every variant choice and every weight is made on
dates before each half-year test window; the last half year (2026-04-28..2026-10-28) is a holdout
that was read twice in the project's life and never used for a decision.  **Caveat first:** these
numbers assume fills at 1-2.5 bps per side; the NBBO check (section "Realistic fills") finds a
median half-spread of ~21 bps at 09:31 in the names the books trade, at which crossing the spread
turns every minute-bar book negative.  The signals hold at the mid-quote; the execution does not.

| portfolio (8 families, picks and weights from prior dates only)   | dev folds 2018-07-01..2026-04-27 | holdout 2026-04-28..2026-10-28 |
|--------------------------------------------------------------------|----------------------------------|--------------------------------|
| **$3 per order, $0.4M equity per book, inverse-vol weights**       | **4.32** (ret 37.6%, vol 8.7%, maxDD -9.1%) | **3.51** (ret 30.5%, vol 8.7%, maxDD -2.8%) |
| $3 per order, $0.4M per book, Sharpe x inverse-vol weights         | 4.66                             | 2.78                           |
| $3 per order, $2M per book, inverse-vol                            | 4.95                             | 3.83                           |
| spread only (1-2.5 bps/side), inverse-vol                          | 5.07                             | 3.92                           |

Dev fold Sharpes at $0.4M (16 half-years): mean 4.50, median 4.74, min 0.37 (2026 Q1), none
negative; by year 2018 4.6, 2019 5.8, 2020 6.5, 2021 4.5, 2022 1.6, 2023 4.0, 2024 5.2, 2025 4.7.
The holdout is 126 trading days (standard error of a half-year Sharpe ~1.4).  Plain inverse-vol
weights hold up out of sample better than Sharpe-weighted ones, which concentrate in the books that
decayed after 2021; use inverse-vol.  Full output: `out/final_eval_phase4.txt`; stitched dev series:
`out/folds_all_oos.csv`.

Reproduce: `bash final_eval.sh --show_holdout` (six runs of `folds.py`: inverse-vol and
Sharpe-weighted, each at spread-only, $0.4M and $2M per book).

## Cost model

* **Spread / impact per side** (`rlib.cost_bps`): 1.0 bps for universe ranks <= 600, 1.5 bps for
  601-1500, 2.5 bps for 1501-2000; 1.5 bps on the VIX ETPs; 15 bps on the monthly momentum book,
  10 bps on the daily-bar ML book, 2 bps on the VIX spread.
* **Borrow** 1%/yr on every short (2% VIXY, 5% UVXY).
* **Commission** $3 per filled order (user's figure).  Every minute-bar book is fully turned over
  every day, so a long/short top-k book holds 2k positions and pays 2k x 2 x $3 per day; on $0.4M of
  equity a 10-name L/S book costs 3.0 bps/day, a 20-name one 6.0 bps/day, an 8-name one 2.4
  bps/day, a 10-name long-only one 1.5 bps/day.  `folds.py --capital_usd 400000 --order_usd 3`
  subtracts this from every minute-bar candidate (position count parsed from the candidate name);
  momentum, VIX spread, daily ML and the VIX-ETP books trade too few orders to matter.
* Fills in the series: official close (market-on-close), the 09:31 bar close, the 09:33 bar close
  and the 10:01 bar close are taken as exact at the spread above.  **The NBBO check below shows the
  real 09:31 half-spread in the chosen names is ~21 bps, not 1-2.5; read the headline table as
  signal quality at the mid, not as achievable net performance.**

## Sleeves (families) and what the chooser picks

`folds.py` holds eight families.  In each fold it picks, per family, the candidate with the best
Sharpe over all prior dates (5-day gap, >= 200 non-zero days) and weights the families by 1/vol
of that prior window (a family with negative prior Sharpe gets weight 0).  Picks and weights at
$0.4M in the last dev fold (2026-01..04):

| family   | pick in the last fold                 | weight | family fold-OOS at $0.4M | by year 2018..2026 |
|----------|----------------------------------------|--------|--------------------------|--------------------|
| onml     | onml_f26calx_u2000_q3_k10              | 0.09   | 3.98 (ret 95%, vol 24%, maxDD -23%) | 6.4 4.3 6.3 4.1 1.7 4.1 3.9 4.3 2.1 |
| dayml    | dayml2_ens_e30_k8                      | 0.12   | 2.35 (ret 49%, vol 21%, maxDD -16%) | 6.0 4.0 4.8 2.6 1.0 1.3 1.1 1.3 1.1 |
| lgb_on   | onml_a1200q_pred_lgb_LS_top20          | 0.13   | 1.95 (maxDD -43%)        | 2.7 3.2 2.7 3.2 0.4 1.4 2.0 3.1 0.6 |
| xsrule   | xsr_cAx_lo10_s15_u150                  | 0.18   | 1.78 (maxDD -19%)        | 1.2 2.6 3.3 1.0 0.4 1.0 2.8 1.0 2.1 |
| etfmacro | vixon_r95_ls                           | 0.14   | 1.15 (maxDD -27%)        | -0.4 1.4 2.6 1.2 0.4 2.0 1.7 0.9 -1.7 |
| mom      | mom_E4_top25                           | 0.10   | 1.06 (maxDD -30%)        | -1.1 1.5 1.9 1.1 -0.2 0.9 2.0 2.0 0.8 |
| vol      | spread_thr14_thu1                      | 0.17   | 1.03 (maxDD -19%)        | 0.0 0.7 1.5 3.2 0.7 1.6 0.0 1.0 0.4 |
| ml       | ml_k25_h5_d_eq_buf0_H_sm5              | 0.07   | 0.98 (maxDD -42%)        | 0.0 0.0 0.4 -0.7 0.9 0.3 2.3 1.7 2.2 |

The picks are stable: from 2021 on every fold chooses the same candidate in every family except
onml (k10 / k15 / k20 of the same predictions) and xsrule (lo10 / lo20 / ls20 of the same signal).

### onml: learned overnight book, 2000 names (`agents/onml`, final script `final2000.py`)
* Universe: `rlib.universe(2000)` = `data/univ1500.txt` (1500 CS/ADRC names ranked by median
  dollar volume 2016-10..2026-10) + `data/univ2000x.txt` (ranks 1501-2000 plus liquid names with
  shorter histories such as BKNG, PDD, NIO).
* Per (day, name) features known by the 15:59 bar, built in `rlib.ticker_frame` and
  `agents/onml/olib.py` ("f26calx", 26 features): FEATS18 = z30 z60 z30l z120 zoc zo15 zgap
  rngpos rng vlast_share vday_rel zcc cc5 rv20 zon_prev spy_z60 spy_z30 spy_oc (z-scores are the
  return over the window divided by its lagged 20-day sd); plus z15l zvwap vfirst_share ldv cc21
  rv5_20 spy_zgap; plus calendar dow, business days to month end, quarter-end dummy, days since the
  last |gap| z-score > 2.5.  The volume shares vlast_share / vday_rel / ldv are the ex-auction
  versions (`vlast_x`, `vday_rel_x`, `ldv_x`, cache `data/cache_x`).  The closing print (zmoc), the
  closing-auction volume share (vmoc_share) and the last-5-minute return (z5l) are NOT used:
  a market-on-close order cannot know them.
* Target: per-day percentile rank of the overnight return from the official close to the 09:31
  bar close (split-adjusted).  Model: LightGBM regression, 300 trees, 15 leaves, learning rate
  0.03, min_child_samples 200, expanding window, quarterly retrain from 2017-01-01, labels lagged
  3 days before each cut; three seeds, predictions rank-averaged per day.
* Book: long the top 10, short the bottom 10 by prediction, equal weight, 100% long / 100% short,
  entered at the close, exited at the 09:31 bar.  k = 10 is the optimum under the fixed commission
  (k8-k12 flat; k15/k20 variants are in the chooser).  Only 4-5% of the names repeat in the
  top/bottom bucket the next day, so nothing is held through.
* Alone at $0.4M: Sharpe 4.3 full dev sample, 3.98 stitched fold-OOS, first/second half 4.8 / 3.3.
* Predictions: `agents/onml/out/pred_f26calx_u2000_q3.parquet` (gitignored).  Also in the family:
  `agents/onml2` (`onml2_f26cal_y_k10`, `y3_k10`, `y_k20`): same recipe with yearly retrain and
  the auction-volume share kept (defensible via the exchange imbalance feeds from 15:50/15:55);
  and the 1500-name f28cal candidates (`onml_f28cal_q3_k10/15/20`) which use zmoc and vmoc_share.

### dayml: learned intraday book (`agents/dayml`, `agents/dayml2`)
* Universe 1500 names.  Features (`agents/dayml/dlib.py` EXT, 43): yesterday's close-time
  features (FEATS18 and extras, shifted one day inside the ticker), today's gap, first-minute
  return and first-minute volume relative to its 20-day mean (all known at the end of the 09:31
  bar), market versions from SPY/QQQ, day of week, lagged log dollar volume.
* Two LightGBM models (same parameters as onml, quarterly expanding retrain, label lag 3 days):
  rank targets `y_60d` = return from the 09:33 bar close to the 10:31 bar close and `y_120d` = to
  the 11:31 bar close.  The ensemble averages the per-day percentile ranks of the two models.
* Book `dayml2_ens_e30_k8`: long top 8 / short bottom 8, equal weight, decided at 09:31, filled at
  the 09:33 bar close (bin 3; the 09:32 bar is never used because single-print minute bars there
  harvest bid-ask bounce), exited at the 10:01 bar close (bin 31).  The 09:31 signal is realised
  in the first 30 minutes; holding to 10:31 or 11:31 only adds variance.
* Alone at $0.4M: 2.67 full dev sample, 2.35 stitched fold-OOS, halves 4.2 / 1.4.  The post-2021
  decay is across the whole cross-section (every liquidity tier halves), not a thin-name effect.
* Predictions: `agents/dayml/out/pred_y_{60d,120d}_Q.parquet`.  Other candidates in the family:
  single-model books k6-k20 with exits at 10:31 / 11:31, 2000-name variants (`*_u2000_*`).

### lgb_on: previous learned overnight book (`agents/intraday`, `on_ml.py`)
LightGBM on FEATS18 only, 1200 names (`data/next600d.txt` chain), quarterly retrain, long/short
top 20 equal weight, close -> 09:31.  Kept because it is only 0.6 correlated with onml (the extra
features change about half the picks) and the chooser keeps giving it weight.  Series in
`agents/intraday/logs/onml_a1200q_pred_lgb_*.csv`.

### xsrule: rule overnight book on 150 names (`agents/xsrule`)
Signal with fixed, untuned weights, all inputs known by the 15:59 bar:

    cAx = 0.5*z30 - z60 - 0.5*z5l + 0.3*zon_prev - rank(vlast_x)

z30 = first-30-minute return, z60 = last-60-minute return, z5l = last-5-minute return (each over
its lagged 20-day sd), zon_prev = previous night's return z-score, vlast_x = cross-sectional
percentile rank (demeaned) of the last hour's share of continuous-session volume.  Book
`xsr_cAx_lo10_s15_u150`: long the top 10 of the 150 most liquid names with cAx > 1.5, equal
weight, close -> 09:31 bar.  Variants in the chooser: top 20, threshold 2.0, and long/short 10 or
20.  Fold-OOS 1.78 at $0.4M; correlation 0.3 with the learned overnight books, 0.44 with SPY
close-to-close for the long-only book.

### etfmacro: VIX-ETP overnight book (`agents/etfmacro`)
VIX-ETP decay is an overnight phenomenon (VIXY close->09:31 -11 bps/night, 09:32->close -1 bps).
`vixon_r95_ls`: short 0.5 VIXY from the close to the 09:31 bar when VIX/VIX3M of the previous
close (FRED) < 0.95, no Friday decisions (the weekend loses); plus long 0.5 VIXY overnight when
the ratio >= 0.95 and SPY's intraday signal 0.5*z30 - z60 < -1.  Core variants without the long
leg: `vixon_r95`, `uvxon_r95` (0.25 UVXY, 5% borrow), and 15:45-bar entries.  Fold-OOS 1.15,
correlation ~0 with the stock books.

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
* `final_eval.sh` fixes the candidate grid: `--omit` drops variants that are superseded, use
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
intraday 4.2 -> 1.4).

## Realistic fills: NBBO check (`quotes_check.py`)

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

Consequence: the portfolio numbers above assume fills at 1-2.5 bps per side and are not achievable
by crossing the spread at 09:31.  What is open: passive exits (limit orders at or inside the mid at
09:31, or later when spreads narrow; the overnight edge was measured to decay by 10:31), restricting
the books to ranks 1-300/600 where the spread is 4-8 bps (fewer names, lower Sharpe before costs),
and re-running the fold evaluation with a per-name, per-time spread model from these quotes instead
of the flat tiers.

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
6. VIX / VIX3M daily closes from FRED (`agents/vol/fred_vix.csv`); the Polygon index feed is not
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
bash final_eval.sh --show_holdout > out/final_eval_phase4.txt
```
Candidate series live in `agents/<lane>/cands/*.csv` (date, daily net return at the spread cost)
with a `NOTES.txt` per lane; each lane has a `REPORT.md` with the full exploration record and the
dead ends.  `rlib.py` is the shared library (universe, costs, feature cache, walk-forward
LightGBM, `book()`, `fold_oos()`); `agents/BRIEF.md` is the protocol the agents worked under.

## Layout

* `rlib.py`, `folds.py`, `final_eval.sh`: library, fold validation, final evaluation.
* `agents/onml`, `agents/onml2`, `agents/dayml`, `agents/dayml2`, `agents/xsrule`,
  `agents/etfmacro`: the active lanes (code, reports, candidates).
* `agents/intraday` (`ilib.py`, `logs/`): the previous minute-bar framework; its `onml_a1200q_*`
  series are still a family.  `agents/vol`, `agents/ml`, `logs/mom_grid`, `logs/vol_grid.csv`:
  the vol, daily-ML and momentum families.
* `strategy.py`, `bt.py`, `feats.py`, `portfolio.py`, `sleeves/`: momentum engine and the older
  fixed-weight portfolio (reference only; its in-sample picks score 3.68 on the dev span).
* `quotes_check.py`, `out/quotes_trades.csv`: NBBO fill check.
* Dead lanes kept for the record: `agents/pairs`, `etf_mr.py`, `zoo.py`, `on_ml_pm.py`,
  `regime_var.py` (pairs / residual reversal, ETF spreads, factor zoo, last-hour books, regime
  filters: all net <= 0.5 or negative).
