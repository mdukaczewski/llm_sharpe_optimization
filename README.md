# strat20 — vol-scaled momentum ensemble (long-only US stocks)

Result over 2016-10-28 .. 2026-10-27 (10 years), 15 bps per side costs,
executed at the close one day after the signal:

| | CAGR | Vol | Sharpe | MaxDD | Total |
|---|---|---|---|---|---|
| Strategy (2 tranches) | 24.0% | 22.8% | 1.06 | -30.0% | x8.56 |
| SPY total return | 15.6% | 18.0% | 0.90 | -33.7% | x4.24 |
| QQQ total return | 20.8% | 22.6% | 0.95 | -35.1% | x6.60 |

First 5 years 26-27% CAGR, last 5 years 21%. Calendar years in logs/strategy.log.
Robustness: 21-26% across rebalance-day offsets (0..17 trading days), 21.7%
at 30 bps costs, 18-25% across neighbouring lookbacks / top-N / liquidity
cutoffs (logs/strat4.log, logs/strat5.log). The SPY-200dma regime filter,
vol targeting, MA stops and 52-week-high composites all HURT (strat2/strat3).

## Rules (strategy.py)
* Universe at each rebalance: Polygon type CS/ADRC, close >= $5, 20-day
  average dollar volume >= $50M, >= ~1 year of history.
* Score: for (lookback, skip) in {(252,21),(252,42),(210,21),(189,21)}:
  log(close[t-skip]/close[t-lookback]) / (60-day daily vol * sqrt(252));
  rank across the universe; sum the 4 ranks.
* Hold the top 25, equal weight, fully invested, no leverage.
* Two 50% tranches, each rebalanced monthly: A at month-end, B 10 trading
  days later. Trade at the next day's close.

## Files
* `fetch_ref.py`   Polygon download (splits, dividends, tickers, grouped daily) -- ran on the remote box
* `build_panel.py` grouped json -> dates x tickers npz with split adj + dividends
* `bt.py`, `feats.py`  backtest engine, features, universe
* `strat1..5.py`, `verify.py`  the iteration record; logs in `logs/`
* `strategy.py`   the final strategy + report
* `out/equity_curve.csv`, `out/current_portfolio.json`, `out/holdings_log.json`

## Data (on the remote box 192.168.0.150)
`~/strat20/data/`: `panel.npz` (3121 days x 24080 tickers, 2014-06..2026-10-27),
`grouped/` (2014-06..2016-09, adjusted=false; later dates come from
`~/roadrunner_v2/data/raw/grouped`), `splits.json`, `dividends.json`
(month-sharded fetch), `tickers.json`. Run with
`/home/trumpet/anaconda3/envs/py38/bin/python strategy.py` in `~/strat20`.

## Caveats
* Parameters were chosen on the same 10 years (in-sample). The neighbourhood is
  flat (see above) and the rules are standard momentum, but expect lower
  live returns. The decade favoured large-cap growth; momentum crashes in sharp
  rebounds (2020 Q2 was fine here, 2023 lagged).
* A holding with no bar for 15 straight days is written off at its last price.
* Ticker renames are two separate series (the new one needs a year of history).

# Phase 2 (2026-10-29): multi-sleeve portfolio, target Sharpe > 2

`python3 portfolio.py` combines daily NET return series in `sleeves/` with
inverse-vol weights (no return-based selection). 2016-10-28..2026-10-27:

| Portfolio | Sharpe | ret/yr | vol | maxDD | CAGR |
|---|---|---|---|---|---|
| MAIN (5 sleeves) inverse-vol | 2.41 | 16.6% | 6.9% | -8.0% | 17.8% |
| MAIN walk-forward weights (from 2018-11) | 2.16 | 17.4% | 8.0% | -8.2% | 18.6% |
| CONSERVATIVE (overnight book on the 40 untouched names only) | 2.04 | 20.0% | 9.8% | -13.0% | 21.6% |

Sleeves (net Sharpe): momentum long-only 1.07 (strategy.py); VIX spread 1.04
(agents/vol, 0.78 without its in-sample Thursday skip); ML L/S 0.95 (agents/ml);
conditional-overnight top-10 long-only on 50 large caps 2.06 (xs_on.py, 1 bp/side;
1.54 on the 40 out-of-sample names); time-series overnight book on 52 names 1.45
(on_book.py). Pairwise correlations <= 0.09 except momentum/vol 0.43 and the two
overnight books 0.69.

The overnight signal (found by the intraday agent on SPY/QQQ/5 mega-caps, then
fixed): s = 0.5*z30 - z60 at the close, z60 = last-60-min return / its 20-day sd,
z30 = first-30-min return / its 20-day sd; long overnight, exit at the 09:31 bar.
Out of sample it was net-positive on 39 of 40 new large caps. Cost sensitivity:
each extra 1 bp per side costs ~0.25 Sharpe on the overnight books.

Dead lanes (see agents/*/ and logs/): pairs / residual reversal / ETF spreads
(gross <= 1, net < 0), classic cross-sectional factor zoo (net < 0.4), leveraged
ETF pairs (net < 0 after borrow), daily ETF timing (0.6-0.76 each), Gao-style
intraday momentum (negative), SPY-200dma regime filters (hurt everything).

Data on the remote box: `~/strat20/data/minute/{T}_{Y}.json.gz` Polygon minute
bars for ~180 tickers, converted by `agents/intraday/convert.py` to
`~/strat20/data/minute_npz/{T}.npz` (391 bins/day).

## Phase 2 update (2026-10-29 08:30): toward Sharpe 3
`python3 portfolio.py` MAIN now = momentum + VIX ENS2 + ML L/S + three overnight
books (rule top-20 s>1 exit 09:31 on 150 names; rule top-20 exit 10:31; LightGBM
L/S top/bottom 20 exit 09:31) + an intraday LightGBM L/S (decide 09:31, hold to
close): inverse-vol Sharpe 2.95 (walk-forward 2.84), CAGR 24.6%, vol 7.6%,
maxDD -10.3%. Scripts: xs_on.py (rule books), on_ml.py (learned overnight),
on_ml_day.py (learned intraday), on_ml_pm.py (last hour: cost-dominated, dead).
A first intraday run showed net 2.10 but had a lookahead leak (rv20 included the
day's close); the corrected sleeve is 0.88 (second half ~0). Exit-timing: the
overnight rule book improves monotonically with a later exit (09:31 1.63 ->
10:31 1.93 on 150 names, top 20); the learned L/S book does not.

## Phase 2 final (2026-10-29 09:00): Sharpe 3.05

Adding the next 150 liquid names (data/next150b.txt, 300 names total) to the
learned overnight book pushed the LightGBM overnight L/S sleeve from net 1.67
(150 names) to net 1.93 (300 names; `on_ml.py` with exit at the 09:31 bar,
1 bp per side). Swapping that sleeve into the portfolio:

| portfolio.py MAIN (7 sleeves)        | Sharpe | ret   | vol  | maxDD  | CAGR  |
|--------------------------------------|--------|-------|------|--------|-------|
| inverse-vol (full-period weights)    | 3.05   | 23.5% | 7.7% | -10.1% | 26.1% |
| walk-forward (trailing 2y, from 2018-11) | 3.05 | 27.2% | 8.9% | -9.3% | 30.7% |
| conservative (OOS overnight names)   | 2.04   | 20.0% | 9.8% | -13.0% | 21.6% |

Sleeves: mom_longonly, vol_ens2_net, ml_ls, xslo20s1_150, onrule_x60_lo20,
onlgb300_ls20, daylgb_ls20. Worst calendar year Sharpe 0.30 (2022).

Other 300-name results (run300.sh, logs/run300.log): rule book XSLO30 smin=1
on 300 names net 1.86 (adding it on top does not help, 3.00); the second 150
names alone are weaker (XSLO20 1.52); rule book with 10:31 exit on 300 names
1.49 (worse than the 150-name 1.93); intraday LGB L/S on 300 names 0.81.

Caveats (unchanged): sleeve selection and the k / smin / exit-bar choices are
in-sample; overnight books lose ~0.25 Sharpe per extra 1 bp per side and show
second-half decay (overnight LGB L/S halves 2.9 / 1.1); the vol sleeve's
Thursday skip is in-sample. The conservative line (2.0) is the defensible
out-of-sample number.

## Fold validation (2026-10-29): thresholds chosen only on prior dates

`folds.py` cuts the timeline into half-year test windows (17 folds, 2018-07-01
.. 2026-10-27). For each fold every choice that portfolio.py made in-sample
(variant of each sleeve family: top-k, smin, exit bar, universe, lookbacks,
rv10 threshold, Thursday skip, ML k/hold/smoothing; which families to hold;
the weights) is made only from dates before the window (5-day gap), then
frozen for the window. Candidate grids: `logs/mom_grid/` (mom_grid.py, 12),
`logs/vol_grid.csv` (agents/vol/grid.py, 24), agents/ml/logs/net_*.csv (8),
agents/intraday/logs xs_* / onml_* / dayml_* (68). One pick per family,
weights = inverse-vol x max(prior Sharpe, 0).

| 2018-07-01 .. 2026-10-27                       | Sharpe | ret   | vol   | maxDD  | CAGR  |
|------------------------------------------------|--------|-------|-------|--------|-------|
| folds, selection on all prior dates (train+val)| 2.39   | 23.0% | 9.6%  | -14.7% | 25.2% |
| folds, selection on trailing 1y only (val)     | 2.05   | 19.0% | 9.2%  | -16.1% | 20.3% |
| reference: portfolio.py MAIN fixed in-sample picks, same span | 3.04 | 25.0% | 8.2% | -10.1% | 27.9% |

Fold Sharpes (sel=all): mean 2.83, median 2.44, min 0.12 (2022 H2), no
negative fold; sel=val: 2 negative folds (2022). Per-family out-of-sample
(stitched picks, sel=all): rule overnight exit 09:31 1.82, LightGBM overnight
1.47, rule overnight late exit 1.46, momentum 0.97, vol 0.93, daily ML 0.90,
intraday 0.84. The picks are stable after 2020 (XSLO20 on 150 names with
s>1, rule LO top20 exit 10:31, LGB L/S top20 on 300 names, spread rv10>14%
with Thursday skip, momentum E4). Outputs: out/folds_{all,val}_oos.csv and
out/folds_{all,val}_table.csv.

Still in-sample even here: the sleeve families and the ML feature sets were
designed after seeing the whole sample, and the minute-bar universe (150/300
names) was chosen by today's liquidity (survivorship). The 2.0-2.4 range is
the honest number; 3.05 is the in-sample one.

## Holdout design and development ladder (2026-10-29, afternoon)

Holdout = the last half year, 2026-04-28 .. 2026-10-27, never used for any
decision until the single final look reported at the end of this section.
Development = half-year test folds 2018-07-01 .. 2026-04-27 (16 folds), every
variant pick and every weight taken from dates before the fold (5-day gap):
`python3 folds.py --sel all --wmode iv --holdout 2026-04-28 [--show_holdout]`.
Weights = plain inverse vol ("iv"); Sharpe-weighted, ERC, min-var and max-Sharpe
weights were all worse out of fold.

What moved the development number (all lines are fold-OOS, holdout untouched):

| change                                                        | full grid | coarse grid |
|---------------------------------------------------------------|-----------|-------------|
| starting point (150/300-name books, yearly retrain)           | 2.50      | 2.64        |
| + 600-name universe (data/next300c.txt)                       | 2.76      | 2.84        |
| + quarterly retrain of the LightGBM overnight book (retrain=Q)| 2.80      | 2.96        |
| + 1200-name universe (data/next600d.txt, ranks 601-1200)      | 2.77      | **3.17**    |
| coarse grid, trailing-1y selection instead of all prior dates |           | 2.64        |
| coarse grid, 1200-name L/S book costed at 1.5 bps/side        |           | 2.82        |

"Full grid" = every variant ever produced (~90 candidates); "coarse grid" =
one pre-specified knob per family (k = 20/30, exit 09:31 or 10:31, rule or
LightGBM, universe 150/300/600/1200, rv10 threshold 12/14 with Thursday skip,
momentum E4 top 25, daily ML k25 h5), i.e. the omit list in the command below.
More candidates = more selection noise = lower fold-OOS; breadth of the
overnight universe is the lever that actually works.

Things tried that did NOT improve fold-OOS and were left out of the grid:
extra features / bigger models (on_ml2.py: y300 L/S 1.82 vs 1.93 original),
hedged (XSLOH) and regime-switched (XSLOR) rule books, vol ENS3/ENS4 and
contango/backwardation filters, hedged momentum, first-hour intraday books
(exit 10:31 for the learned book: L/S 0.95 on 1200 names), top-40 books,
ridge models, Sharpe-based / ERC / max-Sharpe weighting, vol targeting.

Single-book in-sample Sharpe at 1 bp/side as breadth grows (LightGBM overnight
L/S top 20, exit 09:31): 300 names 1.93 -> 600 names 2.10 -> 600 quarterly
2.39 -> 1200 names (yearly) 2.76 (2.49 at 1.5 bps; second half of the sample
alone 2.30). Learned intraday L/S top 20 (09:32 -> close): 600 names 1.49 ->
1200 names 1.99. Rule book XSLO20 on 1200 names 1.51 (no gain over 150).

Coarse-grid command (dev folds + holdout):
```
python3 folds.py --sel all --wmode iv --holdout 2026-04-28 --show_holdout --omit "XSLOR|XSLOH|ens[34]eq|contango|backwd|mom_hedged|top10|top5$|XSLO5|XSLO10|XSLO15|XSTOP|XSLS|TS$|TSH|ridge|_x15|_x30|thr10|thr0_|shortvixy|S126|S252|E2_|top15|mom_.*top40|k40|k30|h1_|pred_lgbv|pred_lgbbig|pred_ens2|onml2|a300x60|c15"
```
Per-fold picks (coarse grid) after 2020: momentum E4 top25, spread rv10>14%
Thursday skip, daily ML k25 h5 sm5, XSLO20 on 150 names (s>1), rule LO top20
exit 10:31, LightGBM L/S top20 on 1200 names, learned intraday L/S top20 on
1200 names. Family fold-OOS: LightGBM overnight 2.08+, rule overnight 1.95,
intraday 1.52, late rule 1.32, momentum 1.13, vol 1.06, daily ML 0.71.

### Final result with the quarterly 1200-name book, and the one holdout look

The quarterly-retrain step was committed before the 1200-name result was
seen, so it was folded into the grid before looking at the holdout. All
numbers below come from the single command above (output in
out/folds_final_coarse.txt, series in out/folds_all_oos.csv).

| coarse grid, plain inverse-vol weights, picks from prior dates | Sharpe | ret   | vol  | maxDD  | CAGR  |
|----------------------------------------------------------------|--------|-------|------|--------|-------|
| development folds 2018-07-01 .. 2026-04-27 (16 half-years)     | 3.33   | 30.6% | 9.2% | -15.0% | 35.3% |
| **holdout 2026-04-28 .. 2026-10-27 (never used before)**       | **4.61** | 36.8% | 8.0% | -1.9% | 44.0% |
| full grid, same dev folds                                       | 3.23   |       |      |        |       |
| coarse grid, trailing-1y selection, dev folds                  | 2.86   |       |      |        |       |
| fixed single variant per family (variant chosen in-sample), dev| 3.79   |       |      |        |       |

Fold Sharpes: mean 3.65, median 3.21, min 0.35 (2022 H2), no negative fold.
Family fold-OOS: LightGBM overnight L/S 3.20, intraday L/S 1.72, rule
overnight 1.59, late rule 1.46, momentum 1.06, vol 1.03, daily ML 0.98.

portfolio.py MAIN now holds the same seven sleeves with the 1200-name books
(sleeves/onlgb1200q_ls20.csv net 3.22 alone, sleeves/daylgb1200_ls20.csv net
1.99): in-sample inverse-vol Sharpe 3.77, walk-forward 3.92 (previous version
kept as MAIN_V2: 3.05). Conservative line unchanged at 2.04.

Caveats that still apply to the holdout number:
* The holdout is 126 trading days; a Sharpe of 4.6 over half a year has a
  standard error of about 1.4.
* Single-book Sharpes used to decide which runs to launch (600 names, 1200
  names, quarterly retrain) were full-span figures that include the holdout
  months; the fold selections themselves never saw holdout dates.
* The LightGBM feature set and hyperparameters were designed on the whole
  sample earlier in the project.
* Universe ranks 601-1200 by today's median dollar volume: survivorship, and
  1 bp/side is optimistic there (1.5 bps costs the dev number about 0.35).
* Fills at the official close and at the 09:31/09:32 bar are assumed exact.
