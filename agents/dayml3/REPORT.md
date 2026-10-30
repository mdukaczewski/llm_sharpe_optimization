# dayml3 lane: the learned intraday book under REALISTIC execution (spread as a cost and as a feature)

Scope (agents/BRIEF6.md): re-build the learned intraday book (`dayml2_ens_e30_k8`: decide at the 09:31 bar, fill at the
09:33 bar = bin 3, exit at the 10:01 bar = bin 31, L/S top/bottom k, 1500 names) so that it is evaluated and optimised at
the PREDICTED NBBO half-spread of every name it trades (`liq.py`: `hs0931` for the 09:33 entry, `hs1001` for the exit,
cost per side = f x half-spread, f = 1 crosses the spread, 0.5 = midpoint-ish, 0.25 = mostly passive), plus 1 %/yr borrow
and the $3/order commission on $0.4M (2k positions x 2 orders: k4 = 1.2, k6 = 1.8, k8 = 2.4 bps/day).  Metric: fold-OOS
Sharpe (`rlib.fold_oos`, 16 half-year windows 2018-07..2026-04; for a single series this is its Sharpe over that span).
Dates <= 2026-04-27 everywhere (`rlib.DEV_END`); the holdout was never read.

Code: `d3lib.py` (frame = `d2lib.build` + 17 liq columns, per-name spread-costed books, causal cost-aware selection),
`step1.py` (saved predictions costed), `step2.py` / `step2b.py` (cost-aware selection on the saved ensemble), `step3.py` /
`step3b.py` / `step3c.py` (yearly refits with / without the liq features, net targets, two-sided net targets, pairs, f = 1
targets), `test_causal.py`, `fitq.py` (quarterly fit of one configuration, n_jobs = 2), `final.py` (books + candidates from
the saved quarterly predictions), `mk_tables.py` / `build_report.py` (this file).  Logs `log.txt`, `step*.log`, `fitq_*.log`,
`final_*.log`; tables `out/step1.csv`, `out/step1_tiers.csv`, `out/step2.csv`, `out/step2b.csv`, `out/step3_Y.csv`,
`out/step3b_Y.csv`, `out/step3c_Y.csv`, `out/final.csv`; predictions
`out/pred_step3_*_Y.parquet` (yearly) and `out/pred_final_*_Q.parquet` (quarterly); daily book frames `out/*_k*.parquet`
(columns g gross, sc spread cost at f = 1, old old-convention cost, bor borrow, comm commission).

## 0. Summary

* **Nothing is positive at f = 1 on any universe, with any model, at any k.**  Crossing the predicted spread on both legs
  costs 20 bps per trade in ranks 1-300, 34 in 301-600 and 53 beyond 600, against a gross edge of 10-16 bps per trade;
  the best f = 1 book found anywhere is -0.1 (k = 2, 4 positions) and every k >= 4 book is <= -0.5 fold-OOS.  Models
  trained directly on the f = 1 net target (`B1f1+S1f1`) are -1.2 at best.  The x1 candidates are saved as asked and are
  marked NEGATIVE in their notes.
* **At f = 0.5 the book is marginal and only works with cost-aware selection on the liquid half of the universe.**  The
  saved ensemble costed at f = 0.5 is -2.3 (k8) / -1.9 (k6); restricting the universe to ranks <= 300 / 600 does not help
  (-1.7 / -2.1); a cap on the predicted round-trip cost or a linear net score reaches at best +0.2.  The two things that
  work are (a) a gross-target model with the liq features whose top/bottom ranks are re-scored as E[y | rank] - 0.5 c on
  both sides (causal calibration) and (b) two-sided net-target models (long rank(y - 0.5c), short rank(-y - 0.5c)).  Final
  (quarterly retrain): **`dayml3_liqsel_u600_x05_k4` Sh 1.03 / fold-OOS 0.77, halves 2.18 / -0.19, maxDD -44 %**, k6
  0.51, k8 0.19.  The f = 0.5 book has been flat since 2021 (yearly Sharpe 2021-26: -0.4 -0.8 +0.6 -0.9 -0.6 +1.2), so it
  is a "only works at f = 0.5, and only before 2021" book.
* **The liq features are a strong signal improvement at the mid-quote**: the same gross book at the old 1-1.5 bps
  convention goes from 2.07 (no liq features, yearly) to 2.88 fold-OOS (yearly) and 2.97 quarterly at k8 (vs 2.40 for
  `dayml2_ens_e30_k8`), because the model learns which thin names bounce most.  That is exactly why it does not help at
  a realistic spread: the gross-target model allocates 70 % of its positions beyond rank 600, where the round-trip
  half-spread is 46 bps.
* A single rank(y - cost) target is WRONG for a long/short book (step 3 (ii)/(iii) as specified): the costliest names get
  the lowest net rank and the book shorts them (601-1500 shorts: -24 bps/trade).  The cost has to enter as a two-sided
  score or as two models.

## 1. Cost model check

`liq.py` was not broken; no change was made to it.  Sanity check of its predictions against the measured NBBO of this
book's own trades (`out/quotes_trades.csv`, dayml book, dev dates, n = 3,728 round trips):

| tier | n | entry 09:33: measured median / predicted median | measured mean / predicted mean | exit 10:01: measured median / predicted | measured mean / predicted |
|---|---|---|---|---|---|
| 1-300 | 627 | 7.1 / 10.7 | 11.3 / 13.8 | 3.4 / 3.8 | 5.0 / 5.0 |
| 301-600 | 759 | 16.1 / 21.6 | 22.1 / 25.7 | 6.0 / 7.4 | 8.4 / 9.1 |
| 601-1500 | 2342 | 26.0 / 35.0 | 35.2 / 39.5 | 10.2 / 11.4 | 14.0 / 13.6 |
| all | 3728 | 19.6 / 27.6 | 28.5 / 32.3 | 7.7 / 9.2 | 11.3 / 11.3 |

corr(predicted, measured) 0.73 entry / 0.78 exit.  The exit model is unbiased; the entry cost is CONSERVATIVE by ~13 %
on the mean (the "0931" model pools the wider 09:31:00 quotes of the overnight book's exit with the 09:33 quotes of this
book's entry, as `liq.training_rows` relabels 0933 -> 0931).  All numbers below use the brief's convention (f x hs0931
at entry); a 13 % lower entry cost moves a -2.3 book to about -1.9 and does not change any conclusion.  Median predicted
half-spreads over all (date, name) rows: 09:31 10.9 / 19.5 / 29.7 bps and 10:01 3.7 / 6.1 / 8.7 bps by tier.

## 2. Step 1: the saved predictions costed at the predicted half-spreads

Ensemble = average per-day pct rank of `agents/dayml/out/pred_y_60d_Q.parquet` and `pred_y_120d_Q.parquet`, exit bin
31, equal weight; `u600` / `u300` select top/bottom k within ranks <= 600 / <= 300; `tier..` books select within the
tier.  "old" = the 1 / 1.5 bps per side convention of the earlier lanes, "gross-comm" = mid-to-mid minus commission and
borrow.  All at the $0.4M commission.

| book | Sh f=1.0 | fo f=1.0 | Sh f=0.5 | fo f=0.5 | Sh f=0.25 | fo f=0.25 | Sh f=old | fo f=old | Sh f=gross-comm | fo f=gross-comm | halves f=0.5 | mdd f=0.5 | halves old | mdd old |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ens_e30_k8_u1500 | -7.56 | -7.63 | -2.13 | -2.33 | 0.63 | 0.35 | 2.72 | 2.40 | 3.38 | 3.02 | -1.01/-3.20 | -98% | 4.18/1.37 | -16% |
| ens_e30_k6_u1500 | -6.48 | -6.57 | -1.67 | -1.86 | 0.77 | 0.52 | 2.64 | 2.36 | 3.22 | 2.91 | -0.78/-2.52 | -98% | 3.83/1.55 | -22% |
| ens_e30_k8_u600 | -5.92 | -6.24 | -1.67 | -2.11 | 0.48 | -0.02 | 2.01 | 1.49 | 2.64 | 2.08 | -0.28/-3.21 | -96% | 3.27/0.65 | -27% |
| ens_e30_k6_u600 | -5.26 | -5.63 | -1.32 | -1.78 | 0.67 | 0.16 | 2.10 | 1.57 | 2.67 | 2.11 | 0.20/-2.95 | -95% | 3.52/0.61 | -30% |
| ens_e30_k8_u300 | -5.18 | -5.32 | -1.46 | -1.69 | 0.43 | 0.14 | 1.54 | 1.24 | 2.31 | 1.97 | -0.56/-2.39 | -88% | 2.39/0.66 | -16% |
| ens_e30_k6_u300 | -4.40 | -4.61 | -1.00 | -1.28 | 0.71 | 0.39 | 1.72 | 1.40 | 2.41 | 2.05 | -0.03/-1.98 | -85% | 2.68/0.77 | -19% |
| ens_e30_k8_tier1-300 | -5.18 | -5.32 | -1.46 | -1.69 | nan | nan | 1.54 | 1.24 | 2.31 | 1.97 | -0.56/-2.39 | -88% | 2.39/0.66 | -16% |
| ens_e30_k6_tier1-300 | -4.40 | -4.61 | -1.00 | -1.28 | nan | nan | 1.72 | 1.40 | 2.41 | 2.05 | -0.03/-1.98 | -85% | 2.68/0.77 | -19% |
| ens_e30_k8_tier301-600 | -8.18 | -8.48 | -2.90 | -3.32 | nan | nan | 1.82 | 1.30 | 2.48 | 1.93 | -1.24/-4.86 | -99% | 3.20/0.24 | -29% |
| ens_e30_k6_tier301-600 | -7.20 | -7.53 | -2.48 | -2.91 | nan | nan | 1.74 | 1.22 | 2.32 | 1.77 | -0.88/-4.31 | -99% | 3.11/0.20 | -34% |
| ens_e30_k8_tier601-1500 | -9.78 | -9.62 | -3.39 | -3.42 | nan | nan | 2.33 | 2.14 | 3.09 | 2.87 | -2.79/-3.98 | -100% | 3.41/1.31 | -19% |
| ens_e30_k6_tier601-1500 | -8.11 | -7.95 | -2.56 | -2.56 | nan | nan | 2.40 | 2.24 | 3.05 | 2.86 | -2.01/-3.08 | -100% | 3.48/1.44 | -18% |

By tier of the full-universe books (share of positions, median predicted half-spreads of the positions, bps per trade):

| book | f | tier | share of positions | median hs 09:33 (bps) | median hs 10:01 | gross bps/trade | spread cost bps/trade | net bps/trade |
|---|---|---|---|---|---|---|---|---|
| ens_e30_k8_u1500 | 1.00 | 1-300 | 0.17 | 11.50 | 4.00 | 10.50 | 20.10 | -9.60 |
| ens_e30_k8_u1500 | 1.00 | 301-600 | 0.21 | 21.40 | 7.40 | 13.80 | 34.40 | -20.60 |
| ens_e30_k8_u1500 | 1.00 | 601-1500 | 0.62 | 34.70 | 11.20 | 16.30 | 53.20 | -36.90 |
| ens_e30_k8_u1500 | 0.50 | 1-300 | 0.17 | 11.50 | 4.00 | 10.50 | 10.10 | 0.40 |
| ens_e30_k8_u1500 | 0.50 | 301-600 | 0.21 | 21.40 | 7.40 | 13.80 | 17.20 | -3.40 |
| ens_e30_k8_u1500 | 0.50 | 601-1500 | 0.62 | 34.70 | 11.20 | 16.30 | 26.60 | -10.30 |
| ens_e30_k6_u1500 | 1.00 | 1-300 | 0.17 | 11.60 | 4.10 | 13.70 | 20.30 | -6.60 |
| ens_e30_k6_u1500 | 1.00 | 301-600 | 0.21 | 21.40 | 7.40 | 14.90 | 34.50 | -19.70 |
| ens_e30_k6_u1500 | 1.00 | 601-1500 | 0.63 | 35.00 | 11.40 | 16.40 | 53.90 | -37.50 |
| ens_e30_k6_u1500 | 0.50 | 1-300 | 0.17 | 11.60 | 4.10 | 13.70 | 10.20 | 3.60 |
| ens_e30_k6_u1500 | 0.50 | 301-600 | 0.21 | 21.40 | 7.40 | 14.90 | 17.30 | -2.40 |
| ens_e30_k6_u1500 | 0.50 | 601-1500 | 0.63 | 35.00 | 11.40 | 16.40 | 27.00 | -10.50 |

Reading: the thin names carry the largest gross edge (16 bps) but a 53 bps round-trip half-spread; only the 1-300 names
are (barely) positive at f = 0.5 (+0.4 / +3.6 bps per trade before commission).  The book needs f <= 0.25 to be positive
(k6: Sh 0.77, fold-OOS 0.52), i.e. it must capture three quarters of the spread on both legs.

## 3. Step 2: cost-aware selection on the saved ensemble

Cap on f x (hs0931 + hs1001): fold-OOS Sharpe at $0.4M by k (rows: f, cap in bps):

| f | cap | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| 0.50 | 5.00 | -0.27 | -0.38 | -0.95 | -1.32 |
| 0.50 | 10.00 | 0.21 | -0.23 | -0.65 | -0.96 |
| 0.50 | 15.00 | -0.20 | -0.68 | -1.02 | -1.35 |
| 0.50 | 20.00 | -0.17 | -0.57 | -1.05 | -1.50 |
| 0.50 | 30.00 | -0.62 | -1.12 | -1.63 | -2.08 |
| 1.00 | 5.00 | -1.86 | -2.81 | -3.45 | -4.16 |
| 1.00 | 10.00 | -1.62 | -1.96 | -2.70 | -3.26 |
| 1.00 | 15.00 | -1.72 | -2.14 | -2.56 | -3.02 |
| 1.00 | 20.00 | -1.70 | -2.53 | -3.18 | -3.67 |
| 1.00 | 30.00 | -2.46 | -3.34 | -4.00 | -4.61 |

Share of (date, name) rows eligible: f=0.5 cap5: 0.13, f=0.5 cap10: 0.34, f=0.5 cap15: 0.54, f=0.5 cap20: 0.69, f=0.5 cap30: 0.87, f=1.0 cap5: 0.03, f=1.0 cap10: 0.13, f=1.0 cap15: 0.23, f=1.0 cap20: 0.34, f=1.0 cap30: 0.54

Net score = E[y_30d | rank] - f x round-trip cost (linear or bucket calibration), top-k of each side:

| book | calibration | Sh | fold-OOS | ret % | maxDD % | h1 | h2 | share ranks<=300 | share >600 |
|---|---|---|---|---|---|---|---|---|---|
| netscore_f1.0_k4 | linear | -1.33 | -1.42 | -12.50 | -73.00 | -0.72 | -1.88 | 0.82 | 0.05 |
| netscore_f1.0_k6 | linear | -1.92 | -1.99 | -15.80 | -79.00 | -1.05 | -2.73 | 0.80 | 0.05 |
| netscore_f1.0_k8 | linear | -2.49 | -2.59 | -18.40 | -83.00 | -1.57 | -3.37 | 0.79 | 0.06 |
| netscore_f1.0_k10 | linear | -2.87 | -2.93 | -20.00 | -85.00 | -1.96 | -3.76 | 0.78 | 0.06 |
| netscore_f0.5_k4 | linear | 0.12 | -0.04 | 1.30 | -51.00 | 1.20 | -0.86 | 0.81 | 0.06 |
| netscore_f0.5_k6 | linear | -0.19 | -0.38 | -1.80 | -58.00 | 1.11 | -1.46 | 0.79 | 0.06 |
| netscore_f0.5_k8 | linear | -0.47 | -0.68 | -4.10 | -60.00 | 0.83 | -1.80 | 0.78 | 0.07 |
| netscore_f0.5_k10 | linear | -0.75 | -0.95 | -6.10 | -64.00 | 0.59 | -2.13 | 0.78 | 0.07 |
| netscore_bucket_f1.0_k4 | bucket | -0.79 | -0.96 | -9.80 | -75.00 | 0.18 | -1.74 | 0.76 | 0.09 |
| netscore_bucket_f1.0_k6 | bucket | -1.42 | -1.58 | -14.90 | -79.00 | -0.38 | -2.43 | 0.76 | 0.09 |
| netscore_bucket_f1.0_k8 | bucket | -1.75 | -1.93 | -16.30 | -80.00 | -0.61 | -2.86 | 0.76 | 0.09 |
| netscore_bucket_f1.0_k10 | bucket | -2.22 | -2.33 | -18.80 | -84.00 | -1.13 | -3.27 | 0.75 | 0.09 |
| netscore_bucket_f0.5_k4 | bucket | 0.67 | 0.35 | 10.80 | -57.00 | 2.16 | -0.68 | 0.62 | 0.19 |
| netscore_bucket_f0.5_k6 | bucket | 0.52 | 0.21 | 7.00 | -59.00 | 1.98 | -0.87 | 0.63 | 0.18 |
| netscore_bucket_f0.5_k8 | bucket | 0.35 | 0.03 | 4.20 | -63.00 | 2.01 | -1.25 | 0.64 | 0.17 |
| netscore_bucket_f0.5_k10 | bucket | 0.01 | -0.30 | 0.10 | -67.00 | 1.66 | -1.59 | 0.64 | 0.16 |

The calibration for the net score is in-sample: linear OLS of the traded return on the ensemble rank over every
(date, name) row, `E[y_30d] = -0.15 bps + 12.4 bps x rank` (rank in [-0.5, 0.5]), i.e. +6.0 bps for the top name and
-6.3 for the bottom; the y_60d map is `+0.49 + 13.4 x rank`.  The relation is convex at the extremes (mean y_30d +14.1
bps in the top 1 %, -16.0 in the bottom 1 %, vs +5.3 / -8.1 in the next 4 %), which the "bucket" calibration (1 % buckets
beyond |rank| > 0.45) captures.  Nothing here is positive at f = 1; at f = 0.5 the best construction is 0.35 fold-OOS
(bucket net score, k4) with a second half of -0.7.

## 4. Step 3: refits with the liquidity features (yearly retrain, seed 0, n_jobs = 1, 2.53 M rows)

Features: `dlib.EXT` (43) -> `EXTL` = EXT + lagged `ldv20 lpx hl1 hl5 hlc roll30 n30 nc30 dv0 dvc`, same-morning
`t_hl0 t_hl1 t_n0 t_n1 t_dv0` (bins 0-1 of day t) and the predicted `hs0931 hs1001` (60 features).  Targets: A = rank
of y_60d; B = rank of y_60d - 0.5 (hs0931 + hs1001)/1e4; C = B on ranks <= 600 (trained and predicted there only).
Suffix 0 = EXT, 1 = EXTL.  IC against y_30d (the traded return): A0 0.031 (t 13.5), A1 0.035 (14.4), B0 0.025, B1
0.021, C0 0.029, C1 0.025.  Books: L/S k in 4 / 6 / 8, fill bin 3, exit bin 31, equal weight, $0.4M commission.

| model | universe | k | Sh f=1.0 | fo f=1.0 | Sh f=0.5 | fo f=0.5 | Sh f=0.25 | fo f=0.25 | Sh f=old | fo f=old | halves f=0.5 | mdd f=0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | u1500 | 4 | -6.18 | -6.33 | -1.50 | -1.69 | 0.89 | 0.67 | 2.83 | 2.62 | -0.53/-2.34 | -99% |
| A1 | u1500 | 6 | -7.70 | -7.79 | -2.11 | -2.25 | 0.75 | 0.58 | 3.06 | 2.88 | -1.48/-2.69 | -99% |
| A1 | u1500 | 8 | -8.81 | -8.88 | -2.64 | -2.79 | 0.53 | 0.34 | 3.08 | 2.87 | -1.93/-3.31 | -100% |
| A1 | u600 | 4 | -4.67 | -4.89 | -1.14 | -1.42 | 0.65 | 0.34 | 2.01 | 1.69 | -0.35/-1.92 | -96% |
| A1 | u600 | 6 | -5.47 | -5.66 | -1.44 | -1.72 | 0.60 | 0.28 | 2.12 | 1.79 | -0.45/-2.48 | -96% |
| A1 | u600 | 8 | -6.23 | -6.36 | -1.83 | -2.08 | 0.41 | 0.10 | 2.05 | 1.71 | -0.85/-2.85 | -96% |
| A1 | u300 | 4 | -3.54 | -3.61 | -0.68 | -0.82 | 0.77 | 0.58 | 1.65 | 1.46 | -0.07/-1.22 | -82% |
| A1 | u300 | 6 | -4.42 | -4.51 | -1.14 | -1.33 | 0.50 | 0.27 | 1.49 | 1.24 | -0.40/-1.84 | -87% |
| A1 | u300 | 8 | -5.14 | -5.16 | -1.56 | -1.69 | 0.23 | 0.04 | 1.30 | 1.09 | -1.02/-2.07 | -89% |
| B1 | u1500 | 4 | -9.06 | -9.01 | -3.30 | -3.34 | -0.40 | -0.49 | 1.93 | 1.80 | -3.08/-3.56 | -100% |
| B1 | u1500 | 6 | -11.00 | -10.87 | -4.21 | -4.18 | -0.78 | -0.82 | 1.93 | 1.86 | -3.87/-4.57 | -100% |
| B1 | u1500 | 8 | -12.47 | -12.36 | -4.90 | -4.93 | -1.07 | -1.17 | 1.96 | 1.81 | -4.43/-5.40 | -100% |
| B1 | u600 | 4 | -8.37 | -8.67 | -3.27 | -3.56 | -0.66 | -0.95 | 1.38 | 1.12 | -2.29/-4.19 | -100% |
| B1 | u600 | 6 | -9.83 | -10.11 | -4.03 | -4.32 | -1.07 | -1.36 | 1.23 | 0.95 | -2.95/-5.11 | -100% |
| B1 | u600 | 8 | -10.88 | -11.14 | -4.53 | -4.82 | -1.29 | -1.59 | 1.21 | 0.91 | -3.44/-5.64 | -100% |
| B1 | u300 | 4 | -7.95 | -8.11 | -3.30 | -3.48 | -0.93 | -1.13 | 0.73 | 0.55 | -3.06/-3.54 | -99% |
| B1 | u300 | 6 | -9.14 | -9.22 | -3.97 | -4.12 | -1.34 | -1.52 | 0.48 | 0.29 | -3.65/-4.30 | -99% |
| B1 | u300 | 8 | -9.96 | -9.98 | -4.45 | -4.58 | -1.64 | -1.82 | 0.28 | 0.08 | -4.27/-4.65 | -99% |
| C1 | u1500 | 4 | -6.71 | -7.18 | -2.20 | -2.60 | 0.13 | -0.24 | 1.92 | 1.61 | -0.88/-3.73 | -99% |
| C1 | u1500 | 6 | -7.99 | -8.43 | -2.87 | -3.25 | -0.22 | -0.58 | 1.80 | 1.49 | -1.34/-4.70 | -99% |
| C1 | u1500 | 8 | -8.96 | -9.37 | -3.42 | -3.80 | -0.56 | -0.93 | 1.61 | 1.27 | -1.78/-5.47 | -99% |
| C1 | u600 | 4 | -6.71 | -7.18 | -2.20 | -2.60 | 0.13 | -0.24 | 1.92 | 1.61 | -0.88/-3.73 | -99% |
| C1 | u600 | 6 | -7.99 | -8.43 | -2.87 | -3.25 | -0.22 | -0.58 | 1.80 | 1.49 | -1.34/-4.70 | -99% |
| C1 | u600 | 8 | -8.96 | -9.37 | -3.42 | -3.80 | -0.56 | -0.93 | 1.61 | 1.27 | -1.78/-5.47 | -99% |
| C1 | u300 | 4 | -6.37 | -6.75 | -2.41 | -2.75 | -0.38 | -0.72 | 1.00 | 0.71 | -1.39/-3.55 | -97% |
| C1 | u300 | 6 | -7.39 | -7.63 | -2.96 | -3.21 | -0.70 | -0.95 | 0.81 | 0.59 | -2.04/-4.05 | -98% |
| C1 | u300 | 8 | -8.09 | -8.33 | -3.31 | -3.58 | -0.87 | -1.15 | 0.74 | 0.48 | -2.38/-4.39 | -98% |
| A0 | u1500 | 4 | -5.24 | -5.12 | -1.27 | -1.28 | 0.73 | 0.66 | 2.26 | 2.15 | -0.56/-1.92 | -98% |
| A0 | u1500 | 6 | -6.92 | -6.76 | -2.07 | -2.09 | 0.38 | 0.26 | 2.25 | 2.07 | -1.11/-2.93 | -99% |
| A0 | u1500 | 8 | -8.23 | -8.03 | -2.66 | -2.66 | 0.17 | 0.05 | 2.31 | 2.13 | -1.62/-3.63 | -99% |
| A0 | u600 | 4 | -4.22 | -4.43 | -0.81 | -1.10 | 0.91 | 0.58 | 2.15 | 1.80 | 0.47/-2.02 | -93% |
| A0 | u600 | 6 | -5.48 | -5.61 | -1.52 | -1.76 | 0.48 | 0.18 | 1.91 | 1.58 | -0.19/-2.81 | -95% |
| A0 | u600 | 8 | -6.48 | -6.70 | -2.14 | -2.47 | 0.06 | -0.33 | 1.62 | 1.21 | -0.71/-3.62 | -97% |
| A0 | u300 | 4 | -3.91 | -4.03 | -0.93 | -1.10 | 0.57 | 0.37 | 1.48 | 1.28 | -0.42/-1.44 | -86% |
| A0 | u300 | 6 | -4.74 | -4.76 | -1.42 | -1.53 | 0.25 | 0.10 | 1.25 | 1.08 | -0.89/-1.96 | -89% |
| A0 | u300 | 8 | -5.53 | -5.57 | -1.85 | -1.99 | 0.01 | -0.18 | 1.11 | 0.91 | -1.34/-2.37 | -91% |
| B0 | u1500 | 4 | -7.05 | -7.09 | -2.23 | -2.39 | 0.20 | -0.03 | 2.02 | 1.75 | -0.98/-3.63 | -99% |
| B0 | u1500 | 6 | -8.34 | -8.27 | -2.81 | -2.85 | -0.01 | -0.13 | 2.07 | 1.91 | -1.80/-3.95 | -99% |
| B0 | u1500 | 8 | -9.31 | -9.18 | -3.31 | -3.35 | -0.29 | -0.42 | 1.96 | 1.77 | -2.12/-4.68 | -100% |
| B0 | u600 | 4 | -5.83 | -6.11 | -1.91 | -2.27 | 0.07 | -0.32 | 1.49 | 1.09 | -0.76/-3.24 | -97% |
| B0 | u600 | 6 | -6.98 | -7.19 | -2.48 | -2.77 | -0.20 | -0.54 | 1.43 | 1.06 | -1.10/-4.03 | -98% |
| B0 | u600 | 8 | -7.83 | -8.01 | -2.95 | -3.25 | -0.49 | -0.84 | 1.26 | 0.88 | -1.56/-4.49 | -98% |
| B0 | u300 | 4 | -5.42 | -5.60 | -2.05 | -2.27 | -0.35 | -0.59 | 0.70 | 0.46 | -1.12/-3.00 | -96% |
| B0 | u300 | 6 | -6.34 | -6.51 | -2.50 | -2.73 | -0.55 | -0.83 | 0.63 | 0.36 | -1.71/-3.29 | -96% |
| B0 | u300 | 8 | -7.11 | -7.17 | -2.97 | -3.12 | -0.88 | -1.08 | 0.40 | 0.18 | -2.27/-3.71 | -97% |
| C0 | u1500 | 4 | -4.15 | -4.46 | -0.82 | -1.17 | 0.88 | 0.50 | 2.07 | 1.70 | 0.95/-2.68 | -94% |
| C0 | u1500 | 6 | -5.36 | -5.66 | -1.53 | -1.89 | 0.42 | 0.03 | 1.80 | 1.40 | 0.00/-3.22 | -95% |
| C0 | u1500 | 8 | -6.27 | -6.49 | -2.05 | -2.34 | 0.11 | -0.22 | 1.62 | 1.28 | -0.63/-3.64 | -96% |
| C0 | u600 | 4 | -4.15 | -4.46 | -0.82 | -1.17 | 0.88 | 0.50 | 2.07 | 1.70 | 0.95/-2.68 | -94% |
| C0 | u600 | 6 | -5.36 | -5.66 | -1.53 | -1.89 | 0.42 | 0.03 | 1.80 | 1.40 | 0.00/-3.22 | -95% |
| C0 | u600 | 8 | -6.27 | -6.49 | -2.05 | -2.34 | 0.11 | -0.22 | 1.62 | 1.28 | -0.63/-3.64 | -96% |
| C0 | u300 | 4 | -3.93 | -4.04 | -0.95 | -1.09 | 0.56 | 0.41 | 1.47 | 1.33 | -0.18/-1.77 | -86% |
| C0 | u300 | 6 | -4.96 | -5.01 | -1.63 | -1.75 | 0.06 | -0.09 | 1.06 | 0.92 | -0.76/-2.64 | -91% |
| C0 | u300 | 8 | -5.90 | -5.93 | -2.17 | -2.26 | -0.27 | -0.39 | 0.86 | 0.74 | -1.15/-3.33 | -93% |

* With vs without the liq features on the same book: at the old convention A1 beats A0 everywhere (u1500 k6 fold-OOS
  2.88 vs 2.07, k8 2.87 vs 2.13); at f = 0.5 both are negative and A1 is slightly worse (-2.25 vs -2.09 at k6) because
  it concentrates further in the thin names (71 % of positions beyond rank 600 vs 63 %).
* The single-ranking net targets (B, C) are worse than the gross target at every f: the model learns "cheap = high net
  rank, expensive = low net rank", so the shorts are the most expensive names (B1 u1500 k4, f = 0.5: 601-1500 shorts
  -23 bps per trade).  Restricting to ranks <= 600 (C) softens but does not fix this.

### 4b. Two-sided net targets and cost-aware selection on the refit models (`step3b.py`, `step3c.py`, `test_causal.py`)

`B1+S1`: long the top-k of B1 (rank(y_60d - 0.5c)), short the top-k of S1 (rank(-y_60d - 0.5c)); `C1+S1_600` the same
on ranks <= 600; `A1sel05`: the gross liq model A1 re-scored as E[y_30d | rank] - 0.5c (long) and -E - 0.5c (short),
bucket calibration IN-SAMPLE; `A1sel0.5c` / `A1sel1.0c`: the same with a CAUSAL calibration (half-year cuts, each window
calibrated on the prior predictions only; the selection cost at f = 0.5 resp. 1.0); `B1x120+S1x120`: the pair on the
y_120d net targets; `ens4`: average pct rank of the two long models / the two short models; `B1f1+S1f1`: two-sided models
trained on the f = 1 net target.

| model | universe | k | Sh f=1.0 | fo f=1.0 | Sh f=0.5 | fo f=0.5 | Sh f=0.25 | fo f=0.25 | Sh f=old | fo f=old | halves f=0.5 | mdd f=0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1+S1 | u1500 | 4 | -0.79 | -0.70 | 0.66 | 0.56 | 1.39 | 1.19 | 1.49 | 1.22 | 1.21/0.12 | -43% |
| B1+S1 | u1500 | 6 | -1.44 | -1.31 | 0.25 | 0.15 | 1.09 | 0.87 | 1.20 | 0.90 | 0.93/-0.45 | -52% |
| B1+S1 | u1500 | 8 | -2.08 | -1.96 | -0.20 | -0.35 | 0.74 | 0.45 | 0.85 | 0.48 | 0.62/-1.04 | -64% |
| B1+S1 | u600 | 4 | -1.06 | -1.17 | 0.27 | 0.00 | 0.94 | 0.59 | 0.92 | 0.51 | 1.23/-0.69 | -59% |
| B1+S1 | u600 | 6 | -1.51 | -1.52 | 0.06 | -0.16 | 0.84 | 0.52 | 0.81 | 0.43 | 0.92/-0.82 | -57% |
| B1+S1 | u600 | 8 | -1.86 | -1.85 | -0.09 | -0.31 | 0.78 | 0.45 | 0.75 | 0.34 | 0.86/-1.10 | -58% |
| B1+S1 | u300 | 4 | -1.23 | -1.35 | 0.07 | -0.18 | 0.73 | 0.40 | 0.60 | 0.24 | 1.07/-0.91 | -63% |
| B1+S1 | u300 | 6 | -1.71 | -1.76 | -0.20 | -0.41 | 0.56 | 0.27 | 0.41 | 0.08 | 0.77/-1.19 | -63% |
| B1+S1 | u300 | 8 | -2.20 | -2.18 | -0.50 | -0.67 | 0.34 | 0.08 | 0.18 | -0.13 | 0.23/-1.26 | -63% |
| C1+S1_600 | u600 | 4 | -0.80 | -0.95 | 0.63 | 0.34 | 1.34 | 0.99 | 1.47 | 1.09 | 1.41/-0.22 | -42% |
| C1+S1_600 | u600 | 6 | -1.38 | -1.46 | 0.28 | 0.04 | 1.11 | 0.79 | 1.25 | 0.90 | 1.14/-0.71 | -52% |
| C1+S1_600 | u600 | 8 | -1.81 | -1.83 | -0.03 | -0.22 | 0.86 | 0.58 | 1.01 | 0.68 | 0.83/-1.04 | -60% |
| C1+S1_600 | u300 | 4 | -0.95 | -1.01 | 0.36 | 0.18 | 1.01 | 0.78 | 0.99 | 0.74 | 0.89/-0.25 | -41% |
| C1+S1_600 | u300 | 6 | -1.45 | -1.45 | 0.04 | -0.09 | 0.78 | 0.59 | 0.75 | 0.53 | 0.60/-0.61 | -50% |
| C1+S1_600 | u300 | 8 | -1.87 | -1.84 | -0.21 | -0.34 | 0.62 | 0.41 | 0.59 | 0.34 | 0.35/-0.83 | -54% |
| A1sel05 | u1500 | 4 | -0.84 | -0.83 | 0.75 | 0.65 | 1.54 | 1.39 | 1.72 | 1.54 | 1.65/-0.15 | -41% |
| A1sel05 | u1500 | 6 | -1.34 | -1.32 | 0.48 | 0.37 | 1.39 | 1.21 | 1.57 | 1.36 | 1.12/-0.16 | -37% |
| A1sel05 | u1500 | 8 | -1.62 | -1.58 | 0.36 | 0.24 | 1.34 | 1.15 | 1.51 | 1.29 | 1.09/-0.38 | -41% |
| A1sel05 | u600 | 4 | -0.56 | -0.54 | 0.93 | 0.83 | 1.67 | 1.52 | 1.78 | 1.61 | 1.89/-0.07 | -33% |
| A1sel05 | u600 | 6 | -1.13 | -1.18 | 0.58 | 0.39 | 1.43 | 1.18 | 1.53 | 1.24 | 1.56/-0.42 | -44% |
| A1sel05 | u600 | 8 | -1.67 | -1.70 | 0.15 | -0.02 | 1.06 | 0.81 | 1.13 | 0.86 | 1.19/-0.96 | -56% |

| model | universe | k | Sh f=1.0 | fo f=1.0 | Sh f=0.5 | fo f=0.5 | Sh f=0.25 | fo f=0.25 | Sh f=old | fo f=old | halves f=0.5 | mdd f=0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1+S1_k23 | u1500 | 2 | -0.11 | 0.02 | 0.97 | 0.95 | 1.50 | 1.42 | 1.60 | 1.47 | 1.47/0.54 | -43% |
| B1+S1_k23 | u1500 | 3 | -0.28 | -0.26 | 0.98 | 0.83 | 1.60 | 1.38 | 1.71 | 1.42 | 1.47/0.52 | -41% |
| B1+S1_k23 | u600 | 2 | -0.03 | 0.01 | 0.98 | 0.91 | 1.49 | 1.36 | 1.49 | 1.31 | 1.69/0.27 | -33% |
| B1+S1_k23 | u600 | 3 | -0.65 | -0.66 | 0.54 | 0.39 | 1.13 | 0.91 | 1.12 | 0.85 | 1.55/-0.45 | -54% |
| B1x120+S1x120 | u1500 | 4 | -1.26 | -1.18 | 0.29 | 0.21 | 1.06 | 0.90 | 1.28 | 1.06 | 1.03/-0.38 | -58% |
| B1x120+S1x120 | u1500 | 6 | -1.62 | -1.44 | 0.16 | 0.14 | 1.05 | 0.92 | 1.28 | 1.09 | 0.68/-0.34 | -55% |
| B1x120+S1x120 | u1500 | 8 | -1.91 | -1.68 | 0.02 | 0.02 | 0.98 | 0.87 | 1.22 | 1.03 | 0.72/-0.66 | -58% |
| B1x120+S1x120 | u600 | 4 | -0.66 | -0.73 | 0.70 | 0.47 | 1.37 | 1.06 | 1.43 | 1.07 | 1.71/-0.34 | -52% |
| B1x120+S1x120 | u600 | 6 | -1.04 | -1.04 | 0.49 | 0.32 | 1.26 | 0.99 | 1.31 | 0.99 | 1.37/-0.43 | -51% |
| B1x120+S1x120 | u600 | 8 | -1.41 | -1.32 | 0.29 | 0.17 | 1.13 | 0.92 | 1.19 | 0.91 | 1.26/-0.74 | -52% |
| ens4 | u1500 | 2 | -0.17 | -0.11 | 0.87 | 0.79 | 1.39 | 1.24 | 1.47 | 1.28 | 1.65/0.22 | -49% |
| ens4 | u1500 | 3 | -0.38 | -0.31 | 0.88 | 0.78 | 1.50 | 1.33 | 1.59 | 1.35 | 1.53/0.28 | -43% |
| ens4 | u1500 | 4 | -0.71 | -0.65 | 0.69 | 0.58 | 1.39 | 1.18 | 1.48 | 1.20 | 1.42/0.01 | -40% |
| ens4 | u1500 | 6 | -1.44 | -1.32 | 0.20 | 0.11 | 1.01 | 0.81 | 1.10 | 0.83 | 0.99/-0.57 | -54% |
| ens4 | u1500 | 8 | -1.81 | -1.70 | -0.02 | -0.14 | 0.88 | 0.63 | 0.96 | 0.65 | 0.77/-0.77 | -61% |
| ens4 | u600 | 2 | -0.16 | -0.33 | 0.83 | 0.56 | 1.33 | 0.99 | 1.32 | 0.94 | 2.09/-0.45 | -56% |
| ens4 | u600 | 3 | -0.52 | -0.70 | 0.65 | 0.33 | 1.23 | 0.84 | 1.21 | 0.77 | 1.92/-0.65 | -59% |
| ens4 | u600 | 4 | -0.87 | -1.00 | 0.45 | 0.15 | 1.11 | 0.73 | 1.08 | 0.65 | 1.72/-0.84 | -60% |
| ens4 | u600 | 6 | -1.16 | -1.22 | 0.34 | 0.09 | 1.09 | 0.74 | 1.05 | 0.64 | 1.51/-0.84 | -59% |
| ens4 | u600 | 8 | -1.58 | -1.55 | 0.09 | -0.10 | 0.91 | 0.62 | 0.87 | 0.52 | 1.11/-0.98 | -59% |
| B1f1+S1f1 | u1500 | 4 | -1.21 | -1.23 | 0.07 | -0.07 | 0.70 | 0.50 | 0.30 | 0.07 | 0.64/-0.57 | -47% |
| B1f1+S1f1 | u1500 | 6 | -1.76 | -1.77 | -0.32 | -0.47 | 0.40 | 0.17 | -0.05 | -0.30 | 0.16/-0.82 | -50% |
| B1f1+S1f1 | u1500 | 8 | -2.24 | -2.15 | -0.62 | -0.69 | 0.18 | 0.03 | -0.30 | -0.48 | -0.21/-1.07 | -50% |
| B1f1+S1f1 | u600 | 4 | -1.36 | -1.52 | -0.08 | -0.34 | 0.56 | 0.25 | 0.17 | -0.15 | 0.76/-1.06 | -53% |
| B1f1+S1f1 | u600 | 6 | -1.96 | -2.07 | -0.47 | -0.71 | 0.26 | -0.04 | -0.17 | -0.48 | 0.39/-1.46 | -56% |
| B1f1+S1f1 | u600 | 8 | -2.42 | -2.43 | -0.80 | -0.95 | 0.01 | -0.22 | -0.45 | -0.69 | -0.10/-1.57 | -57% |

Causal calibration check on the yearly A1 predictions (fold-OOS at f = 0.5, k = 2 / 3 / 4 / 6 / 8): u1500 0.73 / 0.68 /
0.56 / 0.31 / 0.00, u600 1.08 / 0.79 / 0.64 / 0.18 / -0.06 (the in-sample bucket version gave 0.83 at u600 k4); with the
selection cost at f = 1 the f = 1 books are still negative (best -0.02 at u600 k2) and the f = 0.5 books are worse.
Findings: (i) the two-sided formulation is the first thing positive at f = 0.5 (B1+S1 u1500 k4: Sh 0.66 / fold-OOS 0.56,
halves 1.2 / 0.1, maxDD -43 %); (ii) the cost-aware selection on the gross liq model is as good or better (A1sel u600
k4: 0.64 causal) and needs one model instead of two; (iii) the y_120d pair adds nothing (ens4 u1500 k4 0.58 vs 0.56);
(iv) the edge is in the very top of the ranking: k = 2 / 3 beat k = 4 at every construction (B1+S1 u1500 k2: 0.95,
second half +0.5), but a 4-position book is not what the brief asks for and its per-fold Sharpe is dominated by single
names; (v) f = 1 cannot be rescued by training on it.

## 5. Final: quarterly retrain, n_jobs = 2

### 5a. `sel_A1`: quarterly A1 (EXTL, rank y_60d; IC(y_30d) 0.038, t 15.8) with the causal cost-aware selection at f = 0.5

| universe | k | Sh f=1.0 | fo f=1.0 | Sh f=0.5 | fo f=0.5 | Sh f=0.25 | fo f=0.25 | Sh f=old | fo f=old | Sh f=gross-comm | fo f=gross-comm | halves f=0.5 | mdd f=0.5 | ret f=0.5 | by year f=0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u1500 | 2 | -0.50 | -0.68 | 0.71 | 0.51 | 1.31 | 1.10 | 1.46 | 1.27 | 1.92 | 1.70 | 1.61/-0.20 | -70% | 18% | 17:+2.7 18:+3.3 19:+1.7 20:+1.8 21:+0.4 22:+1.6 23:-0.1 24:-1.3 25:-1.3 26:-1.5 |
| u1500 | 3 | -0.75 | -0.99 | 0.72 | 0.46 | 1.45 | 1.18 | 1.65 | 1.39 | 2.19 | 1.90 | 1.86/-0.40 | -66% | 15% | 17:+2.9 18:+4.2 19:+0.9 20:+2.9 21:-0.1 22:+1.0 23:-0.4 24:-1.2 25:-1.8 26:+0.1 |
| u1500 | 4 | -0.83 | -1.06 | 0.82 | 0.55 | 1.64 | 1.36 | 1.87 | 1.60 | 2.47 | 2.16 | 1.97/-0.38 | -52% | 16% | 17:+3.2 18:+4.2 19:+1.1 20:+3.0 21:-0.1 22:-0.1 23:-0.1 24:-0.8 25:-0.9 26:-1.0 |
| u1500 | 6 | -1.18 | -1.38 | 0.74 | 0.50 | 1.71 | 1.44 | 1.96 | 1.70 | 2.66 | 2.37 | 1.96/-0.53 | -53% | 12% | 17:+2.2 18:+4.7 19:+0.9 20:+3.4 21:-0.4 22:-0.3 23:-0.2 24:-1.1 25:-1.1 26:-0.6 |
| u1500 | 8 | -1.62 | -1.80 | 0.50 | 0.26 | 1.56 | 1.28 | 1.82 | 1.55 | 2.61 | 2.30 | 1.83/-0.92 | -61% | 7% | 17:+2.6 18:+4.2 19:+1.1 20:+2.9 21:-0.2 22:-0.8 23:-0.9 24:-1.2 25:-1.8 26:-1.1 |
| u600 | 2 | -0.10 | -0.32 | 1.09 | 0.84 | 1.68 | 1.42 | 1.82 | 1.57 | 2.27 | 1.99 | 2.31/-0.20 | -58% | 24% | 17:+3.3 18:+4.3 19:+2.9 20:+2.6 21:-0.1 22:+0.4 23:+0.3 24:-1.5 25:-1.0 26:+1.5 |
| u600 | 3 | -0.35 | -0.59 | 1.04 | 0.76 | 1.74 | 1.43 | 1.90 | 1.60 | 2.43 | 2.10 | 2.15/-0.11 | -45% | 20% | 17:+3.2 18:+4.6 19:+2.1 20:+2.8 21:-0.4 22:-0.5 23:+0.5 24:-0.5 25:-0.6 26:+0.8 |
| u600 | 4 | -0.56 | -0.76 | 1.03 | 0.77 | 1.82 | 1.54 | 1.99 | 1.72 | 2.60 | 2.30 | 2.18/-0.19 | -44% | 17% | 17:+2.7 18:+4.5 19:+2.1 20:+3.0 21:-0.4 22:-0.8 23:+0.6 24:-0.9 25:-0.6 26:+1.2 |
| u600 | 6 | -1.08 | -1.23 | 0.72 | 0.51 | 1.61 | 1.37 | 1.79 | 1.55 | 2.51 | 2.23 | 2.00/-0.65 | -52% | 10% | 17:+2.3 18:+4.3 19:+1.7 20:+2.8 21:-0.4 22:-1.6 23:-0.0 24:-0.8 25:-1.0 26:+0.5 |
| u600 | 8 | -1.60 | -1.70 | 0.37 | 0.19 | 1.34 | 1.13 | 1.51 | 1.29 | 2.32 | 2.06 | 1.71/-1.10 | -60% | 5% | 17:+1.9 18:+3.7 19:+1.6 20:+2.7 21:-0.6 22:-1.6 23:-1.0 24:-1.4 25:-1.3 26:+0.0 |
| u300 | 2 | -0.18 | -0.28 | 0.92 | 0.79 | 1.48 | 1.32 | 1.52 | 1.38 | 2.03 | 1.86 | 1.86/-0.06 | -49% | 18% | 17:+2.1 18:+3.6 19:+1.7 20:+2.3 21:-0.4 22:-1.0 23:+0.0 24:-0.8 25:+0.7 26:+1.2 |
| u300 | 3 | -0.61 | -0.69 | 0.69 | 0.57 | 1.34 | 1.20 | 1.38 | 1.25 | 1.99 | 1.83 | 1.64/-0.31 | -48% | 11% | 17:+1.9 18:+3.3 19:+1.2 20:+2.3 21:-0.4 22:-1.1 23:-0.7 24:-0.2 25:+0.0 26:+0.7 |
| u300 | 4 | -0.69 | -0.76 | 0.76 | 0.64 | 1.47 | 1.33 | 1.50 | 1.37 | 2.19 | 2.02 | 1.69/-0.23 | -41% | 11% | 17:+2.4 18:+3.1 19:+1.4 20:+2.6 21:-0.6 22:-0.8 23:-0.8 24:-0.4 25:+0.3 26:+0.9 |
| u300 | 6 | -1.14 | -1.17 | 0.48 | 0.39 | 1.28 | 1.17 | 1.29 | 1.18 | 2.09 | 1.95 | 1.37/-0.46 | -47% | 6% | 17:+1.9 18:+2.7 19:+0.5 20:+2.8 21:-0.7 22:-0.6 23:-1.8 24:-0.5 25:-0.2 26:+1.0 |
| u300 | 8 | -1.63 | -1.66 | 0.15 | 0.05 | 1.03 | 0.91 | 1.02 | 0.91 | 1.92 | 1.76 | 1.11/-0.86 | -55% | 2% | 17:+1.8 18:+2.4 19:+0.6 20:+2.5 21:-1.4 22:-1.1 23:-2.4 24:-0.5 25:-0.7 26:+0.8 |

The same quarterly model traded as a plain gross book (`gross_A1`, no cost in the selection), u1500: old-convention
fold-OOS k4 2.64 / k6 2.90 / k8 2.97 (dayml2_ens_e30_k8: 2.40); at f = 0.5 -1.72 / -2.31 / -2.78.

### 5b. Saved candidates (`agents/dayml3/cands/`, series = gross - f x spread cost - borrow, BEFORE commission)

| candidate | at $0.4M (commission added) | corr dayml2_ens_e30_k8 / onlgb1200q_ls20 / SPY | tiers: share of positions / net bps per trade at its f | |
|---|---|---|---|---|
| dayml3_liqsel_u600_x05_k4 | Sh 1.03 fold-OOS 0.77 halves 2.18/-0.19 mdd -44% | dayml2_ens_e30_k8 0.43 onlgb -0.04 SPY -0.03 | 1-300:0.70/+3.6bps 301-600:0.30/+5.5bps 601-1500:0.00/- |  |
| dayml3_liqsel_u600_x05_k6 | Sh 0.72 fold-OOS 0.51 halves 2.00/-0.65 mdd -52% | dayml2_ens_e30_k8 0.45 onlgb -0.03 SPY -0.03 | 1-300:0.71/+3.1bps 301-600:0.29/+3.0bps 601-1500:0.00/- |  |
| dayml3_liqsel_u600_x1_k6 | Sh -1.08 fold-OOS -1.23 halves 0.00/-2.25 mdd -85% | dayml2_ens_e30_k8 0.45 onlgb -0.04 SPY -0.02 | 1-300:0.71/-1.0bps 301-600:0.29/-4.1bps 601-1500:0.00/- | NEGATIVE at this spread assumption: not tradable |
| dayml3_liqsel_u600_x05_k8 | Sh 0.37 fold-OOS 0.19 halves 1.71/-1.10 mdd -60% | dayml2_ens_e30_k8 0.45 onlgb -0.05 SPY -0.03 | 1-300:0.72/+2.4bps 301-600:0.28/+2.1bps 601-1500:0.00/- |  |
| dayml3_liqsel_u600_x1_k8 | Sh -1.60 fold-OOS -1.70 halves -0.44/-2.87 mdd -88% | dayml2_ens_e30_k8 0.45 onlgb -0.05 SPY -0.02 | 1-300:0.72/-1.7bps 301-600:0.28/-4.8bps 601-1500:0.00/- | NEGATIVE at this spread assumption: not tradable |

f-sensitivity of the kept k4 / k6 / k8 books (u600, $0.4M): see the `sel_A1` table above (columns f = 1 / 0.5 / 0.25 / old);
at f = 0.25 the k4 book is 1.82 / 1.54 and at the old convention 1.99 / 1.72.  Yearly Sharpe of `dayml3_liqsel_u600_x05_k4`
at $0.4M: 2017 +2.7, 2018 +4.5, 2019 +2.1, 2020 +3.0, 2021 -0.4, 2022 -0.8, 2023 +0.6, 2024 -0.9, 2025 -0.6, 2026 +1.2.
Correlation with `dayml2_ens_e30_k8` is 0.43-0.45 (half the positions are different names), ~0 with the overnight sleeve
and SPY.

### 5c. `pair`: quarterly B1 + S1 (two-sided net targets) -- running, see log.txt / final_pair.log

## 6. Leakage checklist

* Every feature of a row is known at the end of the 09:31 bar of day t: `dlib` close-time features are shifted one day
  inside the ticker; the `o_*` / gap / first-minute features and the `t_hl0 t_hl1 t_n0 t_n1 t_dv0` liq columns use bins 0-1
  of day t only; the lagged liq proxies (`ldv20 ... dvc`) are 20-day means ending at t-1 (`_lag_mean` shifts by one);
  `hs0931` / `hs1001` are functions of those columns only (the spread model itself is a cost calibration fit on
  2017-2026 quotes, in-sample by construction, as the brief states).
* No 09:32 fill anywhere: entry = bin 3 close, exit = bin 31 close (`y_30d` of `d2lib.extra2`).
* Walk-forward: expanding window, label lag 3 days before each cut, yearly (exploration) / quarterly (final) retrain from
  2017-01-01, seed 0; the universe mask of the C / S1_600 models applies to training rows and predictions alike.
* The causal calibration of the net score uses predictions and realised y_30d strictly before each half-year cut minus
  3 days; the first window (2017-01..06, outside the fold-OOS span) uses the fixed linear slope 12.4 bps x rank.
* Books: equal weight 1/k per side, every name's own predicted spread, borrow 1 %/yr on shorts, commission 2k x 2 x $3 /
  $0.4M per day; `|y| > 0.5` rows are NaN (unadjusted splits).  Dates <= DEV_END; `STRAT_END` never set.
* Choices made in-sample on the dev span: the feature set, k, the universe cut (u600) and the f = 0.5 selection cost.
  The in-sample bucket calibration of step 2 / `A1sel05` is reported as such and not used in any candidate.

## 7. What I would keep

* Nothing from this lane as a trading book at f = 1, and nothing at f = 0.5 for the post-2021 period: the realistic
  intraday book is dead since 2021 on every construction (the gross edge at the mid halved while spreads widened from a
  14 bps median entry half-spread in 2017 to 32 in 2026).  `dayml3_liqsel_u600_x05_k4` is the best honest f = 0.5 series
  (0.77 fold-OOS, 2.2 before 2021-10, -0.2 after) and is only worth holding in the chooser if the coordinator wants an
  f = 0.5 sleeve at all; the x1 series exist to document that crossing the spread loses.
* The liq features themselves (`EXTL`): +0.8 fold-OOS at the mid-quote convention on the same book.  If the coordinator
  keeps any mid-quote intraday series for comparison, `gross_A1` (quarterly, k8, old convention 2.97) supersedes
  `dayml2_ens_e30_k8` (2.40); it is NOT saved as a candidate because the brief's convention is the spread-costed series.
* The two-sided treatment of execution cost (score each side as expected gross minus its own cost) is the construction
  to reuse for any other book that is re-costed at the NBBO; the single net-rank target should not be repeated.
