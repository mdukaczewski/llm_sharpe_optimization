# dayml lane — learned intraday books decided at 09:31 (and later) — REPORT

(draft; sections marked TODO are filled when the universe(1500) runs finish)

## Setup
* Code: `agents/dayml/dlib.py` (feature builder + walk-forward wrapper over `rlib`), `exp1.py` (exploration on the
  available universe), `post1.py` (construction variants on saved predictions), `expq.py` (quarterly vs yearly),
  `exp2.py` (universe(1500) runs), `final.py` (regenerates the kept candidates; honours STRAT_END).
* Alignment (built in `dlib.ticker_rows`): every close-time feature of `rlib.ticker_frame` is shifted by one day
  within the ticker BEFORE half days are dropped, so a row dated t carries yesterday's close-time features plus
  today's `o_zgap / o_zm1 / o_vm1_rel` (known at the end of the 09:31 bar).  Extra features from `ilib.Min`
  (opened with `end=DEV_END`): raw gap, realised overnight return close(t-1)->09:31(t) (`ron`, z-scored `o_zron`),
  first-minute return, SPY's gap / first minute / overnight of day t, SPY's lagged z60/z30/zoc, cross-sectional
  percentile ranks of gap / overnight / first minute, market-relative gap (gap - median gap of the day),
  gap / lagged rv20, day of week, log dollar volume (lagged).  43 features ("EXT"); "BASE" = the 23 features of
  `on_ml_day.py`.
* Targets built explicitly from `Min.c` (fill = close of bin m+1 for a decision at bin m; |y| > 0.5 -> NaN):
  `y_day` = close / c[:,2] - 1 (rlib), `y_60` = c[:,60]/c[:,2]-1 (exit 10:30 bar), `y_120` = c[:,120]/c[:,2]-1
  (exit 11:30 bar), `y_1000` = close / c[:,31] - 1 (decide 10:00, features = first 30 minutes), `y_1500` =
  close / c[:,331] - 1 (decide 15:00).  Model target = cross-sectional percentile rank of y per day.
* Model: `rlib.walk_forward` LightGBM (300 trees, 15 leaves, lr 0.03, min_child 200), expanding window,
  label lag 3 days, first OOS 2017-01-01, yearly retrain for exploration, quarterly for the final runs.
* Book: `rlib.book` long top-k / short bottom-k, equal or |pred|-weighted (sum |w| = 1 per side), costs 1 bp/side
  (1.5 bps for names ranked > 600), borrow 1%/yr on shorts.  Half days skipped (minute_frame convention).

## Exploration on the 269 names available at the start (top of univ1500 by dollar volume), yearly retrain
Full-sample Sh / fold-OOS Sh (rlib.fold_oos of the single series, 16 half-year folds from 2018-07):

| decision -> exit | features | k, weight | Sh | halves | mdd | fold-OOS | neg folds | @1.5bp |
|---|---|---|---|---|---|---|---|---|
| 09:31 -> close (baseline on_ml_day.py) | BASE 23 | 20 eq | 1.19 | 1.84/0.54 | -32% | 1.15 | 3/16 | 0.88 |
| 09:31 -> close | BASE | 20 pred | 1.22 | 1.95/0.53 | -32% | 1.18 | 3/16 | 0.94 |
| 09:31 -> close | EXT 43 | 20 eq | 1.20 | 1.73/0.63 | -30% | 1.10 | 1/16 | 0.91 |
| 09:31 -> close | EXT | 20 pred | 1.34 | 2.12/0.55 | -33% | 1.24 | 1/16 | 1.08 |
| 09:31 -> close | EXT | 10 pred | 1.47 | 2.34/0.62 | -32% | 1.39 | 2/16 | |
| 09:31 -> 10:30 (bin 60) | EXT | 20 eq | 1.53 | 2.22/0.76 | -18% | 1.38 | 4/16 | 1.12 |
| 09:31 -> 10:30 | EXT | 20 pred | 1.83 | 2.58/1.05 | -18% | 1.67 | 2/16 | 1.44 |
| 09:31 -> 10:30 | EXT | 10 pred | 2.07 | 2.90/1.28 | -24% | 1.91 | 2/16 | |
| 09:31 -> 11:30 (bin 120) | EXT | 20 eq | 1.57 | 2.31/0.82 | -19% | 1.53 | 3/16 | 1.22 |
| 09:31 -> 11:30 | EXT | 20 pred | 1.83 | 2.71/1.00 | -19% | 1.78 | 3/16 | 1.51 |
| 09:31 -> 11:30 | EXT | 30 pred | 1.90 | 2.81/1.02 | -18% | 1.85 | 3/16 | |
| 09:31 -> 11:30 | EXT | 10 pred | 2.03 | 3.06/1.09 | -23% | 1.97 | 1/16 | |
| **09:31 -> 11:30, QUARTERLY retrain** | EXT | 20 pred | **2.25** | 3.19/1.36 | -19% | **2.06** | 2/16 | |
| 09:31 -> 11:30, quarterly | EXT | 10 pred | 2.28 | 3.32/1.32 | -22% | 2.05 | 1/16 | |
| 09:31 -> 11:30, quarterly | EXT | 30 pred | 2.17 | 3.20/1.18 | -18% | 1.98 | 1/16 | |
| 10:00 (bin 30) -> close, first-30-min feats | F1000 54 | 20 eq/pred | 0.07 / 0.16 | | -39% | -0.06 | 7/16 | -0.32 |
| 15:00 (bin 330) -> close | F1500 54 | 20 eq/pred | -0.18 / 0.06 | | | -0.18 | 11/16 | -1.49 |

Daily IC (rank pred vs raw y): close exit BASE 0.022 (t 6.7), EXT 0.024 (t 7.0); 10:30 exit 0.031 (t 9.0);
11:30 exit 0.031 (t 8.8), quarterly 0.036 (t 9.9); 10:00 decision 0.012 (t 3.9); 15:00 decision 0.028 (t 10.9)
but the one-hour move is too small for 2 bps round trip (dead, as the brief predicted: fold-OOS < 1.2 -> dropped).

Findings:
* The predictable part of the 09:31 signal is realised in the first 1-2 hours: the 11:30 exit has the same IC as
  the close exit on a return with half the variance, so the book Sharpe rises from ~1.3 to ~1.8-2.3 and the
  max drawdown roughly halves (-19% vs -32%).  The close-exit book captures the same alpha plus 4.5 hours of noise.
* The extra 09:31 features (cross-sectional ranks of gap / overnight, market-relative gap, gap/rv20, SPY overnight,
  day of week, prior-day zvwap / vlast_share / vmoc_share) add little on their own at the close horizon
  (1.19 -> 1.20 eq) but combine well with |pred| weighting (1.34).
* |pred|-weighting and smaller k help everywhere (k10 > k20 > k30 > k40); the gain comes from the tails of the
  prediction.  On 1500 names the tails are deeper, so k = 10/20 are the candidates; k10 is kept only as a second
  variant of the same model (same predictions, different concentration).
* Quarterly retrain is the single largest improvement (+0.4 Sh at the 11:30 horizon, 2017 goes from +1.8 to +4.2
  because the model starts learning from the 2016 H2 data within the year).
* Cross-model ensembling of the rank predictions (average of the y_60 / y_120 / y_day models) was worse than any
  single model; the y_60 model traded to the 11:30 exit is as good as the y_120 model (fold-OOS 1.94, 0 negative
  folds), i.e. the two horizons are the same signal.
* Requiring >= 300 names in the cross-section is moot on 269 names; on 1500 names it only trims the first months
  (reported as the "min300" variant in the 1500-name log).
* Regime check (yearly Sharpes, 269 names, 11:30 exit quarterly k20): 17:+4.2 18:+5.3 19:+3.6 20:+3.6 21:+1.4
  22:+1.2 23:-0.0 24:+2.6 25:+1.3 26:-0.1.  The decay starts in 2021, not 2022: the second half is ~1.3 vs ~3.2.
  The close-exit book shows the same pattern with 2022 negative (22:-0.6/-0.9) and 2023 ~0.  So it is not a
  2022-only (rates) effect but a structural halving of the first-hour reversal/continuation edge after 2020.
* Costs: the 11:30-exit book loses ~0.3 Sh per extra 0.5 bp/side (2.25 -> ~1.9 at 1.5 bps on these top names;
  the 1500-name version already pays 1.5 bps on ranks > 600 inside rlib.book).

## universe(1500) runs
Launched unattended (`agents/dayml/drive3.sh` -> `exp2.py y_120,y_60 Q` then `exp2.py y_day Y`, log
`agents/dayml/log3.txt`).  Each run appends its table (Sh, halves, mdd, fold-OOS, 1.5-bp cost, correlation with
sleeves/daylgb1200_ls20.csv, sleeves/onlgb1200q_ls20.csv and SPY close-to-close) to the end of this file and
writes the candidates with rlib.save_cand: dayml_x120q_k10p / k20p / k30p, dayml_x60q_k10p / k20p / k30p
(quarterly), dayml_xdayy_k10p / k20p (close exit, yearly; the reference comparable to the existing sleeve).
Expected: the ~0.5 Sh breadth gain seen for the close-exit book (269 -> 1200 names: 1.2 -> 2.0 in-sample)
should carry over to the 11:30 exit.

## What I would keep
`dayml_x120q_k20p` (09:31 LightGBM, EXT features, quarterly retrain, long/short 20 |pred|-weighted, exit 11:30)
as the primary; `dayml_x60q_k20p` (10:30 exit) as the second if its correlation with the 11:30 book is < 0.8,
otherwise only the k10 variant of the 11:30 book.  The close-exit book (dayml_xdayy_*) is dominated by them on
fold-OOS and drawdown and correlates 0.4-0.5 with the existing daylgb1200 sleeve; the 11:30 book correlates
~0.45 with daylgb1200, ~0.0 with the overnight LightGBM sleeve and ~0.0 with SPY.  `final.py` KEPT lists
exactly these three (x120q k20p, x120q k10p, x60q k20p); drop entries from KEPT if the 1500-name table
below says otherwise.

## final.py
`nohup python3 agents/dayml/final.py > agents/dayml/final.log 2>&1 &` rebuilds everything from the minute npz
(per-ticker caches in data/cache and data/cache_dayml are keyed on the npz mtime and on DEV_END, so a run with
STRAT_END=2026-10-28 rebuilds the dayml extras automatically), fits two quarterly walk-forwards on universe(1500)
with n_jobs=4 and writes the three KEPT candidates.  Expected runtime alone on 4 cores: feature build 5-10 min
(1500 names, ~2 min per 400 names when caches are cold), then ~45-60 min per quarterly model (37 expanding fits
on up to ~3.4M rows x 43 features), i.e. about 2-2.5 hours in total; RAM ~4 GB.

## Leakage checklist
* All close-time features are shifted one day within the ticker before the half-day drop (so the day after a half
  day uses the half day's close-time features, which are known).  o_* and Min extras use only bins 0-1 of day t.
* Z-scores: rlib.zsd (rolling std lagged one day).  rv20 is yesterday's (shifted) - the lookahead that broke the
  first intraday run in the README is excluded by construction (checked: no unshifted rv column in EXT).
* Cross-sectional ranks / medians are over names of the same day at 09:31 (all known at that time); the cross-section
  is the minute universe (survivorship inherent in univ1500, as for every lane).
* Targets: fill at c[:,2] (09:32 bar) for a decision at the end of bin 1; exits at c[:,60] / c[:,120] / official
  close; |y| > 0.5 -> NaN.  Label lag 3 days before each training cut; expanding window; first OOS 2017-01.
* Selection: k, weighting, horizon and retrain frequency were chosen on the 269-name exploration (in-sample for
  those names); the 1500-name numbers are a single run per configuration, no further choice.  The strongest
  in-sample choice is quarterly vs yearly (one comparison) and the exit horizon (three horizons).
