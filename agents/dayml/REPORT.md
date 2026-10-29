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
