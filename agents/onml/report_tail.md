
## 4. What I would keep, and why
* **onml_f28cal_q3_k20** (3-seed average, top/bottom 20 equal weight) as the replacement of sleeves/onlgb1200q_ls20: same
  recipe as the existing sleeve (rank target, quarterly expanding retrain, LightGBM 300 trees / 15 leaves / lr 0.03 /
  min_child_samples 200, k=20 equal weight) with 14 extra close-time / calendar features and the full 1500-name universe.
  The seed average removes the +-0.15 single-seed spread without changing the mean; the single-seed version
  (onml_f28cal_q_k20) is kept so the coordinator can see that the ensemble is not doing the work.
* **onml_f28cal_q3_k30p** (top/bottom 30, prediction-weighted) is the higher-Sharpe / lower-drawdown construction variant;
  pred-weighting was a consistent +0.1 at both 400 and 1500 names and k30 only helps once breadth is 1500. It holds 60 names
  instead of 40, so it is the more capacity-friendly line, but it is one more construction choice made after seeing the data.
* Not kept: the 50-feature 'fall' set (same fold-OOS, -22% drawdown, cross-sectional ranks add nothing at 400 names),
  the 10:31 exit, vol-scaled and classification targets, 1/rv20 weights, |pred| quantile filters, yearly retrain.

## 5. Leakage / robustness checklist
* All features come from rlib.ticker_frame (close-time features, lagged rolling std in zsd) or the calendar; dsgap uses
  |zgap| of day t and earlier (gap of day t is known at the open of t); dme/qend/dow are known in advance.
  No feature uses y_on, the 09:31 bar of t+1, or any forward quantity.  The vol-scaled target (not kept) used a shift(1)
  rolling std of past y_on.
* Training rows end label_lag_days=3 before each quarterly cut; test rows are [cut, next cut).  The per-day rank target is
  built within each day only.  Predictions are rank-normalised per day before averaging (no cross-day information).
* Fills/costs = rlib.book conventions: close fill at the official close, exit at the 09:31 bar, 1 bp/side (1.5 bps for ranks
  > 600), borrow 1%/yr; a 1.5 bps/side flat cost costs 0.12-0.15 Sharpe at 1500 names.
* Universe = data/univ1500.txt ranked by today's median dollar volume (survivorship, same as the existing sleeve);
  minute data with |y| > 0.5 removed (split guard); v2 caches split-adjust the forward targets.
* Selection noise: the feature set was chosen among 5 at 400 names (fold-OOS gap 1.0 vs seed noise 0.15), the construction
  among ~10 cheap variants; only f28cal and fall were run at 1500 names, and only f28cal was carried into final.py.
* Remaining in-sample element: the EXTRA10 features were written into rlib.ticker_frame earlier in the project with the
  whole sample visible, and the LightGBM parameters are the baseline ones (not re-tuned).

## 6. final.py
`nohup python3 agents/onml/final.py > agents/onml/final.log 2>&1 &` rebuilds the three kept candidates from
data/minute_npz via rlib (universe(1500), ONML_NJOBS default 4), writes them with rlib.save_cand("onml", ...) and honours
STRAT_END through rlib.DEV_END (loaders, walk-forward cuts and book() all cut there).  Expected runtime: ~40 s load
(warm cache; ~15 min if the per-ticker cache must be rebuilt) + 3 quarterly walk-forwards of ~15 min each at n_jobs=2 on
the shared box (46 min measured for the dev run; ~25-30 min alone with n_jobs=4), i.e. about 30-50 min; memory ~2 GB.
With STRAT_END=2026-10-28 the frame grows by 6 months and the cuts add two quarters: budget ~1 hour.
