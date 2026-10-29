# onml lane — learned overnight book (LightGBM, close -> 09:31), 2026-10-29

Lane files: `agents/onml/olib.py` (features / targets / book construction), `exp1..5.py` (exploration),
`final.py` + `keep.py` (regenerates only the kept candidates), `rep.py` (report table), `log.txt`, `out/` (series of every run).
All dates <= rlib.DEV_END (2026-04-27); costs = rlib.cost_bps (1 bp/side ranks <= 600, 1.5 bps beyond) + 1%/yr borrow;
fills: decision at the close fills at the official close (bin 390), exit fills at the 09:31 bar (bin 1).

## 0. Baseline reproduction (first 143 names, FEATS18, yearly retrain)
`rlib.minute_frame + add_market + clean`, target = per-day pct-rank of y_on - 0.5, `walk_forward(FEATS18, retrain='Y')`,
`book(k=20, mode='ls')`: **Sharpe 2.00 / fold-OOS 2.09** — identical to logs/baseline_local.log (needs the universe row order
and LightGBM's default seed). Same data, rows sorted alphabetically or other seeds: 1.74 / 1.79 / 1.84 (fold 1.78-1.94).
=> single-run seed/row-order noise on 143 names is about +-0.15 Sharpe; on 400 names the 3-seed spread was 2.21-2.40.

## 1. Exploration at fixed breadth: universe(1500)[:400], yearly retrain, seed 0, L/S top-20 eq-weight (exp1, exp3, exp4)
| feature set (n feats)                                   | full | fold-OOS | halves    |
|---------------------------------------------------------|------|----------|-----------|
| f18 = FEATS18 (18)                                      | 1.43 | 1.57     | 1.72/1.10 |
| f28 = FEATS18 + EXTRA10 (28)                            | 2.16 | 2.36     | 2.53/1.76 |
| f28 + per-day cross-sectional ranks of 13 feats (41)    | 2.06 | 2.18     | 2.60/1.42 |
| f28 + xs ranks + market-relative z60/z30/zcc/zoc (45)   | 2.18 | 2.35     | 2.60/1.72 |
| fall = f28 + xr + mr + CAL + nnames (50)                | 2.40 | 2.55     | 2.96/1.76 |
| fall, seeds 1 / 2                                       | 2.29 / 2.21 | 2.43 / 2.34 | |
| fall, 3-seed prediction average                         | 2.35 | 2.45     | 2.87/1.76 |
| fall without nnames (49)                                | 2.30 | 2.51     | 2.73/1.82 |
| **f28cal = f28 + CAL (32)**                             | 2.37 | 2.54     | 2.82/1.86 |
| f28cal + nnames (33)                                    | 1.98 | 2.17     | 2.24/1.69 |
| fall, quarterly retrain                                 | 2.66 | 2.81     | 3.27/1.98 |
| fall, avg of 3 yearly seeds + quarterly                 | 2.60 | 2.76     | 3.28/1.84 |

EXTRA10 = z15l z5l zmoc zvwap vfirst_share vmoc_share ldv cc21 rv5_20 spy_zgap (all already in rlib.ticker_frame);
CAL = dow, business days to month end (dme), quarter-end dummy (qend), days since last |zgap| > 2.5 (dsgap, capped 60).
Reading: the ten extra close-time features are the whole gain (+0.7..0.8 Sharpe, 4-5x the seed noise); the calendar
dummies add ~+0.2; cross-sectional ranks / market-relative versions add nothing (the target is already a per-day rank, and
LightGBM splits are rank-invariant per feature); nnames (universe size that day) is a time proxy and is noise.
Quarterly retrain adds ~+0.25 as in the README. Seed averaging does not raise the mean (it only removes the spread).

Targets (fall feature set, yearly): rank target 2.40/2.55; vol-scaled target (y_on / lagged 60d std) 1.29/1.55;
decile classification (+1/-1/0 regression) 2.23/2.46; rank-average of the three models 2.45/2.65; rank+vol 2.13/2.30.
=> keep the rank target (ensembling targets is within noise).

Exit at 10:31 (y_on60): model trained on the 10:31 rank target 1.13/1.12; the 09:31 model held to 10:31 1.74/1.84 —
both far below the 09:31 exit (2.40/2.55); confirmed dead for the learned book.

Construction on the fall predictions (exp2, no retraining): k20 eq 2.40/2.55, k30 2.34/2.47, k40 2.30/2.43;
k20 pred-weighted 2.51/2.67, k30 pred-weighted 2.47/2.62; 1/rv20 weights 2.35/2.51 (k20), 2.19/2.33 (k40);
per-day |pred| quantile filter (k30 q60) 2.30/2.42, (k40 q70) 2.18/2.28; even/odd rank halves 2.34 / 1.91 (both
positive, both halves of the sample positive); 1.5 bps/side: k20 2.04/2.21 (-0.35). Pred-weighting is a small, consistent
plus (+0.1 at 400 names, +0.1..0.15 at 1500); the quantile filter and vol scaling do not help.

## 2. Final breadth: universe(1500) = 1497 names, quarterly expanding retrain, seed 0 (exp5)
Same breadth, same construction, same seed; fold-OOS = rlib.fold_oos of the single series (16 half-year folds from 2018-07-01).

| 1497 names, quarterly retrain, seed 0          | full | fold-OOS | halves    | maxDD  | corr sleeve | at 1.5 bps/side |
|------------------------------------------------|------|----------|-----------|--------|-------------|-----------------|
| f18 (= existing sleeve recipe), k20 eq         | 3.20 | 3.22     | 3.71/2.62 | -18.2% | 0.68        | 3.08 / 3.12     |
| f18, k20 pred-weighted                         | 3.31 | 3.35     | 3.91/2.69 | -19.1% | 0.65        |                 |
| **f28cal, k20 eq**                             | 3.93 | 3.99     | 4.37/3.44 | -13.4% | 0.59        | 3.81 / 3.87     |
| f28cal, k20 pred-weighted                      | 3.97 | 4.04     | 4.42/3.49 | -14.0% | 0.58        |                 |
| f28cal, k30 eq                                 | 4.01 | 4.04     | 4.63/3.31 | -12.5% | 0.62        |                 |
| **f28cal, k30 pred-weighted**                  | 4.10 | 4.15     | 4.68/3.46 | -11.6% | 0.61        |                 |
| f28cal, even / odd rank halves (k20)           | 3.18 / 3.57 | 3.11 / 3.75 |      | -7.2% / -11.1% |      |                 |
| fall (50 feats), k20 eq                        | 3.91 | 3.84     | 4.18/3.62 | -21.9% | 0.53        | 3.80 / 3.74     |
| fall, k20 pred-weighted                        | 4.18 | 4.13     | 4.38/3.95 | -22.7% | 0.52        |                 |
| fall, even / odd halves (k20)                  | 3.52 / 3.22 | 3.47 / 3.21 |      |        |             |                 |
| f28cal, yearly retrain, k20 eq                 | 3.65 | 3.74     | 3.89/3.38 | -13.4% | 0.53        | 3.52 / 3.62     |
| rank-average of the four models above, k20     | 3.91 | 3.90     | 4.27/3.49 | -15.4% | 0.62        | 3.79 / 3.78     |
| existing sleeve onlgb1200q_ls20 (dev cut)      | 3.40 | 3.49     | 4.28/2.41 | -19.1% | 1.00        |                 |

Family fold-OOS over the 12 non-cost variants of this table (rlib.fold_oos picks per fold): 3.79, min fold 0.86, no negative fold.
The feature gain carries from 400 names (+1.0) to 1500 names (+0.77 fold-OOS at k20, same seed, same construction), i.e.
it is additive to the breadth gain, which was the only lever found earlier in the project. The sleeve-recipe rerun on 1497
names (3.20) is a little below the shipped 1200-name sleeve (3.40 on the dev window): names 1201-1500 cost 1.5 bps and add
little; the difference is also within seed/row-order noise. Correlation of the new book with the existing sleeve is 0.59,
lower than expected — the extra features change about half of the picks.

## 3. Kept candidates (agents/onml/cands, written by `final.py`)
All three use feature set f28cal (32 features), rank target, LightGBM baseline params, quarterly expanding retrain from
2017-01-01 (label lag 3 days), universe(1500) = 1497 names, L/S book exit at the 09:31 bar, rlib costs.  Numbers from the
final.py dev run (agents/onml/final.log, 2852 s at n_jobs=2 on the shared box); `python3 agents/onml/rep.py` reprints them.

| candidate              | members                    | book            | full | fold-OOS | halves    | maxDD  | ret/vol      | corr sleeve | corr SPY | min fold |
|------------------------|----------------------------|-----------------|------|----------|-----------|--------|--------------|-------------|----------|----------|
| onml_f28cal_q_k20      | seed 0                     | top/bot 20 eq   | 3.93 | 3.99     | 4.37/3.44 | -13.4% | 68.1%/17.3%  | 0.59        | -0.04    | 0.77     |
| onml_f28cal_q3_k20     | seeds 0,1,2 rank-averaged  | top/bot 20 eq   | 3.75 | 3.76     | 4.10/3.35 | -15.3% | 64.6%/17.2%  | 0.60        | -0.03    | 1.10     |
| onml_f28cal_q3_k30p    | seeds 0,1,2 rank-averaged  | top/bot 30 pred-w | 3.84 | 3.83   | 4.34/3.26 | -12.7% | 56.8%/14.8%  | 0.63        | -0.02    | 0.95     |
| family (rlib.fold_oos over the three)                           |                 |      | **3.92** | folds mean 4.05, min 0.77, 0/16 negative |
| existing sleeve onlgb1200q_ls20, same dev window                |                 | 3.40 | 3.49     | 4.28/2.41 | -19.1% | 60.2%/17.7%  | 1.00        |          | 0.07     |

Yearly Sharpe (q3_k20): 17 +3.3, 18 +4.3, 19 +4.1, 20 +5.0, 21 +4.1, 22 +3.7, 23 +3.0, 24 +4.2, 25 +3.8, 26(Jan-Apr) +1.1
(sleeve: 22 +1.2, 23 +2.4, 26 +1.5).  Candidates are 0.93-0.95 correlated with each other.
Cost sensitivity (flat 1.5 bps/side instead of rlib.cost_bps, seed-0 k20 book from exp5): 3.93/3.99 -> 3.81/3.87, i.e.
-0.12 Sharpe (the 1200-name sleeve lost ~0.35 under the same change per README: the new book turns over the same 80 names
a day but its gross return is higher).  Expect the same -0.12..-0.15 for the two ensembles.
Honesty note on the seed-0 line: seed 0 happens to be the best of the three seeds at 1500 names (3.93 vs 3.75 for the
3-seed average); the 3.75-3.85 ensemble numbers are the central estimate, 3.93/3.99 is the optimistic one.  Even so every
variant is 0.3-0.5 above the existing sleeve on the identical dev window with a lower drawdown and a far better 2022-23.

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
the shared box (47.5 min measured for the dev run at n_jobs=2 while sharing the box; ~25-30 min alone with n_jobs=4); memory ~2 GB.
With STRAT_END=2026-10-28 the frame grows by 6 months and the cuts add two quarters: budget ~1 hour.
