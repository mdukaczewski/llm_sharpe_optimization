# dayml2 lane — intraday 09:31 book at a fixed $3/order commission on $0.4M — REPORT

Scope: raise the fold-OOS Sharpe of the learned intraday book (decide at the 09:31 bar, fill at the 09:33 bar = bin 3,
targets y_60d / y_120d of `agents/dayml/dlib.py`) at a fixed commission of $3 per filled order on $0.4M of equity per
book: `r_net = r - P x $6 / $400k` per day (P positions: L/S top-6 = 12 positions = 1.8 bps/day, top-8 = 2.4 bps/day)
on top of rlib's spread cost; metric `rlib.fold_oos(r_net)` (16 half-year folds from 2018-07, dates <= 2026-04-27).
Baseline (agents/dayml/booklab.csv, quarterly LightGBM on dlib.EXT): `dayml_x60dq_k6` L/S top-6 eq fold-OOS **2.02**
($0.4M) / 2.18 (spread-only), `dayml_x120dq_k6` 1.91.

Code: `d2lib.py` (extras + lean walk-forward + commission-aware books), `lab1.py` / `lab1b.py` (construction on the
SAVED quarterly predictions `agents/dayml/out/pred_y_{60d,120d}_Q.parquet`, cut at DEV_END, no refit), `lab2.py`
(yearly-retrain refit sweeps on universe(1500)), `lab3.py` (constructions on the refit predictions), `final.py`
(regenerates only the kept candidates; `--from-saved` builds them from the saved predictions).  Logs: `log.txt`,
`lab1*.log`, `lab2a.log`, `lab3.log`, `final.log`; tables `lab1.csv`, `lab1b.csv`, `out/lab2_*.csv`, `out/lab3_*.csv`.
The 09:32-bar fill is never used (bid-ask bounce leak, see agents/dayml/REPORT.md); all exits are from the bin-3 fill.

## 1. Construction on the saved quarterly predictions (no refit) — lab1 / lab1b
Full-sample Sh / fold-OOS at spread-only and at the $0.4M commission (all L/S, equal weight unless stated):

| book | spread Sh / fo | $0.4M Sh / fo | halves ($0.4M) |
|---|---|---|---|
| y_60d model, k6 eq (baseline) | 2.27 / 2.18 | 2.11 / 2.02 | 2.88 / 1.39 |
| y_60d, k4 eq | 2.13 / 2.05 | 2.04 / 1.97 | |
| y_60d, k8 eq | 2.34 / 2.18 | 2.09 / 1.94 | |
| y_60d, k6 pred-weight | 2.22 / 2.12 | 2.07 / 1.98 | |
| y_60d, long-only k6 | 2.05 / 1.91 | 1.92 / 1.79 | 2.68 / 1.03 |
| y_120d model, k6 eq | 2.27 / 2.04 | 2.14 / 1.91 | 3.37 / 1.07 |
| y_120d, long-only k6 | 1.64 / 1.52 | 1.53 / 1.41 | |
| **y_60d model traded to the bin-31 exit (10:01), k6** | 2.57 / 2.47 | 2.37 / **2.28** | 3.25 / 1.53 |
| **y_120d model traded to bin 31, k6** | 2.72 / 2.49 | 2.54 / **2.31** | 3.57 / 1.67 |
| y_120d model traded to bin 61, k6 | 2.45 / 2.20 | 2.30 / 2.05 | |
| y_60d model traded to bin 91 (11:01), k6 | 2.08 / 2.02 | 1.93 / 1.88 | |
| **ensemble (avg per-day pct rank of the two models) -> bin 31, k6** | 2.84 / 2.55 | 2.64 / **2.36** | 3.83 / 1.55 |
| **ensemble -> bin 31, k8** | 3.02 / 2.68 | 2.72 / **2.40** | 4.18 / 1.37 |
| ensemble -> bin 61, k6 / k8 | 2.50 / 2.26, 2.80 / 2.52 | 2.34 / 2.10, 2.54 / 2.28 | |
| ensemble -> bin 121, k8 | 2.63 / 2.46 | 2.41 / 2.25 | |

* k: at this commission k = 6 is the sweet spot for a single model (k4 1.97, k6 2.02, k8 1.94, k10 1.96); the
  ensemble, whose top of the ranking is cleaner, prefers k = 8.  |pred| weighting is never better than equal weight
  at k <= 8.  Long-only top-k (half the orders) is 0.2-0.5 worse everywhere (1.79 at best): the short side carries
  about half of the edge, so it is not competitive.
* **Exit at bin 31** (10:01) is the single largest robust gain: +0.26 / +0.40 fold-OOS for the y_60d / y_120d models
  and +0.26 for the ensemble, consistent at k6 and k8 and at spread-only (2.18 -> 2.47).  The edge of the 09:31 signal
  is realised in the first 30 minutes; holding to 10:31 / 11:31 adds variance only.  Bin 91 is worse than bin 61.
* **Ensembling** the y_60d and y_120d models (average of per-day percentile ranks) adds +0.08 (k6) / +0.3 (k8) over the
  better single model at every exit: the two targets are the same signal with different label noise.
* Rank tiers (k6, y_60d model, $0.4M): ranks 1-600 only 1.20 (halves 2.51/0.52), 601-1500 only 1.86 (2.69/1.20),
  301-1500 1.95, 601-1000 1.54, 1001-1500 1.22, 1-300 0.92.  The k6 positions sit 62-66 % in ranks > 600 in every year
  (median rank ~770, stable 2017-2026).  **The post-2021 decay is the whole cross-section**: every tier halves in the
  second half (liquid names decay most, 2.5 -> 0.5), so restricting the book does not fix it; restricting to ranks
  > 300 costs nothing but gains nothing either (1.95 vs 2.02).
* Dispersion skip (flat when the cross-sectional std of the predictions is below its trailing median): looks good for
  the y_60d model (k6: 2.02 -> 2.40 with a 60-day median, 2.22 with 120 days, 2.16 with 250 days) but **hurts the
  ensemble** (2.10 -> 1.83) and is mixed for the y_120d model; for the ensemble the would-be-skipped days actually had
  the higher Sharpe (2.48 vs 2.20).  A gap-dispersion skip (cross-sectional std of the day's gaps below its trailing
  60-day median) helps the y_120d model and the ensemble (2.05 -> 2.52, 2.28 -> 2.44) but hurts the y_60d model
  (2.02 -> 1.86).  Rules that halve the trading days and flip sign across near-identical models are selection noise
  at the +-0.3 level, so no skip rule is kept (an always-on book is also what the fixed commission favours).

## 2. Refit sweeps (yearly retrain, universe(1500), target y_60d, 300 trees / 15 leaves unless stated) — lab2
k6 eq fold-OOS at $0.4M (spread-only in brackets), halves at $0.4M:

| config | IC (t) | k6 fo | k4 / k8 fo | halves |
|---|---|---|---|---|
| base (dlib.EXT, 43 feats) | 0.0270 (11.7) | 2.03 (2.18) | 1.76 / 1.72 | 3.06 / 1.28 |
| + 16 new 09:31 features (auction volume share, first-minute range/position, QQQ first minute & gap, cross-sectional dispersion of gaps / overnight / first minute, yesterday's realised-overnight and gap rank) = EXT2 | 0.0270 (11.9) | 1.65 (1.81) | 1.80 / 1.65 | 2.49 / 1.18 |
| EXT2 + overnight model's prediction at yesterday's close (agents/onml/out/pred_f28cal_q3, shifted to the next trading day) | 0.0269 (11.8) | 1.91 (2.08) | 1.82 / 1.87 | 2.25 / **1.76** |
| 600 trees, lr 0.02 | 0.0287 (12.3) | 2.10 (2.25) | 2.03 / 1.88 | 2.99 / 1.44 |
| rolling 3-year training window | 0.0249 (10.7) | 0.96 (1.11) | 1.08 / 0.91 | 2.51 / -0.08 |
| recency weights, half-life 2 years | HL2_LINE |
| num_leaves 31/63 with 600/1000 trees, min_child 500/1000, colsample 0.5; thin-name (rank > 600) training; dedicated bin-31 / bin-91 targets | not run: the box was handed over to the 2000-name runs (configs are in lab2.py: cap31, cap63, thin, y30, y90) |

* The new 09:31 features **do not help**: IC unchanged, the book is worse (more noise in the top ranks).  Adding the
  overnight model's prediction brings the IC back to the baseline level and clearly improves the post-2021 half
  (1.28 -> 1.76 at k6, 1.20 -> 1.54 at k8) at the price of a weaker first half, net fold-OOS 1.91 < 2.03.  This is the
  one feature worth revisiting with a quarterly refit and the bin-31 exit (lab3 below), but it is not a winner here.
* Capacity: 600 trees at lr 0.02 is +0.07 (within noise, IC 0.027 -> 0.029).  The larger-leaf configs were not run.
* **The decay is not a window problem**: a rolling 3-year window destroys the book (0.96, second half -0.08), i.e.
  the old data still helps in the new regime; the expanding window is right.

## 3. Refit predictions under the kept constructions — lab3
LAB3_BLOCK

## 4. Candidates (`rlib.save_cand("dayml2", ...)`), all built from the two quarterly models of the dayml lane
FINAL_BLOCK

## What I would keep
`dayml2_ens_e30_k8` (ensemble of the y_60d and y_120d quarterly models, L/S 8 equal weight, exit at the 10:01 bar) as
the primary at this commission, `dayml2_ens_e30_k6` as the lower-turnover sibling; `dayml2_x120q_e30_k6` is the
single-model version (one fewer model to maintain) and `dayml2_ens_e60_k8` the 10:31-exit fallback if the 30-minute
exit is judged too fast to execute.  All four beat the baseline by 0.25-0.4 fold-OOS at the commission with the same
models, so the gain is pure construction (shorter hold, two-target ensemble), not a new fit.  `final.py` KEPT lists
exactly these four.

## Negative results (do not repeat)
long-only top-k; |pred| weighting at k <= 8; rank-restricted books; rolling training window; the 16 new 09:31
microstructure / dispersion features; prediction-dispersion and gap-dispersion day skips (sign flips across models);
bin-91 exit.  The overnight prediction as a feature is the only "maybe" (helps after 2021 only).

## Leakage checklist
* Same feature alignment as dayml (close-time features shifted one day within the ticker; o_* and Min extras use bins
  0-1 of day t only).  New extras: 09:30-bar volume vs lagged 20-day means, first-two-bar range, QQQ/SPY first minute,
  cross-sectional dispersions of the day's gaps (all known at 09:31), yesterday's ranks (shift(1) within ticker).
* The overnight prediction is dated at the close of t-1 in agents/onml/out/pred_f28cal_q3.parquet and is mapped to the
  NEXT trading day of the full calendar (incl. half days) before the merge, so row t only sees the prediction made at
  the close of t-1.  That file was produced with STRAT_END=2026-10-28; it is cut at DEV_END before use.
* Fills: decision at bin 1, fill at the close of bin 3, exits at the close of bins 31 / 61 / 121 (never the signal
  bar, never bin 2).  |y| > 0.5 -> NaN.  Label lag 3 days, expanding window, first OOS 2017-01.
* Selection: the exit / ensemble / k choices were made on the whole development sample from the saved predictions
  (in-sample for construction); fold-OOS of the 4-candidate family is reported in section 4 as the honest number.
