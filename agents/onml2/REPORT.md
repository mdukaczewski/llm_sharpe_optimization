# onml2 lane — overnight LightGBM L/S book at a fixed $3 commission ($0.4M per book), 2026-10-29/30

Lane files: `agents/onml2/o2lib.py` (walk-forward with window / recency weights, commission-aware book `book3`, evaluation),
`sweep1.py` (model capacity + training window, 2000 names yearly), `constr.py` / `asym.py` (construction on saved predictions),
`final.py` + `keep.py` (regenerates only the kept candidates), `log.txt`, `out/` (every series of every run with the daily position count).
All dates <= rlib.DEV_END (2026-04-27).  Costs: rlib.cost_bps per side (1 / 1.5 / 2.5 bps for ranks <= 600 / <= 1500 / <= 2000) + 1%/yr
borrow, PLUS the commission: $3 per filled order on $0.4M equity, every position traded in and out daily, i.e. a day with P positions
costs P x 6 / 400000 of equity (L/S top-10 = 20 positions = 3 bps/day; top-20 = 6 bps/day).  Every line below is reported both
"spread-only" (rlib conventions) and "comm" (with the commission); fold-OOS = rlib.fold_oos of the single series (16 half-year folds).
Fills unchanged: decision at the close fills at the official close, exit at the 09:31 bar.

## 0. Feature set change (coordinator, 23:35): implementable close features
`zmoc` (closing print vs the 15:59 bar) and `vmoc_share` (closing-auction volume share) are only known AT the close, so a
market-on-close order cannot use them.  Every kept candidate uses **f26cal = f28cal minus {zmoc, vmoc_share}** (26 features =
FEATS18 + z15l z5l zvwap vfirst_share ldv cc21 rv5_20 spy_zgap + dow dme qend dsgap).  z5l / z15l (15:54-15:59, 15:44-15:59
returns) are borderline for a MOC order (NYSE cut-off 15:50); a variant without them is reported in section 4.
Cost of dropping zmoc/vmoc_share on 2000 names, yearly seed 0, k10 at the commission: 4.11/4.19 (f28cal, coordinator's breadth2000.log)
-> 3.89/3.95 (f26cal, sweep1 base), i.e. about -0.25.

## 1. Construction at the commission (c), (d) — predictions fixed, no retraining
### 1a. 1500 names, 3-seed quarterly f28cal predictions (agents/onml/out/pred_f28cal_q3.parquet cut at DEV_END) = the current best
| book (eq-weight unless noted) | positions/day | spread-only full/fold | comm full/fold | halves (comm) | maxDD (comm) |
|---|---|---|---|---|---|
| L/S k6  | 12 | 3.48/3.57 | 3.31/3.42 | 3.37/3.26 | -25.3% |
| L/S k8  | 16 | 3.49/3.61 | 3.24/3.37 | 3.41/3.07 | -20.8% |
| **L/S k10** | 20 | 3.71/3.81 | **3.37/3.48** | 3.73/2.99 | -16.7% |
| L/S k12 | 24 | 3.69/3.75 | 3.25/3.33 | 3.71/2.76 | -17.3% |
| L/S k15 | 30 | 3.80/3.84 | 3.20/3.27 | 3.70/2.65 | -16.6% |
| L/S k20 (= onml_f28cal_q3_k20) | 40 | 3.75/3.76 | 2.87/2.93 | 3.26/2.42 | -15.4% |
| L/S k30 | 60 | 3.83/3.82 | 2.29/2.37 | 2.86/1.63 | -15.1% |
| L/S k10 / k15 / k20 pred-weighted | | 3.72/3.81, 3.80/3.84, 3.75/3.77 | 3.37/3.48, 3.21/3.28, 2.88/2.94 | | |
| L/S k10 / k15 / k20 linear rank weights | | 3.58/3.70, 3.77/3.86, 3.85/3.92 | 3.29/3.43, 3.26/3.37, 3.09/3.20 | | |
| long-only k6 / k10 / k15 / k20 | 6/10/15/20 | 2.47/2.41, 2.59/2.50, 2.55/2.45, 2.46/2.37 | 2.39/2.33, 2.44/2.35, 2.30/2.21, 2.11/2.04 | | -42..-46% |
| asymmetric L10/S5, L15/S5, L15/S10, L20/S10, L12/S6 | 15/20/25/30/18 | | 3.22/3.37, 3.11/3.28, 3.30/3.42, 3.09/3.25, 3.36/3.48 | | -18..-26% |
| half-gross short side (L15/S5 g0.5, L10/S5 g0.5) | | | 3.11/3.17, 3.18/3.22 | | |

* Gross overnight return by prediction rank (3-seed quarterly preds, bps/day): longs ranked 1-5 +32, 6-10 +23, 11-15 +21, 16-20 +19,
  21-30 +17, 31-40 +16; shorts 1-5 -14, 6-10 -7, 11-15 -7, 16-20 -3, 21-30 -5, 31-40 -6.  A position in a top-k book of size k is
  $400k/2k, so the commission is 3 bps of the position at k=10 and 6 bps at k=20 (+2.5 bps spread round trip): names ranked
  beyond ~12 on the short side do not pay for themselves at k=20, which is why the optimum moves from k=20-30 (spread-only) to
  k=8-12 at the commission.  The "pred" weighting of olib is a no-op on rank-averaged predictions (the top-k values are all ~0.5).
* The short side is weaker per name but the asymmetric books lose: the shorts are the overnight market hedge (corr SPY stays
  -0.03..-0.05 for all L/S variants, but the half-size short side raises drawdown to -25..-30%).  Long-only halves the orders
  and drops the borrow, but Sharpe falls to 2.3-2.4 and drawdown to -40% (overnight market beta); dead.
* Hold-through (d): fraction of today's top-k (bottom-k) names that are in the top-k (bottom-k) again the next day:
  k6 4.0% (3.1%), k10 4.8% (3.8%), k15 5.7% (4.6%), k20 6.5% (5.4%) — the signal is a one-day reversal signal, there is nothing
  to hold through even if the book were not flat 09:31-16:00; on 2000 names yearly the numbers are the same (5.1% / 3.7% at k10).
* Family fold-OOS at the commission over the 14 k/weight variants of the table: 3.36 (the fold procedure picks k10 or k6 in every fold).

### 1b. 2000 names, f26cal, yearly seed 0 (sweep1 base raw predictions; raw model outputs available -> quantile / dispersion filters)
| book | P/day | spread-only full/fold | comm full/fold | halves (comm) | maxDD |
|---|---|---|---|---|---|
| L/S k6 / k8 | 12/16 | 3.79/3.92, 4.08/4.13 | 3.64/3.78, 3.86/3.91 | | -29% / -26% |
| **L/S k10** | 20 | 4.20/4.24 | **3.89/3.95** | 4.44/3.27 | -24.9% |
| L/S k12 / k15 | 24/30 | 4.21/4.26, 4.21/4.29 | 3.81/3.89, 3.67/3.78 | | -25% / -28% |
| L/S k20 / k30 | 40/60 | 4.31/4.39, 4.11/4.17 | 3.51/3.64, 2.78/2.92 | | -25% / -20% |
| k10 / k15 / k20 weighted by raw model output | | 4.17/4.23, 4.26/4.36, 4.37/4.47 | 3.87/3.96, 3.75/3.88, 3.62/3.76 | | |
| k10 / k15 / k20 linear rank weights | | | 3.70/3.81, 3.74/3.84, 3.62/3.74 | | |
| variable book size: k15 q0.6, k20 q0.5 / q0.7 / q0.8, k30 q0.8 / q0.9, k40 q0.9 | 29.9, 39.9/39.6/39.1, 58/54, 69 | | 3.64/3.75, 3.49/3.62, 3.40/3.51, 3.36/3.47, 2.67/2.76, 2.66/2.73, 1.94/2.02 | | |
| dispersion filter (trade only when the day's xs-std of the raw prediction > trailing 120d q0.3 / q0.5): k10, k20 | 13.3/26.6, 9.4/18.9 | | 3.05/3.11, 2.79/2.97, 2.61/2.63, 2.45/2.51 | | |
| long-only k6 / k10 / k15 / k20 | | | 2.88/2.86, 2.79/2.68, 2.59/2.50, 2.42/2.34 | | -36..-39% |
| asymmetric L10/S5, L15/S10, L20/S10, L12/S6, L8/S4 | | | 3.59/3.67, 3.74/3.82, 3.62/3.72, 3.62/3.70, 3.55/3.61 | | |

Reading: same picture at 2000 names.  k=10 equal weight is the commission optimum (k8-k12 within +-0.05); the |raw| quantile filter
never helps (it removes names on calm days, which still earn); the dispersion filter is strongly negative (the low-dispersion days
earn nearly as much as the others, skipping them just loses days); long-only and asymmetric books are worse.  Family fold-OOS
at the commission over the 14 k/weight variants: 3.88.  Construction is settled: **L/S top-10 / bottom-10, equal weight**.
