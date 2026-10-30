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

## 2. Model capacity (a) and training window (b): 2000 names, f26cal, yearly retrain, seed 0, L/S k10 / k20 eq-weight
Single-seed noise on this book is about +-0.15 Sharpe (onml REPORT, 3 seeds at 1500 names).  Run killed by the coordinator's
wrap-up during `hl2` (recency weights, half-life 2y); `hl4`, `L31_600`, `L63_600`, `L31_1000`, `L63_1000` and the small+large
ensemble were NOT run (box at load 8).  Fit times at n_jobs=2 on the shared box: 480-660 s per yearly walk-forward.

| model (LightGBM, expanding window unless noted) | IC | k10 spread-only full/fold | k10 comm full/fold | k10 halves (comm) | k20 comm full/fold |
|---|---|---|---|---|---|
| **base** (300 trees, lr 0.03, 15 leaves, mcs 200, colsample 0.8) | 0.0371 | 4.20/4.24 | **3.89/3.95** | 4.44/3.27 | 3.51/3.64 |
| min_child_samples 500 | 0.0368 | 3.98/4.12 | 3.68/3.83 | 3.88/3.45 | 3.23/3.35 |
| min_child_samples 1000 | 0.0371 | 4.14/4.34 | 3.82/4.04 | 4.10/3.51 | 3.08/3.27 |
| colsample_bytree 0.5 | — | 4.23/4.27 | 3.91/3.96 | 4.81/2.96 | 3.22/3.25 |
| rolling 3y window | — | 3.65/3.63 | 3.35/3.35 | 4.28/2.31 | 2.94/3.00 |
| rolling 4y window | — | 3.78/3.77 | 3.47/3.48 | 4.36/2.45 | 3.03/3.10 |
| rank-average of base + mcs1000 + colsample0.5 (3 variants) | — | 4.22/4.36 | 3.90/4.06 | 4.32/3.42 | — |
| reference, 1500 names f28cal (with zmoc) base yearly | 0.0363 | 3.43/3.50 | 3.08/3.17 | 3.21/2.94 | 2.76/2.90 |
| reference, 2000 names f28cal (with zmoc) base yearly (coordinator) | — | 4.42/4.49 | 4.11/4.19 | 4.96/3.82 | 3.89/3.89 |

Reading: none of the regularisation knobs moves the k10 book beyond seed noise (mcs500 -0.12, mcs1000 +0.09, colsample 0.5 +0.01
fold-OOS at the commission); the rank-average of the three variants is +0.1 over base, which is the usual seed-averaging effect.
Rolling windows are clearly negative (-0.5 to -0.6, the second half of the sample collapses to 2.3-2.5): the book wants ALL the
history — consistent with the earlier finding that breadth (more rows) is the main lever.  Bigger trees / more trees were not
reached; given that 2.5-3M rows did not reward even min_child_samples changes, I would not expect them to matter, but it is untested.
Quarterly retrain (+0.25 at 1500 names in the onml lane) was not run on 2000 names in this lane (CPU); keep.py takes retrain="Q".

## 3. Kept candidates (agents/onml2/cands, written by final.py with ONML2_USE_RAW=1 from the sweep1 predictions)
All: f26cal (26 implementable close-time + calendar features), per-day rank target, LightGBM baseline params, yearly expanding
retrain from 2017-01-01 (label lag 3 days), universe(2000) = 1989 names, L/S book exit at the 09:31 bar, rlib costs + 1%/yr borrow
+ $3/order commission on $0.4M.  The saved series are NET of everything (the spread-only series and the daily position count P are
in agents/onml2/out/final_<name>.csv).  Baseline = onml_f28cal_q3_k20 (1500 names, 3-seed quarterly, f28cal incl. zmoc, k20).

| candidate | model | book | P/day | spread-only full/fold | comm full/fold | halves (comm) | maxDD | ret/vol | corr sleeve | corr SPY | min fold | neg folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **onml2_f26cal_y_k10** | base, seed 0 | top/bot 10 eq | 20 | 4.20/4.24 | **3.89/3.95** | 4.44/3.27 | -24.9% | 95%/24.3% | 0.57 | -0.04 | -0.68 | 1/16 |
| **onml2_f26cal_y3_k10** | base+mcs1000+cs05 rank-avg | top/bot 10 eq | 20 | 4.22/4.36 | **3.90/4.06** | 4.32/3.42 | -23.2% | 92%/23.5% | 0.55 | -0.03 | -0.23 | 1/16 |
| onml2_f26cal_y_k20 | base, seed 0 | top/bot 20 eq | 40 | 4.31/4.39 | 3.51/3.64 | 4.28/2.63 | -24.7% | 66%/18.9% | 0.64 | -0.04 | -0.68 | 1/16 |
| family (rlib.fold_oos over the three, comm) | | | | | **3.90** | folds mean 4.04, min -0.68, 1/16 negative | | | | | | |
| baseline onml_f28cal_q3_k20 (1500, comm) | | top/bot 20 eq | 40 | 3.75/3.76 | 2.87/2.93 | 3.26/2.42 | -15.4% | 49%/17.2% | 0.60 | -0.03 | 0.35 | 0/16 |
| baseline predictions, top/bot 10 (comm) | | top/bot 10 eq | 20 | 3.71/3.81 | 3.37/3.48 | 3.73/2.99 | -16.7% | 74%/21.9% | 0.53 | -0.03 | 1.09 | 0/16 |

Yearly Sharpe at the commission (y_k10): 17 +2.8, 18 +5.4, 19 +5.0, 20 +4.9, 21 +5.2, 22 +1.5, 23 +3.7, 24 +3.5, 25 +4.4, 26(Jan-Apr) +3.0.
The one negative fold is 2022-H2 (the y3 ensemble -0.2, single model -0.7; all other folds +1.6..+6.9); the 1500-name books have no negative fold (min 0.35-1.09),
i.e. the 2000-name universe (ranks 1501-2000, 2.5 bps spread) adds +0.5..+0.8 fold-OOS but a worse 2022 and a -25% drawdown
(vs -15..-17% at 1500 names): higher gross return (95% vs 74%) at higher vol (24% vs 22%).
Decomposition of the gain vs the baseline at the commission (fold-OOS 2.93 -> 3.95): construction k20 -> k10 +0.55 (on the baseline
predictions, 3.48), universe 1500 -> 2000 +0.8 (same recipe, 3.17 -> 3.95 yearly), minus zmoc/vmoc_share -0.25, minus quarterly
retrain (not run here) about -0.25 (the baseline is quarterly, the candidates yearly).

## 4. What I would keep, and why; what was not done
* Keep **onml2_f26cal_y3_k10** (and the single-model onml2_f26cal_y_k10 as its control): the top-10 construction is the only
  commission-specific choice, it is the optimum on every prediction set tried (400 / 1500 / 2000 names, yearly and quarterly) with a
  flat k8-k12 plateau, so it is not a fitted knob.  The three-variant average replaces the 3-seed average of the onml lane (same
  +0.1 effect, no extra fits beyond the sweep).
* onml2_f26cal_y_k20 is the reference line (baseline construction on the new universe / feature set); drop it from the family if
  fewer candidates are wanted — it is never picked by the fold procedure.
* Next step for whoever has the CPU: the same KEEP with retrain="Q" and seeds [0,1,2] (keep.py), expected +0.2..0.3; the z5l/z15l-free
  variant (F24CAL in o2lib: MODELS entry `dict(feats=F24CAL)`) for the MOC-implementability question — the coordinator's
  agents/onml/breadth2b.py measures exactly this on 2000 names (its log had only the f28cal lines when this report was written).
* Negative results worth recording: rolling windows (-0.5), per-day |raw| quantile filters (-0.1..-0.9, more names removed = worse),
  dispersion gating (-0.8..-1.3), long-only (-1.1..-1.5, drawdown -40%), asymmetric long/short sizes (-0.1..-0.4), half-size short
  side (-0.3..-0.7), linear-in-rank and |raw| weights (within +-0.1 of equal weight at k10).  Every model knob was within seed noise.

## 5. Leakage / robustness checklist
* Features: rlib.ticker_frame close-time features (lagged rolling std in zsd) minus the two close-only ones + calendar; no feature uses
  y_on, the 09:31 bar of t+1 or any forward quantity; dsgap uses |zgap| of day t and earlier.  Training rows end 3 days before each
  yearly cut; the per-day rank target is built within the day; raw predictions are rank-normalised per day before averaging.
* Costs: rlib.cost_bps (2.5 bps for ranks 1501-2000) + 1%/yr borrow + commission P x 6/400000 per day with P counted from the actual
  filled positions of the day (every name in the top/bottom k with a valid target).  The commission does not depend on fill price,
  so it is a pure daily drag; the spread cost is unchanged from rlib.book.
* Selection: k chosen among 7 values on three prediction sets (the choice is identical on all three); model knobs: 6 variants, none
  kept except as an average; universe 2000 was the coordinator's instruction, not a search.  The remaining in-sample elements are the
  ones of the onml lane (EXTRA10 features designed with the whole sample visible; LightGBM params = project baseline).
* Survivorship: universe files ranked by today's dollar volume (same as every book in the project); |y| > 0.5 removed (split guard).
* One negative fold (2022-H2) at 2000 names; the 1500-name baseline had none — the extra 500 names make the book more volatile.

## 6. final.py
`nohup python3 agents/onml2/final.py > agents/onml2/final.log 2>&1 &` rebuilds the three kept candidates from data/minute_npz via rlib
(universe(2000) / env ONML2_N, ONML2_NJOBS default 2), writes them with rlib.save_cand("onml2", ...) and honours STRAT_END through
rlib.DEV_END.  With ONML2_USE_RAW=1 it reuses agents/onml2/out/raw_{base,mcs1000,cs05}.npy (sweep1 yearly seed-0 predictions, same
code path) instead of refitting — this is how the saved candidates were produced (final.log).  A full refit = 3 yearly walk-forwards
of ~8-10 min each at n_jobs=2 on the shared box (~30 min; memory ~1.5 GB); with STRAT_END=2026-10-28 the frame grows by 6 months.
