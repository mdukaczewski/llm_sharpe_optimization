# onml3: the learned overnight book under realistic execution

Lane `agents/onml3` (code `lib3.py`, `step1.py` .. `step4.py`, log `log.txt`, run logs `step*.log`, tables `out/step*.csv`).
Every number is on dates <= `rlib.DEV_END` = 2026-04-27; STRAT_END was never set.  Commission: $3 per order on $0.4M
equity = 2k x 2 x $3 / $400k per day (3.0 bps/day for k=10, 1.8 for k=6).  "fold-OOS" = `rlib.fold_oos` of the single
series (half-year windows from 2018-07-01).  Candidate series are NET of the modelled execution cost and borrow, BEFORE
commission; "$0.4M" columns subtract the commission.

## Execution model (shared `liq.py`, phase-6 brief)
* MOC entry + MOO exit, per-side auction cost `liq.auction_cost_bps(order, auction_dv, rv_daily, c)` =
  max(1 bp, c x daily vol (bps) x sqrt(order / auction $vol)), order = $400k / (2k).  Opening auction $vol = exp(dv0)
  (lagged 20-day mean $vol of the 09:30 bar); closing auction $vol = max(exp(dvc), 5% x exp(ldv20)) (the 16:00 bar
  understates the closing auction).  rv_daily = rv20 / sqrt(252) (the rlib rv20 is annualised).  c = 0.3 central,
  0.1 / 1.0 reported.  Borrow 1%/yr on shorts.  Target `y_oo` = official open(t+1) / official close(t) - 1 (`liq.add_y_oo`).
* MOC entry + 09:31-bar exit paying f x predicted NBBO half-spread `hs0931` (f = 1 crossing, 0.5 half); the MOC entry
  pays the c = 0.3 closing-auction cost in these cases too.  Target `y_on` (09:31 bar close).
* Old convention for reference: 1 / 1.5 / 2.5 bps per side by rank tier (`rlib.cost_bps`).

Two bugs on the way, both mine, both fixed before any number below was produced: (1) `rlib.clean` clips features to
+-5, which destroyed the liq columns (ldv20, dv0, dvc ~ 14-23, hs0931 in bps) in the first cached frame -> every cost was
~100x too large (the first step-1 lines in `log.txt` are marked INVALID); the liq columns are now merged without clipping
(`lib3.fix_liq`, `remerge_liq`).  (2) `book3` crashed when entry and exit used the same cost column.  `liq.py` itself
needed no change.  (`olib.load` wrote two "load" lines into `agents/onml/log.txt` before I redirected its logger; they
are harmless; I was not allowed to delete them.)

## Step 1: the saved f26calx predictions (quarterly 3-seed, 2000 names, no refit) under every execution case, k = 10

| book (k=10) | exit / cost | pre-commission Sh | $0.4M Sh | fold-OOS | halves | maxDD | ret/yr |
|---|---|---|---|---|---|---|---|
| (a) `onml_f26calx_u2000_q3_k10` reproduced | 09:31 bar, 1-2.5 bps/side | 4.43 | 4.13 | 4.23 | 4.59 / 3.59 | -23% | 103% |
| (a') same, gross (no cost) | 09:31 bar | 5.15 | 4.85 | 4.92 | 5.25 / 4.38 | -23% | 120% |
| (b') gross to the official open | MOO, no cost | 5.82 | 5.49 | 5.68 | 5.68 / 5.31 | -22% | 127% |
| (b) MOC + MOO, c = 0.1 | auction | 2.10 | 1.77 | 1.99 | 2.53 / 0.82 | -31% | 41% |
| (b) MOC + MOO, **c = 0.3** | auction | -5.18 | -5.50 | -5.23 | -3.80 / -7.69 | -100% | -129% |
| (b) MOC + MOO, c = 1.0 | auction | -22.5 | -22.7 | -22.5 | | -100% | |
| (c) MOC + cross the 09:31 spread (f = 1) | hs0931 | -4.50 | -4.80 | -4.33 | -3.94 / -5.86 | -100% | -119% |
| (c') same without the MOC entry cost | hs0931 | -0.89 | -1.19 | -0.74 | | -95% | -30% |
| (d) MOC + half the 09:31 half-spread (f = 0.5) | 0.5 hs0931 | -1.50 | -1.81 | -1.52 | -0.93 / -2.88 | -99% | -45% |
| (d') f = 0.25 | 0.25 hs0931 | 0.00 | -0.30 | -0.11 | 0.59 / -1.38 | -83% | -8% |

k sweep of (b) at c = 0.3 ($0.4M Sharpe): k6 -7.1, k8 -6.2, k12 -5.1, k15 -4.5, k20 -3.9 (more names = less impact per
name, still deeply negative).  The reproduction (a) matches the saved candidate exactly (corr 1.0000, diff < 1e-8).

The gross edge to the official open is larger than to the 09:31 bar (127% vs 120%/yr, Sharpe 5.8 vs 5.2), confirming the
brief.  The problem is where the book trades.  By tier (positions of the k=10 book, bps per trade, 2017-2026):

| rank tier | share of positions | gross bps/trade (y_oo) | MOC+MOO cost at c=0.3 (both sides) | net | case (c), cross at 09:31: cost / net |
|---|---|---|---|---|---|
| 1-300 | 11% | 19.9 | 9.3 | +10.5 | 14.9 / +2.5 |
| 301-600 | 14% | 17.1 | 22.4 | -5.3 | 27.4 / -13.8 |
| 601-1500 | 45% | 24.7 | 56.2 | -31.5 | 52.4 / -27.6 |
| 1501-2000 | 30% | 37.0 | 70.8 | -33.8 | 61.4 / -26.3 |

75% of the positions are beyond rank 600, where the modelled round trip (MOO exit per side at c=0.3: mean 33 bps, median
19; MOC entry mean 18) is 2-3x the gross.  The predicted half-spread of the chosen names at 09:31 is mean 29.6 / median
24.0 bps, in line with the NBBO check (mean 32 / median 21).  Only ranks 1-300 are net positive under any realistic case.

## Step 2: cost-aware selection on the same predictions (no refit), MOC + MOO at c = 0.3

Two selection rules, both using only per-name predicted costs known at 15:59 of day t:
* A. cost cap: keep only names whose mean per-side auction cost (entry, exit) at c = 0.3 is <= cap, then L/S top-k by p.
* B. net score: long the top-k by E[y_oo | p] - lambda x (ce + cx), short the top-k by -E[y_oo | p] - lambda x (ce + cx)
  - borrow/day, where E[y_oo | p] is a 50-bin map of the gross overnight return on the per-day prediction rank.  Step 2
  uses the IN-SAMPLE full-dev-span map (stated here; it is one monotone curve of 50 numbers with no per-name information:
  bins 0 / 25 / 45 / 48 / 49 -> -9 / +5 / +12 / +19 / +33 bps).  Step 2c repeats the best rules with a CAUSAL map
  (expanding: the map used in year Y is fitted on dates before Jan 1 of Y minus 3 days; 2017 uses its own year's map since
  no prior prediction exists - flagged; fold-OOS starts 2018-07 so no fold touches it).

$0.4M Sharpe / fold-OOS (full tables `out/step2.csv`, `step2b.csv`, `step2c.csv`):

| rule | k=6 | k=8 | k=10 | k=12 | k=15 |
|---|---|---|---|---|---|
| no selection (plain top-k) | -7.1 / -6.9 | -6.2 / -5.9 | -5.5 / -5.2 | -5.1 / -4.9 | -4.5 / -4.2 |
| A cap 20 bps (~1070 names/day) | -0.7 / -0.2 | -0.9 / -0.5 | -0.9 / -0.5 | -1.0 / -0.5 | -1.1 / -0.7 |
| A cap 10 (~800) | 0.3 / 0.6 | 0.3 / 0.6 | 0.2 / 0.6 | 0.0 / 0.4 | 0.0 / 0.3 |
| A cap 5 (~460) | 1.3 / 1.5 | 1.4 / 1.6 | 1.1 / 1.3 | 0.9 / 1.1 | 0.6 / 0.9 |
| A cap 3 (~250) | 1.9 / 2.0 | 1.5 / 1.6 | 1.4 / 1.6 | | |
| A cap 2 (~125) | 1.7 / 1.6 | 1.4 / 1.4 | 1.2 / 1.2 | | |
| B lambda 1 (in-sample map) | 1.4 / 1.6 | 1.3 / 1.6 | 1.3 / 1.5 | 1.0 / 1.2 | 0.9 / 1.2 |
| B lambda 2 | 1.9 / 2.1 | 1.9 / 2.0 | 1.7 / 1.8 | 1.5 / 1.7 | 1.4 / 1.5 |
| B lambda 3 | 1.9 / 2.1 | 1.9 / 2.1 | 1.7 / 1.8 | | |
| B lambda 4 | 1.9 / 2.0 | 1.8 / 1.9 | 1.5 / 1.7 | | |
| B per-tier map, lambda 4 | 2.1 / 2.2 | 1.9 / 2.0 | 1.6 / 1.7 | | |
| A cap 5 + B lambda 2 | 2.0 / 2.1 | 2.0 / 2.1 | 1.9 / 2.0 | | |
| **B lambda 3, causal map** | **1.85 / 1.95** | 1.73 / 1.86 | 1.39 / 1.47 | 1.29 / 1.41 | 1.03 / 1.17 |
| B lambda 2, causal map | 1.74 / 1.82 | 1.64 / 1.73 | 1.44 / 1.53 | 1.53 / 1.62 | 1.37 / 1.52 |
| B lambda 2/3, causal, 2017 dropped | 1.90 / 1.85, 2.02 / 2.00 | | 1.61 / 1.64, 1.55 / 1.56 | | |

The selection is what matters: it takes the book from -5 to about +2.  The result is a broad plateau (lambda 2-4, caps
3-5, cap x lambda, per-tier map: all 1.9-2.2 in-sample, 1.7-2.0 with the causal map), so it is not a tuned peak; the
causal map costs ~0.1.  Smaller books are better (k = 6-8): the commission favours them and the cheap names with real
edge are few.  Over-penalising the cost (lambda 2-4 beats 1) helps because the global map overstates the edge in liquid
names (per-tier top-bin mean: +26 bps in ranks 1-300 vs +46 beyond 1500).

Sensitivity of `B lambda 2 causal` to c (selection fixed at the c = 0.3 scores, costs re-evaluated): fold-OOS at
c = 0.1 / 0.3 / 1.0: k6 2.4 / 1.8 / -0.9; k8 2.3 / 1.7 / -1.1; k10 2.1 / 1.5 / -1.5.  At c = 1.0 nothing survives.
Spread exits with the same rule (score uses ce + f x hs0931): half the half-spread k6 1.73 / 1.85, k10 1.45 / 1.53;
crossing k6 1.02 / 1.05, k10 0.67 / 0.72.

## Step 3: refit with liquidity / spread features (yearly retrain, seed 0, n_jobs = 1, universe 2000)

Feature sets: F26X = the f26calx set (26 features of `agents/onml/final2000.py`); F40 = F26X + the 14 liq columns
`ldv20 lpx hl1 hl5 hlc roll30 n30 nc30 dv0 dvc c_hlc c_nc30 hs0931 hs1559` (lagged 20-day means, or same-day values of
bins 360-389, or the calibrated spread predictions from them: all known at 15:59 of day t; the `t_` columns are
next-morning information and were not used).  Targets: `yr` = per-day rank of y_on (09:31 bar), `yr_oo` = rank of y_oo
(official open), `yr_oon` = rank of y_oo - MOO auction cost (c = 0.3, fraction).  Rows: 2.97M; a yearly fit took 8-11 min.
Also two "liquid-universe" models (R6/R7): trained AND predicted only on names whose predicted mean per-side cost at
k = 10, c = 0.3 is <= 10 bps (64% of the rows, ~800 names/day; the rank target recomputed within that set).

$0.4M Sharpe / fold-OOS; "selection" = rule B of step 2 with the causal map (lambda 2 or 3, the better one):

| run | feats / target | IC y_oo | old convention: plain top-10, 09:31 exit, 1-2.5 bps | plain top-10 MOC+MOO c=0.3 | cap 10, k10 | selection k6 | k8 | k10 | selection, half-spread exit k6 / k10 |
|---|---|---|---|---|---|---|---|---|---|
| R5 `f26x_on` | F26X / yr (= the old model, 1 seed) | 0.033 | 3.68 / 3.77 | -5.5 / -5.2 | 0.1 / 0.5 | 1.54 / 1.79 | 1.42 / 1.64 | 1.39 / 1.55 | 1.21 / 1.34, 0.89 / 1.06 |
| R1 `f26x_oo` | F26X / yr_oo | 0.035 | 3.87 / 3.91 | -5.2 / -5.0 | 0.4 / 0.8 | 1.74 / 1.90 | 1.56 / 1.67 | 1.46 / 1.57 | 1.28 / 1.40, 0.81 / 0.97 |
| **R4 `f40_on`** | **F40 / yr** | 0.036 | **5.23 / 5.31** | -7.9 / -7.8 | 1.1 / 1.5 | **1.74 / 1.93** | **1.76 / 1.94** | **1.58 / 1.72** | **1.62 / 1.78, 1.52 / 1.63** |
| R2 `f40_oo` | F40 / yr_oo | 0.038 | 5.06 / 5.09 | -6.7 / -6.6 | 1.2 / 1.5 | 1.82 / 1.71 | 1.64 / 1.53 | 1.64 / 1.60 | 1.28 / 1.21, 1.02 / 1.03 |
| R3 `f40_oon` | F40 / yr_oon | 0.003 | 0.78 / 1.04 | -18.8 / -18.4 | -1.0 / -0.8 | 0.10 / 0.21 | 0.08 / 0.19 | -0.04 / 0.09 | negative |
| R6 `f40_oo_liq` | F40 / yr_oo, liquid universe | 0.032 | 2.34 / 2.49 | 1.04 / 1.38 | (same) | 1.50 / 1.48 | 1.32 / 1.31 | 1.15 / 1.15 | 0.78 / 0.83, 0.34 / 0.41 |
| R7 `f26x_oo_liq` | F26X / yr_oo, liquid universe | 0.030 | 1.94 / 2.04 | 0.20 / 0.54 | (same) | 1.22 / 1.37 | 0.98 / 1.11 | 0.77 / 0.87 | 0.55 / 0.64, 0.17 / 0.20 |
| ref: saved `f26calx_q3` (quarterly, 3 seeds) | F26X / yr | 0.036 | 4.13 / 4.23 | -5.5 / -5.2 | 0.2 / 0.6 | 1.85 / 1.95 | 1.73 / 1.86 | 1.39 / 1.47 | 1.73 / 1.85, 1.38 / 1.38 |

Reading:
* The liq features add a lot at the mid (old convention 5.2-5.3 vs 3.7-3.9 for the same yearly seed-0 setup), but the
  model spends it on the names with the widest spreads (plain MOC+MOO books get worse: -7.9 vs -5.5).  Under the
  cost-aware selection they still add +0.15..0.3 fold-OOS at every k and much more with the half-spread exit
  (1.78 vs 1.34 at k6, 1.63 vs 1.06 at k10): spread-aware features let the model rank well among the names it is allowed
  to trade.
* The official-open target helps the model without liq features a little (+0.1) and not the model with them; with F40
  the old 09:31-bar target is better (yr: 1.93 / 1.94 / 1.72 vs yr_oo 1.71 / 1.53 / 1.60) - the y_oo prints of thin
  names are noisier labels, and the liq features already carry the spread information.
* Target net of cost (R3) is dead: the per-name cost term is symmetric in the target but the book is not (it pushes
  expensive names to the BOTTOM of the ranking, i.e. into the shorts), IC collapses to 0.003.  Do not repeat.
* Training only on the liquid universe (R6/R7) loses: fewer rows, weaker model (1.48 vs 1.7-1.9 under the same
  selection), even though its plain books are the only plain books that are positive.  Train on everything, select after.
* The cap-10 universe + plain top-k (what the brief suggested as the simplest variant) is far below the lambda rule
  at every setting (1.5 vs 1.9).

Chosen final configurations (step 4): A = the saved quarterly 3-seed f26calx predictions with the lambda = 3 selection
(no refit; the reference that isolates the selection effect), B = `f40_on` (F40 features, 09:31-bar rank target)
quarterly 3-seed with the same selection.
