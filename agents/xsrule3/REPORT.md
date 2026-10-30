# xsrule3 lane: the cAx rule overnight book re-built for realistic execution (2026-10-30)

Scripts (all in `agents/xsrule3/`, single core, no model fitting): `x3lib.py` (frame, book, cost model, reporting),
`s0_frame.py` (frame + reproduction check), `s1_books.py` (steps 1-2: books x universes x executions, `out/s1_table.csv`),
`s2_liqfeat.py` (step 3: liquidity filters and tilts, `out/s2_table.csv`), `s2b_extra.py` (universe 50/75/100, c-sensitivity,
per-fold tables, correlations), `s3_etf.py` (step 4: VIX-ETP book with the official-open exit), `s4_save.py` (step 5:
regenerates and saves the 6 candidates).  Logs: `log.txt` (everything), `s0.log`..`s4.log` (per script).  No change to
`liq.py` was needed; the closing-auction volume floor from the brief (max(exp(dvc), 0.05 exp(ldv20))) lives in `x3lib.cost_sides`.

Signal, book and inputs are exactly those of `agents/xsrule` (cAx = 0.5 z30 - z60 - 0.5 z5l + 0.3 zon_prev - rank(vlast_x), fixed
weights, all known at the 15:59 bar; rank(vlast_x) over universe(1500) as in `final.py` / `k10.py`).  Reproduction check: my
old-convention series of `lo10_s15_u150` equals the saved `xsr_cAx_lo10_s15_u150` exactly (Sh 2.265 = 2.265, corr 1.0000, 2286 rows).

Execution model (per (date, name), order = $400k / positions, rv = rv20 / sqrt(252), all liquidity inputs lagged 20-day means):
MOC entry cost = `liq.auction_cost_bps(order, max(exp(dvc), 0.05 exp(ldv20)), rv, c)`; official-open exit (`y_oo`, MOO) cost =
`liq.auction_cost_bps(order, exp(dv0), rv, c)`; 09:31-bar exit (`y_on`) cost = f x predicted `hs0931`; old convention = 1 bp/side,
09:31 exit.  Borrow 1%/yr on shorts.  All Sharpes below are AT the $3 / $0.4M commission (1.5 bps/day for lo10, 3.0 for lo20 and
ls10, 6.0 for ls20) unless marked "pre-comm"; saved series are before commission.  "OOS" = `rlib.fold_oos` on the single series
(2018-07-01 onward); "family" = the chooser over a set.

## Result in one paragraph
The official-open exit works as the brief expected but the auction cost is not small on 150 names: at c = 0.3 the long-only
top-10 book pays 2.3 bps (MOC) + 3.8 bps (MOO) per round trip against a 15 bps gross edge, and its fold-OOS Sharpe goes from
1.72 (old 1 bp convention) to **1.08** (MOO, c = 0.3), 1.66 at c = 0.1, negative at c = 1.0; crossing the 09:31 spread leaves
0.09, paying half of it 0.88.  y_oo and y_on carry the same edge (15.3 vs 15.5 bps per pick; corr 0.94 on the panel).  The L/S
books are dead at c = 0.3 (ls10 0.47, ls20 negative: 7 bps edge per side against ~4.5 bps of auction cost per side).  Breadth
is the wrong direction with realistic costs: on 300 / 600 names the exit auction costs 5.8 / 10.5 bps per side and the lo10
book falls to 0.46 / -0.62.  Liquidity as a feature helps, in the only way that makes sense: by keeping the book in the names
where the auction cost is small.  A pre-ranking filter on the predicted 09:31 half-spread (<= 10 bps) lifts the 150-name lo10
to 1.28 (auction cost <= 5 bps: 1.28), and restricting the universe to the 50 / 75 / 100 most liquid names gives 1.39 / 1.30 /
1.26 (gross per pick unchanged at 15 bps, cost halved).  A fixed-weight liquidity term in the composite does nothing (1.03-1.18
vs 1.08: inside the noise).  The VIX-ETP book is worse with the official open: VIXY's close-to-open decay is -7.5 bps/night
vs -12.1 to the 09:31 bar (the first minute carries a third of the overnight decay), so `vixon_r95_ls` drops from 1.23 to 1.06;
nothing saved from it.  Six candidates saved; the family fold-OOS over them is 1.10 (the chooser alternates between the u50 and
the fhs10 books, both ~0.85 correlated with the base: selection noise between near-duplicates costs 0.2-0.3).

## Step 1-2: books x universe x execution (Sharpe at commission / fold-OOS at commission)

| book (k, threshold) | names | MOO c=0.1 | **MOO c=0.3** | MOO c=1.0 | 09:31 cross (x1) | 09:31 half (x05) | old 1 bp |
|---|---|---|---|---|---|---|---|
| lo10 cAx>1.5 | 150 | 1.90 / 1.66 | **1.29 / 1.08** | -1.15 / -1.25 | 0.29 / 0.09 | 1.11 / 0.88 | 1.97 / 1.72 |
| lo10 cAx>1.5 | 300 | 1.46 / 1.36 | 0.54 / 0.46 | -2.83 / -2.79 | -0.49 / -0.54 | 0.47 / 0.39 | 1.63 / 1.50 |
| lo10 cAx>1.5 | 600 | 1.14 / 0.92 | -0.43 / -0.62 | -5.78 / -5.87 | -1.32 / -1.47 | -0.13 / -0.31 | 1.56 / 1.36 |
| lo10 cAx>2.0 | 150 | 1.84 / 1.63 | 1.28 / 1.09 | -0.99 / -1.08 | 0.35 / 0.18 | 1.11 / 0.91 | 1.91 / 1.68 |
| lo10 cAx>2.0 | 300 | 1.46 / 1.34 | 0.56 / 0.47 | -2.75 / -2.71 | -0.49 / -0.56 | 0.47 / 0.36 | 1.61 / 1.47 |
| lo10 cAx>2.0 | 600 | 1.18 / 0.94 | -0.37 / -0.58 | -5.68 / -5.76 | -1.31 / -1.46 | -0.10 / -0.30 | 1.59 / 1.36 |
| lo20 cAx>1.5 | 150 | 1.48 / 1.30 | 1.10 / 0.93 | -0.68 / -0.77 | -0.14 / -0.27 | 0.74 / 0.57 | 1.54 / 1.35 |
| lo20 cAx>1.5 | 300 | 1.31 / 1.22 | 0.69 / 0.63 | -1.80 / -1.74 | -0.76 / -0.76 | 0.27 / 0.22 | 1.33 / 1.24 |
| lo20 cAx>1.5 | 600 | 1.11 / 0.88 | -0.04 / -0.23 | -4.18 / -4.24 | -1.61 / -1.71 | -0.28 / -0.45 | 1.32 / 1.10 |
| lo20 cAx>2.0 | 150 | 1.14 / 0.99 | 0.80 / 0.66 | -0.78 / -0.86 | -0.29 / -0.41 | 0.49 / 0.35 | 1.21 / 1.05 |
| lo20 cAx>2.0 | 300 | 1.17 / 1.10 | 0.59 / 0.54 | -1.77 / -1.71 | -0.83 / -0.83 | 0.17 / 0.13 | 1.20 / 1.12 |
| lo20 cAx>2.0 | 600 | 1.10 / 0.87 | -0.02 / -0.20 | -4.05 / -4.09 | -1.59 / -1.68 | -0.27 / -0.43 | 1.29 / 1.08 |
| ls10 | 150 | 1.41 / 1.38 | 0.49 / 0.47 | -3.69 / -3.58 | -2.71 / -2.72 | -0.68 / -0.75 | 1.23 / 1.15 |
| ls10 | 300 | 0.97 / 0.94 | -0.47 / -0.49 | -6.03 / -5.97 | -3.82 / -3.83 | -1.49 / -1.53 | 0.97 / 0.92 |
| ls10 | 600 | 0.73 / 0.53 | -1.66 / -1.89 | -6.32 / -6.31 | -4.82 / -5.00 | -2.09 / -2.28 | 1.27 / 1.13 |
| ls20 | 150 | 0.39 / 0.47 | -0.27 / -0.18 | -4.04 / -3.81 | -4.68 / -4.57 | -2.04 / -2.01 | 0.27 / 0.25 |
| ls20 | 300 | 0.24 / 0.38 | -0.84 / -0.68 | -5.69 / -5.40 | -5.51 / -5.28 | -2.59 / -2.45 | 0.19 / 0.28 |
| ls20 | 600 | 0.36 / 0.19 | -1.74 / -1.91 | -8.19 / -8.21 | -6.89 / -6.94 | -3.29 / -3.43 | 0.60 / 0.43 |

Picks are identical across the execution columns (a pick needs both y_oo and y_on).  Mean gross per pick (y_oo / y_on, bps):
lo10 150 names 15.3 / 15.5, 300 names 14.0 / 14.1, 600 names 14.7 / 14.5; ls10 (per side) 7.4 / 6.9, 7.0 / 6.5, 8.1 / 7.8.  Mean
auction cost per side at c = 0.3 (entry / exit): lo10 150 names 2.3 / 3.8, 300 names 3.3 / 5.8, 600 names 5.5 / 10.5; lo20 1.7 /
2.6, 2.2 / 3.9, 3.7 / 7.0; ls10 1.7 / 2.8, 2.4 / 4.2, 4.0 / 7.8.  Predicted 09:31 half-spread of the picks: median 7.0 (150),
9.4 (300), 13.1 (600) bps, mean 9.8 / 12.4 / 16.6 (the brief's 6 / 10 figure is the NBBO measurement on the sampled days).
Family fold-OOS of the six MOO c = 0.3 books per universe (the chooser's view): 150 names 1.03, 300 names 0.50, 600 names -0.24.
The breadth result of the 09:31-exit lane (LO best on 150 names) is reinforced: with the auction model every step of breadth
adds cost faster than edge.  The opening auction is the expensive leg: exp(dv0) (20-day mean $ volume of the 09:30 bar) is
$6-25M for ranks 100-300 against a $40k order, and the cost scales with sqrt(order / auction volume) x daily vol.

Liquidity tiers of the base lo10 book on 150 names (y_oo, c = 0.3): ranks 1-50 37% of picks, gross 17.2 bps, cost 1.5 + 2.1
bps, hs0931 5.8; ranks 51-100 32%, 14.4 bps, 2.5 + 4.1, hs 10.6; ranks 101-150 31%, 14.0 bps, 3.1 + 5.5, hs 13.6.  The most
liquid third has the best gross and a third of the cost, which is what step 3 and the u50 book exploit.

## Step 2b: breadth in the other direction (lo10 cAx>1.5, MOO)

| names | picks/day | gross bps | cost/side c=0.3 (in / out) | MOO c=0.3 Sh / OOS | MOO c=0.1 | x05 | old 1 bp |
|---|---|---|---|---|---|---|---|
| 50 | 6.7 | 14.8 | 1.4 / 2.1 | **1.62 / 1.39** | 1.88 / 1.64 | 1.52 / 1.26 | 1.93 / 1.67 |
| 75 | 7.6 | 15.1 | 1.7 / 2.7 | 1.55 / 1.30 | 1.94 / 1.68 | 1.41 / 1.16 | 1.95 / 1.70 |
| 100 | 8.0 | 15.3 | 1.9 / 3.0 | 1.48 / 1.26 | 1.94 / 1.70 | 1.32 / 1.10 | 1.99 / 1.75 |
| 150 | 8.7 | 15.3 | 2.3 / 3.8 | 1.29 / 1.08 | 1.90 / 1.66 | 1.11 / 0.88 | 1.97 / 1.72 |
| 300 | 9.2 | 14.0 | 3.3 / 5.8 | 0.54 / 0.46 | 1.46 / 1.36 | 0.47 / 0.39 | 1.63 / 1.50 |
| 600 | 9.7 | 14.7 | 5.5 / 10.5 | -0.43 / -0.62 | 1.14 / 0.92 | -0.13 / -0.31 | 1.56 / 1.36 |

Under the old convention the universe size does not matter (1.67-1.75 from 50 to 150 names); under the auction model it is
the main knob.  The 50-name book has fewer picks (6.7/day: the cAx > 1.5 threshold bites) and 9.5% vol.

## Step 3: liquidity as a feature (MOO, c = 0.3, fold-OOS at commission; Sharpe in `s2.log`)

(a) filters applied before ranking: `fhs` = predicted hs0931 <= cap (bps); `fac` = max(entry, exit) auction cost per side at
c = 0.3 and the book's order size <= cap.  (b) tilts: cAx - w x rank(hs0931) (`rhs`) or - w x rank(hl1) (`rhl`), rank =
cross-sectional pct rank demeaned inside the book's universe, w = 0.3 / 0.5 fixed (3-value check incl. 0).

| book, names | base | fhs5 | fhs10 | fhs20 | fac5 | fac10 | fac20 | rhs0.3 | rhs0.5 | rhl0.3 | rhl0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lo10 s1.5, 150 | 1.08 | 1.10 | **1.28** | 1.14 | **1.28** | 1.23 | 1.08 | 1.09 | 1.18 | 1.03 | 1.05 |
| lo10 s1.5, 300 | 0.46 | 1.11 | 1.09 | 0.74 | 1.13 | 0.82 | 0.63 | 0.52 | 0.58 | 0.51 | 0.50 |
| lo10 s1.5, 600 | -0.62 | 1.02 | 0.80 | 0.12 | 1.19 | 0.63 | -0.02 | -0.49 | -0.41 | -0.58 | -0.62 |
| lo20 s1.5, 150 | 0.93 | 0.41 | 0.72 | 0.92 | 0.81 | 0.87 | 0.93 | 0.94 | 1.01 | 0.91 | 0.95 |
| lo20 s1.5, 300 | 0.63 | 0.29 | 0.86 | 0.65 | 0.67 | 0.65 | 0.64 | 0.69 | 0.69 | 0.56 | 0.50 |
| lo20 s1.5, 600 | -0.23 | 0.35 | 0.76 | 0.38 | 0.56 | 0.31 | -0.10 | -0.21 | -0.14 | -0.23 | -0.29 |
| ls10, 150 | 0.47 | -0.02 | 0.64 | 0.66 | 0.56 | 0.45 | 0.46 | 0.59 | 0.57 | 0.26 | 0.23 |
| ls10, 300 | -0.49 | -0.10 | 0.20 | -0.03 | -0.05 | -0.31 | -0.36 | -0.46 | -0.46 | -0.61 | -0.64 |
| ls10, 600 | -1.89 | 0.30 | 0.22 | -0.64 | -0.17 | -0.71 | -1.37 | -1.86 | -1.90 | -1.93 | -2.09 |
| ls20, 150 | -0.18 | -1.91 | -0.62 | -0.07 | -0.37 | -0.23 | -0.19 | -0.11 | -0.14 | -0.32 | -0.36 |
| ls20, 300 | -0.68 | -1.74 | -0.65 | -0.56 | -0.65 | -0.69 | -0.64 | -0.59 | -0.63 | -0.87 | -0.99 |
| ls20, 600 | -1.91 | -1.29 | -0.50 | -0.95 | -0.95 | -1.59 | -1.76 | -2.12 | -2.08 | -2.10 | -2.21 |

Share of universe rows kept by the filters on 150 names: hs <= 5 28%, <= 10 61%, <= 20 87%; auction <= 5 78%, <= 10 96%,
<= 20 99% (on 600 names hs <= 10 keeps 33%, auction <= 5 73%).  Reading:
* A filter helps the 10-name LO book: +0.2 on 150 names (1.28 for hs <= 10 and for auction <= 5, with 8.0 / 8.2 picks per day
  instead of 8.7), and it brings the 300 / 600-name books back to ~1.1-1.2, i.e. to the level of the 150-name book, never above
  it.  The two filters select similar names (corr 0.87 between the two 150-name books); hs <= 10 is the one saved because it is
  the directly measured quantity (NBBO calibration) and does not depend on c or the order size.
* The filter does not help the 20-name books (there are not enough names with cAx > 1.5 among the liquid ones to fill 20 slots:
  fhs5 drops lo20 to 0.41) and does not rescue the L/S books (best 0.66, with 9 of 16 folds negative).
* The tilt (rank(hs0931) or rank(hl1) with a fixed -0.3 / -0.5 weight) changes nothing: 1.03-1.18 vs 1.08 on the base book, the
  same +-0.1 on every other book; the composite's z-scored terms dominate a bounded rank term, and the 1.5 threshold already
  selects on magnitude.  No gain: not saved.
* The equivalent of the filter by construction is the universe cut (step 2b): u50 1.39 > fhs10 on 150 names 1.28 > base 1.08.

## Step 4: VIX-ETP book with the official-open exit (`s3_etf.py`)
VIXY close -> 09:31 bar: -12.07 bps/night; close -> official open: -7.54 bps (sd 290 / 295, corr 0.989, n 2413).  UVXY -24.9
vs -19.9.  The first minute of trading carries about a third of the ETP's overnight decay (the official open print sits at the
futures' overnight level; the decay appears as the first bars trade).  At 1.5 bps per side, borrow 2% / 5%:

| book | 09:31 exit (as saved) Sh / OOS | official-open exit Sh / OOS |
|---|---|---|
| vixon_r95 | 0.96 / 0.94 | 0.70 / 0.71 |
| vixon_r95_ls | 1.23 / 1.25 (saved file 1.225; my rebuild corr 1.0000) | 1.06 / 1.11 |
| uvxon_r95 | 1.05 / 1.03 | 0.87 / 0.88 |

The MOO exit is materially worse for the VIX ETPs, so the existing 09:31 convention should stay for them (the 1-2 bps VIXY
spread makes the 09:31 crossing realistic).  Nothing saved.

## Saved candidates (`agents/xsrule3/cands`, series net of auction cost at c = 0.3 / half-spread and borrow, BEFORE commission)

| candidate | book | pre-comm Sh | Sh @ $3/$0.4M | fold-OOS | halves | maxDD | ret / vol | folds neg | picks/day | corr xsr_cAx_lo10_s15_u150 | corr onml_f26calx_u2000_q3_k10 | corr SPY c->c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xsr3_cAx_u150_moo_lo10 | LO top-10 cAx>1.5, 150 names, MOC+MOO | 1.59 | 1.29 | 1.08 | 2.19 / 0.12 | -23.6% | 16.3% / 12.6% | 7/16 | 8.7 | 0.97 | 0.25 | 0.46 |
| xsr3_cAx_u150_moo_lo20 | LO top-20 cAx>1.5, 150 names, MOC+MOO | 1.87 | 1.10 | 0.93 | 1.94 / -0.03 | -16.9% | 10.9% / 9.9% | 6/16 | 14.6 | 0.91 | 0.26 | 0.46 |
| **xsr3_cAx_u50_moo_lo10** | LO top-10 cAx>1.5, 50 names, MOC+MOO | 2.02 | **1.62** | **1.39** | 2.43 / 0.66 | -15.3% | 15.5% / 9.5% | 6/16 | 6.7 | 0.79 | 0.24 | 0.44 |
| **xsr3_cAx_u150_fhs10_moo_lo10** | LO top-10 cAx>1.5, 150 names, hs0931 <= 10 filter, MOC+MOO | 1.88 | **1.54** | **1.28** | 2.45 / 0.48 | -18.9% | 17.1% / 11.1% | 4/16 | 8.0 | 0.87 | 0.25 | 0.42 |
| xsr3_cAx_u150_fhs10_moo_ls10 | L/S top/bottom 10, 150 names, hs0931 <= 10 filter, MOC+MOO | 1.31 | 0.67 | 0.64 | 1.93 / -0.43 | -33.5% | 7.8% / 11.7% | 9/16 | 20 | 0.36 | 0.15 | 0.01 |
| xsr3_cAx_u150_x05_lo10 | LO top-10 cAx>1.5, 150 names, MOC + 09:31 exit at half the half-spread | 1.40 | 1.11 | 0.88 | 1.99 / -0.04 | -27.1% | 14.4% / 13.0% | 6/16 | 8.7 | 1.00 | 0.26 | 0.46 |

Sensitivity of every saved book over the execution assumption (Sh / fold-OOS at commission; the series as saved is c = 0.3 or x05):

| candidate | c = 0.1 | c = 0.3 | c = 1.0 | 09:31 cross | 09:31 half | old 1 bp |
|---|---|---|---|---|---|---|
| xsr3_cAx_u150_moo_lo10 | 1.90 / 1.66 | 1.29 / 1.08 | -1.15 / -1.25 | 0.29 / 0.09 | 1.11 / 0.88 | 1.97 / 1.72 |
| xsr3_cAx_u150_moo_lo20 | 1.48 / 1.30 | 1.10 / 0.93 | -0.68 / -0.77 | -0.14 / -0.27 | 0.74 / 0.57 | 1.54 / 1.35 |
| xsr3_cAx_u50_moo_lo10 | 1.88 / 1.64 | 1.62 / 1.39 | 0.23 / 0.02 | 1.01 / 0.75 | 1.52 / 1.26 | 1.93 / 1.67 |
| xsr3_cAx_u150_fhs10_moo_lo10 | 2.07 / 1.79 | 1.54 / 1.28 | -0.68 / -0.84 | 1.08 / 0.83 | 1.54 / 1.26 | 2.01 / 1.71 |
| xsr3_cAx_u150_fhs10_moo_ls10 | 1.45 / 1.42 | 0.67 / 0.64 | -3.14 / -3.04 | -0.84 / -0.84 | 0.26 / 0.20 | 1.18 / 1.09 |
| xsr3_cAx_u150_x05_lo10 (same picks as the base) | 1.90 / 1.66 | 1.29 / 1.08 | -1.15 / -1.25 | 0.29 / 0.09 | 1.11 / 0.88 | 1.97 / 1.72 |

The u50 book is the only one that survives c = 1.0 (0.02) and crossing the 09:31 spread (0.75): its names have a 5.9 bps
predicted half-spread and $25M+ opening-auction volume.  For the filtered 150-name book the 09:31 exit at half the spread
(1.26) is as good as the MOO exit (1.28): with the filter, the exit venue stops mattering.

Per-fold Sharpe (half-years from 2018H2, at commission):

| candidate | 18H2 | 19H1 | 19H2 | 20H1 | 20H2 | 21H1 | 21H2 | 22H1 | 22H2 | 23H1 | 23H2 | 24H1 | 24H2 | 25H1 | 25H2 | 26H1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u150_moo_lo10 | -0.2 | 1.9 | 2.0 | 2.6 | 4.7 | 3.0 | -1.5 | -0.5 | -0.1 | 0.7 | -0.9 | 0.9 | 3.5 | -0.4 | 1.2 | -2.6 |
| u150_moo_lo20 | -0.8 | 1.2 | 1.3 | 3.1 | 3.9 | 2.9 | -1.5 | -0.1 | -0.7 | 0.1 | 0.3 | 0.8 | 2.6 | -1.3 | 2.1 | -3.1 |
| u50_moo_lo10 | -0.4 | 1.9 | 0.8 | 3.7 | 5.0 | 2.6 | 0.6 | -0.1 | -0.9 | -0.3 | 2.6 | 0.8 | 3.8 | -0.2 | 3.1 | -1.6 |
| u150_fhs10_moo_lo10 | -0.5 | 2.0 | 1.1 | 2.7 | 4.8 | 3.1 | -0.2 | 1.0 | -0.9 | 0.6 | 0.6 | 0.5 | 3.2 | 0.1 | 1.6 | -1.6 |
| u150_fhs10_moo_ls10 | -0.1 | -0.1 | -0.3 | 4.0 | 6.2 | 1.8 | 0.7 | 0.9 | -1.8 | -0.7 | -1.8 | -1.8 | 2.3 | 0.3 | -0.7 | -1.5 |
| u150_x05_lo10 | 0.2 | 1.8 | 1.5 | 2.5 | 4.1 | 2.0 | -2.0 | -0.5 | -0.1 | 0.2 | -0.5 | 0.2 | 3.0 | -0.6 | 0.3 | -1.2 |

2026H1 is negative for every variant (-1.2 to -3.1; the old-convention book has -0.1 there): the 2026 Q1 loss is mostly the
widened spreads / auction costs in a thin quarter, consistent with the README's note that spreads widened in 2025-26.
Family fold-OOS over the saved set 1.10 (picks: u150_fhs10 8 folds, u50 8 folds); over the four LO MOO books 1.12; the L/S
book alone 0.64.  Candidate-candidate correlations: LO books 0.79-0.97 with each other (u50 vs base 0.81), the L/S 0.33-0.45.

## Leakage checklist
* Signal: unchanged from `agents/xsrule` (z30, z60 = r(329,389), z5l = r(384,389), zon_prev, vlast_x over bins 0..389); z-scores
  over a lagged 20-day sd; cross-sectional ranks on the same day's names only; everything known at 15:59.  Exact reproduction
  of the saved `xsr_cAx_lo10_s15_u150` under the old convention (corr 1.0000).
* Target y_oo = official open(t+1) / official close(t) - 1 from `data/panel_liquid.npz` (split-adjusted by `Min.sf`; falls back
  to the 09:30 bar open when the panel has no open), |y| > 0.5 -> NaN.  Entry at the official close needs an MOC order entered
  by 15:50 / 15:55, so the last 5-10 minutes of z60 / z5l are not available in practice (same caveat as every overnight book).
* Costs: `dv0`, `dvc`, `ldv20` are lagged 20-day means (known before the open of t); `rv20` is the 20-day sd of close-to-close
  returns through the close of t (it uses the closing print of t, but only to scale the cost, not in the signal); `hs0931` at
  date t uses lagged features plus bins 0/1 of day t (known at 09:31 of t, i.e. before the 15:59 decision).  The spread model
  itself is a LightGBM fit on 2017-2026 quotes (`logs/spread_fit.log`): a cost calibration that is in-sample by construction.
  The `fhs10` filter therefore uses an in-sample-calibrated *cost* model as a feature; it predicts spreads, not returns, from
  lagged liquidity, and the auction-cost filter (no quote model, 1.28 as well) and the plain universe cut (1.39) give the same
  conclusion without it.
* Universe ranks are 2016-2026 median dollar volume (survivorship and look-ahead in the list, as in every sleeve); the u50 book
  leans on this more than the 150-name book does (it holds only names that are among today's 50 most liquid).
* The series row at DEV_END (2026-04-27) uses the open of 2026-04-28, exactly as every existing candidate's y_on row uses the
  09:31 bar of 04-28 (series indexed at the decision day); no other holdout date was read.  Reference series were cut at DEV_END.
* Selection: the tables above show 6 books x 3 universes x 6 executions, 11 liquidity variants x 12 book-universes and three
  extra universe sizes; the six saved were chosen with those numbers in view.  The honest number for "this lane's realistic LO
  book" is the family fold-OOS over the saved LO set, 1.12, not the best single series (1.39).

## What I would keep
1. `xsr3_cAx_u150_fhs10_moo_lo10` (1.28 fold-OOS, 4/16 negative folds, the least execution-sensitive of the 150-name books:
   1.26 at the 09:31 half-spread exit, 0.83 even crossing it) as the realistic replacement of `xsr_cAx_lo10_s15_u150`.
2. `xsr3_cAx_u50_moo_lo10` (1.39, 9.5% vol, survives c = 1.0) as the conservative version; it is 0.85 correlated with (1), so
   they are one family for the chooser, not two sleeves.  Together with the base (1.08) they bracket the cost of realism:
   1.72 (old convention) -> 1.08 (same book, MOC + MOO at c = 0.3) -> 1.28-1.39 (same signal, liquid names only).
3. Drop from consideration: the L/S variants (0.64 at best, 9 negative folds), the 20-name books (0.93), breadth beyond 150
   names, the liquidity tilt.  The x05 book (0.88) is there only to document the 09:31 alternative.
4. Keep the VIX-ETP book on its 09:31 exit; the official open is worse for it.
