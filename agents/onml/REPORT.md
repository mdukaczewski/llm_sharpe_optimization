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
