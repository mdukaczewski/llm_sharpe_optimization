#!/usr/bin/env python3
"""assemble REPORT.md from the out/*.csv tables and the candidate notes"""
import sys, os, re; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/dayml3")
import pandas as pd, numpy as np, mk_tables as T
from mk_tables import md, OUT
HERE = os.path.dirname(os.path.abspath(__file__)); R = "/home/user/llm_sharpe_optimization"
def final_tab(cfg, unis=("u1500", "u600", "u300")):
    F = pd.read_csv(f"{OUT}/final.csv") if cfg != "pair" else pd.read_csv(f"{OUT}/final_pair.csv")
    F = F[(F.cfg == cfg) & F.uni.isin(unis)]; F["f"] = F.f.astype(str); rows = []
    for (uni, k), g in F.groupby(["uni", "k"], sort=False):
        d = {"universe": uni, "k": k}
        for f in ["1.0", "0.5", "0.25", "old", "gross-comm"]:
            x = g[g.f == f]
            if len(x): d[f"Sh f={f}"] = x.sh.iloc[0]; d[f"fo f={f}"] = x.fo.iloc[0]
        x = g[g.f == "0.5"].iloc[0]; d["halves f=0.5"] = f"{x.h1:.2f}/{x.h2:.2f}"; d["mdd f=0.5"] = f"{x.mdd*100:.0f}%"; d["ret f=0.5"] = f"{x.ret*100:.0f}%"
        d["by year f=0.5"] = x.yrs; rows.append(d)
    Rr = pd.DataFrame(rows); return md(Rr, list(Rr.columns))
def notes():
    s = ""
    for line in open(f"{HERE}/cands/NOTES.txt"):
        name, sh, note = line.rstrip("\n").split("\t"); m = re.search(r"\$0\.4M: (.*?) \| corr (.*?) \| tiers (.*?)(?: \| (NEGATIVE.*))?$", note)
        s += f"| {name} | {m.group(1)} | {m.group(2)} | {m.group(3).replace('+nanbps','-')} | {m.group(4) or ''} |\n"
    return s
pair_done = os.path.exists(f"{OUT}/final_pair.csv")
doc = f"""# dayml3 lane: the learned intraday book under REALISTIC execution (spread as a cost and as a feature)

Scope (agents/BRIEF6.md): re-build the learned intraday book (`dayml2_ens_e30_k8`: decide at the 09:31 bar, fill at the
09:33 bar = bin 3, exit at the 10:01 bar = bin 31, L/S top/bottom k, 1500 names) so that it is evaluated and optimised at
the PREDICTED NBBO half-spread of every name it trades (`liq.py`: `hs0931` for the 09:33 entry, `hs1001` for the exit,
cost per side = f x half-spread, f = 1 crosses the spread, 0.5 = midpoint-ish, 0.25 = mostly passive), plus 1 %/yr borrow
and the $3/order commission on $0.4M (2k positions x 2 orders: k4 = 1.2, k6 = 1.8, k8 = 2.4 bps/day).  Metric: fold-OOS
Sharpe (`rlib.fold_oos`, 16 half-year windows 2018-07..2026-04; for a single series this is its Sharpe over that span).
Dates <= 2026-04-27 everywhere (`rlib.DEV_END`); the holdout was never read.

Code: `d3lib.py` (frame = `d2lib.build` + 17 liq columns, per-name spread-costed books, causal cost-aware selection),
`step1.py` (saved predictions costed), `step2.py` / `step2b.py` (cost-aware selection on the saved ensemble), `step3.py` /
`step3b.py` / `step3c.py` (yearly refits with / without the liq features, net targets, two-sided net targets, pairs, f = 1
targets), `test_causal.py`, `fitq.py` (quarterly fit of one configuration, n_jobs = 2), `final.py` (books + candidates from
the saved quarterly predictions), `mk_tables.py` / `build_report.py` (this file).  Logs `log.txt`, `step*.log`, `fitq_*.log`,
`final_*.log`; tables `out/step1.csv`, `out/step1_tiers.csv`, `out/step2.csv`, `out/step2b.csv`, `out/step3_Y.csv`,
`out/step3b_Y.csv`, `out/step3c_Y.csv`, `out/final.csv`{', `out/final_pair.csv`' if pair_done else ''}; predictions
`out/pred_step3_*_Y.parquet` (yearly) and `out/pred_final_*_Q.parquet` (quarterly); daily book frames `out/*_k*.parquet`
(columns g gross, sc spread cost at f = 1, old old-convention cost, bor borrow, comm commission).

## 0. Summary

* **Nothing is positive at f = 1 on any universe, with any model, at any k.**  Crossing the predicted spread on both legs
  costs 20 bps per trade in ranks 1-300, 34 in 301-600 and 53 beyond 600, against a gross edge of 10-16 bps per trade;
  the best f = 1 book found anywhere is -0.1 (k = 2, 4 positions) and every k >= 4 book is <= -0.5 fold-OOS.  Models
  trained directly on the f = 1 net target (`B1f1+S1f1`) are -1.2 at best.  The x1 candidates are saved as asked and are
  marked NEGATIVE in their notes.
* **At f = 0.5 the book is marginal and only works with cost-aware selection on the liquid half of the universe.**  The
  saved ensemble costed at f = 0.5 is -2.3 (k8) / -1.9 (k6); restricting the universe to ranks <= 300 / 600 does not help
  (-1.7 / -2.1); a cap on the predicted round-trip cost or a linear net score reaches at best +0.2.  The two things that
  work are (a) a gross-target model with the liq features whose top/bottom ranks are re-scored as E[y | rank] - 0.5 c on
  both sides (causal calibration) and (b) two-sided net-target models (long rank(y - 0.5c), short rank(-y - 0.5c)).  Final
  (quarterly retrain): **`dayml3_liqsel_u600_x05_k4` Sh 1.03 / fold-OOS 0.77, halves 2.18 / -0.19, maxDD -44 %**, k6
  0.51, k8 0.19.  The f = 0.5 book has been flat since 2021 (yearly Sharpe 2021-26: -0.4 -0.8 +0.6 -0.9 -0.6 +1.2), so it
  is a "only works at f = 0.5, and only before 2021" book.
* **The liq features are a strong signal improvement at the mid-quote**: the same gross book at the old 1-1.5 bps
  convention goes from 2.07 (no liq features, yearly) to 2.88 fold-OOS (yearly) and 2.97 quarterly at k8 (vs 2.40 for
  `dayml2_ens_e30_k8`), because the model learns which thin names bounce most.  That is exactly why it does not help at
  a realistic spread: the gross-target model allocates 70 % of its positions beyond rank 600, where the round-trip
  half-spread is 46 bps.
* A single rank(y - cost) target is WRONG for a long/short book (step 3 (ii)/(iii) as specified): the costliest names get
  the lowest net rank and the book shorts them (601-1500 shorts: -24 bps/trade).  The cost has to enter as a two-sided
  score or as two models.

## 1. Cost model check

`liq.py` was not broken; no change was made to it.  Sanity check of its predictions against the measured NBBO of this
book's own trades (`out/quotes_trades.csv`, dayml book, dev dates, n = 3,728 round trips):

| tier | n | entry 09:33: measured median / predicted median | measured mean / predicted mean | exit 10:01: measured median / predicted | measured mean / predicted |
|---|---|---|---|---|---|
| 1-300 | 627 | 7.1 / 10.7 | 11.3 / 13.8 | 3.4 / 3.8 | 5.0 / 5.0 |
| 301-600 | 759 | 16.1 / 21.6 | 22.1 / 25.7 | 6.0 / 7.4 | 8.4 / 9.1 |
| 601-1500 | 2342 | 26.0 / 35.0 | 35.2 / 39.5 | 10.2 / 11.4 | 14.0 / 13.6 |
| all | 3728 | 19.6 / 27.6 | 28.5 / 32.3 | 7.7 / 9.2 | 11.3 / 11.3 |

corr(predicted, measured) 0.73 entry / 0.78 exit.  The exit model is unbiased; the entry cost is CONSERVATIVE by ~13 %
on the mean (the "0931" model pools the wider 09:31:00 quotes of the overnight book's exit with the 09:33 quotes of this
book's entry, as `liq.training_rows` relabels 0933 -> 0931).  All numbers below use the brief's convention (f x hs0931
at entry); a 13 % lower entry cost moves a -2.3 book to about -1.9 and does not change any conclusion.  Median predicted
half-spreads over all (date, name) rows: 09:31 10.9 / 19.5 / 29.7 bps and 10:01 3.7 / 6.1 / 8.7 bps by tier.

## 2. Step 1: the saved predictions costed at the predicted half-spreads

Ensemble = average per-day pct rank of `agents/dayml/out/pred_y_60d_Q.parquet` and `pred_y_120d_Q.parquet`, exit bin
31, equal weight; `u600` / `u300` select top/bottom k within ranks <= 600 / <= 300; `tier..` books select within the
tier.  "old" = the 1 / 1.5 bps per side convention of the earlier lanes, "gross-comm" = mid-to-mid minus commission and
borrow.  All at the $0.4M commission.

{T.step1()}
By tier of the full-universe books (share of positions, median predicted half-spreads of the positions, bps per trade):

{T.step1_tiers()}
Reading: the thin names carry the largest gross edge (16 bps) but a 53 bps round-trip half-spread; only the 1-300 names
are (barely) positive at f = 0.5 (+0.4 / +3.6 bps per trade before commission).  The book needs f <= 0.25 to be positive
(k6: Sh 0.77, fold-OOS 0.52), i.e. it must capture three quarters of the spread on both legs.

## 3. Step 2: cost-aware selection on the saved ensemble

{T.step2()}
The calibration for the net score is in-sample: linear OLS of the traded return on the ensemble rank over every
(date, name) row, `E[y_30d] = -0.15 bps + 12.4 bps x rank` (rank in [-0.5, 0.5]), i.e. +6.0 bps for the top name and
-6.3 for the bottom; the y_60d map is `+0.49 + 13.4 x rank`.  The relation is convex at the extremes (mean y_30d +14.1
bps in the top 1 %, -16.0 in the bottom 1 %, vs +5.3 / -8.1 in the next 4 %), which the "bucket" calibration (1 % buckets
beyond |rank| > 0.45) captures.  Nothing here is positive at f = 1; at f = 0.5 the best construction is 0.35 fold-OOS
(bucket net score, k4) with a second half of -0.7.

## 4. Step 3: refits with the liquidity features (yearly retrain, seed 0, n_jobs = 1, 2.53 M rows)

Features: `dlib.EXT` (43) -> `EXTL` = EXT + lagged `ldv20 lpx hl1 hl5 hlc roll30 n30 nc30 dv0 dvc`, same-morning
`t_hl0 t_hl1 t_n0 t_n1 t_dv0` (bins 0-1 of day t) and the predicted `hs0931 hs1001` (60 features).  Targets: A = rank
of y_60d; B = rank of y_60d - 0.5 (hs0931 + hs1001)/1e4; C = B on ranks <= 600 (trained and predicted there only).
Suffix 0 = EXT, 1 = EXTL.  IC against y_30d (the traded return): A0 0.031 (t 13.5), A1 0.035 (14.4), B0 0.025, B1
0.021, C0 0.029, C1 0.025.  Books: L/S k in 4 / 6 / 8, fill bin 3, exit bin 31, equal weight, $0.4M commission.

{T.step3("step3_Y.csv")}
* With vs without the liq features on the same book: at the old convention A1 beats A0 everywhere (u1500 k6 fold-OOS
  2.88 vs 2.07, k8 2.87 vs 2.13); at f = 0.5 both are negative and A1 is slightly worse (-2.25 vs -2.09 at k6) because
  it concentrates further in the thin names (71 % of positions beyond rank 600 vs 63 %).
* The single-ranking net targets (B, C) are worse than the gross target at every f: the model learns "cheap = high net
  rank, expensive = low net rank", so the shorts are the most expensive names (B1 u1500 k4, f = 0.5: 601-1500 shorts
  -23 bps per trade).  Restricting to ranks <= 600 (C) softens but does not fix this.

### 4b. Two-sided net targets and cost-aware selection on the refit models (`step3b.py`, `step3c.py`, `test_causal.py`)

`B1+S1`: long the top-k of B1 (rank(y_60d - 0.5c)), short the top-k of S1 (rank(-y_60d - 0.5c)); `C1+S1_600` the same
on ranks <= 600; `A1sel05`: the gross liq model A1 re-scored as E[y_30d | rank] - 0.5c (long) and -E - 0.5c (short),
bucket calibration IN-SAMPLE; `A1sel0.5c` / `A1sel1.0c`: the same with a CAUSAL calibration (half-year cuts, each window
calibrated on the prior predictions only; the selection cost at f = 0.5 resp. 1.0); `B1x120+S1x120`: the pair on the
y_120d net targets; `ens4`: average pct rank of the two long models / the two short models; `B1f1+S1f1`: two-sided models
trained on the f = 1 net target.

{T.step3("step3b_Y.csv")}
{T.step3("step3c_Y.csv")}
Causal calibration check on the yearly A1 predictions (fold-OOS at f = 0.5, k = 2 / 3 / 4 / 6 / 8): u1500 0.73 / 0.68 /
0.56 / 0.31 / 0.00, u600 1.08 / 0.79 / 0.64 / 0.18 / -0.06 (the in-sample bucket version gave 0.83 at u600 k4); with the
selection cost at f = 1 the f = 1 books are still negative (best -0.02 at u600 k2) and the f = 0.5 books are worse.
Findings: (i) the two-sided formulation is the first thing positive at f = 0.5 (B1+S1 u1500 k4: Sh 0.66 / fold-OOS 0.56,
halves 1.2 / 0.1, maxDD -43 %); (ii) the cost-aware selection on the gross liq model is as good or better (A1sel u600
k4: 0.64 causal) and needs one model instead of two; (iii) the y_120d pair adds nothing (ens4 u1500 k4 0.58 vs 0.56);
(iv) the edge is in the very top of the ranking: k = 2 / 3 beat k = 4 at every construction (B1+S1 u1500 k2: 0.95,
second half +0.5), but a 4-position book is not what the brief asks for and its per-fold Sharpe is dominated by single
names; (v) f = 1 cannot be rescued by training on it.

## 5. Final: quarterly retrain, n_jobs = 2

### 5a. `sel_A1`: quarterly A1 (EXTL, rank y_60d; IC(y_30d) 0.038, t 15.8) with the causal cost-aware selection at f = 0.5

{final_tab("sel_A1")}
The same quarterly model traded as a plain gross book (`gross_A1`, no cost in the selection), u1500: old-convention
fold-OOS k4 2.64 / k6 2.90 / k8 2.97 (dayml2_ens_e30_k8: 2.40); at f = 0.5 -1.72 / -2.31 / -2.78.

### 5b. Saved candidates (`agents/dayml3/cands/`, series = gross - f x spread cost - borrow, BEFORE commission)

| candidate | at $0.4M (commission added) | corr dayml2_ens_e30_k8 / onlgb1200q_ls20 / SPY | tiers: share of positions / net bps per trade at its f | |
|---|---|---|---|---|
{notes()}
f-sensitivity of the kept k4 / k6 / k8 books (u600, $0.4M): see the `sel_A1` table above (columns f = 1 / 0.5 / 0.25 / old);
at f = 0.25 the k4 book is 1.82 / 1.54 and at the old convention 1.99 / 1.72.  Yearly Sharpe of `dayml3_liqsel_u600_x05_k4`
at $0.4M: 2017 +2.7, 2018 +4.5, 2019 +2.1, 2020 +3.0, 2021 -0.4, 2022 -0.8, 2023 +0.6, 2024 -0.9, 2025 -0.6, 2026 +1.2.
Correlation with `dayml2_ens_e30_k8` is 0.43-0.45 (half the positions are different names), ~0 with the overnight sleeve
and SPY.

{"### 5c. `pair`: quarterly two-sided net-target models B1 (long) + S1 (short), 1500 names" + chr(10) + chr(10) + final_tab("pair") if pair_done else "### 5c. `pair`: quarterly B1 + S1 (two-sided net targets) -- running, see log.txt / final_pair.log"}

## 6. Leakage checklist

* Every feature of a row is known at the end of the 09:31 bar of day t: `dlib` close-time features are shifted one day
  inside the ticker; the `o_*` / gap / first-minute features and the `t_hl0 t_hl1 t_n0 t_n1 t_dv0` liq columns use bins 0-1
  of day t only; the lagged liq proxies (`ldv20 ... dvc`) are 20-day means ending at t-1 (`_lag_mean` shifts by one);
  `hs0931` / `hs1001` are functions of those columns only (the spread model itself is a cost calibration fit on
  2017-2026 quotes, in-sample by construction, as the brief states).
* No 09:32 fill anywhere: entry = bin 3 close, exit = bin 31 close (`y_30d` of `d2lib.extra2`).
* Walk-forward: expanding window, label lag 3 days before each cut, yearly (exploration) / quarterly (final) retrain from
  2017-01-01, seed 0; the universe mask of the C / S1_600 models applies to training rows and predictions alike.
* The causal calibration of the net score uses predictions and realised y_30d strictly before each half-year cut minus
  3 days; the first window (2017-01..06, outside the fold-OOS span) uses the fixed linear slope 12.4 bps x rank.
* Books: equal weight 1/k per side, every name's own predicted spread, borrow 1 %/yr on shorts, commission 2k x 2 x $3 /
  $0.4M per day; `|y| > 0.5` rows are NaN (unadjusted splits).  Dates <= DEV_END; `STRAT_END` never set.
* Choices made in-sample on the dev span: the feature set, k, the universe cut (u600) and the f = 0.5 selection cost.
  The in-sample bucket calibration of step 2 / `A1sel05` is reported as such and not used in any candidate.

## 7. What I would keep

* Nothing from this lane as a trading book at f = 1, and nothing at f = 0.5 for the post-2021 period: the realistic
  intraday book is dead since 2021 on every construction (the gross edge at the mid halved while spreads widened from a
  14 bps median entry half-spread in 2017 to 32 in 2026).  `dayml3_liqsel_u600_x05_k4` is the best honest f = 0.5 series
  (0.77 fold-OOS, 2.2 before 2021-10, -0.2 after) and is only worth holding in the chooser if the coordinator wants an
  f = 0.5 sleeve at all; the x1 series exist to document that crossing the spread loses.
* The liq features themselves (`EXTL`): +0.8 fold-OOS at the mid-quote convention on the same book.  If the coordinator
  keeps any mid-quote intraday series for comparison, `gross_A1` (quarterly, k8, old convention 2.97) supersedes
  `dayml2_ens_e30_k8` (2.40); it is NOT saved as a candidate because the brief's convention is the spread-costed series.
* The two-sided treatment of execution cost (score each side as expected gross minus its own cost) is the construction
  to reuse for any other book that is re-costed at the NBBO; the single net-rank target should not be repeated.
"""
open(f"{HERE}/REPORT.md", "w").write(doc); print("REPORT.md written", len(doc))
