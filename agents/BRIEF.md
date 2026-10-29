# Agent brief (2026-10-29 session) — goal: portfolio Sharpe >= 4 out of sample

Read `rlib.py` first (docstring = the protocol).  Short version:

* Data is local under `data/` (daily panel for every US ticker, minute bars for the
  1500 most liquid common stocks + ~30 ETFs, 2016-06..2026-10-28).  Loaders in
  `rlib.py` cut at DEV_END = 2026-04-27.  Never read dates after that, never set
  STRAT_END.  The holdout (2026-04-28..2026-10-28) is looked at once by the
  coordinator.
* Deliverable per agent: a handful (<= 8) of CAUSAL daily NET return series written
  with `rlib.save_cand(lane, name, series, note)`, plus a short report
  `agents/<lane>/REPORT.md` with: what was tried, full-sample Sharpe, FOLD-OOS Sharpe
  (`rlib.fold_oos`), halves, correlation with the existing best sleeve
  (`sleeves/onlgb1200q_ls20.csv`, overnight LightGBM L/S) and with SPY, and what you
  would keep.  Fewer, better-motivated candidates beat a grid: every extra candidate
  adds selection noise to the fold procedure.
* Costs: 1 bp per side on names ranked <= 600 by dollar volume, 1.5 bps beyond, borrow
  1%/yr on shorts (built into `rlib.book`).  ETFs: SPY/QQQ 0.5 bp, others 1.5 bps.
  Fills: a decision at minute bin m fills at the close of bin m+1; a decision at the
  close fills at the official close (bin 390) and the overnight exit fills at the
  09:31 bar (bin 1).  Never use the signal bar as the fill.
* Compute: 4 cores shared by 4 agents.  LightGBM with n_jobs=2, quarterly retrain
  only for your final 1-2 runs (yearly for exploration).  Cache feature frames
  (`rlib.minute_frame` caches per ticker).  Log progress to `agents/<lane>/log.txt`.
* Existing results to beat (fold-OOS, from README): overnight LightGBM L/S top20 on
  1200 names 3.2; intraday learned L/S 1.7; rule overnight 1.6; momentum 1.1; vol 1.0.
  Portfolio of 7 sleeves: 3.33 fold-OOS.  Dead lanes (do not repeat): pairs,
  residual reversal, ETF spreads, classic daily factor zoo, leveraged ETF pairs,
  Gao intraday momentum, SPY-200dma regime filters, vol targeting, last-hour books.
* Leakage checklist: z-scores use a lagged rolling std; every rolling window ends at
  the decision time; labels end before the training cutoff (label_lag_days); no
  survivorship beyond the universe file itself; targets with |y| > 0.5 are NaN
  (unadjusted splits in minute data).
