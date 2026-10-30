# Agent brief, phase 6 (2026-10-30): realistic execution — spread as a feature and as a cost

Read `agents/BRIEF.md` (protocol, leakage checklist, compute rules) and `README.md` (section
"Realistic fills") first.  What changed:

## The finding
`quotes_check.py` looked up the real NBBO at the decision times of the three new books (every 10th
day since 2017).  The signals hold at the mid-quote (mid-to-mid returns equal the bar-based
backtest), but the median NBBO half-spread at 09:31 in the names the books pick is ~21 bps (7.5
bps in ranks 1-300, 16 in 301-600, 26 beyond 600), ten times the 1-2.5 bps per side in every
series so far.  Crossing the spread at 09:31 makes every minute-bar book negative.  Spreads at
09:33 are ~20 bps (entry of the intraday book), ~8 bps at 10:01.

What survives: the overnight edge is at least as large to the OFFICIAL OPEN as to the 09:31 bar
(saved onml predictions, L/S top-10: 51.8 bps/day gross to the open vs 49.9 to the 09:31 bar,
Sharpe 5.7 vs 5.1 full sample).  Market-on-close entry + market-on-open exit are two auction
prints: no spread to cross, only commission and auction impact.

## New shared code: `liq.py`
* `liq.liq_frame(tk)`: per-day liquidity proxies from minute bars, LAGGED (known before the open)
  unless prefixed `t_` (known at the end of the 09:31 bar) or `c_` (known at 15:59): `ldv20 lpx
  hl1 hl5 hlc roll30 n30 nc30 dv0 dvc t_hl0 t_hl1 t_n0 t_n1 t_dv0 c_hlc c_nc30`, and the
  calibrated predicted NBBO half-spreads in bps `hs0931`, `hs1001`, `hs1559` (LightGBM fit on the
  measured quotes; the fit report is in `logs/spread_fit.log`; it is a cost calibration, in-sample
  by construction).  `liq.add_liq(D)` merges them into a stacked (date, tk) frame.
* `liq.overnight_targets(tk)` / `liq.add_y_oo(D)`: `y_oo` = official open(t+1) / official close(t)
  - 1, split-adjusted (MOC entry, MOO exit).
* `liq.auction_cost_bps(order_usd, auction_dv_usd, rv_daily, c)`: per-side cost of an auction
  order = max(1 bp, c x daily vol in bps x sqrt(order / expected auction dollar volume)); use
  exp(dv0) for the opening auction (20-day lagged mean dollar volume of the 09:30 bar), exp(dvc)
  for the closing auction (16:00 bar), rv20 for the vol; order = $400k / (2k) for an L/S top-k
  book.  c = 0.3 is the central assumption; ALWAYS report c = 0.1 / 0.3 / 1.0.
* `liq.spread_cost_bps(hs, f)`: per-side cost of a continuous-market order paying fraction f of
  the predicted half-spread (f = 1 crosses the spread, f = 0.5 is the passive/midpoint case).

## Conventions for the candidates you save (`rlib.save_cand(lane, name, series, note)`)
* The series is NET of the modelled execution cost of its own fills (auction impact at c = 0.3
  and/or spread fraction f, say which in the note) and of borrow, but BEFORE the $3 commission
  (the coordinator adds commission by position count; name the book `..._k10` / `..._lo10` so the
  count parses).  Put the main execution assumption in the name (`moo` for MOC+MOO, `x1` for
  crossing the spread, `x05` for half the spread).
* Report in `agents/<lane>/REPORT.md`, for every kept candidate: full-sample and fold-OOS Sharpe
  (`rlib.fold_oos`) at the $0.4M commission, halves, maxDD, by tier of liquidity, and the
  sensitivity table over c (auction) or f (spread).  Also the same book at the OLD convention
  (1-2.5 bps/side) so the coordinator sees the cost of realism.
* Dates <= rlib.DEV_END (2026-04-27).  Never set STRAT_END.  Never use the 09:32 bar as a fill.
* Compute: 4 cores shared by 3 agents and a quote download; LightGBM n_jobs=1 (2 for the final
  run only), yearly retrain for exploration, quarterly only for the final 1-2 runs.
  Log to `agents/<lane>/log.txt`; the coordinator reads it.
