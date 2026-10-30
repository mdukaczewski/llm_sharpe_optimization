# dml2: daily ML sleeve rebuilt for MOC execution, $3 per order on $0.4M

(Written by the coordinator from the lane's hand-back; numbers are in `log.txt` and `out/res_*.csv`.)

**Result:** rank-hysteresis ("band") books send ~4 orders/day instead of 64 and reach **fold-OOS 0.92-0.95**, vs **0.47**
for the existing `agents/ml` sleeve re-costed under the same execution (0.83 as booked, which ignores commission and uses a
next-open fill at 10 bps). Family fold-OOS of the 4 saved candidates: **0.92**.

## Execution / cost model (inside the series; folds.py adds no commission for family `dml`)
* Trade at the official close (MOC) of day t on information to the close of t-1 (order sent before the 15:50 cutoff).
* Impact per side `liq.auction_cost_bps(order, max(exp(dvc), 0.05 exp(ldv20)), vol20, c=0.3)`, floor 1 bp; names outside the
  liq universe 3 bps. Sensitivity at c = 0.1 / 1.0.
* $3 per order actually sent (after netting, SPY hedge included) on $0.4M; borrow 1%/yr on shorts.
* Exit-dated: holdings set at close t earn t+1 and are dated t+1. Holdings drift; a name trades on entry/exit/target change or >30% drift.

## Findings
| series (dev 2016-10..2026-04) | Sh | fold-OOS | orders/day | impact / comm. bps/day |
|---|---|---|---|---|
| agents/ml as booked | 0.73 | 0.83 | – | – / 0 |
| agents/ml re-costed (MOC, t-1 score, c=0.3, $3) | 0.35 | 0.47 | 64.3 | 1.34 / 4.82 |
| **dml2_b5b10_band_n15_r10_b20_H** | 0.91 | 0.95 | 3.7 | 1.37 / 0.27 |
| dml2_b5b10_band_n20_r10_b45_H | 0.90 | 0.92 | 4.2 | 1.04 / 0.32 |
| dml2_b5b10_band_n12_r1_b10_H | 0.82 | 0.94 | 7.3 | 3.61 / 0.55 |
| dml2_b5b10_band_n15_r5_b20_H | 0.78 | 0.92 | 5.2 | 1.92 / 0.39 |

* Commission was 78% of the old sleeve's cost (tranche adjustments of ~$3.2k pay ~9 bps each).
* Plateau: band books k = 10-20, rebalance 1-10 days, buffer 10-45 all give fold-OOS 0.8-0.95; k >= 25 is worse.
* Signal: h5 + h10 LightGBM models (126-day retrain), score smoothed 5 days. The 1-day model's edge is gone with the one-day delay.
* Minute-bar close features and the onml3 overnight prediction add little IC at 5-10 days and lower the books: not kept.
* SPY hedge capped at 0.5 beats unhedged in all 48 pairs by 0.03-0.1.
* Risks: vol 37-47% at 100/100 gross, maxDD -62% to -72% (inverse-vol weighting absorbs the vol, not the drawdown shape);
  first half weak (~0.3); 2026 fold negative. Correlation ~0 with mom, onml4 and SPY.

## Leakage notes
Panel cut at DEV_END on load; features at close t traded at close t+1; labels t+h <= cut-1; causal universe, lagged cost
inputs and hedge beta. In-sample choices: features and LightGBM params inherited from agents/ml (chosen on a span that included
the holdout); k / reb / buf / smoothing chosen on the full dev span (wide plateau); c = 0.3 assumed.
