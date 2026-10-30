# onml4 lane: overnight book execution routing at realistic cost (2026-10-30)

Written by the coordinator from the agent's hand-back (the agent's own write was blocked).  No LightGBM refit; everything
runs on the saved onml3 predictions (`agents/onml3/out/pred_f40_on.parquet` = p40, `pred_f26calx_q3.parquet` = p26).
Code `lib4.py`, `step0..4.py`, `final_cfg.txt`; logs `log.txt`, `step*.log`; tables `out/*.csv`.  Dates <= 2026-04-27.
Shared code: `liq.py` gained an additive 09:35 spread model (`FEATS["0935"]`, `add_spread_model`, `python3 liq.py add0935`;
grouped-CV R² 0.776, MAE 12.9 bps; the 0931 / 1001 / 1559 boosters are byte-identical; backup `out/spread_model_before.pkl`).

## 0. Cost bug in onml3 (fixed here)
The onml3 frame's `rv20` went through `rlib.clean`, which fills NaN with 0; 1,698 rows in 2017-18 (names with < 20 days of
history) then got the 1 bp auction floor and the cost-aware selection piled into those nearly free thin names.  Exact
reproduction of `onml3_f40_on_lam3_moo_k6` (corr 1.0000); with rv20 left NaN (row untradable) the series differs only in
2017-18.  Corrected (k6, lam 3, MOC + MOO, c = 0.3, $0.4M full Sharpe / fold-OOS): p40 1.82/1.70 -> **1.62/1.54**, p26
1.85/1.95 -> **1.77/1.84**.  The saved onml3 MOO candidates keep inflated 2017-18 prior Sharpes and should not be in the
chooser.

## 1. Edge decay by exit time (gross bps per trade from the official close, corrected p40 k6 cost-aware book)
| tier | pos% | official open | 09:31 | 09:35 | 10:01 | 10:31 | predicted hs 09:31 / 09:35 / 10:01 | MOO cost/side |
|---|---|---|---|---|---|---|---|---|
| all | 100 | 9.2 | 8.8 | 7.8 | 5.9 | 6.0 | 6.8 / 4.7 / 2.9 | 1.8 |
| 1-300 | 84 | 9.1 | 8.6 | 8.0 | 6.8 | 6.8 | 6.2 / 4.3 / 2.6 | 1.6 |
| 301-600 | 6 | 0.1 | 0.7 | -2.0 | -4.8 | -2.6 | 10.0 / 7.3 / 4.4 | 3.1 |
| 601-1500 | 7 | 9.5 | 7.8 | 6.3 | 3.1 | 1.4 | 7.8 / 5.3 / 3.2 | 2.2 |
| 1501-2000 | 3 | 26.8 | 31.0 | 23.7 | 10.2 | 12.9 | 13.7 / 10.3 / 6.9 | 4.4 |

Ranks 1-300, net per trade on the exit leg: MOO 7.5 bps; 09:35 3.7 (f = 1) / 5.9 (f = 0.5); 10:01 4.2 / 5.5.  Waiting for
narrower spreads loses less edge than it saves in spread, but still costs more than the opening auction at c = 0.3.
Ranks 1-300 decay from 10.7 bps (2017-21) to 7.0 (2022-26) to the open.

## 2. Routes (k6, lam 3, each route's own cost in its own selection; $0.4M full Sharpe / fold-OOS, p40 | p26)
| route | p40 | p26 |
|---|---|---|
| MOC -> MOO, c = 0.1 / 0.3 / 1.0 | 3.02/3.16, 1.62/1.54, -0.54/-0.51 | 2.37/2.49, 1.77/1.84, -0.58/-0.39 |
| MOC -> 09:35, f = 1 / 0.5 | 0.98/0.85, 1.61/1.54 | 1.05/1.19, 1.25/1.27 |
| MOC -> 10:01, f = 1 / 0.5 | 0.78/0.74, 1.03/0.92 | 0.79/0.92, 0.96/1.12 |
| MOC -> ½ MOO + ½ 10:01, c = 0.3 | 1.27/1.17 | 1.18/1.28 |
| MOC -> per-name exit route, c = 0.3 / 1.0 | 1.57/1.55, 0.23/0.19 | 1.40/1.51, 0.05/0.30 |
| 15:59:30 cross -> MOO, f = 1 | 1.70/1.63 | 1.66/1.80 |
| **entry route (MOC or 15:59:30 cross per name) -> MOO** | **1.73/1.66** | **1.82/1.94** |
| entry route -> exit route, c = 0.3 / c = 1.0 | 1.70/1.71, **1.27/1.27** | 1.46/1.72, 0.65/0.94 |

Per-name entry routing adds ~+0.1 (about 85% of entries become 15:59:30 crosses at ~1 bp).  No fixed continuous exit or
split beats MOO at c = 0.3; per-name exit routing adds nothing there but is the only thing that survives c = 1.0 (83-92%
of exits move to continuous fills).  The 15:59:30 cross is modelled at the 15:59-bar close with the hs1559 cost.

## 3. One book (fold-OOS at $0.4M, entry route -> MOO, c = 0.3)
| book | k6, lam 2 / 3 / 4 | k8 (lam 3) | k10 | k12 |
|---|---|---|---|---|
| p40 / p26 | 1.66 / 1.94 (lam 3) | 1.67 / 1.84 | 1.62 / 1.61 | 1.54 / 1.50 |
| **pR = per-day rank average of p40 and p26** | 2.06 / **1.88** / 1.92 | 1.71 | 1.73 | 1.53 |
| pR, routed exit | 1.93 / **2.07** / 1.78 | 1.72 | | |
| pR + 0.25 / 0.5 / 1 x z(cAx) | 1.93 / 1.88 / 1.16 (lam 3) | | | |
| p40 and p26 books netted 50/50 | 1.75 / 1.68 / 1.87 | 1.57 | 1.37 | 1.20 |

Plateau pR k6-8, lam 2-4: 1.71-2.07 (median ~1.85), +0.1-0.2 over p26 alone; lam 3 kept from onml3.  The rule signal adds
nothing; netting two books at fixed capital adds names (commission) and loses; onml3's 1.97 for the 50/50 mix used $0.8M.

## 4. Kept candidates (`cands/`; series net of their own route's cost at c = 0.3 / f = 1 and borrow, before commission)
| candidate | $0.4M Sh | fold-OOS | halves | maxDD | neg folds | c = 0.1 / c = 1.0 fold (reselected) |
|---|---|---|---|---|---|---|
| `onml4_pR_lam3_rte_moo_k6` | 1.89 | **1.88** | 2.48 / 1.43 | -21.5% | 4 | 2.85 / 0.69 |
| `onml4_pR_lam3_rte_moo_k8` | 1.69 | 1.71 | 2.54 / 0.99 | -21.5% | 3 | |
| `onml4_pR_lam3_rte_rtx_k6` | 1.99 | **2.07** | 2.48 / 1.53 | -16.7% | 3 | 2.66 / 1.03 |
| `onml4_pR_lam3_rte_rtx_k8` | 1.63 | 1.72 | 2.23 / 1.07 | -22.4% | 4 | |
| `onml4_p26_lam3_rte_moo_k6` | 1.82 | 1.94 | 2.29 / 1.44 | -20.3% | 2 | 2.69 / 0.68 |
| `onml4_p40_lam3_rte_moo_k6` | 1.73 | 1.66 | 2.90 / 0.79 | -26.0% | 6 | 3.06 / 0.49 |

Family fold-OOS 1.90.  By year (pR rte_moo k6, 2017-2026): 0.8 1.4 0.3 6.8 2.1 0.5 0.0 3.8 2.7 -0.9.  Tier mix: 83.5% of
positions in ranks 1-300 (net 6.6 bps per position), 2026 H1 negative for every variant.  Correlation with the onml3
books 0.7-0.9, with the 50-name rule book 0.45-0.49.

## 5. Leakage checklist
Selection uses only day-t information: saved predictions, lagged liquidity inputs, rv20 to the close of t (NaN rows
untradable), day-t predicted spreads, z(cAx).  t+1 spread predictions only charge the realised exit.  Calibration map and
decay ratios are expanding yearly (fitted on dates before Jan 1 minus 3 days; 2017 uses its own year and no fold touches
it).  The 15:59:30 cross carries the same few-minutes look-ahead the project accepts for MOC entries.  lam, k, the route
menu and the decay definition were chosen on the dev span; spread models are full-span calibrations.

## 6. Keep
`onml4_pR_lam3_rte_moo_k6` replaces the onml3 MOO books; `onml4_pR_lam3_rte_rtx_k6` as the hedge against a high auction
coefficient (the only book still ~1.0 at c = 1.0).  Dead: fixed continuous exits, MOO + 10:01 splits, split entries, cAx
as an extra ranking signal, netted two-book mix at fixed capital, k >= 10.  Next: measure c from auction prints vs the
pre-auction mid; NBBO-check the 15:59:30 cross that now carries ~80% of entries.
