# onml6: the overnight book with a feasible information cutoff (coordinator write-up of the lane hand-back)

MOC / limit-on-close orders must be entered by 15:50 (NYSE) / 15:55 (Nasdaq). The onml3/onml4/onml4x books traded a 15:59
signal into the MOC. This lane rebuilds every feature at a 15:49:00 cutoff (bin 378 = the 15:48 bar, closing 15:49:00) and
re-runs the onml4 pipeline ($0.4M, $3 per filled order, k = 6, lam = 3, MOO exit, c = 0.3, +1.85 bps cross adverse selection).

## Leak found on the way
The "15:59" features already contained the official close of day t (`ilib.Min` sets c[390] to the panel close): zoc, zcc, cc5,
cc21, rv20, rv5_20, rngpos, rng, zvwap, spy_oc; ldv_x and vfirst_share used daily volume incl. the closing auction. The closing
print reverts overnight (rank IC of close / c389 - 1 with y_oo: -0.019 .. -0.035 by tier, vs the model's 0.036), so this was a
real look-ahead. About a fifth of the 1.80 -> 1.21 loss is this leak (cut 388 = features to 15:59:00: 1.70), the rest is the
last ten minutes. (ldv_x is a constant 5.0 after `rlib.clean` clipping; it was never a working feature.)

## Predictions (daily rank IC with y_oo)
| | original | cut 388 (15:59:00) | cut 378 (15:49:00) |
|---|---|---|---|
| f40 quarterly | 0.036 | 0.033 | 0.029 (3 seeds) |
| f26 quarterly | – | 0.030 | 0.025 (3 seeds) |

## Books (fold-OOS at c = 0.1 / 0.3 / 1.0, selection re-run at each c)
| book | fold-OOS | full Sh | halves | maxDD | corr onml4x |
|---|---|---|---|---|---|
| ref onml4x (15:59 + official close into the MOC; infeasible) | 2.78 / **1.80** / -0.02 | 1.80 | 2.62 / 1.17 | -20.9% | 1.00 |
| MOC only, pR cut 378 (`onml6_pR49_lam3_moc_moo_k6`) | 1.92 / **1.21** / -0.85 | 1.17 | 2.23 / 0.40 | -33.6% | 0.71 |
| MOC only, p40 cut 378 (`onml6_p40c49_lam3_moc_moo_k6`) | 2.26 / **1.29** / -0.93 | 1.32 | 2.30 / 0.53 | -35.3% | 0.71 |
| two-stage MOC 15:49 + 15:59:30 cross (`onml6_pR49x59_lam3_two_moo_k6`) | 1.92 / **1.22** / -0.22 | 1.19 | 2.17 / 0.42 | -36.3% | 0.72 |
| **LOC at the 15:49 bid, pR** (`onml6_pRcut_lamThree_locbid_moo_kSix`) | 2.02 / **1.62** / 0.99 | 1.60 | 1.98 / 1.17 | -15.8% | 0.49 |
| **LOC at the 15:49 bid, p40** (`onml6_pForty_lamThree_locbid_moo_kSix`) | 2.30 / **1.70** / 0.80 | 1.72 | 1.82 / 1.60 | -13.1% | 0.49 |

LOC rule: buy limit = round(c378 x (1 - hs49/1e4), 2) (sell: +), hs49 = the hs1559 spread model on inputs to 15:49:00; filled
iff the official close is strictly through the limit, at the close + MOC impact; unfilled = no trade, no commission. Fill ~45%;
filled gross 12.9 bps vs unfilled 3.8 (p40): the 15:49 -> close move reverts overnight. The kSix series include commission.

Verification on the real 15:49:00 NBBO (Polygon, 3,346 quotes, `quotes/`): proxy bid minus real bid median 0.00 bps; fill
agreement 95-97%; on the books' own positions (58 days) the proxy is slightly conservative (mean daily PnL 11.4 vs 12.4 bps).
Limit placement s = 0 / 1 / 1.2 x hs: 1.71 / 1.70 / 1.77. Seed noise ~0.15 (p40 LOC: 1.90 yearly, 1.85 quarterly 1 seed, 1.70 3 seeds).

xsrule3 at 15:49: fold-OOS 1.39 -> 0.42 (50 names), 1.08 -> 0.12 (150 names); not saved, family dropped.
Nasdaq 15:54 cutoff skipped: the only listing-exchange field is a 2026-10-29 snapshot (not causal).

In-sample choices: lam, k, limit placement, p40 over pR, all on the dev span; spread models are full-span calibrations.
