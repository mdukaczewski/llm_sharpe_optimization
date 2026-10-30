# lmt: passive limit orders at the touch (user's "post at the bid") — coordinator write-up of the lane hand-back

Data: real NBBO on the 59 quote-panel days <= DEV_END (109k name-days); fills decided from minute-bar low/high (strict = a print
through the limit, touch = at the limit, upper bound). Full 2017-2026 history via a minute-bar proxy (limit = prior bar close x
(1 -/+ predicted half-spread)), which matches the exact fill rate within 0-3 points and conditional returns within ~1-2 bps.
(The lane could not reach /v3/trades because POLYGON_API_KEY was not exported in its shell; the coordinator has since checked the
endpoint works with the key.) $3 per FILLED order on $0.4M inside the series.

## Adverse selection (exact, strict)
* Continuous session: fills are the losers. 09:31 buy at bid, 4 min: fill 45% (69% in ranks 1-300); filled mid move -31 bps,
  unfilled +29.5. Passive beats crossing by 30-40 bps but still loses to the mid.
* Closing auction: the opposite. A close printing below the 15:59:30 bid reverts: filled +15.2 bps to the open vs +4.6 unfilled.

## Books ($0.4M, $3 per filled order)
| book | fill | Sh | fold-OOS | maxDD | note |
|---|---|---|---|---|---|
| onml4x route / MOO (reference) | 100% | 1.79 | 1.80 | -21% | |
| onml4x entries, cross 09:31 exit (user's "sell at bid") | 99% | -0.10 | -0.05 | | |
| route / passive exit 09:32 ask 30 min, else 10:01 cross | 86% | 1.16 | 1.21 | -35% | |
| **limit at the 15:59:30 bid into the close / MOO** (`lmt_on_pbid_moo_kSix`) | 48% | 1.62 | 1.57 | -16% | corr 0.67 with onml4x; **timing infeasible, see below** |
| same entry / passive exit (`lmt_on_pbid_pask_kSix`) | 47% | 1.40 | 1.35 | -19% | fold-OOS 0.67 at c = 1.0 (MOO book -0.06) |
| dayml3 intraday, cross both legs | 100% | -3.59 | -3.82 | | |
| **dayml3, user's rule**: bid 2 min, unfilled = no trade, sell at bid 10:01 (`lmt_day_u300_pbid_kFour`) | 58% | 0.59 | 0.54 | | second half -0.08 |
| dayml3, passive both legs (`lmt_day_u300_pbid_pask_kFour`) | 65% | 0.72 | 0.67 | | second half -0.25 |

## Portfolio (dev folds, coordinator, `out/eval7_lmt_dev.txt`; base = onml4x + dml2 = 2.30)
| added | dev fold-OOS |
|---|---|
| lmtday | 2.42 |
| lmton | 2.38 |
| both | 2.49 |

## Caveats
* The close-limit books post at the 15:59:30 bid, but limit-on-close orders must be entered by 15:50 (NYSE) / 15:55 (Nasdaq):
  as built they are NOT executable. Same issue as the MOC entries of the overnight books (README caveat); being re-done at 15:49.
* lmtday: proxy ~1.5 bps optimistic (fold-OOS 0.26-0.37 with that haircut); every strict variant has a non-positive second half;
  sizes chosen on the dev span. Its portfolio gain is diversification of a weak, fragile sleeve.
