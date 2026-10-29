# etfmacro lane report (2026-10-29) — ETF / vol / macro time-series sleeves, dev data <= 2026-04-27

Files: `agents/etfmacro/elib.py` (helpers: daily ETF data cut at rlib.DEV_END, FRED VIX, split-safe overnight ETP legs from
minute bars), exploration scripts `x1_..x10_*.py` (all output appended to `log.txt`), `final.py` (regenerates ONLY the kept
candidates, honours STRAT_END via rlib.DEV_END, runtime ~30-60 s, run with `nohup python3 agents/etfmacro/final.py`),
candidates in `agents/etfmacro/cands/`.

## What was found: VIX-ETP decay is an overnight phenomenon

VIXY 2016-10..2026-04 (minute bars, split-adjusted): mean close->09:31 return **-11.2 bp/night (Sh -0.61)**, mean
09:32->close **-0.9 bp (Sh -0.04)**. Same for VXX/UVXY. Conditioning the overnight leg on the FRED VIX/VIX3M ratio of the
previous close: ratio < 0.85: -22.6 bp (Sh -1.82, halves -1.4/-2.3); 0.85-0.95: -21.7 bp (Sh -1.30); 0.95-1.0: -12 bp;
> 1.05: +75 bp (long-vol nights). Decisions taken on Fridays (hold Fri close -> Mon 09:31) lose (Sh -0.27, both halves
negative; the VIX calendar-day weekend effect). Holding the same regime close-to-close instead of overnight gives Sh 0.12
(the intraday session gives the carry back), exit at 10:31 gives 0.39 vs 1.0 at 09:31: the split matters.

## Kept candidates (all: 1.5 bps/side, fill at official close / 15:45 bar, exit at the 09:31 bar, regime from t-1 close)

| candidate | full Sh | FOLD-OOS | fold min / neg | halves | ret | vol | maxDD | corr onlgb1200q | corr vol_ens2 | corr SPY | time in mkt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vixon_r95 (short 0.5 VIXY, ratio<0.95, no Fri) | 0.96 | 0.94 | -2.36 / 3 of 16 | 1.20 / 0.69 | 11.9% | 12.3% | -18.4% | -0.01 | 0.15 | 0.27 | 63% |
| uvxon_r95 (short 0.25 UVXY, borrow 5%) | 1.05 | 1.03 | -2.15 / 3 | 1.29 / 0.76 | 10.0% | 9.5% | -13.0% | -0.02 | 0.14 | 0.27 | 63% |
| vixon_r95_1545 (entry 15:45 bar) | 0.97 | 1.04 | -1.56 / 3 | 0.95 / 0.99 | 12.6% | 12.9% | -18.0% | -0.02 | 0.14 | 0.27 | 63% |
| uvxon_r95_1545 | 1.02 | 1.09 | -1.60 / 3 | 0.99 / 1.05 | 10.1% | 10.0% | -13.6% | -0.01 | 0.14 | 0.26 | 63% |
| vixon_r95_ls (+ long 0.5 VIXY o/n if ratio>=0.95 & SPY s<-1) | 1.22 | 1.25 | -3.33 / 3 | 1.60 / 0.81 | 20.7% | 16.9% | -24.7% | 0.01 | -0.17 | 0.03 | 66% |

Family fold-OOS (`rlib.fold_oos` over the 5): **1.22** (folds mean 1.10, 3 negative: 2018H2 -0.4, 2022H1 -0.3, 2026Q1 -3.3),
stitched series corr 0.01 with onlgb1200q, -0.17 with vol_ens2, 0.02 with SPY. Core family without the `_ls` leg: 1.03.
The four core variants are 0.95-0.99 correlated with each other (same book, two instruments x two entry bars); selection
among them adds little noise. Worst single days of the stitched series: -5.8% (2021-11-26, Omicron, a Thursday-close
decision), -5.0% (2026-04-08), -4.8% (2020-06-11).

What I would keep: **uvxon_r95** (or vixon_r95 if UVXY borrow/locates are a problem: same book, 0.99 correlated) as the
core, plus **vixon_r95_ls** as a second, lower-correlation entry (its long leg is negatively correlated with the existing
vol sleeve, -0.40). The `_1545` entries have better second halves but the h1/h2 flip versus the close entry says the
15-minute difference is noise; I include them so the fold procedure can choose, not because I believe in them.

## Why it is (only) a ~1.0 sleeve and what did not work

* Sizing the short by the ratio (ramp), by VIX-RV10 (variance risk premium), by 1/VIX, by trailing overnight vol: no
  fold-OOS gain over the binary ratio<0.95 rule (0.90-1.09 vs 0.94-1.03). VRP>0 alone: fold-OOS 1.09, union with the
  ratio: 1.23 with the unlagged ratio but 0.89 once the regime is lagged properly (see leakage), so dropped.
* Hedging the SPY-overnight beta (VIXY o/n vs SPY o/n: corr -0.83, beta -3.4) with short SPY overnight destroys it (50%
  hedge: Sh 0.53, full: -0.50): SPY's own overnight drift is a third of the carry.  A causal long-VIXM close-to-close
  overlay (the existing spread idea) also hurts (0.62-0.85).
* Intraday short-VIXY legs (post-spike / flat-ratio / VIX-level / gap-conditioned): best 0.4-0.6 and half-inconsistent.
  Post-spike triggers (VIXY 5-day momentum > 5/10%, vol-of-vol buckets, VIX 25-35) on the overnight leg: 0.3-0.6.
* Tilting the short by SPY's intraday signal s: no gain. The long leg of `_ls` is the only use of s that survived.
* Sector-ETF cross-sectional overnight/intraday reversal (11 sectors + SMH/KRE/GDX, and 23 ETFs; s, z60, zcc, z30; k=2-4;
  y_on and y_cc): L/S <= 0.5 full-sample, second halves negative everywhere. Dead.
* Bond/gold/equity overnight splits: GLD overnight 5.4 bp/night (Sh 1.12 gross) but 3 bps of daily round-trip cost
  leaves ~0.5, below simply holding GLD (0.89); TLT overnight -2.1 bp, Fri->Mon -7.6 bp (TLT/IEF/HYG/EEM all negative over
  the weekend) but t-stats ~2 and cost-dominated. Prior-day-move conditioning adds nothing consistent. Dead.
* Cross-asset trend/carry weekly (20 ETFs, 1.5 bps): L/S time-series momentum 63/126/252d: -0.2..0.2; cross-sectional
  top5/bot5: ~0; long-only versions 0.75-1.0 fold-OOS but that is long beta in a bull decade (vol 4.5%). Not kept.
* Continuously sized daily VIXY/VIXM spread (ratio ramp, VRP ramp, ratio<0.95 filters) at 1.5 bps: all worse than the
  existing rv10>12% spread (fold-OOS 0.2-0.9 vs 0.83). Confirms the vol lane's "contango filters alone fail".

## Leakage / implementation concerns

* The FRED VIX "close" is the 16:15 value, so the regime uses **ratio(t-1)**; with the same-day ratio the book is 1.01 /
  halves 1.32/0.65 (reported for comparison only, not used).
* The 0.95 threshold, the Friday skip and the 0.5/0.25 sizes were chosen from full-sample bucket tables (x2/x4 logs);
  fold-OOS does not protect against that choice. Neighbours: ratio<0.90 fold-OOS 0.84, <1.0 0.84, ramp 0.97. The `_ls`
  long leg (ratio>=0.95 & s<-1) is the most data-mined piece: alone it is 0.76 (halves 1.05/0.45).
* Overnight legs are built from minute bars with the split factor (ilib.Min.sf): VIXY/UVXY reverse splits (4:1..10:1) are
  handled; rlib.ticker_frame's |y|>0.5 cap would have dropped UVXY's two worst nights (2020-03-06 +51%, 2024-08-02 +66%),
  so the candidates use elib.on_ret (no cap). Both nights are outside the regime (ratio(t-1) 1.22 and 0.971) — the latter
  by a hair: with ratio<1.0 the book takes -22%/-16% that night. Tail risk of overnight gaps is the real risk here.
* Fills assumed exact at the official close (or the 15:45 bar close) and at the 09:31 bar close; UVXY borrow 5%/yr and
  continuous availability assumed (realistically harder than VIXY). Cost 1.5 bps/side on both.
* UVXY changed from 2x to 1.5x on 2018-02-28 (0.25 UVXY ~ 0.375 VIXY-equivalent since); SVXY (not used) has a bogus -49%
  in ticker_frame y_on on 2017-07-14 (2:1 split not caught by the cap) — noted for other agents.
* rv10/VRP variants (not kept) also used t-1 values. No survivorship (ETFs fixed by name). The 2026 Q1 fold (-3.3) is the
  latest window and is a real drawdown of the regime rule, not a data issue (2026-03-06 and 2026-04-08 are two of the
  five worst nights).
