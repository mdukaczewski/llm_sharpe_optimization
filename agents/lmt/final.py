#!/usr/bin/env python3
"""lmt final: candidates + sensitivities.  All series NET of spread / impact / borrow AND the $3-per-filled-order commission.
Overnight series are DECISION-dated (night after day t dated t, like onml4); intraday series are same-day.
  python3 final.py save=1"""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/lmt")
import books_on as bo, books_day as bd
from lmtlib import *
SAVE = "save=1" in sys.argv
REF = pd.read_csv(f"{R}/agents/onml4x/cands/onml4x_pR_lam3_rte_moo_k6.csv", index_col=0, parse_dates=True).iloc[:, 0] - 24 * 3 / CAP
pd.set_option("display.width", 250)
def corr(a, b): return round(pd.concat([a, b], axis=1).dropna().corr().iloc[0, 1], 2)
rows = []; S = {}
def rec(name, P, r, extra=None):
    m = metrics(r); row = dict(name=name, **per_trade(P), **m, corr_ref=corr(r, REF))
    if extra: row.update(extra)
    rows.append(row); S[name] = r; log(str({a: b for a, b in row.items() if a != "yrs"})); return row
# ---------------- overnight
H = bo.setup(); log("=== final (lmt)")
for c in [0.3, 1.0, 0.1]:
    r0, L = bo.route_book(H, 6, 3.0, "route", "moo", c, 1.0, ret_pos=True); d = r0.index      # selection re-run at this c
    for ent, ex in [("route", "moo"), ("pbid", "moo"), ("pbid", "p0932w30"), ("route", "p0932w30")]:
        for f in ([1.0, 0.75, 1.25] if (c == 0.3 and ex != "moo") else [1.0]):
            P = bo.execute(H, L, 6, ent, ex, c=c, f=f); r = daily(P, 6, dates=d)
            rec(f"on {ent}+{ex} c={c} f={f} (resel)", P, r, dict(exit_fill=round(P.loc[P.fill, "xfill"].mean() * 100, 1)))
            if c == 0.3 and f == 1.0 and ent == "pbid":
                if SAVE:
                    nm = {"moo": "lmt_on_pbid_moo_kSix", "p0932w30": "lmt_on_pbid_pask_kSix"}[ex]
                    note = (f"onml4x_pR_lam3_rte_moo_k6 positions (pR, L/S 6+6, cost-aware selection at c=0.3 with the +1.85 bps cross adverse selection); ENTRY = limit at the 15:59:30 bid "
                            f"(short: ask) resting into the closing auction, filled iff official close strictly through it (bid proxied by c388 x (1 - hs1559); exact on panel days: same 50% fill, "
                            f"filled +20.6 vs proxy +20.1 bps), fill at the close + MOC impact c=0.3, unfilled = no trade (~48% fill); EXIT = "
                            + ("MOO (c=0.3)" if ex == "moo" else "passive: sell limit at the ask / cover at the bid posted 09:32:00 (09:31-bar close x (1 +- hs0931)), strict fill within 30 min, else cross at the 10:01 bar close paying hs1001")
                            + f"; DECISION-dated; NET of impact/spread, borrow 1% AND $3 per filled order on $0.4M (do not charge commission again) | Sh {rows[-1]['sh']} fold-OOS {rows[-1]['fold']} halves {rows[-1]['h1']}/{rows[-1]['h2']} corr onml4x {rows[-1]['corr_ref']}")
                    rlib.save_cand("lmt", nm, r, note)
# ---------------- intraday
D = bd.frame(); dd = pd.DatetimeIndex(sorted(D.date.unique()))
for U, k in [(300, 4), (300, 8), (600, 4)]:
    L = bd.positions(D, k, U)
    for mode in ["x1", "x05", "pW2", "pW4px4"]:
        for rule in (["strict", "touch"] if mode.startswith("p") else ["strict"]):
            for s in ([1.0, 0.75, 1.25] if (mode.startswith("p") and rule == "strict" and U == 300 and k == 4) else [1.0]):
                for f in ([1.0, 0.5] if (mode.startswith("p") and rule == "strict" and s == 1.0 and U == 300 and k == 4) else [1.0]):
                    P = bd.execute(L, mode, s=s, rule=rule, f=f); r = daily(P, k, dates=dd)
                    # proxy-optimism haircut: study1 shows the proxy 1.7 bps too favourable on filled 09:31 buys (0.6 sells)
                    row = rec(f"day u{U} k{k} {mode} {rule} s={s} f={f}", P, r, dict(tiers=bd.tiers(P)))
                    if mode.startswith("p") and rule == "strict" and s == 1.0 and f == 1.0:
                        Ph = P.copy(); Ph["cost"] = Ph.cost + 1.5; rh = daily(Ph, k, dates=dd); mh = metrics(rh)
                        rows[-1].update(haircut15_sh=mh["sh"], haircut15_fold=mh["fold"])
                        if SAVE and U == 300 and k == 4:
                            nm = {"pW2": "lmt_day_u300_pbid_kFour", "pW4px4": "lmt_day_u300_pbid_pask_kFour"}[mode]
                            note = (f"dayml3 A1 quarterly prediction (pred_final_A1_Q), L/S top/bottom 4 within ranks <= 300, eq 1/4 per position; decision 09:32:00; ENTRY = buy limit at the bid "
                                    f"(short: sell at the ask), proxy 09:31-bar close x (1 -/+ hs0931), strict fill (bar low strictly below) within "
                                    + ("2 min, unfilled = no trade; EXIT = cross at the 10:01 bar close paying hs1001 (the user's rule)" if mode == "pW2" else
                                       "4 min, unfilled = no trade; EXIT = passive limit at the ask (cover at the bid) posted 10:01:00, strict within 4 min, else cross at the 10:04 bar close paying hs1001")
                                    + f"; same-day dated; NET of spread, borrow AND $3 per filled order on $0.4M (do not charge commission again) | Sh {rows[-1]['sh']} fold-OOS {rows[-1]['fold']} halves {rows[-1]['h1']}/{rows[-1]['h2']} "
                                    f"(+1.5 bps/trade proxy haircut: fold {mh['fold']}) MARGINAL: second half <= 0")
                            rlib.save_cand("lmt", nm, r, note)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/final.csv", index=False)
log("--- final\n" + T.drop(columns=["yrs", "tiers"], errors="ignore").to_string(index=False))
log("--- years\n" + T[["name", "yrs"]].to_string(index=False))
log("--- tiers (intraday)\n" + T.loc[T.tiers.notna(), ["name", "tiers"]].to_string(index=False))
# correlations among the saved-style series
keys = [k_ for k_ in S if ("c=0.3 f=1.0" in k_ and "pbid" in k_) or ("s=1.0 f=1.0" in k_ and "u300 k4" in k_ and "strict" in k_ and k_.split()[3].startswith("p"))]
M = pd.DataFrame({k_: S[k_] for k_ in keys}); M["onml4x_ref"] = REF.loc[:rlib.DEV_END]
log("--- correlation\n" + M.corr().round(2).to_string())
log("final DONE")
