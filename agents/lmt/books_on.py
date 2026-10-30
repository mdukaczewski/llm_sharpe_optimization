#!/usr/bin/env python3
"""lmt step 3 (overnight): passive entry / exit variants on the positions of the onml4x books.
Positions (date t = decision day, tk, side) from lib4.route_book with the onml4x cross adverse selection (+1.85 bps).
Entry options:
  route  : as onml4x (MOC at c=0.3 or 15:59:30 cross at hs1559 + 1.85 bps, cheaper per name)
  pbid   : the user's rule: a buy limit at the 15:59:30 bid (short: sell at the ask) that rests into the closing auction;
           filled iff official close < limit (strict) [touch: <=], at the official close + MOC impact; unfilled = no trade.
           The 15:59:30 bid is proxied by c388 x (1 -/+ s hs1559) (calibrated in study1; exact on panel days in study1).
Exit options (day t+1):
  moo    : market-on-open (c=0.3)
  x0931  : cross at the 09:31 bar close paying the predicted 09:31 half-spread (the user's 'sell also at bid')
  p{T}w{W}: passive: sell limit at the ask (cover: buy at the bid) posted at T (0932: mid = 09:31-bar close, hs0931; 1001: 10:00-bar
           close, hs1001), strict fill within W bars at the limit; unfilled -> cross at the close of the last window bar
           paying the predicted half-spread of that time.
Series: decision-dated (PnL of the night after day t is dated t), like onml4; net of spread / impact / borrow AND of the $3
per filled order commission (2 orders per filled position)."""
import sys; sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/lmt"); sys.path.insert(0, "/home/user/llm_sharpe_optimization/agents/onml4")
from lmtlib import *
import lib4
from lib4 import load_G, prep, route_book, route_costs, auction
ADV = 1.85
_rc = lib4.route_costs
def rc(G, k, c=0.3, f=1.0, ls=True):
    C = _rc(G, k, c, f, ls); C["x1559"] = (C["x1559"][0] + ADV, C["x1559"][1] + ADV); return C
lib4.route_costs = rc

def setup():
    G = load_G(); G["pR"] = (((G.p40 + G.p26) / 2).groupby(G.date).rank(pct=True) - 0.5).astype(np.float32)
    H = prep(G, "pR")
    E = ext(); NC = ["o_off", "c0", "c1", "c2", "c3", "c4", "c5", "c29", "c30", "c31", "c34", "c60"] + [c for c in E.columns if c.startswith(("lo_", "hi_"))] + ["hs0931", "hs0935", "hs1001"]
    N = next_day(E, NC); X = pd.concat([E[["date", "tk", "c388", "c389", "c_off", "hs1559"]], N], axis=1)
    H = H.merge(X, on=["date", "tk"], how="left", suffixes=("", "_e"))
    return H

FB = {("0932", 2): ("n_c3", "n_hs0931"), ("0932", 4): ("n_c5", "n_hs0935"), ("0932", 30): ("n_c31", "n_hs1001"),
      ("0931", 4): ("n_c4", "n_hs0935"), ("0931", 30): ("n_c30", "n_hs1001"),
      ("1001", 4): ("n_c34", "n_hs1001"), ("1001", 30): ("n_c60", "n_hs1001")}
MID = {"0931": ("n_c0", "n_hs0931"), "0932": ("n_c1", "n_hs0931"), "1001": ("n_c30", "n_hs1001")}

def execute(H, L, k, entry="route", exit="moo", s=1.0, rule="strict", c=0.3, f=1.0, s_close=1.0):
    """L: positions (index into H) from route_book.  Returns positions with ret (signed price return of a long), cost (bps,
    both legs), fill flag, norders, and diagnostics (exit_fill)."""
    h = H.loc[L.index]; w = L.w.values; lg = w > 0; n = len(h)
    C = rc(h, k, c=c, f=f)
    # ---- entry
    if entry == "route":
        e_cost = np.where(L.entry.values == "x1559", C["x1559"][1].values, C["moc"][1].values)
        e_adj = np.where(L.entry.values == "x1559", h.r_moc.fillna(0).values, 0.0); e_fill = np.ones(n, bool)
    elif entry == "moc":
        e_cost = C["moc"][1].values; e_adj = np.zeros(n); e_fill = np.ones(n, bool)
    elif entry == "pbid":
        Lm = np.where(lg, lim(h.c388.values, h.hs1559.values, "buy", s_close), lim(h.c388.values, h.hs1559.values, "sell", s_close))
        cl = h.c_off.values
        e_fill = np.where(lg, (cl < Lm) if rule == "strict" else (cl <= Lm + 1e-6), (cl > Lm) if rule == "strict" else (cl >= Lm - 1e-6))
        e_cost = C["moc"][1].values; e_adj = np.zeros(n)
    # ---- exit: price ratio vs the official open of t+1 (long direction), cost in bps
    oo = h.n_o_off.values
    if exit == "moo":
        x_ratio = np.ones(n); x_cost = C["moo"][1].values; x_fill = np.ones(n, bool)
    elif exit == "x0931":
        x_ratio = h.n_c1.values / oo; x_cost = f * h.n_hs0931.values; x_fill = np.ones(n, bool)
    else:
        T, W = exit[1:].split("w"); W = int(W); mc, hc = MID[T]; fbc, fbh = FB[(T, W)]
        # long exits by SELLING at the ask; short covers by BUYING at the bid
        Lx = np.where(lg, lim(h[mc].values, h[hc].values, "sell", s), lim(h[mc].values, h[hc].values, "buy", s))
        lo = h[f"n_lo_{T}_{W}"].values; hi = h[f"n_hi_{T}_{W}"].values
        x_fill = np.where(lg, filled(lo, hi, Lx, "sell", rule), filled(lo, hi, Lx, "buy", rule))
        x_ratio = np.where(x_fill, Lx, h[fbc].values) / oo
        x_cost = np.where(x_fill, 0.0, f * h[fbh].values)
    ret = (1 + h.y_oo.values) * (1 + e_adj) * x_ratio - 1
    P = pd.DataFrame({"date": h.date.values, "tk": h.tk.values, "tier": h.tier.values, "w": w, "ret": ret, "cost": e_cost + x_cost,
                      "fill": e_fill & np.isfinite(ret) & np.isfinite(e_cost + x_cost), "xfill": x_fill, "y_oo": h.y_oo.values}, index=L.index)
    P["norders"] = 2 * P.fill.astype(int)
    return P

if __name__ == "__main__":
    K = 6; H = setup(); log(f"=== books_on: H rows {len(H)}  next-day ext coverage {H.n_o_off.notna().mean():.3f}")
    REF = pd.read_csv(f"{R}/agents/onml4x/cands/onml4x_pR_lam3_rte_moo_k6.csv", index_col=0, parse_dates=True).iloc[:, 0]
    books = {"rte_moo": ("route", "moo", 3.0), "lam0_x0931": ("route", "x0931", 0.0)}
    res = []; S = {}
    for bk, (ent, ex, lam) in books.items():
        r0, L = route_book(H, K, lam, ent, ex, 0.3, 1.0, ret_pos=True)
        dates = r0.index
        variants = [("route", "moo"), ("route", "x0931"), ("route", "p0932w2"), ("route", "p0932w4"), ("route", "p0932w30"), ("route", "p0931w4"), ("route", "p0931w30"),
                    ("route", "p1001w4"), ("route", "p1001w30"), ("pbid", "moo"), ("pbid", "x0931"), ("pbid", "p0932w4"), ("pbid", "p0932w30"), ("moc", "moo")]
        for ent2, ex2 in variants:
            for rule in (["strict", "touch"] if (ent2 == "pbid" or ex2.startswith("p")) else ["strict"]):
                P = execute(H, L, K, ent2, ex2, rule=rule); r = daily(P, K, dates=dates); nm = f"{bk}|{ent2}|{ex2}|{rule}"; S[nm] = r
                m = metrics(r); pt = per_trade(P); xf = round(P.loc[P.fill, "xfill"].mean() * 100, 1)
                row = dict(book=bk, entry=ent2, exit=ex2, rule=rule, **pt, exit_fill=xf, **m, corr_ref=round(pd.concat([r, REF - 24 * 3 / CAP], axis=1).dropna().corr().iloc[0, 1], 2))
                res.append(row); log(str({a: b for a, b in row.items() if a != "yrs"}))
    T = pd.DataFrame(res); T.to_csv(f"{OUT}/books_on.csv", index=False)
    pd.set_option("display.width", 250); log("--- books_on\n" + T.drop(columns=["yrs"]).to_string(index=False))
    pd.to_pickle(S, f"{OUT}/books_on_series.pkl")
    log("books_on DONE")
